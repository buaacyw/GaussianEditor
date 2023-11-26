#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#


import torch
import torch.nn.functional as F
import numpy as np
from gaussiansplatting.utils.general_utils import (
    inverse_sigmoid,
    get_expon_lr_func,
    build_rotation,
)
from torch import nn
import os
from gaussiansplatting.utils.system_utils import mkdir_p
from plyfile import PlyData, PlyElement
from gaussiansplatting.utils.sh_utils import RGB2SH
from simple_knn._C import distCUDA2
from gaussiansplatting.utils.graphics_utils import BasicPointCloud
from gaussiansplatting.utils.general_utils import (
    strip_symmetric,
    build_scaling_rotation,
)
from gaussiansplatting.gaussian_renderer import camera2rasterizer

from gaussiansplatting.knn import K_nearest_neighbors
# from threestudio.utils.typing import Bool, Tensor

MAX_ANCHOR_WEIGHT = 10


class GaussianModel:
    def setup_functions(self):
        def build_covariance_from_scaling_rotation(scaling, scaling_modifier, rotation):
            L = build_scaling_rotation(scaling_modifier * scaling, rotation)
            actual_covariance = L @ L.transpose(1, 2)
            symm = strip_symmetric(actual_covariance)
            return symm

        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log

        self.covariance_activation = build_covariance_from_scaling_rotation

        self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = inverse_sigmoid

        self.rotation_activation = torch.nn.functional.normalize

    def __init__(
        self,
        sh_degree: int,
        anchor_weight_init_g0: float,
        anchor_weight_init: float,
        anchor_weight_multiplier: float,
    ):
        self.active_sh_degree = 0
        self.anchor_weight_init = anchor_weight_init
        self.anchor_weight_multiplier = anchor_weight_multiplier
        self._anchor_loss_schedule = torch.tensor(
            [anchor_weight_init_g0], device="cuda"
        )  # generation 0 begin from weight 0
        # self._anchor_loss_schedule[x] = y means weight y will be multiplied to the anchor loss of generation x
        self.max_sh_degree = sh_degree
        self._generation = torch.empty(0)  # begin from 0
        self._xyz = torch.empty(0)
        self._features_dc = torch.empty(0)
        self._features_rest = torch.empty(0)
        self._scaling = torch.empty(0)
        self._rotation = torch.empty(0)
        self._opacity = torch.empty(0)
        self.max_radii2D = torch.empty(0)
        self.xyz_gradient_accum = torch.empty(0)
        self.denom = torch.empty(0)
        self.optimizer = None
        self.percent_dense = 0
        self.spatial_lr_scale = 0
        self.setup_functions()
        self.anchor = {}
        self.localize=False

    def capture(self):
        return (
            self.active_sh_degree,
            self._xyz,
            self._features_dc,
            self._features_rest,
            self._scaling,
            self._rotation,
            self._opacity,
            self.max_radii2D,
            self.xyz_gradient_accum,
            self.denom,
            self.optimizer.state_dict(),
            self.spatial_lr_scale,
        )

    def update_anchor(self):
        self.anchor = dict(
            _xyz=self._xyz.detach().clone(),
            _features_dc=self._features_dc.detach().clone(),
            _features_rest=self._features_rest.detach().clone(),
            _scaling=self._scaling.detach().clone(),
            _rotation=self._rotation.detach().clone(),
            _opacity=self._opacity.detach().clone(),
        )

    def update_anchor_loss_schedule(self):
        for generation_idx, weight in enumerate(self._anchor_loss_schedule):
            self._anchor_loss_schedule[generation_idx] = min(
                self.anchor_weight_multiplier * weight, MAX_ANCHOR_WEIGHT
            )

        if self.generation_num > 1:
            assert self._anchor_loss_schedule[-1] == 0
            self._anchor_loss_schedule[-1] = self.anchor_weight_init
            # generation_0 begins with 1 anchor loss weight, generations after it begin with self.anchor_weight_init
            # the overall anchor loss can be modified through lambda_anchor_xxx
        self._anchor_loss_schedule = torch.cat(
            [self._anchor_loss_schedule, torch.tensor([0], device="cuda")]
        )  # firstborn generation won't be applied anchor loss

    # anchor loss
    def anchor_loss(self):
        out = {
            "loss_anchor_color": 0,
            "loss_anchor_geo": 0,
            "loss_anchor_opacity": 0,
            "loss_anchor_scale": 0,
        }

        target_generation = self._generation[self.mask]
        anchor_weight_list = torch.gather(
            self._anchor_loss_schedule, dim=0, index=target_generation
        )

        for key, value in self.anchor.items():
            delta = torch.nn.functional.mse_loss(
                getattr(self, key)[self.mask], value[self.mask], reduction="none"
            )
            if "feature" in key:
                delta *= anchor_weight_list[:, None, None]
            else:
                delta *= anchor_weight_list[:, None]
            delta = torch.mean(delta)
            if key in ["_xyz", "_rotation"]:
                out["loss_anchor_geo"] += delta
            elif key in ["_features_dc", "_features_rest"]:
                out["loss_anchor_color"] += delta
            elif key in ["_opacity"]:
                out["loss_anchor_opacity"] += delta
            elif key == "_scaling":
                out["loss_anchor_scale"] += delta
            else:
                raise
        return out

    def restore(self, model_args, training_args):
        (
            self.active_sh_degree,
            self._xyz,
            self._features_dc,
            self._features_rest,
            self._scaling,
            self._rotation,
            self._opacity,
            self.max_radii2D,
            xyz_gradient_accum,
            denom,
            opt_dict,
            self.spatial_lr_scale,
        ) = model_args
        # self.training_setup(training_args)
        self.xyz_gradient_accum = xyz_gradient_accum
        self.denom = denom
        self.optimizer.load_state_dict(opt_dict)

    def prune_with_mask(self, new_mask = None):
        self.prune_points(self.mask) # all the mask with value 1 are pruned
        if new_mask is not None:
            self.mask= new_mask
        else:
            self.mask[:] = 1 #all updatable
        self.remove_grad_mask()
        self.apply_grad_mask(self.mask)
        self.update_anchor()


    @property
    def generation_num(self):
        return len(self._anchor_loss_schedule)

    @property
    def get_scaling(self):
        if self.localize:
            return self.scaling_activation(self._scaling[self.mask])
        else:
            return self.scaling_activation(self._scaling)

    @property
    def get_rotation(self):
        if self.localize:
            return self.rotation_activation(self._rotation[self.mask])
        else:
            return self.rotation_activation(self._rotation)

    @property
    def get_xyz(self):
        if self.localize:
            return self._xyz[self.mask]
        else:
            return self._xyz

    @property
    def get_features(self):
        if self.localize:
            features_dc = self._features_dc[self.mask]
            features_rest = self._features_rest[self.mask]
        else:
            features_dc = self._features_dc
            features_rest = self._features_rest

        return torch.cat((features_dc, features_rest), dim=1)

    @property
    def get_opacity(self):
        if self.localize:
            return self.opacity_activation(self._opacity[self.mask])
        else:
            return self.opacity_activation(self._opacity)

    def get_covariance(self, scaling_modifier=1):
        if self.localize:
            return self.covariance_activation(
                self.get_scaling[self.mask], scaling_modifier, self._rotation[self.mask]
            )
        else:
            return self.covariance_activation(
                self.get_scaling, scaling_modifier, self._rotation
            )


    def oneupSHdegree(self):
        if self.active_sh_degree < self.max_sh_degree:
            self.active_sh_degree += 1

    def create_from_pcd(self, pcd: BasicPointCloud, spatial_lr_scale: float):
        self.spatial_lr_scale = spatial_lr_scale
        fused_point_cloud = torch.tensor(np.asarray(pcd.points)).float().cuda()
        fused_color = RGB2SH(torch.tensor(np.asarray(pcd.colors)).float().cuda())
        features = (
            torch.zeros((fused_color.shape[0], 3, (self.max_sh_degree + 1) ** 2))
            .float()
            .cuda()
        )
        features[:, :3, 0] = fused_color
        features[:, 3:, 1:] = 0.0

        print("Number of points at initialisation : ", fused_point_cloud.shape[0])

        dist2 = torch.clamp_min(
            distCUDA2(torch.from_numpy(np.asarray(pcd.points)).float().cuda()),
            0.0000001,
        )
        scales = torch.log(torch.sqrt(dist2))[..., None].repeat(1, 3)
        rots = torch.zeros((fused_point_cloud.shape[0], 4), device="cuda")
        rots[:, 0] = 1

        # opacities = inverse_sigmoid(
        #     1.0
        #     * torch.ones(
        #         (fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda"
        #     )
        # )
        opacities = 1.0   * torch.ones(
                (fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda"
            )

        self._xyz = nn.Parameter(fused_point_cloud.requires_grad_(True))
        self._features_dc = nn.Parameter(
            features[:, :, 0:1].transpose(1, 2).contiguous().requires_grad_(True)
        )
        self._features_rest = nn.Parameter(
            features[:, :, 1:].transpose(1, 2).contiguous().requires_grad_(True)
        )
        self._scaling = nn.Parameter(scales.requires_grad_(True))
        self._rotation = nn.Parameter(rots.requires_grad_(True))
        self._opacity = nn.Parameter(opacities.requires_grad_(True))
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")
        self.active_sh_degree
        self._generation = torch.zeros(
            self._opacity.shape[0],
            dtype=torch.int64,
            device="cuda",
            requires_grad=False,
        )  # generation list, begin from zero
        self.set_mask(
            torch.ones(
                self._opacity.shape[0],
                dtype=torch.bool,
                device="cuda",
                requires_grad=False,
            )
        )
        self.apply_grad_mask(self.mask)

        self.update_anchor()


    def training_setup(self, training_args):
        self.percent_dense = training_args.percent_dense
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")

        l = [
            {
                "params": [self._xyz],
                "lr": training_args.position_lr_init * self.spatial_lr_scale,
                "name": "xyz",
            },
            {
                "params": [self._features_dc],
                "lr": training_args.feature_lr,
                "name": "f_dc",
            },
            {
                "params": [self._features_rest],
                "lr": training_args.feature_lr / 20.0,
                "name": "f_rest",
            },
            {
                "params": [self._opacity],
                "lr": training_args.opacity_lr,
                "name": "opacity",
            },
            {
                "params": [self._scaling],
                "lr": training_args.scaling_lr,
                "name": "scaling",
            },
            {
                "params": [self._rotation],
                "lr": training_args.rotation_lr,
                "name": "rotation",
            },
        ]
        self.params_list = l
        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
        self.xyz_scheduler_args = get_expon_lr_func(
            lr_init=training_args.position_lr_init * self.spatial_lr_scale,
            lr_final=training_args.position_lr_final * self.spatial_lr_scale,
            lr_delay_mult=training_args.position_lr_delay_mult,
            max_steps=training_args.position_lr_max_steps,
        )

    def update_learning_rate(self, iteration):
        """Learning rate scheduling per step"""
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "xyz":
                lr = self.xyz_scheduler_args(iteration)
                param_group["lr"] = lr
                # print("xyz lr : ", lr)
            # if param_group["name"] == "f_dc":
            #     # import pdb; pdb.set_trace()
            #     param_group['lr'] = param_group['lr'] * ((0.5) ** (1.0 / 1200.0))
            # if param_group["name"] == "f_rest":
            #     # import pdb; pdb.set_trace()
            #     param_group['lr'] = param_group['lr'] * ((0.5) ** (1.0 / 1200.0))

    def construct_list_of_attributes(self):
        l = ["x", "y", "z", "nx", "ny", "nz"]
        # All channels except the 3 DC
        for i in range(self._features_dc.shape[1] * self._features_dc.shape[2]):
            l.append("f_dc_{}".format(i))
        for i in range(self._features_rest.shape[1] * self._features_rest.shape[2]):
            l.append("f_rest_{}".format(i))
        l.append("opacity")
        for i in range(self._scaling.shape[1]):
            l.append("scale_{}".format(i))
        for i in range(self._rotation.shape[1]):
            l.append("rot_{}".format(i))
        return l

    def save_ply(self, path):
        mkdir_p(os.path.dirname(path))

        xyz = self._xyz.detach().cpu().numpy()
        normals = np.zeros_like(xyz)
        f_dc = (
            self._features_dc.detach()
            .transpose(1, 2)
            .flatten(start_dim=1)
            .contiguous()
            .cpu()
            .numpy()
        )
        f_rest = (
            self._features_rest.detach()
            .transpose(1, 2)
            .flatten(start_dim=1)
            .contiguous()
            .cpu()
            .numpy()
        )
        opacities = self._opacity.detach().cpu().numpy()
        scale = self._scaling.detach().cpu().numpy()
        rotation = self._rotation.detach().cpu().numpy()

        dtype_full = [
            (attribute, "f4") for attribute in self.construct_list_of_attributes()
        ]

        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        attributes = np.concatenate(
            (xyz, normals, f_dc, f_rest, opacities, scale, rotation), axis=1
        )
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, "vertex")
        PlyData([el]).write(path)

    def reset_opacity(self):
        opacities_new = inverse_sigmoid(
            torch.min(self.get_opacity, torch.ones_like(self.get_opacity) * 0.01)
        )
        optimizable_tensors = self.replace_tensor_to_optimizer(opacities_new, "opacity")
        self._opacity = optimizable_tensors["opacity"]

    # load ply
    def load_ply(self, path):
        plydata = PlyData.read(path)

        xyz = np.stack(
            (
                np.asarray(plydata.elements[0]["x"]),
                np.asarray(plydata.elements[0]["y"]),
                np.asarray(plydata.elements[0]["z"]),
            ),
            axis=1,
        )
        opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]

        features_dc = np.zeros((xyz.shape[0], 3, 1))
        features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
        features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
        features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])

        extra_f_names = [
            p.name
            for p in plydata.elements[0].properties
            if p.name.startswith("f_rest_")
        ]
        extra_f_names = sorted(extra_f_names, key=lambda x: int(x.split("_")[-1]))
        # assert len(extra_f_names)==3*(self.max_sh_degree + 1) ** 2 - 3
        self.max_sh_degree = int(((len(extra_f_names) + 3) / 3) ** 0.5 - 1)
        features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
        for idx, attr_name in enumerate(extra_f_names):
            features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
        # Reshape (P,F*SH_coeffs) to (P, F, SH_coeffs except DC)
        features_extra = features_extra.reshape(
            (features_extra.shape[0], 3, (self.max_sh_degree + 1) ** 2 - 1)
        )

        scale_names = [
            p.name
            for p in plydata.elements[0].properties
            if p.name.startswith("scale_")
        ]
        scale_names = sorted(scale_names, key=lambda x: int(x.split("_")[-1]))
        scales = np.zeros((xyz.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

        rot_names = [
            p.name for p in plydata.elements[0].properties if p.name.startswith("rot")
        ]
        rot_names = sorted(rot_names, key=lambda x: int(x.split("_")[-1]))
        rots = np.zeros((xyz.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name])

        self._xyz = nn.Parameter(
            torch.tensor(xyz, dtype=torch.float, device="cuda").requires_grad_(True)
        )
        self._features_dc = nn.Parameter(
            torch.tensor(features_dc, dtype=torch.float, device="cuda")
            .transpose(1, 2)
            .contiguous()
            .requires_grad_(True)
        )
        self._features_rest = nn.Parameter(
            torch.tensor(features_extra, dtype=torch.float, device="cuda")
            .transpose(1, 2)
            .contiguous()
            .requires_grad_(True)
        )
        self._opacity = nn.Parameter(
            torch.tensor(opacities, dtype=torch.float, device="cuda").requires_grad_(
                True
            )
        )
        self._scaling = nn.Parameter(
            torch.tensor(scales, dtype=torch.float, device="cuda").requires_grad_(True)
        )
        self._rotation = nn.Parameter(
            torch.tensor(rots, dtype=torch.float, device="cuda").requires_grad_(True)
        )

        self.active_sh_degree = self.max_sh_degree
        self._generation = torch.zeros(
            self._opacity.shape[0],
            dtype=torch.int64,
            device="cuda",
            requires_grad=False,
        )  # generation list, begin from zero
        self.set_mask(
            torch.ones(
                self._opacity.shape[0],
                dtype=torch.bool,
                device="cuda",
                requires_grad=False,
            )
        )
        self.apply_grad_mask(self.mask)

        self.update_anchor()

    def replace_tensor_to_optimizer(self, tensor, name):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] == name:
                stored_state = self.optimizer.state.get(group["params"][0], None)
                stored_state["exp_avg"] = torch.zeros_like(tensor)
                stored_state["exp_avg_sq"] = torch.zeros_like(tensor)

                del self.optimizer.state[group["params"][0]]
                group["params"][0] = nn.Parameter(tensor.requires_grad_(True))
                self.optimizer.state[group["params"][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def _prune_optimizer(self, mask):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            stored_state = self.optimizer.state.get(group["params"][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = stored_state["exp_avg"][mask]
                stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask]

                del self.optimizer.state[group["params"][0]]
                group["params"][0] = nn.Parameter(
                    (group["params"][0][mask].requires_grad_(True))
                )
                self.optimizer.state[group["params"][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(
                    group["params"][0][mask].requires_grad_(True)
                )
                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def prune_points(self, mask):
        valid_points_mask = ~mask
        optimizable_tensors = self._prune_optimizer(valid_points_mask)

        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

        self.xyz_gradient_accum = self.xyz_gradient_accum[valid_points_mask]

        self.denom = self.denom[valid_points_mask]
        self.max_radii2D = self.max_radii2D[valid_points_mask]

        self.mask = self.mask[valid_points_mask]
        self._generation = self._generation[valid_points_mask]

    def cat_tensors_to_optimizer(self, tensors_dict):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            assert len(group["params"]) == 1
            extension_tensor = tensors_dict[group["name"]]
            stored_state = self.optimizer.state.get(group["params"][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = torch.cat(
                    (stored_state["exp_avg"], torch.zeros_like(extension_tensor)), dim=0
                )
                stored_state["exp_avg_sq"] = torch.cat(
                    (stored_state["exp_avg_sq"], torch.zeros_like(extension_tensor)),
                    dim=0,
                )

                del self.optimizer.state[group["params"][0]]
                group["params"][0] = nn.Parameter(
                    torch.cat(
                        (group["params"][0], extension_tensor), dim=0
                    ).requires_grad_(True)
                )
                self.optimizer.state[group["params"][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(
                    torch.cat(
                        (group["params"][0], extension_tensor), dim=0
                    ).requires_grad_(True)
                )
                optimizable_tensors[group["name"]] = group["params"][0]

        return optimizable_tensors

    def densification_postfix(
        self,
        new_xyz,
        new_features_dc,
        new_features_rest,
        new_opacities,
        new_scaling,
        new_rotation,
    ):
        d = {
            "xyz": new_xyz,
            "f_dc": new_features_dc,
            "f_rest": new_features_rest,
            "opacity": new_opacities,
            "scaling": new_scaling,
            "rotation": new_rotation,
        }

        optimizable_tensors = self.cat_tensors_to_optimizer(d)
        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

    def densify_and_split(self, grads, grad_threshold, scene_extent, N=2):
        n_init_points = self.get_xyz.shape[0]
        # Extract points that satisfy the gradient condition
        padded_grad = torch.zeros((n_init_points), device="cuda")
        padded_grad[: grads.shape[0]] = grads.squeeze()
        selected_pts_mask = torch.where(padded_grad >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(
            selected_pts_mask,
            torch.max(self.get_scaling, dim=1).values
            > self.percent_dense * scene_extent,
        )

        stds = self.get_scaling[selected_pts_mask].repeat(N, 1)
        means = torch.zeros((stds.size(0), 3), device="cuda")
        samples = torch.normal(mean=means, std=stds)
        rots = build_rotation(self._rotation[selected_pts_mask]).repeat(N, 1, 1)
        new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self.get_xyz[
            selected_pts_mask
        ].repeat(N, 1)
        new_scaling = self.scaling_inverse_activation(
            self.get_scaling[selected_pts_mask].repeat(N, 1) / (0.8 * N)
        )
        new_rotation = self._rotation[selected_pts_mask].repeat(N, 1)
        new_features_dc = self._features_dc[selected_pts_mask].repeat(N, 1, 1)
        new_features_rest = self._features_rest[selected_pts_mask].repeat(N, 1, 1)
        new_opacity = self._opacity[selected_pts_mask].repeat(N, 1)

        self.densification_postfix(
            new_xyz,
            new_features_dc,
            new_features_rest,
            new_opacity,
            new_scaling,
            new_rotation,
        )

        new_mask = torch.cat([self.mask[selected_pts_mask]] * N, dim=0)
        self.mask = torch.cat([self.mask, new_mask], dim=0)

        new_generation = torch.zeros_like(
            selected_pts_mask.nonzero()[:, 0], dtype=torch.int64
        )
        new_generation[:] = self.generation_num
        new_generation = torch.cat([new_generation] * N, dim=0)
        self._generation = torch.cat([self._generation, new_generation])
        assert self._generation.shape == self.mask.shape

        prune_filter = torch.cat(
            (
                selected_pts_mask,
                torch.zeros(N * selected_pts_mask.sum(), device="cuda", dtype=bool),
            )
        )

        self.prune_points(prune_filter)
        assert self._generation.shape == self.mask.shape

    def densify_and_clone(self, grads, grad_threshold, scene_extent):
        # Extract points that satisfy the gradient condition
        selected_pts_mask = torch.where(
            torch.norm(grads, dim=-1) >= grad_threshold, True, False
        )
        selected_pts_mask = torch.logical_and(
            selected_pts_mask,
            torch.max(self.get_scaling, dim=1).values
            <= self.percent_dense * scene_extent,
        )

        new_xyz = self._xyz[selected_pts_mask]
        new_features_dc = self._features_dc[selected_pts_mask]
        new_features_rest = self._features_rest[selected_pts_mask]
        new_opacities = self._opacity[selected_pts_mask]
        new_scaling = self._scaling[selected_pts_mask]
        new_rotation = self._rotation[selected_pts_mask]

        self.densification_postfix(
            new_xyz,
            new_features_dc,
            new_features_rest,
            new_opacities,
            new_scaling,
            new_rotation,
        )
        assert (
            len(torch.nonzero(self.mask[selected_pts_mask] == 0)) == 0
        ), "nontarget area should not be densified"
        # selected_pts_mask [points_num,]
        self.mask = torch.cat([self.mask, self.mask[selected_pts_mask]], dim=0)
        new_generation = torch.zeros_like(
            selected_pts_mask.nonzero()[:, 0], dtype=torch.int64
        )
        new_generation[:] = self.generation_num
        self._generation = torch.cat([self._generation, new_generation])
        # firstborn generation won't be applied anchor loss

    # densify
    def densify_and_prune(
        self, max_grad, max_densify_percent, min_opacity, extent, max_screen_size
    ):
        grads = self.xyz_gradient_accum / self.denom
        grads[grads.isnan()] = 0.0
        grads[~self.mask] = 0.0  # the hook didn't set grads to zero here. Bug fixed
        if max_densify_percent < 1:
            valid_percent = len(grads.nonzero()) * max_densify_percent / grads.shape[0]
            thresold_value = torch.quantile(grads, 1 - valid_percent)
            grads[grads < thresold_value] = 0.0
        # grads

        before = self.get_xyz.shape[0]
        self.densify_and_clone(grads, max_grad, extent)
        clone = self.get_xyz.shape[0]

        self.densify_and_split(grads, max_grad, extent)
        split = self.get_xyz.shape[0]

        prune_mask = (self.get_opacity < min_opacity).squeeze()
        if max_screen_size:
            big_points_vs = self.max_radii2D > max_screen_size
            big_points_ws = self.get_scaling.max(dim=1).values > 0.1 * extent
            prune_mask = torch.logical_or(
                torch.logical_or(prune_mask, big_points_vs), big_points_ws
            )
        prune_mask = torch.logical_and(prune_mask, self.mask)  # fix bug
        self.prune_points(prune_mask)
        prune = self.get_xyz.shape[0]
        assert self._generation.shape == self.mask.shape

        print(
            f"Generation_{self.generation_num}: before: {before} - clone: {clone} - split: {split} - prune: {prune} "
        )

        self.remove_grad_mask()
        self.apply_grad_mask(self.mask)

        self.update_anchor()
        self.update_anchor_loss_schedule()

        torch.cuda.empty_cache()

    def add_densification_stats(self, viewspace_point_tensor, update_filter):
        self.xyz_gradient_accum[update_filter] += torch.norm(
            viewspace_point_tensor[update_filter, :2], dim=-1, keepdim=True
        )
        self.denom[update_filter] += 1

    def apply_weights(self, camera, weights, weights_cnt, image_weights):
        rasterizer = camera2rasterizer(
            camera, torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32, device="cuda")
        )
        rasterizer.apply_weights(
            self.get_xyz,
            None,
            self.get_opacity,
            None,
            weights,
            self.get_scaling,
            self.get_rotation,
            None,
            weights_cnt,
            image_weights,
        )

    def set_mask(self, mask):
        self.mask = mask

    def apply_grad_mask(self, mask):
        assert self.mask.shape[0] == self._xyz.shape[0]
        self.set_mask(mask)

        def hook(grad):
            final_grad = grad * (
                self.mask[:, None] if grad.ndim == 2 else self.mask[:, None, None]
            )
            # print(final_grad.abs().max())
            # print(final_grad.abs().mean())
            return final_grad

        fields = ["_xyz", "_features_dc", "_features_rest", "_opacity", "_scaling"]

        self.hooks = []

        for field in fields:
            this_field = getattr(self, field)
            assert this_field.is_leaf and this_field.requires_grad
            self.hooks.append(this_field.register_hook(hook))

    def remove_grad_mask(self):
        # assert hasattr(self, "hooks")
        for hook in self.hooks:
            hook.remove()

        del self.hooks

    def move_penalty(self):
        if not hasattr(self, "prev_xyz") or self.prev_xyz is None:
            self.prev_xyz = self.get_xyz.data.clone()
            return 0.0

        loss = F.mse_loss(self.get_xyz, self.prev_xyz)

        self.prev_xyz = self.get_xyz.data.clone()

        return loss

    def alpha_penalty(self):
        xyz = self.get_xyz.detach()
        alpha = self.get_opacity

        return torch.mean(xyz.norm(dim=-1, keepdim=True) * alpha)

    def scale_penalty(self):
        scale = self.get_scaling
        volume = scale.prod(dim=-1)

        return torch.sum(volume)

    def get_near_gaussians_by_mask(
        self, mask, dist_thresh: float = 0.1
    ):
        mask = mask.squeeze()
        object_xyz = self._xyz[mask]
        remaining_xyz = self._xyz[~mask]

        bbox_3D = torch.stack([torch.quantile(object_xyz[:,0],0.03), torch.quantile(object_xyz[:,0],0.97),
                               torch.quantile(object_xyz[:,1],0.03), torch.quantile(object_xyz[:,1],0.97),
                               torch.quantile(object_xyz[:,2],0.03), torch.quantile(object_xyz[:,2],0.97)])
        scale = bbox_3D[1::2] - bbox_3D[0::2]
        mid = (bbox_3D[1::2] + bbox_3D[0::2])/2
        scale *= 1.3
        bbox_3D[0::2] = mid - scale/2
        bbox_3D[1::2] = mid + scale/2

        in_bbox = (remaining_xyz[:, 0] >= bbox_3D[0]) & (remaining_xyz[:, 0] <= bbox_3D[1]) & \
                  (remaining_xyz[:, 1] >= bbox_3D[2]) & (remaining_xyz[:, 1] <= bbox_3D[3]) & \
                  (remaining_xyz[:, 2] >= bbox_3D[4]) & (remaining_xyz[:, 2] <= bbox_3D[5])
        in_box_remaining_xyz = remaining_xyz[in_bbox]

        _, _, nn_dist = K_nearest_neighbors(
            object_xyz, 1, query=in_box_remaining_xyz, return_dist=True
        )
        nn_dist = nn_dist.squeeze()
        valid_mask = (nn_dist <= dist_thresh)

        mask_to_update = torch.zeros_like(remaining_xyz[:,0],dtype=torch.bool)
        true_indices = torch.nonzero(in_bbox)
        true_indices = true_indices[valid_mask,0]
        mask_to_update[true_indices]=True
        # valid_remaining_idx = remaining_idx[valid_mask]

        return mask_to_update

    def concat_gaussians(self, another_gaussian):
        # return a mask
        new_xyz = another_gaussian._xyz
        new_features_dc = another_gaussian._features_dc
        new_features_rest = another_gaussian._features_rest
        new_opacities = another_gaussian._opacity
        new_scaling = another_gaussian._scaling
        new_rotation = another_gaussian._rotation
        self.densification_postfix(
            new_xyz,
            new_features_dc,
            new_features_rest,
            new_opacities,
            new_scaling,
            new_rotation,
        )
        self.mask = ~self.mask
        self.mask = torch.cat([self.mask, torch.ones_like(new_opacities[:,0],dtype=torch.bool)], dim=0)
        self.remove_grad_mask()
        self.apply_grad_mask(self.mask)

        self._generation = torch.cat([self._generation, torch.zeros_like(new_opacities[:,0],dtype=torch.int64)], dim=0)
        self.update_anchor()