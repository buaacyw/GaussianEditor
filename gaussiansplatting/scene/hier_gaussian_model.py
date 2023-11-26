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
import numpy as np
from gaussiansplatting.utils.general_utils import inverse_sigmoid, get_expon_lr_func, build_rotation
from torch import nn
import os
from gaussiansplatting.utils.system_utils import mkdir_p
from plyfile import PlyData, PlyElement
from gaussiansplatting.utils.sh_utils import RGB2SH
from simple_knn._C import distCUDA2
from gaussiansplatting.utils.graphics_utils import BasicPointCloud
from gaussiansplatting.utils.general_utils import strip_symmetric, build_scaling_rotation
from gaussiansplatting.scene.gaussian_model import GaussianModel

class HierarchicalGaussianModel(GaussianModel):
    def __init__(self):
        super().__init__(sh_degree =0)

    def training_setup(self, training_args):
        self.percent_dense = training_args.percent_dense
        self.xyz_gradient_accum = torch.zeros((self.get_target_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_target_xyz.shape[0], 1), device="cuda")

        l = [
            {'params': [self.target_xyz], 'lr': training_args.position_lr_init * self.spatial_lr_scale , "name": "xyz"},
            {'params': [self.target_features_dc], 'lr': training_args.feature_lr, "name": "f_dc"},
            {'params': [self.target_features_rest], 'lr': training_args.feature_lr / 20.0, "name": "f_rest"},
            {'params': [self.target_opacity], 'lr': training_args.opacity_lr, "name": "opacity"},
            {'params': [self.target_scaling], 'lr': training_args.scaling_lr, "name": "scaling"},
            {'params': [self.target_rotation], 'lr': training_args.rotation_lr, "name": "rotation"}
        ]
        self.params_list = l
        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
        self.xyz_scheduler_args = get_expon_lr_func(lr_init=training_args.position_lr_init*self.spatial_lr_scale,
                                                    lr_final=training_args.position_lr_final*self.spatial_lr_scale,
                                                    lr_delay_mult=training_args.position_lr_delay_mult,
                                                    max_steps=training_args.position_lr_max_steps)

    @property
    def get_target_scaling(self):
        return self.scaling_activation(self.target_scaling)

    @property
    def get_target_rotation(self):
        return self.rotation_activation(self.target_rotation)

    @property
    def get_target_xyz(self):
        return self.target_xyz

    @property
    def get_target_features(self):
        features_dc = self.target_features_dc
        features_rest = self.target_features_rest
        return torch.cat((features_dc, features_rest), dim=1)

    @property
    def get_target_opacity(self):
        return self.opacity_activation(self.target_opacity)

    def reset_opacity(self):
        opacities_new = inverse_sigmoid(torch.min(self.get_target_opacity, torch.ones_like(self.get_target_opacity)*0.01))
        optimizable_tensors = self.replace_tensor_to_optimizer(opacities_new, "opacity")
        self.target_opacity = optimizable_tensors["opacity"]
        self.update_whole_scene_property()

    def load_ply(self, path):
        plydata = PlyData.read(path)

        xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                        np.asarray(plydata.elements[0]["y"]),
                        np.asarray(plydata.elements[0]["z"])),  axis=1)
        opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]

        features_dc = np.zeros((xyz.shape[0], 3, 1))
        features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
        features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
        features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])

        extra_f_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_")]
        extra_f_names = sorted(extra_f_names, key = lambda x: int(x.split('_')[-1]))
        # assert len(extra_f_names)==3*(self.max_sh_degree + 1) ** 2 - 3
        self.max_sh_degree = int(((len(extra_f_names)+3)/3) ** 0.5 -1)
        features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
        for idx, attr_name in enumerate(extra_f_names):
            features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
        # Reshape (P,F*SH_coeffs) to (P, F, SH_coeffs except DC)
        features_extra = features_extra.reshape((features_extra.shape[0], 3, (self.max_sh_degree + 1) ** 2 - 1))

        scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
        scale_names = sorted(scale_names, key = lambda x: int(x.split('_')[-1]))
        scales = np.zeros((xyz.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

        rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
        rot_names = sorted(rot_names, key = lambda x: int(x.split('_')[-1]))
        rots = np.zeros((xyz.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name])
        # dummy
        semantic_mask = np.zeros_like(torch.tensor(opacities),dtype=bool)[:,0]
        semantic_mask[:100000] = 1
        if semantic_mask is None:
            semantic_mask = np.ones_like(torch.tensor(opacities), dtype=bool)[:, 0]

        self.target_xyz = nn.Parameter(torch.tensor(xyz[semantic_mask], dtype=torch.float, device="cuda").requires_grad_(True))
        self.target_features_dc = nn.Parameter(torch.tensor(features_dc[semantic_mask], dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self.target_features_rest = nn.Parameter(torch.tensor(features_extra[semantic_mask], dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self.target_opacity = nn.Parameter(torch.tensor(opacities[semantic_mask], dtype=torch.float, device="cuda").requires_grad_(True))
        self.target_scaling = nn.Parameter(torch.tensor(scales[semantic_mask], dtype=torch.float, device="cuda").requires_grad_(True))
        self.target_rotation = nn.Parameter(torch.tensor(rots[semantic_mask], dtype=torch.float, device="cuda").requires_grad_(True))

        self.bkg_xyz = torch.tensor(xyz[~semantic_mask], dtype=torch.float, device="cuda")
        self.bkg_features_dc = torch.tensor(features_dc[~semantic_mask], dtype=torch.float, device="cuda").transpose(1, 2).contiguous()
        self.bkg_features_rest = torch.tensor(features_extra[~semantic_mask], dtype=torch.float, device="cuda").transpose(1, 2).contiguous()
        self.bkg_opacity = torch.tensor(opacities[~semantic_mask], dtype=torch.float, device="cuda")
        self.bkg_scaling = torch.tensor(scales[~semantic_mask], dtype=torch.float, device="cuda")
        self.bkg_rotation = torch.tensor(rots[~semantic_mask], dtype=torch.float, device="cuda")

        self.bkg_num = self.bkg_xyz.shape[0]

        self.update_whole_scene_property()
        self.active_sh_degree = self.max_sh_degree

    def update_whole_scene_property(self):
        self._xyz =torch.concatenate([self.bkg_xyz, self.target_xyz], dim=0)
        self._features_dc=torch.concatenate([self.bkg_features_dc, self.target_features_dc], dim=0)
        self._features_rest=torch.concatenate([self.bkg_features_rest, self.target_features_rest], dim=0)
        self._opacity=torch.concatenate([self.bkg_opacity, self.target_opacity], dim=0)
        self._scaling=torch.concatenate([self.bkg_scaling, self.target_scaling], dim=0)
        self._rotation=torch.concatenate([self.bkg_rotation, self.target_rotation], dim=0)

    def replace_tensor_to_optimizer(self, tensor, name):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] == name:
                stored_state = self.optimizer.state.get(group['params'][0], None)
                stored_state["exp_avg"] = torch.zeros_like(tensor)
                stored_state["exp_avg_sq"] = torch.zeros_like(tensor)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(tensor.requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def _prune_optimizer(self, mask):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = stored_state["exp_avg"][mask]
                stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask]

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter((group["params"][0][mask].requires_grad_(True)))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(group["params"][0][mask].requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def prune_points(self, mask):
        valid_points_mask = ~mask
        optimizable_tensors = self._prune_optimizer(valid_points_mask)

        self.target_xyz = optimizable_tensors["xyz"]
        self.target_features_dc = optimizable_tensors["f_dc"]
        self.target_features_rest = optimizable_tensors["f_rest"]
        self.target_opacity = optimizable_tensors["opacity"]
        self.target_scaling = optimizable_tensors["scaling"]
        self.target_rotation = optimizable_tensors["rotation"]
        self.update_whole_scene_property()

        self.xyz_gradient_accum = self.xyz_gradient_accum[valid_points_mask]

        self.denom = self.denom[valid_points_mask]
        self.max_radii2D = self.max_radii2D[valid_points_mask]

    def cat_tensors_to_optimizer(self, tensors_dict):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            assert len(group["params"]) == 1
            extension_tensor = tensors_dict[group["name"]]
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:

                stored_state["exp_avg"] = torch.cat((stored_state["exp_avg"], torch.zeros_like(extension_tensor)), dim=0)
                stored_state["exp_avg_sq"] = torch.cat((stored_state["exp_avg_sq"], torch.zeros_like(extension_tensor)), dim=0)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]

        return optimizable_tensors

    def densification_postfix(self, new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation):
        d = {"xyz": new_xyz,
        "f_dc": new_features_dc,
        "f_rest": new_features_rest,
        "opacity": new_opacities,
        "scaling" : new_scaling,
        "rotation" : new_rotation}

        optimizable_tensors = self.cat_tensors_to_optimizer(d)
        self.target_xyz = optimizable_tensors["xyz"]
        self.target_features_dc = optimizable_tensors["f_dc"]
        self.target_features_rest = optimizable_tensors["f_rest"]
        self.target_opacity = optimizable_tensors["opacity"]
        self.target_scaling = optimizable_tensors["scaling"]
        self.target_rotation = optimizable_tensors["rotation"]
        self.update_whole_scene_property()

        self.xyz_gradient_accum = torch.zeros((self.get_target_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_target_xyz.shape[0], 1), device="cuda")
        self.max_radii2D = torch.zeros((self.get_target_xyz.shape[0]), device="cuda")

    def densify_and_split(self, grads, grad_threshold, scene_extent, N=2):
        n_init_points = self.get_target_xyz.shape[0]
        # Extract points that satisfy the gradient condition
        padded_grad = torch.zeros((n_init_points), device="cuda")
        padded_grad[:grads.shape[0]] = grads.squeeze()
        selected_pts_mask = torch.where(padded_grad >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_target_scaling, dim=1).values > self.percent_dense*scene_extent)

        stds = self.get_target_scaling[selected_pts_mask].repeat(N,1)
        means =torch.zeros((stds.size(0), 3),device="cuda")
        samples = torch.normal(mean=means, std=stds)
        rots = build_rotation(self.target_rotation[selected_pts_mask]).repeat(N,1,1)
        new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self.get_target_xyz[selected_pts_mask].repeat(N, 1)
        new_scaling = self.scaling_inverse_activation(self.get_target_scaling[selected_pts_mask].repeat(N,1) / (0.8*N))
        new_rotation = self.target_rotation[selected_pts_mask].repeat(N,1)
        new_features_dc = self.target_features_dc[selected_pts_mask].repeat(N,1,1)
        new_features_rest = self.target_features_rest[selected_pts_mask].repeat(N,1,1)
        new_opacity = self.target_opacity[selected_pts_mask].repeat(N,1)

        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacity, new_scaling, new_rotation)

        prune_filter = torch.cat((selected_pts_mask, torch.zeros(N * selected_pts_mask.sum(), device="cuda", dtype=bool)))
        self.prune_points(prune_filter)

    def densify_and_clone(self, grads, grad_threshold, scene_extent):
        # Extract points that satisfy the gradient condition
        selected_pts_mask = torch.where(torch.norm(grads, dim=-1) >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_target_scaling, dim=1).values <= self.percent_dense*scene_extent)

        new_xyz = self.target_xyz[selected_pts_mask]
        new_features_dc = self.target_features_dc[selected_pts_mask]
        new_features_rest = self.target_features_rest[selected_pts_mask]
        new_opacities = self.target_opacity[selected_pts_mask]
        new_scaling = self.target_scaling[selected_pts_mask]
        new_rotation = self.target_rotation[selected_pts_mask]

        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation)

    def densify_and_prune(self, max_grad, min_opacity, extent, max_screen_size):
        grads = self.xyz_gradient_accum / self.denom
        grads[grads.isnan()] = 0.0
        before=self.target_xyz.shape[0]
        self.densify_and_clone(grads, max_grad, extent)
        clone=self.get_target_xyz.shape[0]

        self.densify_and_split(grads, max_grad, extent)
        split=self.get_target_xyz.shape[0]

        prune_mask = (self.get_target_opacity < min_opacity).squeeze()
        if max_screen_size:
            big_points_vs = self.max_radii2D > max_screen_size
            big_points_ws = self.get_target_scaling.max(dim=1).values > 0.1 * extent
            prune_mask = torch.logical_or(torch.logical_or(prune_mask, big_points_vs), big_points_ws)
        self.prune_points(prune_mask)
        prune=self.get_target_xyz.shape[0]
        print(f"before: {before} - clone: {clone} - split: {split} - prune: {prune} ")
        torch.cuda.empty_cache()

    def add_densification_stats(self, viewspace_point_tensor, update_filter):
        self.xyz_gradient_accum[update_filter] += torch.norm(viewspace_point_tensor[update_filter,:2], dim=-1, keepdim=True)
        self.denom[update_filter] += 1
