import time
import numpy as np
import torch
import torchvision
import rembg
from gaussiansplatting.scene.colmap_loader import qvec2rotmat
from gaussiansplatting.scene.cameras import Simple_Camera
from threestudio.utils.dpt import DPT
from torchvision.ops import masks_to_boxes
from gaussiansplatting.utils.graphics_utils import fov2focal

import viser
import viser.transforms as tf
from dataclasses import dataclass, field
from viser.theme import TitlebarButton, TitlebarConfig, TitlebarImage

from PIL import Image
from tqdm import tqdm
import cv2
import numpy as np
import sys
import shutil
import torch
from torchvision.transforms.functional import to_pil_image, to_tensor
from kornia.geometry.quaternion import Quaternion

# import threestudio
# import os
# from threestudio.systems.base import BaseLift3DSystem
# from pathlib import Path
# import subprocess
# import rembg
# from threestudio.utils.clip_metrics import ClipSimilarity

# from threestudio.utils.lama import InpaintModel
# from threestudio.utils.ops import binary_cross_entropy
from threestudio.utils.typing import *
from threestudio.utils.transform import rotate_gaussians
from gaussiansplatting.gaussian_renderer import render, point_cloud_render
from gaussiansplatting.scene import GaussianModel

from gaussiansplatting.scene.vanilla_gaussian_model import (
    GaussianModel as VanillaGaussianModel,
)

# from gaussiansplatting.utils.graphics_utils import fov2focal
from gaussiansplatting.arguments import (
    PipelineParams,
    OptimizationParams,
)
from omegaconf import OmegaConf

# from gaussiansplatting.utils.general_utils import inverse_sigmoid
# from gaussiansplatting.gaussian_renderer import camera2rasterizer
from argparse import ArgumentParser
from threestudio.utils.misc import (
    get_device,
    step_check,
    dilate_mask,
    erode_mask,
    fill_closed_areas,
)
from threestudio.utils.sam import LangSAMTextSegmentor
from threestudio.utils.camera import camera_ray_sample_points, project, unproject

# from threestudio.utils.dpt import DPT
# from threestudio.utils.config import parse_structured
from gaussiansplatting.scene.camera_scene import CamScene
import math
from GUI.EditGuidance import EditGuidance
from GUI.DelGuidance import DelGuidance

# from GUI.AddGuidance import AddGuidance
import os
import random
import ui_utils

import datetime
import subprocess
from pathlib import Path
from threestudio.utils.transform import (
    rotate_gaussians,
    translate_gaussians,
    scale_gaussians,
    default_model_mtx,
)


class WebUI:
    def __init__(self, cfg) -> None:
        self.gs_source = cfg.gs_source
        self.colmap_dir = cfg.colmap_dir
        self.port = 8084
        # training cfg

        self.use_sam = False
        self.guidance = None
        self.stop_training = False
        self.inpaint_end_flag = False
        self.scale_depth = True
        self.depth_end_flag = False
        self.seg_scale = True
        self.seg_scale_end = False
        # from original system
        self.points3d = []
        self.gaussian = GaussianModel(
            sh_degree=0,
            anchor_weight_init_g0=1.0,
            anchor_weight_init=0.1,
            anchor_weight_multiplier=2,
        )
        # load
        self.gaussian.load_ply(self.gs_source)
        self.gaussian.max_radii2D = torch.zeros(
            (self.gaussian.get_xyz.shape[0]), device="cuda"
        )
        # front end related
        self.colmap_cameras = None
        self.render_cameras = None

        # diffusion model
        self.ip2p = None
        self.ctn_ip2p = None

        self.ctn_inpaint = None
        self.ctn_ip2p = None
        self.training = False
        if self.colmap_dir is not None:
            scene = CamScene(self.colmap_dir, h=512, w=512)
            self.cameras_extent = scene.cameras_extent
            self.colmap_cameras = scene.cameras

        self.background_tensor = torch.tensor(
            [0, 0, 0], dtype=torch.float32, device="cuda"
        )
        self.edit_frames = {}
        self.origin_frames = {}
        self.masks_2D = {}
        self.text_segmentor = LangSAMTextSegmentor().to(get_device())
        self.sam_predictor = self.text_segmentor.model.sam
        self.sam_predictor.is_image_set = True
        self.sam_features = {}
        self.semantic_gauassian_masks = {}
        self.semantic_gauassian_masks["ALL"] = torch.ones_like(self.gaussian._opacity)

        self.parser = ArgumentParser(description="Training script parameters")
        self.pipe = PipelineParams(self.parser)

        # status
        self.display_semantic_mask = False
        self.display_point_prompt = False

        self.viewer_need_update = False
        self.system_need_update = False
        self.inpaint_again = True
        self.scale_depth = True

        self.server = viser.ViserServer(port=self.port)
        self.add_theme()
        self.draw_flag = True
        with self.server.add_gui_folder("Render Setting"):
            self.resolution_slider = self.server.add_gui_slider(
                "Resolution", min=384, max=4096, step=2, initial_value=2048
            )

            self.FoV_slider = self.server.add_gui_slider(
                "FoV Scaler", min=0.2, max=2, step=0.1, initial_value=1
            )

            self.fps = self.server.add_gui_text(
                "FPS", initial_value="-1", disabled=True
            )
            self.renderer_output = self.server.add_gui_dropdown(
                "Renderer Output",
                [
                    "comp_rgb",
                ],
            )
            self.save_button = self.server.add_gui_button("Save Gaussian")

            self.frame_show = self.server.add_gui_checkbox(
                "Show Frame", initial_value=False
            )

        with self.server.add_gui_folder("Semantic Tracing"):
            self.sam_enabled = self.server.add_gui_checkbox(
                "Enable SAM",
                initial_value=False,
            )
            self.add_sam_points = self.server.add_gui_checkbox(
                "Add SAM Points", initial_value=False
            )
            self.sam_group_name = self.server.add_gui_text(
                "SAM Group Name", initial_value="table"
            )
            self.clear_sam_pins = self.server.add_gui_button(
                "Clear SAM Pins",
            )
            self.text_seg_prompt = self.server.add_gui_text(
                "Text Seg Prompt", initial_value="a bike"
            )
            self.semantic_groups = self.server.add_gui_dropdown(
                "Semantic Group",
                options=["ALL"],
            )

            self.seg_cam_num = self.server.add_gui_slider(
                "Seg Camera Nums", min=6, max=200, step=1, initial_value=24
            )

            self.mask_thres = self.server.add_gui_slider(
                "Seg Threshold", min=0.2, max=0.99999, step=0.00001, initial_value=0.7, visible=False
            )

            self.show_semantic_mask = self.server.add_gui_checkbox(
                "Show Semantic Mask", initial_value=False
            )
            self.seg_scale_end_button = self.server.add_gui_button(
                "End Seg Scale!",
                visible=False,
            )
            self.submit_seg_prompt = self.server.add_gui_button("Tracing Begin!")

        with self.server.add_gui_folder("Edit Setting"):
            self.edit_type = self.server.add_gui_dropdown(
                "Edit Type", ("Edit", "Delete", "Add")
            )
            self.guidance_type = self.server.add_gui_dropdown(
                "Guidance Type", ("InstructPix2Pix", "ControlNet-Pix2Pix")
            )
            self.edit_frame_show = self.server.add_gui_checkbox(
                "Show Edit Frame", initial_value=True, visible=False
            )
            self.edit_text = self.server.add_gui_text(
                "Text",
                initial_value="",
                visible=True,
            )
            self.draw_bbox = self.server.add_gui_checkbox(
                "Draw Bounding Box", initial_value=False, visible=False
            )
            self.left_up = self.server.add_gui_vector2(
                "Left UP",
                initial_value=(0, 0),
                step=1,
                visible=False,
            )
            self.right_down = self.server.add_gui_vector2(
                "Right Down",
                initial_value=(0, 0),
                step=1,
                visible=False,
            )

            self.inpaint_seed = self.server.add_gui_slider(
                "Inpaint Seed", min=0, max=1000, step=1, initial_value=0, visible=False
            )

            self.refine_text = self.server.add_gui_text(
                "Refine Text",
                initial_value="",
                visible=False,
            )
            self.inpaint_end = self.server.add_gui_button(
                "End 2D Inpainting!",
                visible=False,
            )

            self.depth_scaler = self.server.add_gui_slider(
                "Depth Scale", min=0.0, max=5.0, step=0.01, initial_value=1.0, visible=False
            )
            self.depth_end = self.server.add_gui_button(
                "End Depth Scale!",
                visible=False,
            )
            self.edit_begin_button = self.server.add_gui_button("Edit Begin!")
            self.edit_end_button = self.server.add_gui_button(
                "End Editing!", visible=False
            )

            with self.server.add_gui_folder("Advanced Options"):
                self.edit_cam_num = self.server.add_gui_slider(
                    "Camera Num", min=12, max=200, step=1, initial_value=48
                )
                self.edit_train_steps = self.server.add_gui_slider(
                    "Total Step", min=0, max=5000, step=100, initial_value=1500
                )
                self.densify_until_step = self.server.add_gui_slider(
                    "Densify Until Step",
                    min=0,
                    max=5000,
                    step=50,
                    initial_value=1300,
                )

                self.densification_interval = self.server.add_gui_slider(
                    "Densify Interval",
                    min=25,
                    max=1000,
                    step=25,
                    initial_value=100,
                )
                self.max_densify_percent = self.server.add_gui_slider(
                    "Max Densify Percent",
                    min=0.0,
                    max=1.0,
                    step=0.001,
                    initial_value=0.01,
                )
                self.min_opacity = self.server.add_gui_slider(
                    "Min Opacity",
                    min=0.0,
                    max=0.1,
                    step=0.0001,
                    initial_value=0.005,
                )

                self.per_editing_step = self.server.add_gui_slider(
                    "Edit Interval", min=4, max=48, step=1, initial_value=10
                )
                self.edit_begin_step = self.server.add_gui_slider(
                    "Edit Begin Step", min=0, max=5000, step=100, initial_value=0
                )
                self.edit_until_step = self.server.add_gui_slider(
                    "Edit Until Step", min=0, max=5000, step=100, initial_value=1000
                )

                self.inpaint_scale = self.server.add_gui_slider(
                    "Inpaint Scale", min=0.1, max=10, step=0.1, initial_value=1, visible=False
                )

                self.mask_dilate = self.server.add_gui_slider(
                    "Mask Dilate", min=1, max=30, step=1, initial_value=15, visible=False
                )
                self.fix_holes = self.server.add_gui_checkbox(
                    "Fix Holes", initial_value=True, visible=False
                )
                with self.server.add_gui_folder("Learning Rate Scaler"):
                    self.gs_lr_scaler = self.server.add_gui_slider(
                        "XYZ LR Init", min=0.0, max=10.0, step=0.1, initial_value=3.0
                    )
                    self.gs_lr_end_scaler = self.server.add_gui_slider(
                        "XYZ LR End", min=0.0, max=10.0, step=0.1, initial_value=2.0
                    )
                    self.color_lr_scaler = self.server.add_gui_slider(
                        "Color LR", min=0.0, max=10.0, step=0.1, initial_value=3.0
                    )
                    self.opacity_lr_scaler = self.server.add_gui_slider(
                        "Opacity LR", min=0.0, max=10.0, step=0.1, initial_value=2.0
                    )
                    self.scaling_lr_scaler = self.server.add_gui_slider(
                        "Scale LR", min=0.0, max=10.0, step=0.1, initial_value=2.0
                    )
                    self.rotation_lr_scaler = self.server.add_gui_slider(
                        "Rotation LR", min=0.0, max=10.0, step=0.1, initial_value=2.0
                    )

                with self.server.add_gui_folder("Loss Options"):
                    self.lambda_l1 = self.server.add_gui_slider(
                        "Lambda L1", min=0, max=100, step=1, initial_value=10
                    )
                    self.lambda_p = self.server.add_gui_slider(
                        "Lambda Perceptual", min=0, max=100, step=1, initial_value=10
                    )

                    self.anchor_weight_init_g0 = self.server.add_gui_slider(
                        "Anchor Init G0", min=0., max=10., step=0.05, initial_value=0.05
                    )
                    self.anchor_weight_init = self.server.add_gui_slider(
                        "Anchor Init", min=0., max=10., step=0.05, initial_value=0.1
                    )
                    self.anchor_weight_multiplier = self.server.add_gui_slider(
                        "Anchor Multiplier", min=1., max=10., step=0.1, initial_value=1.3
                    )

                    self.lambda_anchor_color = self.server.add_gui_slider(
                        "Lambda Anchor Color", min=0, max=500, step=1, initial_value=0
                    )
                    self.lambda_anchor_geo = self.server.add_gui_slider(
                        "Lambda Anchor Geo", min=0, max=500, step=1, initial_value=50
                    )
                    self.lambda_anchor_scale = self.server.add_gui_slider(
                        "Lambda Anchor Scale", min=0, max=500, step=1, initial_value=50
                    )
                    self.lambda_anchor_opacity = self.server.add_gui_slider(
                        "Lambda Anchor Opacity", min=0, max=500, step=1, initial_value=50
                    )
                    self.anchor_term = [self.anchor_weight_init_g0, self.anchor_weight_init,
                                        self.anchor_weight_multiplier,
                                        self.lambda_anchor_color, self.lambda_anchor_geo,
                                        self.lambda_anchor_scale, self.lambda_anchor_opacity, ]

        @self.inpaint_seed.on_update
        def _(_):
            self.inpaint_again = True

        @self.depth_scaler.on_update
        def _(_):
            self.scale_depth = True

        @self.mask_thres.on_update
        def _(_):
            self.seg_scale = True


        @self.edit_type.on_update
        def _(_):
            if self.edit_type.value == "Edit":
                self.edit_text.visible = True
                self.refine_text.visible = False
                for term in self.anchor_term:
                    term.visible = True
                self.inpaint_scale.visible = False
                self.mask_dilate.visible = False
                self.fix_holes.visible = False
                self.per_editing_step.visible = True
                self.edit_begin_step.visible = True
                self.edit_until_step.visible = True
                self.draw_bbox.visible = False
                self.left_up.visible = False
                self.right_down.visible = False
                self.inpaint_seed.visible = False
                self.inpaint_end.visible = False
                self.depth_scaler.visible = False
                self.depth_end.visible = False
                self.edit_frame_show.visible = True
                self.guidance_type.visible = True

            elif self.edit_type.value == "Delete":
                self.edit_text.visible = True
                self.refine_text.visible = False
                for term in self.anchor_term:
                    term.visible = True
                self.inpaint_scale.visible = True
                self.mask_dilate.visible = True
                self.fix_holes.visible = True
                self.edit_cam_num.value = 24
                self.densification_interval.value = 50
                self.per_editing_step.visible = False
                self.edit_begin_step.visible = False
                self.edit_until_step.visible = False
                self.draw_bbox.visible = False
                self.left_up.visible = False
                self.right_down.visible = False
                self.inpaint_seed.visible = False
                self.inpaint_end.visible = False
                self.depth_scaler.visible = False
                self.depth_end.visible = False
                self.edit_frame_show.visible = True
                self.guidance_type.visible = False

            elif self.edit_type.value == "Add":
                self.edit_text.visible = True
                self.refine_text.visible = False
                for term in self.anchor_term:
                    term.visible = False
                self.inpaint_scale.visible = False
                self.mask_dilate.visible = False
                self.fix_holes.visible = False
                self.per_editing_step.visible = True
                self.edit_begin_step.visible = True
                self.edit_until_step.visible = True
                self.draw_bbox.visible = True
                self.left_up.visible = True
                self.right_down.visible = True
                self.inpaint_seed.visible = False
                self.inpaint_end.visible = False
                self.depth_scaler.visible = False
                self.depth_end.visible = False
                self.edit_frame_show.visible = False
                self.guidance_type.visible = False

        @self.save_button.on_click
        def _(_):
            current_time = datetime.datetime.now()
            formatted_time = current_time.strftime("%Y-%m-%d-%H:%M")
            self.gaussian.save_ply(os.path.join("ui_result", "{}.ply".format(formatted_time)))
        @self.inpaint_end.on_click
        def _(_):
            self.inpaint_end_flag = True

        @self.seg_scale_end_button.on_click
        def _(_):
            self.seg_scale_end = True


        @self.depth_end.on_click
        def _(_):
            self.depth_end_flag = True

        @self.edit_end_button.on_click
        def _(event: viser.GuiEvent):
            self.stop_training = True

        @self.edit_begin_button.on_click
        def _(event: viser.GuiEvent):
            self.edit_begin_button.visible = False
            self.edit_end_button.visible = True
            if self.training:
                return
            self.training = True
            self.configure_optimizers()
            self.gaussian.update_anchor_term(
                anchor_weight_init_g0=self.anchor_weight_init_g0.value,
                anchor_weight_init=self.anchor_weight_init.value,
                anchor_weight_multiplier=self.anchor_weight_multiplier.value,
            )

            if self.edit_type.value == "Add":
                # self.add(self.camera)
                self.add(self.camera)
            else:
                self.edit_frame_show.visible = True

                edit_cameras, train_frames, train_frustums = ui_utils.sample_train_camera(self.colmap_cameras,
                                                                                          self.edit_cam_num.value,
                                                                                          self.server)
                if self.edit_type.value == "Edit":
                    self.edit(edit_cameras, train_frames, train_frustums)

                elif self.edit_type.value == "Delete":
                    self.delete(edit_cameras, train_frames, train_frustums)

                ui_utils.remove_all(train_frames)
                ui_utils.remove_all(train_frustums)
                self.edit_frame_show.visible = False
            self.guidance = None
            self.training = False
            self.gaussian.anchor_postfix()
            self.edit_begin_button.visible = True
            self.edit_end_button.visible = False

        @self.submit_seg_prompt.on_click
        def _(_):
            if not self.sam_enabled.value:
                text_prompt = self.text_seg_prompt.value
                print("[Segmentation Prompt]", text_prompt)
                _, semantic_gaussian_mask = self.update_mask(text_prompt)
            else:
                text_prompt = self.sam_group_name.value
                # buggy here, if self.sam_enabled == True, will raise strange errors. (Maybe caused by multi-threading access to the same SAM model)
                self.sam_enabled.value = False
                self.add_sam_points.value = False
                # breakpoint()
                _, semantic_gaussian_mask = self.update_sam_mask_with_point_prompt(
                    save_mask=True
                )

            self.semantic_gauassian_masks[text_prompt] = semantic_gaussian_mask
            if text_prompt not in self.semantic_groups.options:
                self.semantic_groups.options += (text_prompt,)
            self.semantic_groups.value = text_prompt

        @self.semantic_groups.on_update
        def _(_):
            semantic_mask = self.semantic_gauassian_masks[self.semantic_groups.value]
            self.gaussian.set_mask(semantic_mask)
            self.gaussian.apply_grad_mask(semantic_mask)

        @self.edit_frame_show.on_update
        def _(_):
            if self.guidance is not None:
                for _ in self.guidance.train_frames:
                    _.visible = self.edit_frame_show.value
                for _ in self.guidance.train_frustums:
                    _.visible = self.edit_frame_show.value
                self.guidance.visible = self.edit_frame_show.value

        with torch.no_grad():
            self.frames = []
            random.seed(0)
            frame_index = random.sample(
                range(0, len(self.colmap_cameras)),
                min(len(self.colmap_cameras), 20),
            )
            for i in frame_index:
                self.make_one_camera_pose_frame(i)

        @self.frame_show.on_update
        def _(_):
            for frame in self.frames:
                frame.visible = self.frame_show.value
            self.server.world_axes.visible = self.frame_show.value

        @self.server.on_scene_click
        def _(pointer):
            self.click_cb(pointer)

        @self.clear_sam_pins.on_click
        def _(_):
            self.clear_points3d()

    def make_one_camera_pose_frame(self, idx):
        cam = self.colmap_cameras[idx]
        # wxyz = tf.SO3.from_matrix(cam.R.T).wxyz
        # position = -cam.R.T @ cam.T

        T_world_camera = tf.SE3.from_rotation_and_translation(
            tf.SO3(cam.qvec), cam.T
        ).inverse()
        wxyz = T_world_camera.rotation().wxyz
        position = T_world_camera.translation()

        # breakpoint()
        frame = self.server.add_frame(
            f"/colmap/frame_{idx}",
            wxyz=wxyz,
            position=position,
            axes_length=0.2,
            axes_radius=0.01,
            visible=False,
        )
        self.frames.append(frame)

        @frame.on_click
        def _(event: viser.GuiEvent):
            client = event.client
            assert client is not None
            T_world_current = tf.SE3.from_rotation_and_translation(
                tf.SO3(client.camera.wxyz), client.camera.position
            )

            T_world_target = tf.SE3.from_rotation_and_translation(
                tf.SO3(frame.wxyz), frame.position
            ) @ tf.SE3.from_translation(np.array([0.0, 0.0, -0.5]))

            T_current_target = T_world_current.inverse() @ T_world_target

            for j in range(5):
                T_world_set = T_world_current @ tf.SE3.exp(
                    T_current_target.log() * j / 4.0
                )

                with client.atomic():
                    client.camera.wxyz = T_world_set.rotation().wxyz
                    client.camera.position = T_world_set.translation()

                time.sleep(1.0 / 15.0)
            client.camera.look_at = frame.position

        if not hasattr(self, "begin_call"):

            def begin_trans(client):
                assert client is not None
                T_world_current = tf.SE3.from_rotation_and_translation(
                    tf.SO3(client.camera.wxyz), client.camera.position
                )

                T_world_target = tf.SE3.from_rotation_and_translation(
                    tf.SO3(frame.wxyz), frame.position
                ) @ tf.SE3.from_translation(np.array([0.0, 0.0, -0.5]))

                T_current_target = T_world_current.inverse() @ T_world_target

                for j in range(5):
                    T_world_set = T_world_current @ tf.SE3.exp(
                        T_current_target.log() * j / 4.0
                    )

                    with client.atomic():
                        client.camera.wxyz = T_world_set.rotation().wxyz
                        client.camera.position = T_world_set.translation()
                client.camera.look_at = frame.position

            self.begin_call = begin_trans

    def configure_optimizers(self):
        opt = OptimizationParams(
            parser = ArgumentParser(description="Training script parameters"),
            max_steps= self.edit_train_steps.value,
            lr_scaler = self.gs_lr_scaler.value,
            lr_final_scaler = self.gs_lr_end_scaler.value,
            color_lr_scaler = self.color_lr_scaler.value,
            opacity_lr_scaler = self.opacity_lr_scaler.value,
            scaling_lr_scaler = self.scaling_lr_scaler.value,
            rotation_lr_scaler = self.rotation_lr_scaler.value,

        )
        opt = OmegaConf.create(vars(opt))
        # opt.update(self.training_args)
        self.gaussian.spatial_lr_scale = self.cameras_extent
        self.gaussian.training_setup(opt)

    def render(
        self,
        cam,
        local=False,
        sam=False,
        train=False,
    ) -> Dict[str, Any]:
        self.gaussian.localize = local

        render_pkg = render(cam, self.gaussian, self.pipe, self.background_tensor)
        image, viewspace_point_tensor, _, radii = (
            render_pkg["render"],
            render_pkg["viewspace_points"],
            render_pkg["visibility_filter"],
            render_pkg["radii"],
        )
        if train:
            self.viewspace_point_tensor = viewspace_point_tensor
            self.radii = radii
            self.visibility_filter = self.radii > 0.0

        semantic_map = render(
            cam,
            self.gaussian,
            self.pipe,
            self.background_tensor,
            override_color=self.gaussian.mask[..., None].float().repeat(1, 3),
        )["render"]
        semantic_map = torch.norm(semantic_map, dim=0)
        semantic_map = semantic_map > 0.0  # 1, H, W
        semantic_map_viz = image.detach().clone()  # C, H, W
        semantic_map_viz = semantic_map_viz.permute(1, 2, 0)  # 3 512 512 to 512 512 3
        semantic_map_viz[semantic_map] = 0.50 * semantic_map_viz[
            semantic_map
        ] + 0.50 * torch.tensor([1.0, 0.0, 0.0], device="cuda")
        semantic_map_viz = semantic_map_viz.permute(2, 0, 1)  # 512 512 3 to 3 512 512

        render_pkg["sam_masks"] = []
        render_pkg["point2ds"] = []
        if sam:
            if hasattr(self, "points3d") and len(self.points3d) > 0:
                sam_output = self.sam_predict(image, cam)
                if sam_output is not None:
                    render_pkg["sam_masks"].append(sam_output[0])
                    render_pkg["point2ds"].append(sam_output[1])

        self.gaussian.localize = False  # reverse

        render_pkg["semantic"] = semantic_map_viz[None]
        render_pkg["masks"] = semantic_map[None]  # 1, 1, H, W

        image = image.permute(1, 2, 0)[None]  # C H W to 1 H W C
        render_pkg["comp_rgb"] = image  # 1 H W C

        depth = render_pkg["depth_3dgs"]
        depth = depth.permute(1, 2, 0)[None]
        render_pkg["depth"] = depth
        render_pkg["opacity"] = depth / (depth.max() + 1e-5)

        return {
            **render_pkg,
        }

    @torch.no_grad()
    def update_mask(self, text_prompt) -> None:

        masks = []
        weights = torch.zeros_like(self.gaussian._opacity)
        weights_cnt = torch.zeros_like(self.gaussian._opacity, dtype=torch.int32)

        total_view_num = len(self.colmap_cameras)
        random.seed(0)  # make sure same views
        view_index = random.sample(
            range(0, total_view_num),
            min(total_view_num, self.seg_cam_num.value),
        )

        for idx in tqdm(view_index):
            cur_cam = self.colmap_cameras[idx]
            this_frame = render(
                cur_cam, self.gaussian, self.pipe, self.background_tensor
            )["render"]

            # breakpoint()
            # this_frame [c h w]
            this_frame = this_frame.moveaxis(0, -1)[None, ...]
            mask = self.text_segmentor(this_frame, text_prompt)[0].to(get_device())
            if self.use_sam:
                print("Using SAM")
                self.sam_features[idx] = self.sam_predictor.features

            masks.append(mask)
            self.gaussian.apply_weights(cur_cam, weights, weights_cnt, mask)

        weights /= weights_cnt + 1e-7
        self.seg_scale_end_button.visible = True
        self.mask_thres.visible = True
        self.show_semantic_mask.value = True
        while True:
            if self.seg_scale:
                selected_mask = weights > self.mask_thres.value
                selected_mask = selected_mask[:, 0]
                self.gaussian.set_mask(selected_mask)
                self.gaussian.apply_grad_mask(selected_mask)

                self.seg_scale = False
            if self.seg_scale_end:
                self.seg_scale_end = False
                break
            time.sleep(0.01)

        self.seg_scale_end_button.visible = False
        self.mask_thres.visible = False
        return masks, selected_mask

    @property
    def camera(self):
        if len(list(self.server.get_clients().values())) == 0:
            return None
        if self.render_cameras is None and self.colmap_dir is not None:
            self.aspect = list(self.server.get_clients().values())[0].camera.aspect
            self.render_cameras = CamScene(
                self.colmap_dir, h=-1, w=-1, aspect=self.aspect
            ).cameras
            self.begin_call(list(self.server.get_clients().values())[0])
        viser_cam = list(self.server.get_clients().values())[0].camera
        # viser_cam.up_direction = tf.SO3(viser_cam.wxyz) @ np.array([0.0, -1.0, 0.0])
        # viser_cam.look_at = viser_cam.position
        R = tf.SO3(viser_cam.wxyz).as_matrix()
        T = -R.T @ viser_cam.position
        # T = viser_cam.position
        if self.render_cameras is None:
            fovy = viser_cam.fov * self.FoV_slider.value
        else:
            fovy = self.render_cameras[0].FoVy * self.FoV_slider.value

        fovx = 2 * math.atan(math.tan(fovy / 2) * self.aspect)
        # fovy = self.render_cameras[0].FoVy
        # fovx = self.render_cameras[0].FoVx
        # math.tan(self.render_cameras[0].FoVx / 2) / math.tan(self.render_cameras[0].FoVy / 2)
        # math.tan(fovx/2) / math.tan(fovy/2)

        # aspect = viser_cam.aspect
        width = int(self.resolution_slider.value)
        height = int(width / self.aspect)
        return Simple_Camera(0, R, T, fovx, fovy, height, width, "", 0)

    def click_cb(self, pointer):
        if self.sam_enabled.value and self.add_sam_points.value:
            assert hasattr(pointer, "click_pos"), "please install our forked viser"
            click_pos = pointer.click_pos  # tuple (float, float)  W, H from 0 to 1
            click_pos = torch.tensor(click_pos)

            self.add_points3d(self.camera, click_pos)

            self.viwer_need_update = True
        elif self.draw_bbox.value:
            assert hasattr(pointer, "click_pos"), "please install our forked viser"
            click_pos = pointer.click_pos
            click_pos = torch.tensor(click_pos)
            cur_cam = self.camera
            if self.draw_flag:
                self.left_up.value = [
                    int(cur_cam.image_width * click_pos[0]),
                    int(cur_cam.image_height * click_pos[1]),
                ]
                self.draw_flag = False
            else:
                new_value = [
                    int(cur_cam.image_width * click_pos[0]),
                    int(cur_cam.image_height * click_pos[1]),
                ]
                if (self.left_up.value[0] < new_value[0]) and (
                    self.left_up.value[1] < new_value[1]
                ):
                    self.right_down.value = new_value
                    self.draw_flag = True
                else:
                    self.left_up.value = new_value

    def set_system(self, system):
        self.system = system

    def clear_points3d(self):
        self.points3d = []

    def add_points3d(self, camera, points2d, update_mask=False):
        depth = render(camera, self.gaussian, self.pipe, self.background_tensor)[
            "depth_3dgs"
        ]
        unprojected_points3d = unproject(camera, points2d, depth)
        self.points3d += unprojected_points3d.unbind(0)

        if update_mask:
            self.update_sam_mask_with_point_prompt(self.points3d)

    # no longer needed since can be extracted from langsam
    # def sam_encode_all_view(self):
    #     assert hasattr(self, "sam_predictor")
    #     self.sam_features = {}
    #     # NOTE: assuming all views have the same size
    #     for id, frame in self.origin_frames.items():
    #         # TODO: check frame dtype (float32 or uint8) and device
    #         self.sam_predictor.set_image(frame)
    #         self.sam_features[id] = self.sam_predictor.features

    @torch.no_grad()
    def update_sam_mask_with_point_prompt(
        self, points3d=None, save_mask=False, save_name="point_prompt_mask"
    ):
        points3d = points3d if points3d is not None else self.points3d
        masks = []
        weights = torch.zeros_like(self.gaussian._opacity)
        weights_cnt = torch.zeros_like(self.gaussian._opacity, dtype=torch.int32)

        total_view_num = len(self.colmap_cameras)
        random.seed(0)  # make sure same views
        view_index = random.sample(
            range(0, total_view_num),
            min(total_view_num, self.seg_cam_num.value),
        )
        for idx in tqdm(view_index):
            cur_cam = self.colmap_cameras[idx]
            assert len(points3d) > 0
            points2ds = project(cur_cam, points3d)
            img = render(cur_cam, self.gaussian, self.pipe, self.background_tensor)[
                "render"
            ]

            self.sam_predictor.set_image(
                np.asarray(to_pil_image(img.cpu())),
            )
            self.sam_features[idx] = self.sam_predictor.features
            # print(points2ds)
            mask, _, _ = self.sam_predictor.predict(
                point_coords=points2ds.cpu().numpy(),
                point_labels=np.array([1] * points2ds.shape[0], dtype=np.int64),
                box=None,
                multimask_output=False,
            )
            mask = torch.from_numpy(mask).to(torch.bool).to(get_device())
            self.gaussian.apply_weights(
                cur_cam, weights, weights_cnt, mask.to(torch.float32)
            )
            masks.append(mask)

        weights /= weights_cnt + 1e-7

        self.seg_scale_end_button.visible = True
        self.mask_thres.visible = True
        self.show_semantic_mask.value = True
        while True:
            if self.seg_scale:
                selected_mask = weights > self.mask_thres.value
                selected_mask = selected_mask[:, 0]
                self.gaussian.set_mask(selected_mask)
                self.gaussian.apply_grad_mask(selected_mask)

                self.seg_scale = False
            if self.seg_scale_end:
                self.seg_scale_end = False
                break
            time.sleep(0.01)

        self.seg_scale_end_button.visible = False
        self.mask_thres.visible = False

        if save_mask:
            for id, mask in enumerate(masks):
                mask = mask.cpu().numpy()[0, 0]
                img = Image.fromarray(mask)
                os.makedirs("tmp",exist_ok=True)
                img.save(f"./tmp/{save_name}-{id}.jpg")

        return masks, selected_mask

    @torch.no_grad()
    def sam_predict(self, image, cam):
        img = np.asarray(to_pil_image(image.cpu()))
        self.sam_predictor.set_image(img)
        if len(self.points3d) == 0:
            return
        _points2ds = project(cam, self.points3d)
        _mask, _, _ = self.sam_predictor.predict(
            point_coords=_points2ds.cpu().numpy(),
            point_labels=np.array([1] * _points2ds.shape[0], dtype=np.int64),
            box=None,
            multimask_output=False,
        )
        _mask = torch.from_numpy(_mask).to(torch.bool).to(get_device())

        return _mask.squeeze(), _points2ds

    @torch.no_grad()
    def prepare_output_image(self, output):
        out_key = self.renderer_output.value
        out_img = output[out_key][0]  # H W C
        if out_key == "comp_rgb":
            if self.show_semantic_mask.value:
                out_img = output["semantic"][0].moveaxis(0, -1)
        elif out_key == "masks":
            out_img = output["masks"][0].to(torch.float32)[..., None].repeat(1, 1, 3)
        if out_img.dtype == torch.float32:
            out_img = out_img.clamp(0, 1)
            out_img = (out_img * 255).to(torch.uint8).cpu().to(torch.uint8)
            out_img = out_img.moveaxis(-1, 0)  # C H W

        if self.sam_enabled.value:
            if "sam_masks" in output and len(output["sam_masks"]) > 0:
                try:
                    out_img = torchvision.utils.draw_segmentation_masks(
                        out_img, output["sam_masks"][0]
                    )

                    out_img = torchvision.utils.draw_keypoints(
                        out_img,
                        output["point2ds"][0][None, ...],
                        colors="blue",
                        radius=5,
                    )
                except Exception as e:
                    print(e)

        if (
            self.draw_bbox.value
            and self.draw_flag
            and (self.left_up.value[0] < self.right_down.value[0])
            and (self.left_up.value[1] < self.right_down.value[1])
        ):
            out_img[
                :,
                self.left_up.value[1] : self.right_down.value[1],
                self.left_up.value[0] : self.right_down.value[0],
            ] = 0

        self.renderer_output.options = list(output.keys())
        return out_img.cpu().moveaxis(0, -1).numpy().astype(np.uint8)

    def render_loop(self):
        while True:
            # if self.viewer_need_update:
            self.update_viewer()
            time.sleep(1e-2)

    @torch.no_grad()
    def update_viewer(self):
        gs_camera = self.camera
        if gs_camera is None:
            return
        output = self.render(gs_camera, sam=self.sam_enabled.value)

        out = self.prepare_output_image(output)
        self.server.set_background_image(out, format="jpeg")

    def delete(self, edit_cameras, train_frames, train_frustums):
        if not self.ctn_inpaint:
            from diffusers import (
                StableDiffusionControlNetInpaintPipeline,
                ControlNetModel,
                DDIMScheduler,
            )

            controlnet = ControlNetModel.from_pretrained(
                "lllyasviel/control_v11p_sd15_inpaint", torch_dtype=torch.float16
            )
            pipe = StableDiffusionControlNetInpaintPipeline.from_pretrained(
                "runwayml/stable-diffusion-v1-5",
                controlnet=controlnet,
                torch_dtype=torch.float16,
            )
            pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)

            pipe.enable_model_cpu_offload()

            self.ctn_inpaint = pipe
            self.ctn_inpaint.set_progress_bar_config(disable=True)
            self.ctn_inpaint.safety_checker = None

        num_channels_latents = self.ctn_inpaint.vae.config.latent_channels
        shape = (
            1,
            num_channels_latents,
            edit_cameras[0].image_height // self.ctn_inpaint.vae_scale_factor,
            edit_cameras[0].image_height // self.ctn_inpaint.vae_scale_factor,
        )

        latents = torch.zeros(shape, dtype=torch.float16, device="cuda")

        dist_thres = (
            self.inpaint_scale.value * self.cameras_extent * self.gaussian.percent_dense
        )
        valid_remaining_idx = self.gaussian.get_near_gaussians_by_mask(
            self.gaussian.mask, dist_thres
        )
        # Prune and update mask to valid_remaining_idx
        self.gaussian.prune_with_mask(new_mask=valid_remaining_idx)

        inpaint_2D_mask, origin_frames = self.render_all_view_with_mask(
            edit_cameras, train_frames, train_frustums
        )

        self.guidance = DelGuidance(
            guidance=self.ctn_inpaint,
            latents=latents,
            gaussian=self.gaussian,
            text_prompt=self.edit_text.value,
            lambda_l1=self.lambda_l1.value,
            lambda_p=self.lambda_p.value,
            lambda_anchor_color=self.lambda_anchor_color.value,
            lambda_anchor_geo=self.lambda_anchor_geo.value,
            lambda_anchor_scale=self.lambda_anchor_scale.value,
            lambda_anchor_opacity=self.lambda_anchor_opacity.value,
            train_frames=train_frames,
            train_frustums=train_frustums,
            cams=edit_cameras,
            server=self.server,
        )

        view_index_stack = list(range(len(edit_cameras)))
        for step in tqdm(range(self.edit_train_steps.value)):
            if not view_index_stack:
                view_index_stack = list(range(len(edit_cameras)))
            view_index = random.choice(view_index_stack)
            view_index_stack.remove(view_index)

            rendering = self.render(edit_cameras[view_index], train=True)["comp_rgb"]

            loss = self.guidance(
                rendering,
                origin_frames[view_index],
                inpaint_2D_mask[view_index],
                view_index,
                step,
            )
            loss.backward()

            self.densify_and_prune(step)

            self.gaussian.optimizer.step()
            self.gaussian.optimizer.zero_grad(set_to_none=True)
            if self.stop_training:
                self.stop_training = False
                return

    # In local edit, the whole gaussian don't need to be visible for edit
    def edit(self, edit_cameras, train_frames, train_frustums):
        if self.guidance_type.value == "InstructPix2Pix":
            if not self.ip2p:
                from threestudio.models.guidance.instructpix2pix_guidance import (
                    InstructPix2PixGuidance,
                )

                self.ip2p = InstructPix2PixGuidance(
                    OmegaConf.create({"min_step_percent": 0.02, "max_step_percent": 0.98})
                )
            cur_2D_guidance = self.ip2p
            print("using InstructPix2Pix!")
        elif self.guidance_type.value == "ControlNet-Pix2Pix":
            if not self.ctn_ip2p:
                from threestudio.models.guidance.controlnet_guidance import (
                    ControlNetGuidance,
                )

                self.ctn_ip2p = ControlNetGuidance(
                    OmegaConf.create({"min_step_percent": 0.05,
                                      "max_step_percent": 0.8,
                                        "control_type": "p2p"})
                )
            cur_2D_guidance = self.ctn_ip2p
            print("using ControlNet-InstructPix2Pix!")

        origin_frames = self.render_cameras_list(edit_cameras)
        self.guidance = EditGuidance(
            guidance=cur_2D_guidance,
            gaussian=self.gaussian,
            origin_frames=origin_frames,
            text_prompt=self.edit_text.value,
            per_editing_step=self.per_editing_step.value,
            edit_begin_step=self.edit_begin_step.value,
            edit_until_step=self.edit_until_step.value,
            lambda_l1=self.lambda_l1.value,
            lambda_p=self.lambda_p.value,
            lambda_anchor_color=self.lambda_anchor_color.value,
            lambda_anchor_geo=self.lambda_anchor_geo.value,
            lambda_anchor_scale=self.lambda_anchor_scale.value,
            lambda_anchor_opacity=self.lambda_anchor_opacity.value,
            train_frames=train_frames,
            train_frustums=train_frustums,
            cams=edit_cameras,
            server=self.server,
        )
        view_index_stack = list(range(len(edit_cameras)))
        for step in tqdm(range(self.edit_train_steps.value)):
            if not view_index_stack:
                view_index_stack = list(range(len(edit_cameras)))
            view_index = random.choice(view_index_stack)
            view_index_stack.remove(view_index)

            rendering = self.render(edit_cameras[view_index], train=True)["comp_rgb"]

            loss = self.guidance(rendering, view_index, step)
            loss.backward()

            self.densify_and_prune(step)

            self.gaussian.optimizer.step()
            self.gaussian.optimizer.zero_grad(set_to_none=True)
            if self.stop_training:
                self.stop_training = False
                return

    @torch.no_grad()
    def add(self, cam):
        self.draw_bbox.value = False
        self.inpaint_seed.visible = True
        self.inpaint_end.visible = True
        self.refine_text.visible = True
        self.draw_bbox.visible = False
        self.left_up.visible = False
        self.right_down.visible = False

        if not self.ctn_inpaint:
            from diffusers import (
                StableDiffusionControlNetInpaintPipeline,
                ControlNetModel,
                DDIMScheduler,
            )

            controlnet = ControlNetModel.from_pretrained(
                "lllyasviel/control_v11p_sd15_inpaint", torch_dtype=torch.float16
            )
            pipe = StableDiffusionControlNetInpaintPipeline.from_pretrained(
                "runwayml/stable-diffusion-v1-5",
                controlnet=controlnet,
                torch_dtype=torch.float16,
            )
            pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)

            pipe.enable_model_cpu_offload()

            self.ctn_inpaint = pipe
            self.ctn_inpaint.set_progress_bar_config(disable=True)
            self.ctn_inpaint.safety_checker = None

        with torch.no_grad():
            render_pkg = render(cam, self.gaussian, self.pipe, self.background_tensor)

        image_in = to_pil_image(torch.clip(render_pkg["render"], 0.0, 1.0))
        # image_in = to_pil_image(torch.clip(render_pkg["render"], 0.0, 1.0)*255)
        origin_size = image_in.size  # W, H

        frustum = None
        frame = None
        while True:
            if self.inpaint_again:
                if frustum is not None:
                    frustum.remove()
                    frame.remove()
                mask_in = torch.zeros(
                    (origin_size[1], origin_size[0]),
                    dtype=torch.float32,
                    device=get_device(),
                )  # H, W
                mask_in[
                    self.left_up.value[1] : self.right_down.value[1],
                    self.left_up.value[0] : self.right_down.value[0],
                ] = 1.0

                image_in_pil = to_pil_image(
                    ui_utils.resize_image_ctn(np.asarray(image_in), 512)
                )
                mask_in = to_pil_image(mask_in)  # .resize((1024, 1024))
                mask_in_pil = to_pil_image(
                    ui_utils.resize_image_ctn(np.asarray(mask_in)[..., None], 512)
                )

                image = np.array(image_in_pil.convert("RGB")).astype(np.float32) / 255.0
                image_mask = (
                    np.array(mask_in_pil.convert("L")).astype(np.float32) / 255.0
                )

                image[image_mask > 0.5] = -1.0  # set as masked pixel
                image = np.expand_dims(image, 0).transpose(0, 3, 1, 2)
                control_image = torch.from_numpy(image).to("cuda")
                generator = torch.Generator(device="cuda").manual_seed(
                    self.inpaint_seed.value
                )
                out = self.ctn_inpaint(
                    self.edit_text.value+", high quality, extremely detailed",
                    num_inference_steps=25,
                    generator=generator,
                    eta=1.0,
                    image=image_in_pil,
                    mask_image=mask_in_pil,
                    control_image=control_image,
                ).images[0]
                out = cv2.resize(
                    np.asarray(out),
                    origin_size,
                    interpolation=cv2.INTER_LANCZOS4
                    if out.width / origin_size[0] > 1
                    else cv2.INTER_AREA,
                )
                out = to_pil_image(out)
                frame, frustum = ui_utils.new_frustum_from_cam(
                    list(self.server.get_clients().values())[0].camera,
                    self.server,
                    np.asarray(out),
                )
                self.inpaint_again = False
            else:
                if self.stop_training:
                    self.stop_training = False
                    return
                time.sleep(0.1)
            if self.inpaint_end_flag:
                self.inpaint_end_flag = False
                break
        self.inpaint_seed.visible = False
        self.inpaint_end.visible = False
        self.edit_text.visible = False

        removed_bg = rembg.remove(out)
        inpainted_image = to_tensor(out).to("cuda")
        frustum.remove()
        frame.remove()
        frame, frustum = ui_utils.new_frustum_from_cam(
            list(self.server.get_clients().values())[0].camera,
            self.server,
            np.asarray(removed_bg),
        )

        cache_dir = Path("tmp_add").absolute().as_posix()
        os.makedirs(cache_dir, exist_ok=True)
        mv_image_dir = os.path.join(cache_dir, "multiview_pred_images")
        os.makedirs(mv_image_dir, exist_ok=True)
        inpaint_path = os.path.join(cache_dir, "inpainted.png")
        removed_bg_path = os.path.join(cache_dir, "removed_bg.png")
        mesh_path = os.path.join(cache_dir, "inpaint_mesh.obj")
        gs_path = os.path.join(cache_dir, "inpaint_gs.obj")
        out.save(inpaint_path)
        removed_bg.save(removed_bg_path)

        p1 = subprocess.Popen(
            f"{sys.prefix}/bin/accelerate launch --config_file 1gpu.yaml test_mvdiffusion_seq.py "
            f"--save_dir {mv_image_dir} --config configs/mvdiffusion-joint-ortho-6views.yaml"
            f" validation_dataset.root_dir={cache_dir} validation_dataset.filepaths=[removed_bg.png]".split(
                " "
            ),
            cwd="threestudio/utils/wonder3D",
        )
        p1.wait()

        print(
            f"{sys.prefix}/bin/python launch.py --config configs/neuralangelo-ortho-wmask.yaml --save_dir {cache_dir} --gpu 0 --train dataset.root_dir={os.path.dirname(mv_image_dir)} dataset.scene={os.path.basename(mv_image_dir)}"
        )
        cmd = f"{sys.prefix}/bin/python launch.py --config configs/neuralangelo-ortho-wmask.yaml --save_dir {cache_dir} --gpu 0 --train dataset.root_dir={os.path.dirname(mv_image_dir)} dataset.scene={os.path.basename(mv_image_dir)}".split(
            " "
        )
        p2 = subprocess.Popen(
            cmd,
            cwd="threestudio/utils/wonder3D/instant-nsr-pl",
        )
        p2.wait()

        p3 = subprocess.Popen(
            [
                f"{sys.prefix}/bin/python",
                "train_from_mesh.py",
                "--mesh",
                mesh_path,
                "--save_path",
                gs_path,
                "--prompt",
                self.refine_text.value,
            ]
        )
        p3.wait()

        frustum.remove()
        frame.remove()

        object_mask = np.array(removed_bg)
        object_mask = object_mask[:, :, 3] > 0
        object_mask = torch.from_numpy(object_mask)
        bbox = masks_to_boxes(object_mask[None])[0].to("cuda")

        depth_estimator = DPT(get_device(), mode="depth")

        estimated_depth = depth_estimator(
            inpainted_image.moveaxis(0, -1)[None, ...]
        ).squeeze()
        # ui_utils.vis_depth(estimated_depth.cpu())
        object_center = (bbox[:2] + bbox[2:]) / 2

        fx = fov2focal(cam.FoVx, cam.image_width)
        fy = fov2focal(cam.FoVy, cam.image_height)

        object_center = (
            object_center
            - torch.tensor([cam.image_width, cam.image_height]).to("cuda") / 2
        ) / torch.tensor([fx, fy]).to("cuda")

        rendered_depth = render_pkg["depth_3dgs"][..., ~object_mask]

        inpainted_depth = estimated_depth[~object_mask]
        object_depth = estimated_depth[..., object_mask]

        min_object_depth = torch.quantile(object_depth, 0.05)
        max_object_depth = torch.quantile(object_depth, 0.95)
        obj_depth_scale = (max_object_depth - min_object_depth) * 1

        min_valid_depth_mask = (min_object_depth - obj_depth_scale) < inpainted_depth
        max_valid_depth_mask = inpainted_depth < (max_object_depth + obj_depth_scale)
        valid_depth_mask = torch.logical_and(min_valid_depth_mask, max_valid_depth_mask)
        valid_percent = valid_depth_mask.sum() / min_valid_depth_mask.shape[0]
        print("depth valid percent: ", valid_percent)

        rendered_depth = rendered_depth[0, valid_depth_mask]
        inpainted_depth = inpainted_depth[valid_depth_mask.squeeze()]

        ## assuming rendered_depth = a * estimated_depth + b
        y = rendered_depth
        x = inpainted_depth
        a = (torch.sum(x * y) - torch.sum(x) * torch.sum(y)) / (
            torch.sum(x**2) - torch.sum(x) ** 2
        )
        b = torch.sum(y) - a * torch.sum(x)

        z_in_cam = object_depth.min() * a + b

        self.depth_scaler.visible = True
        self.depth_end.visible = True
        self.refine_text.visible = True

        new_object_gaussian = None
        while True:
            if self.scale_depth:
                if new_object_gaussian is not None:
                    self.gaussian.prune_with_mask()
                scaled_z_in_cam = z_in_cam * self.depth_scaler.value
                x_in_cam, y_in_cam = (object_center.cuda()) * scaled_z_in_cam
                T_in_cam = torch.stack([x_in_cam, y_in_cam, scaled_z_in_cam], dim=-1)

                bbox = bbox.cuda()
                real_scale = (
                    (bbox[2:] - bbox[:2])
                    / torch.tensor([fx, fy], device="cuda")
                    * scaled_z_in_cam
                )

                new_object_gaussian = VanillaGaussianModel(self.gaussian.max_sh_degree)
                new_object_gaussian.load_ply(gs_path)
                new_object_gaussian._opacity.data = (
                    torch.ones_like(new_object_gaussian._opacity.data) * 99.99
                )

                new_object_gaussian._xyz.data -= new_object_gaussian._xyz.data.mean(
                    dim=0, keepdim=True
                )
                rotate_gaussians(new_object_gaussian, default_model_mtx.T)

                object_scale = (
                    new_object_gaussian._xyz.data.max(dim=0)[0]
                    - new_object_gaussian._xyz.data.min(dim=0)[0]
                )[:2]

                relative_scale = (real_scale / object_scale).mean()
                print(relative_scale)

                scale_gaussians(new_object_gaussian, relative_scale)

                new_object_gaussian._xyz.data += T_in_cam

                R = torch.from_numpy(cam.R).float().cuda()
                T = -R @ torch.from_numpy(cam.T).float().cuda()

                rotate_gaussians(new_object_gaussian, R)
                translate_gaussians(new_object_gaussian, T)

                self.gaussian.concat_gaussians(new_object_gaussian)
                self.scale_depth = False
            else:
                if self.stop_training:
                    self.stop_training = False
                    return
                time.sleep(0.01)
            if self.depth_end_flag:
                self.depth_end_flag = False
                break
        self.depth_scaler.visible = False
        self.depth_end.visible = False

    def densify_and_prune(self, step):
        if step <= self.densify_until_step.value:
            self.gaussian.max_radii2D[self.visibility_filter] = torch.max(
                self.gaussian.max_radii2D[self.visibility_filter],
                self.radii[self.visibility_filter],
            )
            self.gaussian.add_densification_stats(
                self.viewspace_point_tensor.grad, self.visibility_filter
            )

            if step > 0 and step % self.densification_interval.value == 0:
                self.gaussian.densify_and_prune(
                    max_grad=1e-7,
                    max_densify_percent=self.max_densify_percent.value,
                    min_opacity=self.min_opacity.value,
                    extent=self.cameras_extent,
                    max_screen_size=5,
                )

    @torch.no_grad()
    def render_cameras_list(self, edit_cameras):
        origin_frames = []
        for cam in edit_cameras:
            out = self.render(cam)["comp_rgb"]
            origin_frames.append(out)

        return origin_frames

    @torch.no_grad()
    def render_all_view_with_mask(self, edit_cameras, train_frames, train_frustums):
        inpaint_2D_mask = []
        origin_frames = []

        for idx, cam in enumerate(edit_cameras):
            res = self.render(cam)
            rgb, mask = res["comp_rgb"], res["masks"]
            mask = dilate_mask(mask.to(torch.float32), self.mask_dilate.value)
            if self.fix_holes.value:
                mask = fill_closed_areas(mask)
            inpaint_2D_mask.append(mask)
            origin_frames.append(rgb)
            train_frustums[idx].remove()
            mask_view = torch.stack([mask] * 3, dim=3)  # 1 H W C
            train_frustums[idx] = ui_utils.new_frustums(
                idx, train_frames[idx], cam, mask_view, True, self.server
            )
        return inpaint_2D_mask, origin_frames

    def add_theme(self):
        buttons = (
            TitlebarButton(
                text="Getting Started",
                icon=None,
                href="https://github.com/buaacyw/GaussianEditor/blob/master/docs/webui.md",
            ),
            TitlebarButton(
                text="Github",
                icon="GitHub",
                href="https://github.com/buaacyw/GaussianEditor",
            ),
            TitlebarButton(
                text="Yiwen Chen",
                icon=None,
                href="https://buaacyw.github.io/",
            ),
            TitlebarButton(
                text="Zilong Chen",
                icon=None,
                href="https://scholar.google.com/citations?user=2pbka1gAAAAJ&hl=en",
            ),
        )
        image = TitlebarImage(
            image_url_light="https://github.com/buaacyw/gaussian-editor/raw/master/static/images/logo.png",
            image_alt="GaussianEditor Logo",
            href="https://buaacyw.github.io/gaussian-editor/",
        )
        titlebar_theme = TitlebarConfig(buttons=buttons, image=image)
        brand_color = self.server.add_gui_rgb("Brand color", (7, 0, 8), visible=False)

        self.server.configure_theme(
            titlebar_content=titlebar_theme,
            show_logo=True,
            brand_color=brand_color.value,
        )


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--gs_source", type=str, required=True)  # gs ply or obj file?
    parser.add_argument("--colmap_dir", type=str, required=True)  #

    args = parser.parse_args()
    webui = WebUI(args)
    webui.render_loop()
