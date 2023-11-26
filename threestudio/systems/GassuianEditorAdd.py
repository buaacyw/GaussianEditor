from dataclasses import dataclass

from PIL import Image
import numpy as np
import sys
import torch
import threestudio
import os
from pathlib import Path
import subprocess
import rembg

from gaussiansplatting.gaussian_renderer import render
from gaussiansplatting.scene import GaussianModel
from gaussiansplatting.scene.vanilla_gaussian_model import (
    GaussianModel as VanillaGaussianModel,
)
from gaussiansplatting.utils.graphics_utils import fov2focal

from threestudio.utils.misc import get_device
from threestudio.utils.perceptual import PerceptualLoss
from threestudio.utils.sam import LangSAMTextSegmentor
from threestudio.utils.dpt import DPT

from threestudio.systems.GassuianEditor import GaussianEditor

@threestudio.register("gsedit-system-add")
class GaussianEditor_Add(GaussianEditor):
    @dataclass
    class Config(GaussianEditor.Config):
        inpaint_prompt: str = "a dog on the bench"
        refine_prompt: str = "make it a dog"

    cfg: Config

    def configure(self) -> None:
        super().configure()
        if len(self.cfg.cache_dir) > 0:
            self.cache_dir = os.path.join("add_cache", self.cfg.cache_dir)
        else:
            self.cache_dir = os.path.join("add_cache", self.cfg.gs_source.replace("/", "-"))

    @torch.no_grad()
    def on_fit_start(self) -> None:
        super().on_fit_start()

        print(self.gaussian.max_sh_degree)
        from torchvision.transforms.functional import to_pil_image, to_tensor
        from torchvision.ops import masks_to_boxes
        from diffusers import AutoPipelineForInpainting

        align_cache_dir = os.path.join(
            self.cache_dir, f"inpaint_{self.cfg.inpaint_prompt.replace(' ', '_')}"
        )
        align_cache_dir = Path(align_cache_dir).absolute().as_posix()
        print(align_cache_dir)
        os.makedirs(align_cache_dir, exist_ok=True)
        print("\n\n")
        print(self.cfg.inpaint_prompt)
        print("\n\n")
        render_path = os.path.join(align_cache_dir, "origin_rendering.png")
        inpaint_path = os.path.join(align_cache_dir, "inpainted.png")
        removed_bg_path = os.path.join(align_cache_dir, "removed_bg.png")
        mask_path = os.path.join(align_cache_dir, "mask.png")
        mv_image_dir = os.path.join(align_cache_dir, "multiview_pred_images")
        os.makedirs(mv_image_dir, exist_ok=True)
        mesh_path = os.path.join(align_cache_dir, "inpaint_mesh.obj")
        gs_path = os.path.join(align_cache_dir, "inpaint_gs.obj")
        merged_ply_path = os.path.join(align_cache_dir, "merged.ply")

        id = 24  # test for bicycle
        cam = self.trainer.datamodule.train_dataset.scene.cameras[id].HW_scale(
            1024, 1024
        )

        render_pkg = render(cam, self.gaussian, self.pipe, self.background_tensor)

        if self.cfg.cache_overwrite or not (
                os.path.exists(inpaint_path) and os.path.exists(inpaint_path)
        ):
            inpaint_model = AutoPipelineForInpainting.from_pretrained(
                "diffusers/stable-diffusion-xl-1.0-inpainting-0.1",
                torch_dtype=torch.float16,
                variant="fp16",
            ).to("cuda")
            generator = torch.Generator(device="cuda").manual_seed(0)
            image_in = to_pil_image(torch.clip(render_pkg["render"], 0.0, 1.0))

            mask_in = torch.zeros(
                image_in.size, dtype=torch.float32, device=get_device()
            )
            h, w = mask_in.shape[0], mask_in.shape[1]
            mask_in[int(h * 0.25): int(h * 0.75), int(w * 0.25): int(w * 0.75)] = 1.0
            mask_in = to_pil_image(mask_in).resize((1024, 1024))
            mask_in.save(mask_path)
            height, width = 1024, 1024

            image_in.save(render_path)
            out = inpaint_model(
                prompt=self.cfg.inpaint_prompt,
                image=image_in,
                mask_image=mask_in,
                guidance_scale=7.5,
                num_inference_steps=20,
                strength=0.99,
                generator=generator,
            ).images[0]

            out = out.resize((height, width))
            out.save(inpaint_path)

            removed_bg = rembg.remove(out)
            removed_bg.save(removed_bg_path)
        else:
            out = Image.open(inpaint_path)
            removed_bg = Image.open(removed_bg_path)

        inpainted_image = to_tensor(out).to("cuda")

        if self.cfg.cache_overwrite or len(os.listdir(mv_image_dir)) != 14:
            p1 = subprocess.Popen(
                f"{sys.prefix}/bin/accelerate launch --config_file 1gpu.yaml test_mvdiffusion_seq.py "
                f"--save_dir {mv_image_dir} --config configs/mvdiffusion-joint-ortho-6views.yaml"
                f" validation_dataset.root_dir={align_cache_dir} validation_dataset.filepaths=[removed_bg.png]".split(
                    " "
                ),
                cwd="threestudio/utils/wonder3D",
            )
            p1.wait()

        if self.cfg.cache_overwrite or not os.path.exists(mesh_path):
            print(
                f"{sys.prefix}/bin/python launch.py --config configs/neuralangelo-ortho-wmask.yaml --save_dir {align_cache_dir} --gpu 0 --train dataset.root_dir={os.path.dirname(mv_image_dir)} dataset.scene={os.path.basename(mv_image_dir)}"
            )
            cmd = f"{sys.prefix}/bin/python launch.py --config configs/neuralangelo-ortho-wmask.yaml --save_dir {align_cache_dir} --gpu 0 --train dataset.root_dir={os.path.dirname(mv_image_dir)} dataset.scene={os.path.basename(mv_image_dir)}".split(
                " "
            )
            p2 = subprocess.Popen(
                cmd,
                cwd="threestudio/utils/wonder3D/instant-nsr-pl",
            )
            p2.wait()

        if self.cfg.cache_overwrite or not os.path.exists(gs_path):
            p3 = subprocess.Popen(
                [
                    f"{sys.prefix}/bin/python",
                    "train_from_mesh.py",
                    "--mesh",
                    mesh_path,
                    "--save_path",
                    gs_path,
                    "--prompt",
                    self.cfg.refine_prompt,
                ]
            )
            p3.wait()

        object_mask = np.array(removed_bg)
        object_mask = object_mask[:, :, 3] > 0
        object_mask = torch.from_numpy(object_mask)
        bbox = masks_to_boxes(object_mask[None])[0].to("cuda")
        # repo = "isl-org/ZoeDepth"
        # depth_estimator = torch.hub.load(repo, "ZoeD_N", pretrained=True).to("cuda")

        # from transformers import pipeline
        # checkpoint = "vinvino02/glpn-nyu"
        # depth_estimator = pipeline("depth-estimation", model=checkpoint)
        # repo = "isl-org/ZoeDepth"
        # # depth_estimator = torch.hub.load(repo, "ZoeD_NK", pretrained=False)
        # depth_estimator = torch.hub.load("threestudio/utils/ZoeDepth", "ZoeD_NK", source="local", pretrained=False)
        #
        # pretrained_dict = torch.hub.load_state_dict_from_url(
        #     'https://github.com/isl-org/ZoeDepth/releases/download/v1.0/ZoeD_M12_NK.pt', map_location='cpu')
        # depth_estimator.load_state_dict(pretrained_dict['model'], strict=False)
        # for b in depth_estimator.core.core.pretrained.model.blocks:
        #     b.drop_path = torch.nn.Identity()
        # depth_estimator.to("cuda")

        # res = depth_estimator(to_pil_image(inpainted_image))
        # estimated_depth = res["predicted_depth"][0].to("cuda")
        depth_estimator = DPT(get_device(), mode="depth")

        estimated_depth = depth_estimator(
            inpainted_image.moveaxis(0, -1)[None, ...]
        ).squeeze()
        object_center = (bbox[:2] + bbox[2:]) / 2

        fx = fov2focal(cam.FoVx, cam.image_width)
        fy = fov2focal(cam.FoVy, cam.image_height)

        ## assume center = reso // 2, i.e. centered camera
        object_center = (
                                object_center - torch.tensor([cam.image_width, cam.image_height]).to("cuda") / 2
                        ) / torch.tensor([fx, fy]).to("cuda")

        rendered_depth = render_pkg["depth_3dgs"][..., ~object_mask]

        inpainted_depth = estimated_depth[~object_mask]
        object_depth = estimated_depth[..., object_mask]

        min_object_depth = torch.quantile(object_depth, 0.05)
        max_object_depth = torch.quantile(object_depth, 0.95)
        obj_depth_scale = (max_object_depth - min_object_depth) * 2

        min_valid_depth_mask = (min_object_depth - obj_depth_scale) < inpainted_depth
        max_valid_depth_mask = inpainted_depth < (max_object_depth + obj_depth_scale)
        valid_depth_mask = torch.logical_and(min_valid_depth_mask, max_valid_depth_mask)
        valid_percent = valid_depth_mask.sum() / min_valid_depth_mask.shape[0]
        print("depth valid percent: ", valid_percent)
        # TODO check this
        # quantile = 0.5
        # valid_depth_mask = rendered_depth < torch.quantile(rendered_depth, quantile)

        rendered_depth = rendered_depth[0, valid_depth_mask]
        inpainted_depth = inpainted_depth[valid_depth_mask.squeeze()]

        ## assuming rendered_depth = a * estimated_depth + b
        y = rendered_depth
        x = inpainted_depth
        a = (torch.sum(x * y) - torch.sum(x) * torch.sum(y)) / (
                torch.sum(x ** 2) - torch.sum(x) ** 2
        )
        b = torch.sum(y) - a * torch.sum(x)

        # depth_scale *= 0.7

        # z_in_cam = object_depth.min() * depth_scale

        z_in_cam = object_depth.min() * a + b
        x_in_cam, y_in_cam = (object_center.cuda()) * z_in_cam
        T_in_cam = torch.stack([x_in_cam, y_in_cam, z_in_cam], dim=-1)

        bbox = bbox.cuda()
        real_scale = (
                (bbox[2:] - bbox[:2]) / torch.tensor([fx, fy], device="cuda") * z_in_cam
        )

        new_object_gaussian = VanillaGaussianModel(self.gaussian.max_sh_degree)
        new_object_gaussian.load_ply(gs_path)
        new_object_gaussian._opacity.data = (
                torch.ones_like(new_object_gaussian._opacity.data) * 99.99
        )

        from threestudio.utils.transform import (
            rotate_gaussians,
            translate_gaussians,
            scale_gaussians,
            default_model_mtx,
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

        self.gaussian.save_ply(merged_ply_path)

        self.render_all_view(cache_name="inpainted_render")
