import os
import numpy as np
import torch
from random import randint
from gaussiansplatting.utils.loss_utils import l1_loss, ssim
from gaussiansplatting.gaussian_renderer import render
import sys
from threestudio.utils.perceptual import PerceptualLoss

# from gaussiansplatting.scene import Scene, GaussianModel
from gaussiansplatting.scene.vanilla_gaussian_model import GaussianModel as Vanilla_GaussianModel
from gaussiansplatting.scene.gaussian_model import GaussianModel

from gaussiansplatting.utils.general_utils import safe_state
import uuid
from tqdm import tqdm, trange
from gaussiansplatting.utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from gaussiansplatting.arguments import ModelParams, PipelineParams, OptimizationParams
from gaussiansplatting.utils.graphics_utils import BasicPointCloud
from mediapy import write_image

from threestudio.utils.render import render_multiview_images_from_mesh
from threestudio.utils.mesh import load_mesh_as_pcd_trimesh
from omegaconf import OmegaConf
from threestudio.models.prompt_processors.stable_diffusion_prompt_processor import StableDiffusionPromptProcessor


def replace_filename(path, new_filename):
    dir_name = os.path.dirname(path)
    new_path = os.path.join(dir_name, new_filename)
    return new_path


def get_scene_radius(c2ws, scale: float = 1.1):
    camera_centers = c2ws[..., :3, 3]
    camera_centers = np.linalg.norm(
        camera_centers - np.mean(camera_centers, axis=0, keepdims=True), axis=-1
    )
    return np.max(camera_centers) * scale


def training(
        dataset,
        opt,
        pipe,
        den_percent,
        prompt,
        per_editing_step,
        coarse_iteration,
        coarse_path,
        save_path,
        hori_cams,
        cams,
        gt_images,
        mesh_file,
        camera_extent,
        args
):
    # poses = None
    # cams = []
    # gt_images = []
    val_dir = replace_filename(save_path, "val")
    os.makedirs(val_dir, exist_ok=True)
    gt_images = torch.from_numpy(gt_images)
    gt_images = gt_images.moveaxis(-1, 1)

    opt.position_lr_init = 0
    opt.position_lr_final = 0
    opt.scaling_lr = 0
    opt.rotation_lr = 0
    opt.opacity_lr = 0
    opt.feature_lr = 0.00625

    background = torch.tensor([1.0, 1.0, 1.0], dtype=torch.float32, device="cuda")
    gaussians = GaussianModel(3, 1, 1, 1.5)

    xyz, color = load_mesh_as_pcd_trimesh(mesh_file, 200000)
    pcd = BasicPointCloud(xyz, color, None)

    gaussians.create_from_pcd(pcd, camera_extent)
    # scene = Scene(dataset, gaussians)
    gaussians.training_setup(opt)

    prompt_processor = StableDiffusionPromptProcessor(
        {
            "pretrained_model_name_or_path": "runwayml/stable-diffusion-v1-5",
            "prompt": prompt,
        }
    )
    prompt_utils = prompt_processor()
    prompt_processor_1024 = StableDiffusionPromptProcessor(
        {
            "pretrained_model_name_or_path": "stabilityai/stable-diffusion-2-1-base",
            "prompt": prompt,
        }
    )
    prompt_utils_1024 = prompt_processor_1024()

    if "n2n" in args.guide:
        from threestudio.models.guidance.instructpix2pix_guidance import InstructPix2PixGuidance
        pix2pix = InstructPix2PixGuidance(OmegaConf.create(
            {
                "min_step_percent": 0.02,
                "max_step_percent": 0.98
            }))

    edit_frames = {}
    origin_frames = {}
    H, W = cams[0].image_height, cams[0].image_width
    device = "cuda"

    perceptual_loss = PerceptualLoss().eval().to(device)

    for iteration in trange(opt.iterations + 1):
        gaussians.update_learning_rate(iteration)
        if iteration < coarse_iteration:
            cur_index = randint(0, len(cams) - 1)
            cam = cams[cur_index]
            gt_image = gt_images[cur_index].cuda()
        else:
            cur_index = randint(0, len(hori_cams) - 1)
            cam = hori_cams[cur_index]

        render_pkg = render(cam, gaussians, pipe, background)
        image, viewspace_point_tensor, visibility_filter, radii = (
            render_pkg["render"],
            render_pkg["viewspace_points"],
            render_pkg["visibility_filter"],
            render_pkg["radii"],
        )
        loss = 0

        if iteration < coarse_iteration:
            Ll1 = l1_loss(image, gt_image)
            loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (
                    1.0 - ssim(image, gt_image)
            )
            loss.backward()
        else:
            if "n2n" in args.guide:
                if cur_index not in edit_frames or (
                        iteration % per_editing_step == 0
                ):
                    result = pix2pix(
                        image[None].permute(0, 2, 3, 1),  #
                        origin_frames[cur_index][None].permute(0, 2, 3, 1),
                        prompt_utils,
                    )
                    edited_image = torch.nn.functional.interpolate(
                        result["edit_images"].permute(0, 3, 1, 2),
                        (H, W),
                        mode="bilinear",
                        align_corners=False,
                    )[0]
                    edit_frames[cur_index] = edited_image.detach()
                    print("edited image index", cur_index)

                gt_image = edit_frames[cur_index]

                loss_l1 = torch.nn.functional.l1_loss(image, gt_image)
                loss_p = perceptual_loss(
                    image[None].contiguous(),
                    gt_image[None].contiguous(),
                ).sum()
                loss += 10 * loss_l1 + 10 * loss_p

            if "anchor" in args.guide:
                anchor_out = gaussians.anchor_loss()
                loss += 0 * anchor_out['loss_anchor_color'] + 5 * anchor_out['loss_anchor_geo'] + 0 * anchor_out[
                    'loss_anchor_opacity'] + 5 * anchor_out['loss_anchor_scale']

            loss.backward()

        with torch.no_grad():
            if iteration == coarse_iteration - 1:
                print("Begin Refine")
                gaussians.save_ply(coarse_path)
                for cam_index, cam in enumerate(hori_cams):
                    render_pkg = render(cam, gaussians, pipe, background)
                    origin_frames[cam_index] = render_pkg["render"]

            # Progress bar
            if iteration % 100 == 0:
                print("\n[ITER {}] Saving Rendered Images".format(iteration))
                # scene.save(iteration)
                out_image = (
                        image.moveaxis(0, -1).cpu().numpy().clip(0.0, 1.0) * 255
                ).astype(np.uint8)
                gt_out_image = (
                        gt_image.detach().moveaxis(0, -1).cpu().numpy().clip(0.0, 1.0) * 255
                ).astype(np.uint8)
                write_image(os.path.join(val_dir, f"gt_{iteration}.png"), gt_out_image)
                write_image(os.path.join(val_dir, f"{iteration}.png"), out_image)

            # Densification
            if iteration < opt.iterations-300:
                # Keep track of max radii in image-space for pruning
                gaussians.max_radii2D[visibility_filter] = torch.max(
                    gaussians.max_radii2D[visibility_filter], radii[visibility_filter]
                )
                gaussians.add_densification_stats(
                    viewspace_point_tensor.grad, visibility_filter
                )

                if (
                        iteration > opt.densify_from_iter
                        and iteration % opt.densification_interval == 0
                ):
                    size_threshold = (
                        20 if iteration > coarse_iteration else None
                    )
                    gaussians.densify_and_prune(
                        opt.densify_grad_threshold,
                        den_percent,
                        0.005,
                        # TODO: calculate camera extent
                        camera_extent,
                        size_threshold,
                    )

                if (iteration % opt.opacity_reset_interval == 0 and iteration > 0) or (
                        dataset.white_background and iteration == opt.densify_from_iter
                ):
                    gaussians.reset_opacity()

            # Optimizer step
            if iteration < opt.iterations:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none=True)

    gaussians.save_ply(save_path)
    # if ".obj" in save_path:
    #     gaussians.save_ply(save_path.replace(".obj",".ply"))


if __name__ == "__main__":
    max_steps = 1800
    parser = ArgumentParser()
    lp = ModelParams(parser)
    op = OptimizationParams(parser, max_steps= max_steps)
    pp = PipelineParams(parser)


    parser.add_argument("--save_path", type=str, required=True)
    parser.add_argument("--prompt", type=str, required=True)
    parser.add_argument("--mesh", type=str, required=True)
    parser.add_argument("--guide", type=str, default="n2n")

    parser.add_argument("--port", type=int, default=6009)
    parser.add_argument("--per_editing_step", type=int, default=50000000)
    parser.add_argument("--den_percent", type=float, default=0.0)
    parser.add_argument("--coarse_iteration", type=int, default=900)


    args = parser.parse_args(sys.argv[1:])
    os.makedirs(os.path.dirname(args.save_path), exist_ok=True)

    coarse_path = replace_filename(args.save_path, "coarse_gs.obj")
    # TODO: add more arguments
    cams, images, camera_extent = render_multiview_images_from_mesh(args.mesh,
                                                                    save_path=replace_filename(args.save_path,
                                                                                               "mesh_rendering.mp4"))
    hori_cams, _, _ = render_multiview_images_from_mesh(args.mesh, horizontal=True,
                                                        save_path=replace_filename(args.save_path,
                                                                                   "horizontal_mesh_rendering.mp4"))
    # hard code
    args.iterations = max_steps
    args.densify_grad_threshold = 0.00005
    args.opacity_reset_interval = 100000000

    # if not os.path.exists(coarse_path):w
    training(
        lp.extract(args),
        op.extract(args),
        pp.extract(args),
        args.den_percent,
        args.prompt,
        args.per_editing_step,
        args.coarse_iteration,
        coarse_path,
        args.save_path,
        hori_cams,
        cams,
        images,
        args.mesh,
        camera_extent,
        args
    )
