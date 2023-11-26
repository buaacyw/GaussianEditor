from dataclasses import dataclass

from tqdm import tqdm
import cv2
import numpy as np
import torch
import threestudio
import os

from threestudio.utils.misc import dilate_mask, fill_closed_areas
from threestudio.systems.GassuianEditor import GaussianEditor


@threestudio.register("gsedit-system-del")
class GaussianEditor_Del(GaussianEditor):
    @dataclass
    class Config(GaussianEditor.Config):
        fix_holes: bool = True
        mask_dilate: int = 5
        inpaint_scale: float = 0.25
        inpaint_prompt: str = ""

    cfg: Config

    def configure(self) -> None:
        super().configure()
        if len(self.cfg.cache_dir) > 0:
            self.cache_dir = os.path.join("del_cache", self.cfg.cache_dir)
        else:
            self.cache_dir = os.path.join("del_cache", self.cfg.gs_source.replace("/", "-"))

        self.masks_2D = {}

    def on_fit_start(self) -> None:
        super().on_fit_start()
        self.render_all_view(cache_name="origin_render")
        assert len(self.cfg.seg_prompt) > 0, "please specify prompt system.seg_prompt for Delete"
        self.update_mask()
        dist_thres = (
                self.cfg.inpaint_scale
                * self.cameras_extent
                * self.gaussian.percent_dense
        )
        valid_remaining_idx = self.gaussian.get_near_gaussians_by_mask(
            self.gaussian.mask, dist_thres
        )
        # Prune and update mask to valid_remaining_idx
        self.gaussian.prune_with_mask(new_mask=valid_remaining_idx)
        # Provide pruned mask for inpainting
        self.render_all_view_with_mask(
            cache_name=self.cfg.seg_prompt + f"_pruned_mask_scale_{self.cfg.inpaint_scale}_{self.view_num}_view")
        # Provide render for inpainting
        self.render_all_view(
            cache_name=self.cfg.seg_prompt + f"_pruned_{self.view_num}_view"
        )
        self.inpaint_with_mask_ctn(
            cache_name=self.cfg.seg_prompt
                       + f"_scale_{self.cfg.inpaint_scale}_{self.view_num}_view_inpaint_ctn",
        )

        if len(self.cfg.prompt_processor) > 0:
            self.prompt_processor = threestudio.find(self.cfg.prompt_processor_type)(
                self.cfg.prompt_processor
            )
        if self.cfg.loss.lambda_l1 > 0 or self.cfg.loss.lambda_p > 0:
            self.guidance = threestudio.find(self.cfg.guidance_type)(self.cfg.guidance)

    @torch.no_grad()
    def inpaint_with_mask_ctn(self, cache_name) -> None:
        pipe = None
        threestudio.info(f"CTN Inpaint with masks: {self.cfg.seg_prompt}")
        inpaint_cache_dir = os.path.join(self.cache_dir, cache_name)
        os.makedirs(inpaint_cache_dir, exist_ok=True)

        for i, id in tqdm(enumerate(self.view_list)):
            cur_path = os.path.join(inpaint_cache_dir, "{:0>4d}.png".format(id))
            if not os.path.exists(cur_path) or self.cfg.cache_overwrite:
                height, width, _ = self.origin_frames[id][0].shape
                if pipe is None:
                    from diffusers import StableDiffusionControlNetInpaintPipeline, ControlNetModel, DDIMScheduler
                    controlnet = ControlNetModel.from_pretrained(
                        "lllyasviel/control_v11p_sd15_inpaint", torch_dtype=torch.float16
                    )
                    pipe = StableDiffusionControlNetInpaintPipeline.from_pretrained(
                        "runwayml/stable-diffusion-v1-5", controlnet=controlnet, torch_dtype=torch.float16
                    )
                    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)

                    pipe.enable_model_cpu_offload()
                    num_channels_latents = pipe.vae.config.latent_channels
                    shape = (1, num_channels_latents, height // pipe.vae_scale_factor, width // pipe.vae_scale_factor)
                    latents = torch.zeros(shape, dtype=torch.float16, device="cuda")
                    pipe.safety_checker = None

                from torchvision.transforms.functional import to_pil_image
                image_in = to_pil_image(self.origin_frames[id][0].permute(2, 0, 1))
                mask_in = to_pil_image(torch.concatenate([self.masks_2D[id] * 255] * 3))

                def make_inpaint_condition(image, image_mask):
                    image = np.array(image.convert("RGB")).astype(np.float32) / 255.0
                    image_mask = np.array(image_mask.convert("L")).astype(np.float32) / 255.0

                    assert image.shape[0:1] == image_mask.shape[
                                               0:1], "image and image_mask must have the same image size"
                    image[image_mask > 0.5] = -1.0  # set as masked pixel
                    image = np.expand_dims(image, 0).transpose(0, 3, 1, 2)
                    image = torch.from_numpy(image)
                    return image

                control_image = make_inpaint_condition(image_in, mask_in).to("cuda")
                generator = torch.Generator(device="cuda").manual_seed(0)

                out = pipe(
                    self.cfg.inpaint_prompt,
                    num_inference_steps=20,
                    generator=generator,
                    eta=1.0,
                    image=image_in,
                    mask_image=mask_in,
                    control_image=control_image,
                    latents=latents,
                ).images[0]

                out.save(cur_path)

            cached_image = cv2.cvtColor(cv2.imread(cur_path), cv2.COLOR_BGR2RGB)
            self.edit_frames[id] = torch.tensor(
                cached_image / 255, device="cuda", dtype=torch.float32
            )[None]

    def render_all_view_with_mask(self, cache_name):
        cache_dir = os.path.join(self.cache_dir, cache_name)
        os.makedirs(cache_dir, exist_ok=True)
        with torch.no_grad():
            for id in tqdm(self.view_list):
                cur_path = os.path.join(cache_dir, "{:0>4d}.png".format(id))
                if not os.path.exists(cur_path) or self.cfg.cache_overwrite:
                    cur_cam = self.trainer.datamodule.train_dataset.scene.cameras[id]
                    cur_batch = {
                        "index": id,
                        "camera": [cur_cam],
                        "height": self.trainer.datamodule.train_dataset.height,
                        "width": self.trainer.datamodule.train_dataset.width,
                    }
                    out = self(cur_batch)["masks"]
                    out = dilate_mask(out.to(torch.float32), self.cfg.mask_dilate)
                    if self.cfg.fix_holes:
                        out = fill_closed_areas(out)
                    out_to_save = (
                            out[0].cpu().detach().numpy().clip(0.0, 1.0) * 255.0
                    ).astype(np.uint8)
                    out_to_save = cv2.cvtColor(out_to_save, cv2.COLOR_RGB2BGR)
                    cv2.imwrite(cur_path, out_to_save)
                cached_image = cv2.cvtColor(cv2.imread(cur_path), cv2.COLOR_BGR2RGB)
                self.masks_2D[id] = torch.tensor(
                    cached_image / 255, device="cuda", dtype=torch.uint8
                )[..., 0][None]

    def training_step(self, batch, batch_idx):
        self.gaussian.update_learning_rate(self.true_global_step)

        batch_index = batch["index"]
        if isinstance(batch_index, int):
            batch_index = [batch_index]
        out = self(batch)

        images = out["comp_rgb"]

        loss = 0.0

        if self.cfg.loss.lambda_l1 > 0 or self.cfg.loss.lambda_p > 0:

            inpainted_images = []
            for img_index, cur_index in enumerate(batch_index):
                inpainted_images.append(self.edit_frames[cur_index])

            inpainted_images = torch.concatenate(inpainted_images, dim=0)

            inpaint_guidance_out = {
                "loss_l1": torch.nn.functional.l1_loss(images, inpainted_images),
                "loss_p": self.perceptual_loss(
                    images.permute(0, 3, 1, 2).contiguous(),
                    inpainted_images.permute(0, 3, 1, 2).contiguous(),
                ).sum(),
            }
            for name, value in inpaint_guidance_out.items():
                self.log(f"train/{name}", value)
                if name.startswith("loss_"):
                    loss += value * self.C(
                        self.cfg.loss[name.replace("loss_", "lambda_")]
                    )

        if (
                self.cfg.loss.lambda_anchor_color > 0
                or self.cfg.loss.lambda_anchor_geo > 0
                or self.cfg.loss.lambda_anchor_scale > 0
                or self.cfg.loss.lambda_anchor_opacity > 0
        ):
            anchor_out = self.gaussian.anchor_loss()
            for name, value in anchor_out.items():
                self.log(f"train/{name}", value)
                if name.startswith("loss_"):
                    loss += value * self.C(
                        self.cfg.loss[name.replace("loss_", "lambda_")]
                    )

        for name, value in self.cfg.loss.items():
            self.log(f"train_params/{name}", self.C(value))

        return {"loss": loss}
