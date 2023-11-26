import torch
from tqdm import tqdm
import numpy as np
import ui_utils
from threestudio.utils.misc import get_device, step_check, dilate_mask, erode_mask, fill_closed_areas
from threestudio.utils.perceptual import PerceptualLoss
from torchvision.transforms.functional import to_pil_image
from torchvision.transforms import ToTensor

from threestudio.models.prompt_processors.stable_diffusion_prompt_processor import StableDiffusionPromptProcessor

# Diffusion model (cached) + prompts + edited_frames + training config

class DelGuidance:
    def __init__(self, guidance, latents, gaussian, text_prompt,
                 lambda_l1, lambda_p, lambda_anchor_color, lambda_anchor_geo, lambda_anchor_scale, lambda_anchor_opacity,
                 train_frames, train_frustums, cams, server,):
        self.guidance = guidance # ctn-inpaint guidance
        self.latents = latents
        self.lambda_l1 = lambda_l1
        self.lambda_p = lambda_p
        self.lambda_anchor_color = lambda_anchor_color
        self.lambda_anchor_geo = lambda_anchor_geo
        self.lambda_anchor_scale = lambda_anchor_scale
        self.lambda_anchor_opacity = lambda_anchor_opacity
        self.gaussian = gaussian
        self.edit_frames = {}
        self.text_prompt = text_prompt
        self.cams = cams
        self.server = server
        self.train_frames = train_frames
        self.train_frustums = train_frustums
        self.visible = True

        self.prompt_utils = StableDiffusionPromptProcessor(
            {
                "pretrained_model_name_or_path": "runwayml/stable-diffusion-v1-5",
                "prompt": text_prompt,
            }
        )
        self.step = 0
        self.perceptual_loss = PerceptualLoss().eval().to(get_device())
        self.to_tensor = ToTensor()

    @torch.no_grad()
    def inpaint_with_mask_ctn(self, image_in, mask_in, view_index) -> None:
        image_in_pil = to_pil_image(image_in[0].permute(2, 0, 1)) # 1, H, W, C to C, H, W
        mask_in_pil = to_pil_image(torch.stack([mask_in[0].to(torch.uint8) * 255] * 3)) # C, H, W 255

        def make_inpaint_condition(image, image_mask):
            image = np.array(image.convert("RGB")).astype(np.float32) / 255.0
            image_mask = np.array(image_mask.convert("L")).astype(np.float32) / 255.0

            assert image.shape[0:1] == image_mask.shape[
                                       0:1], "image and image_mask must have the same image size"
            image[image_mask > 0.5] = -1.0  # set as masked pixel
            image = np.expand_dims(image, 0).transpose(0, 3, 1, 2)
            image = torch.from_numpy(image)
            return image

        control_image = make_inpaint_condition(image_in_pil, mask_in_pil).to("cuda")
        generator = torch.Generator(device="cuda").manual_seed(0)
        out = self.guidance(
            self.text_prompt,
            num_inference_steps=20,
            generator=generator,
            eta=1.0,
            image=image_in_pil,
            mask_image=mask_in_pil,
            control_image=control_image,
            latents=self.latents,
        ).images[0]

        self.edit_frames[view_index] = self.to_tensor(out).to("cuda")[None].permute(0,2,3,1) # 1 C H W to 1 H W C

        self.train_frustums[view_index].remove()
        self.train_frustums[view_index] = ui_utils.new_frustums(view_index, self.train_frames[view_index],
                                                                self.cams[view_index], self.edit_frames[view_index],
                                                                self.visible, self.server)
    def __call__(self, rendering, image_in, mask_in, view_index, step):
        self.gaussian.update_learning_rate(step)
        if view_index not in self.edit_frames:
            self.inpaint_with_mask_ctn(image_in, mask_in, view_index)

        gt_image = self.edit_frames[view_index]

        loss = self.lambda_l1 * torch.nn.functional.l1_loss(rendering, gt_image) + \
               self.lambda_p * self.perceptual_loss(rendering.permute(0, 3, 1, 2).contiguous(),
                                                        gt_image.permute(0, 3, 1, 2).contiguous(), ).sum() # 1 H W C to 1 C H W
        # anchor loss
        if (
                self.lambda_anchor_color > 0
                or self.lambda_anchor_geo > 0
                or self.lambda_anchor_scale > 0
                or self.lambda_anchor_opacity > 0
        ):
            anchor_out = self.gaussian.anchor_loss()
            loss += self.lambda_anchor_color * anchor_out['loss_anchor_color'] + \
                    self.lambda_anchor_geo * anchor_out['loss_anchor_geo'] + \
                    self.lambda_anchor_opacity * anchor_out['loss_anchor_opacity'] + \
                    self.lambda_anchor_scale * anchor_out['loss_anchor_scale']

        return loss