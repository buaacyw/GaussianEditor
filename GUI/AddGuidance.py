import torch

from threestudio.utils.misc import get_device, step_check, dilate_mask, erode_mask, fill_closed_areas
from threestudio.utils.perceptual import PerceptualLoss

from threestudio.models.prompt_processors.stable_diffusion_prompt_processor import StableDiffusionPromptProcessor


# Diffusion model (cached) + prompts + edited_frames + training config

class AddGuidance:
    def __init__(self, guidance, gaussian, origin_frames, text_prompt, cfg):
        self.guidance = guidance
        self.cfg = cfg
        self.gaussian = gaussian
        self.origin_frames = origin_frames
        self.edit_frames = {}
        self.prompt_utils = StableDiffusionPromptProcessor(
            {
                "pretrained_model_name_or_path": "runwayml/stable-diffusion-v1-5",
                "prompt": text_prompt,
            }
        )
        self.step = 0
        self.perceptual_loss = PerceptualLoss().eval().to(get_device())

    def train_step(self, rendering, view_index):
        self.gaussian.update_learning_rate(self.step)

        # nerf2nerf loss
        if view_index not in self.edit_frames or (
                self.cfg.per_editing_step > 0
                and self.cfg.edit_begin_step
                < self.step
                < self.cfg.edit_until_step
                and self.step % self.cfg.per_editing_step == 0
        ):
            result = self.guidance(
                rendering,
                self.origin_frames[view_index],
                self.prompt_utils,
            )
            self.edit_frames[view_index] = result["edit_images"].detach().clone()
            # print("edited image index", cur_index)

        gt_image = self.edit_frames[view_index]

        loss = self.cfg.lambda_l1 * torch.nn.functional.l1_loss(rendering, gt_image) + \
               self.cfg.lambda_p * self.perceptual_loss(rendering.permute(0, 3, 1, 2).contiguous(),
                                                        gt_image.permute(0, 3, 1, 2).contiguous(), ).sum()

        # anchor loss
        if (
                self.cfg.loss.lambda_anchor_color > 0
                or self.cfg.loss.lambda_anchor_geo > 0
                or self.cfg.loss.lambda_anchor_scale > 0
                or self.cfg.loss.lambda_anchor_opacity > 0
        ):
            anchor_out = self.gaussian.anchor_loss()
            loss += self.cfg.lambda_anchor_color * anchor_out['loss_anchor_color'] + \
                    self.cfg.lambda_anchor_geo * anchor_out['loss_anchor_geo'] + \
                    self.cfg.lambda_anchor_opacity * anchor_out['loss_anchor_opacity'] + \
                    self.cfg.lambda_anchor_scale * anchor_out['loss_anchor_scale']

        loss.backward()

        self.step += 1
