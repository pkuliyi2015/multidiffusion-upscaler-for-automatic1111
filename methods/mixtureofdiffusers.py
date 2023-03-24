import torch
from torch import Tensor

from modules import devices, shared, extra_networks
from modules.shared import state
from modules.prompt_parser import MulticondLearnedConditioning

from methods.abstractdiffusion import TiledDiffusion, BlendMode
from methods.utils import gaussian_weights

from ldm.models.diffusion.ddpm import LatentDiffusion


class MixtureOfDiffusers(TiledDiffusion):
    """
        Mixture-of-Diffusers Implementation
        https://github.com/albarji/mixture-of-diffusers
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.custom_weights = []
        self.per_tile_weights = gaussian_weights(self.tile_w, self.tile_h)

    @property
    def init_tile_weights(self) -> Tensor:
        return self.per_tile_weights

    def hook(self):
        if not hasattr(shared.sd_model, 'apply_model_original_md'):
            shared.sd_model.apply_model_original_md = shared.sd_model.apply_model
        shared.sd_model.apply_model = self.apply_model_hijack

    @staticmethod
    def unhook():
        if hasattr(shared.sd_model, 'apply_model_original_md'):
            shared.sd_model.apply_model = shared.sd_model.apply_model_original_md
            del shared.sd_model.apply_model_original_md

    def init_custom_bbox(self, draw_background, bbox_control_states):
        super().init_custom_bbox(draw_background, bbox_control_states)

        for bbox in self.custom_bboxes:
            if bbox.blend_mode == BlendMode.BACKGROUND:
                custom_weights = gaussian_weights(bbox.w, bbox.h)
                self.weights[bbox.slicer] += custom_weights
                self.custom_weights.append(custom_weights.unsqueeze(0).unsqueeze(0))
            else:
                self.custom_weights.append(None)

    def reset_buffer(self, x_in:Tensor):
        super().reset_buffer(x_in)

        if not hasattr(self, 'rescale_factor'):
            # The original gaussian weights can be extremely small, so we rescale them for numerical stability
            self.rescale_factor = 1 / self.weights
            # Meanwhile, we rescale the custom weights in advance to save time of slicing
            for bbox_id, bbox in enumerate(self.custom_bboxes):
                if bbox.blend_mode == BlendMode.BACKGROUND:
                    self.custom_weights[bbox_id] = self.custom_weights[bbox_id].to(device=x_in.device, dtype=x_in.dtype)
                    self.custom_weights[bbox_id] *= self.rescale_factor[bbox.slicer]

    def custom_apply_model(self, x_in, t_in, c_in, bbox_id, bbox, cond, uncond):
        if self.is_kdiff:
            return self.kdiff_custom_forward(x_in, c_in, cond, uncond, t_in, bbox_id, bbox, forward_func=shared.sd_model.apply_model_original_md)
        else:
            def forward_func(x, c, ts, unconditional_conditioning, *args, **kwargs):
                # copy from p_sample_ddim in ddim.py
                c_in = dict()
                for k in c:
                    if isinstance(c[k], list):
                        c_in[k] = [torch.cat([unconditional_conditioning[k][i], c[k][i]]) for i in range(len(c[k]))]
                    else:
                        c_in[k] = torch.cat([unconditional_conditioning[k], c[k]])
                self.set_controlnet_tensors(bbox_id, x.shape[0])
                return shared.sd_model.apply_model_original_md(x, ts, c_in)
            return self.ddim_custom_forward(x_in, c_in, cond, uncond, bbox, ts=t_in, forward_func=forward_func)

    @torch.no_grad()
    def apply_model_hijack(self, x_in:Tensor, t_in:Tensor, cond:MulticondLearnedConditioning):
        ''' Hook to UNet when predicting noise '''

        assert LatentDiffusion.apply_model

        # KDiffusion Compatibility
        c_in = cond
        N, C, H, W = x_in.shape
        assert H == self.h and W == self.w

        self.reset_buffer(x_in)

        breakpoint()

        # Global sampling
        if self.draw_background:
            for batch_id, bboxes in enumerate(self.batched_bboxes):
                if state.interrupted: return x_in

                x_tile_list = []
                t_tile_list = []
                attn_tile_list = []
                image_cond_list = []
                for bbox in bboxes:
                    x_tile_list.append(x_in[bbox.slicer])
                    t_tile_list.append(t_in)
                    if c_in is not None and isinstance(cond, dict):
                        image_cond = cond['c_concat'][0]
                        if image_cond.shape[2] == self.h and image_cond.shape[3] == self.w:
                            image_cond = image_cond[bbox.slicer]
                        image_cond_list.append(image_cond)
                        attn_tile = cond['c_crossattn'][0]
                        attn_tile_list.append(attn_tile)
                x_tile = torch.cat(x_tile_list, dim=0)
                t_tile = torch.cat(t_tile_list, dim=0)
                attn_tile = torch.cat(attn_tile_list, dim=0)
                image_cond_tile = torch.cat(image_cond_list, dim=0)
                c_tile = {'c_concat': [image_cond_tile], 'c_crossattn': [attn_tile]}
                
                # Controlnet tiling
                self.switch_controlnet_tensors(batch_id, N, len(bboxes), is_denoise=True)
                x_tile_out = shared.sd_model.apply_model_original_md(x_tile, t_tile, c_tile)  # here the x is the noise

                for i, bbox in enumerate(bboxes):
                    # This weights can be calcluated in advance, but will cost a lot of vram 
                    # when you have many tiles. So we calculate it here.
                    w = self.per_tile_weights * self.rescale_factor[bbox.slicer]
                    self.x_buffer[bbox.slicer] += x_tile_out[i*N:(i+1)*N, :, :, :] * w

                self.update_pbar()
        
        breakpoint()

        # Custom region sampling
        if len(self.custom_bboxes) > 0 and bbox.blend_mode == BlendMode.FOREGROUND:
            x_feather_buffer = torch.zeros_like(self.x_buffer)
            x_feather_mask   = torch.zeros_like(self.x_buffer)
            x_feather_count  = torch.zeros_like(self.x_buffer)
        else:
            x_feather_buffer = None

        if len(self.custom_bboxes) > 0:
            for bbox_id, bbox in enumerate(self.custom_bboxes):
                if not self.p.disable_extra_networks:
                    with devices.autocast():
                        extra_networks.activate(self.p, bbox.extra_network_data)

                x_tile = x_in[bbox.slicer]
                x_tile_out = self.custom_apply_model(x_tile, t_in, c_in, bbox_id, bbox, bbox.cond, bbox.uncond)

                if bbox.blend_mode == BlendMode.BACKGROUND:
                    self.x_buffer[bbox.slicer] += x_tile_out * self.custom_weights[bbox_id]
                elif bbox.blend_mode == BlendMode.FOREGROUND:
                    x_feather_buffer[bbox.slicer] += x_tile_out
                    x_feather_mask  [bbox.slicer] += bbox.feather_mask
                    x_feather_count [bbox.slicer] += 1

                self.update_pbar()

                if not self.p.disable_extra_networks:
                    with devices.autocast():
                        extra_networks.deactivate(self.p, bbox.extra_network_data)
        breakpoint()

        x_out = self.x_buffer
        if x_feather_buffer is not None:
            # Average overlapping feathered regions
            x_feather_buffer = torch.where(x_feather_count > 1, x_feather_buffer / x_feather_count, x_feather_buffer)
            x_feather_mask   = torch.where(x_feather_count > 1, x_feather_mask / x_feather_count, x_feather_mask)
            # Weighted average with original x_buffer
            x_out = torch.where(x_feather_count > 0, x_out * (1 - x_feather_mask) + x_feather_buffer * x_feather_mask, x_out)

        breakpoint()

        return x_out
