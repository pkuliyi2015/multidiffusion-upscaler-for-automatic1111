from typing import List, Dict, Callable

import torch
from torch import Tensor

from modules import devices, extra_networks
from modules.shared import state
from modules.prompt_parser import ScheduledPromptConditioning
from modules.sd_samplers_kdiffusion import CFGDenoiser
from modules.sd_samplers_compvis import VanillaStableDiffusionSampler

from k_diffusion.external import CompVisDenoiser

from methods.abstractdiffusion import TiledDiffusion, CustomBBox, BlendMode


class MultiDiffusion(TiledDiffusion):
    """
        Multi-Diffusion Implementation
        https://arxiv.org/abs/2302.08113
    """

    def __init__(self, p, *args, **kwargs):
        super().__init__(p, *args, **kwargs)
        assert p.sampler_name != 'UniPC', 'MultiDiffusion is not compatible with UniPC!'

        # For ddim sampler we need to cache the pred_x0
        self.x_pred_buffer = None

    def hook(self):
        if self.is_kdiff:
            # For K-Diffusion sampler with uniform prompt, we hijack into the inner model for simplicity
            # Otherwise, the masked-redraw will break due to the init_latent
            sampler: CFGDenoiser = self.sampler
            self.sampler_forward = sampler.inner_model.forward
            sampler.inner_model.forward = self.kdiff_forward
        else:
            sampler: VanillaStableDiffusionSampler = self.sampler
            self.sampler_forward = sampler.orig_p_sample_ddim
            sampler.orig_p_sample_ddim = self.ddim_forward

    @staticmethod
    def unhook():
        # NOTE: no need to unhook MultiDiffusion as it only hook the sampler,
        # which will be destroyed after the painting is done
        pass
    
    def init_custom_bbox(self, draw_background:bool, bbox_control_states):
        super().init_custom_bbox(draw_background, bbox_control_states)

        for bbox in self.custom_bboxes:
            if bbox.blend_mode == BlendMode.BACKGROUND:
                self.weights[bbox.slicer] += 1

    def reset_buffer(self, x_in: Tensor):
        super().reset_buffer(x_in)
        
        # ddim needs to cache pred0
        if not self.is_kdiff:
            if self.x_pred_buffer is None:
                self.x_pred_buffer = torch.zeros_like(x_in, device=x_in.device)
            else:
                self.x_pred_buffer.zero_()

    def repeat_cond_dict(self, cond_input, bboxes:List[CustomBBox]):
        cond = cond_input['c_crossattn'][0]
        # repeat the condition on its first dim
        cond_shape = cond.shape
        cond = cond.repeat((len(bboxes),) + (1,) * (len(cond_shape) - 1))
        image_cond = cond_input['c_concat'][0]
        if image_cond.shape[2] == self.h and image_cond.shape[3] == self.w:
            image_cond_list = []
            for bbox in bboxes:
                image_cond_list.append(image_cond[bbox.slicer])
            image_cond_tile = torch.cat(image_cond_list, dim=0)
        else:
            image_cond_shape = image_cond.shape
            image_cond_tile = image_cond.repeat((len(bboxes),) + (1,) * (len(image_cond_shape) - 1))
        return {"c_crossattn": [cond], "c_concat": [image_cond_tile]}

    @torch.no_grad()
    def kdiff_forward(self, x_in:Tensor, sigma_in:Tensor, cond:Dict[str, Tensor]):
        '''
        This function hijacks `k_diffusion.external.CompVisDenoiser.forward()`
        So its signature should be the same as the original function, especially the "cond" should be with exactly the same name
        '''

        assert CompVisDenoiser.forward
        # x_in: [B, C=4, H=64, W=64]
        # sigma_in： [1]
        # cond['c_crossattn'][0]: [1, 77, 768]

        def repeat_func(x_tile, bboxes):
            # For kdiff sampler, the dim 0 of input x_in is:
            #   = batch_size * (num_AND + 1)   if not an edit model
            #   = batch_size * (num_AND + 2)   otherwise
            sigma_in_tile = sigma_in.repeat(len(bboxes))
            new_cond = self.repeat_cond_dict(cond, bboxes)
            x_tile_out = self.sampler_forward(x_tile, sigma_in_tile, cond=new_cond)
            return x_tile_out

        def custom_func(x, custom_cond, uncond, bbox_id, bbox):
            return self.kdiff_custom_forward(x, cond, custom_cond, uncond, sigma_in, bbox_id, bbox, self.sampler_forward)

        return self.sample_one_step(x_in, repeat_func, custom_func)

    @torch.no_grad()
    def ddim_forward(self, x_in:Tensor, cond_in, ts:Tensor, unconditional_conditioning:ScheduledPromptConditioning, *args, **kwargs):
        '''
        This function will replace the original p_sample_ddim function in ldm/diffusionmodels/ddim.py
        So its signature should be the same as the original function,
        Particularly, the unconditional_conditioning should be with exactly the same name
        '''

        assert VanillaStableDiffusionSampler.p_sample_ddim_hook
        breakpoint()

        def repeat_func(x_tile, bboxes):
            if isinstance(cond_in, dict):
                ts_tile    = ts.repeat(len(bboxes))
                cond_tile  = self.repeat_cond_dict(cond_in, bboxes)
                ucond_tile = self.repeat_cond_dict(unconditional_conditioning, bboxes)
            else:
                ts_tile = ts.repeat(len(bboxes))
                cond_shape  = cond_in.shape
                cond_tile   = cond_in.repeat((len(bboxes),) + (1,) * (len(cond_shape) - 1))
                ucond_shape = unconditional_conditioning.shape
                ucond_tile  = unconditional_conditioning.repeat((len(bboxes),) + (1,) * (len(ucond_shape) - 1))
            x_tile_out, x_pred = self.sampler_forward(
                x_tile, cond_tile, ts_tile, 
                unconditional_conditioning=ucond_tile, 
                *args, **kwargs)
            return x_tile_out, x_pred

        def custom_func(x, cond, uncond, bbox_id, bbox):
            # before the final forward, we can set the control tensor
            def forward_func(x, *args, **kwargs):
                self.set_controlnet_tensors(bbox_id, 2*x.shape[0])
                return self.sampler_forward(x, *args, **kwargs)
            return self.ddim_custom_forward(x, cond_in, cond, uncond, bbox, ts, forward_func, *args, **kwargs)

        return self.sample_one_step(x_in, repeat_func, custom_func)

    def sample_one_step(self, x_in:Tensor, repeat_func:Callable, custom_func:Callable):
        '''
        this method splits the whole latent and process in tiles
            - x_in: current whole U-Net latent
            - denoise_func: one step denoiser for grid tile
            - denoise_custom_func: one step denoiser for custom tile
        '''

        N, C, H, W = x_in.shape
        assert H == self.h and W == self.w

        # clear buffer canvas
        self.reset_buffer(x_in)

        # Background sampling
        if self.draw_background:
            for batch_id, bboxes in enumerate(self.batched_bboxes):
                if state.interrupted: return x_in

                x_tile_list = []
                for bbox in bboxes:
                    x_tile_list.append(x_in[bbox.slicer])
                x_tile = torch.cat(x_tile_list, dim=0)

                # controlnet tiling
                self.switch_controlnet_tensors(batch_id, N, len(bboxes))

                # compute tiles
                if self.is_kdiff:
                    x_tile_out = repeat_func(x_tile, bboxes)
                    for i, bbox in enumerate(bboxes):
                        self.x_buffer[bbox.slicer] += x_tile_out[i*N:(i+1)*N, :, :, :]
                else:
                    x_tile_out, x_tile_pred = repeat_func(x_tile, bboxes)
                    for i, bbox in enumerate(bboxes):
                        self.x_buffer     [bbox.slicer] += x_tile_out [i*N:(i+1)*N, :, :, :]
                        self.x_pred_buffer[bbox.slicer] += x_tile_pred[i*N:(i+1)*N, :, :, :]

                # update progress bar
                self.update_pbar()

        # Custom region sampling
        x_feather_buffer      = None
        x_feather_mask        = None
        x_feather_count       = None
        x_feather_pred_buffer = None
        if len(self.custom_bboxes) > 0:
            for index, bbox in enumerate(self.custom_bboxes):
                if state.interrupted: return x_in

                x_tile = x_in[bbox.slicer]

                if not self.p.disable_extra_networks:
                    with devices.autocast():
                        extra_networks.activate(self.p, bbox.extra_network_data)

                if self.is_kdiff:
                    # retrieve original x_in from construncted input
                    x_tile_out = custom_func(x_tile, bbox.cond, bbox.uncond, index, bbox)

                    if bbox.blend_mode == BlendMode.BACKGROUND:
                        self.x_buffer[bbox.slicer] += x_tile_out
                    elif bbox.blend_mode == BlendMode.FOREGROUND:
                        if x_feather_buffer is None:
                            x_feather_buffer = torch.zeros_like(self.x_buffer)
                            x_feather_mask   = torch.zeros_like(self.x_buffer)
                            x_feather_count  = torch.zeros_like(self.x_buffer)
                        x_feather_buffer[bbox.slicer] += x_tile_out
                        x_feather_mask  [bbox.slicer] += bbox.feather_mask
                        x_feather_count [bbox.slicer] += 1
                else:
                    x_tile_out, x_tile_pred = custom_func(x_tile, bbox.cond, bbox.uncond, index, bbox)
                    if bbox.blend_mode == BlendMode.BACKGROUND:
                        self.x_buffer     [bbox.slicer] += x_tile_out
                        self.x_pred_buffer[bbox.slicer] += x_tile_pred
                    elif bbox.blend_mode == BlendMode.FOREGROUND:
                        if x_feather_buffer is None:
                            x_feather_buffer      = torch.zeros_like(self.x_buffer)
                            x_feather_mask        = torch.zeros_like(self.x_buffer)
                            x_feather_count       = torch.zeros_like(self.x_buffer)
                            x_feather_pred_buffer = torch.zeros_like(self.x_pred_buffer)
                        x_feather_buffer     [bbox.slicer] += x_tile_out
                        x_feather_mask       [bbox.slicer] += bbox.feather_mask
                        x_feather_count      [bbox.slicer] += 1
                        x_feather_pred_buffer[bbox.slicer] += x_tile_pred

                if not self.p.disable_extra_networks:
                    with devices.autocast():
                        extra_networks.deactivate(self.p,bbox.extra_network_data)

                # update progress bar
                self.update_pbar()

        # Averaging background buffer
        x_out = torch.where(self.weights > 1, self.x_buffer / self.weights, self.x_buffer)
        if not self.is_kdiff:
            x_pred_out = torch.where(self.weights > 1, self.x_pred_buffer / self.weights, self.x_pred_buffer)
        
        # Foreground Feather blending
        if x_feather_buffer is not None:
            # Average overlapping feathered regions
            x_feather_buffer = torch.where(x_feather_count > 1, x_feather_buffer / x_feather_count, x_feather_buffer)
            x_feather_mask   = torch.where(x_feather_count > 1, x_feather_mask   / x_feather_count, x_feather_mask)
            # Weighted average with original x_buffer
            
            x_out = torch.where(x_feather_count > 0, x_out * (1 - x_feather_mask) + x_feather_buffer * x_feather_mask, x_out)
            if not self.is_kdiff:
                x_feather_pred_buffer = torch.where(x_feather_count > 1, x_feather_pred_buffer / x_feather_count, x_feather_pred_buffer)
                x_pred_out            = torch.where(x_feather_count > 0, x_pred_out * (1 - x_feather_mask) + x_feather_pred_buffer * x_feather_mask, x_pred_out)
        
        return x_out if self.is_kdiff else (x_out, x_pred_out)
