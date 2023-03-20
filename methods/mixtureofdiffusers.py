from methods.abstractdiffusion import TiledDiffusion
import numpy as np
from numpy import pi, exp, sqrt
import torch

from modules import devices, shared
from modules.shared import state
from modules.script_callbacks import before_image_saved_callback, remove_callbacks_for_function


class MixtureOfDiffusers(TiledDiffusion):
    """
        MixtureOfDiffusers Implementation
        Hijack the UNet for latent noise tiling and fusion
    """
    def __init__(self, *args, **kwargs):
        super().__init__("Mixture of Diffusers", *args, **kwargs)
        self.custom_weights = []

    def _gaussian_weights(self, tile_w=None, tile_h=None):
        '''
        Gaussian weights to smooth the noise of each tile.
        This is critical for this method to work.
        '''
        if tile_w is None:
            tile_w = self.tile_w
        if tile_h is None:
            tile_h = self.tile_h
        var = 0.01
        # -1 because index goes from 0 to latent_width - 1
        midpoint = (tile_w - 1) / 2
        x_probs = [exp(-(x-midpoint)*(x-midpoint)/(tile_w*tile_w) /
                       (2*var)) / sqrt(2*pi*var) for x in range(tile_w)]
        midpoint = tile_h / 2
        y_probs = [exp(-(y-midpoint)*(y-midpoint)/(tile_h*tile_h) /
                       (2*var)) / sqrt(2*pi*var) for y in range(tile_h)]
        weights = torch.from_numpy(
            np.outer(y_probs, x_probs)).to(devices.device)
        return weights

    def add_global_weights(self, input):
        if not hasattr(self, 'per_tile_weights'):
            self.per_tile_weights = self._gaussian_weights().to(input.device)
        input += self.per_tile_weights

    def prepare_custom_bbox(self, global_multiplier, bbox_control_states):
        super().prepare_custom_bbox(global_multiplier, bbox_control_states)
        for bbox, _, _, m in self.custom_bboxes:
            # multiply the gaussian weights in advance to save time
            gaussian_weights = self._gaussian_weights(
                bbox[2] - bbox[0], bbox[3] - bbox[1]).to(devices.device) * m
            self.weights[:, :, bbox[1]:bbox[3],
                         bbox[0]:bbox[2]] += gaussian_weights
            self.custom_weights.append(gaussian_weights)

    def hook(self):
        if not hasattr(shared.sd_model, 'md_org_apply_model'):
            shared.sd_model.md_org_apply_model = shared.sd_model.apply_model
            shared.sd_model.apply_model = self.apply_model

        def remove_hook(_):
            MixtureOfDiffusers.unhook()
            remove_callbacks_for_function(MixtureOfDiffusers.unhook)
        before_image_saved_callback(remove_hook)

    @staticmethod
    def unhook():
        if hasattr(shared.sd_model, 'md_org_apply_model'):
            shared.sd_model.apply_model = shared.sd_model.md_org_apply_model
            del shared.sd_model.md_org_apply_model

    def custom_apply_model(self, x_in, t_in, c_in, bbox_id, bbox, cond, uncond):
        if self.is_kdiff:
            if not hasattr(self, 'is_edit_model'):
                self.is_edit_model = shared.sd_model.cond_stage_key == "edit" and self.sampler.image_cfg_scale is not None and self.sampler.image_cfg_scale != 1.0
            return self.kdiff_custom_forward(x_in, c_in, cond, uncond, bbox_id, bbox, sigma_in=t_in, forward_func=shared.sd_model.md_org_apply_model, batched=True)
        else:
            def forward_func(x, c, ts, unconditional_conditioning, *args, **kwargs):
                # copy from p_sample_ddim in ddim.py
                x_in = torch.cat([x] * 2)
                t_in = torch.cat([ts] * 2)
                if isinstance(c, dict):
                    assert isinstance(unconditional_conditioning, dict)
                    c_in = dict()
                    for k in c:
                        if isinstance(c[k], list):
                            c_in[k] = [torch.cat([
                                unconditional_conditioning[k][i],
                                c[k][i]]) for i in range(len(c[k]))]
                        else:
                            c_in[k] = torch.cat([
                                    unconditional_conditioning[k],
                                    c[k]])
                elif isinstance(c, list):
                    c_in = list()
                    assert isinstance(unconditional_conditioning, list)
                    for i in range(len(c)):
                        c_in.append(torch.cat([unconditional_conditioning[i], c[i]]))
                else:
                    c_in = torch.cat([unconditional_conditioning, c])
                return shared.sd_model.md_org_apply_model(x_in, t_in, c_in)
            
            return self.ddim_custom_forward(x_in, c_in, cond, uncond, bbox_id, bbox, ts=t_in, forward_func=forward_func)

    @torch.no_grad()
    def apply_model(self, x_in, t_in, cond):
        '''
        Hook to UNet when predicting noise
        '''
        # KDiffusion Compatibility
        c_in = cond
        N, C, H, W = x_in.shape
        assert H == self.h and W == self.w

        self.init(x_in)

        # Global sampling
        if self.global_multiplier > 0:
            for batch_id, bboxes in enumerate(self.batched_bboxes):
                if state.interrupted:
                    return x_in
                x_tile_list = []
                t_tile_list = []
                attn_tile_list = []
                image_cond_list = []
                for bbox in bboxes:
                    x_tile_list.append(
                        x_in[:, :, bbox[1]:bbox[3], bbox[0]:bbox[2]])
                    t_tile_list.append(t_in)
                    if c_in is not None and isinstance(cond, dict):
                        image_cond = cond['c_concat'][0]
                        if image_cond.shape[2] == self.h and image_cond.shape[3] == self.w:
                            image_cond = image_cond[:, :,
                                                    bbox[1]:bbox[3], bbox[0]:bbox[2]]
                        image_cond_list.append(image_cond)
                        attn_tile = cond['c_crossattn'][0]
                        attn_tile_list.append(attn_tile)
                x_tile = torch.cat(x_tile_list, dim=0)
                t_tile = torch.cat(t_tile_list, dim=0)
                attn_tile = torch.cat(attn_tile_list, dim=0)
                image_cond_tile = torch.cat(image_cond_list, dim=0)
                c_tile = {'c_concat': [image_cond_tile],
                          'c_crossattn': [attn_tile]}
                # Controlnet tiling
                self.switch_controlnet_tensors(batch_id, N, len(bboxes))
                x_tile_out = shared.sd_model.md_org_apply_model(
                    x_tile, t_tile, c_tile)  # here the x is the noise

                for i, bbox in enumerate(bboxes):
                    self.x_buffer[:, :, bbox[1]:bbox[3], bbox[0]:bbox[2]
                                  ] += x_tile_out[i*N:(i+1)*N, :, :, :] * self.per_tile_weights

                self.update_pbar()

        # Custom region sampling
        if len(self.custom_bboxes) > 0:
            if self.global_multiplier > 0 and abs(self.global_multiplier - 1.0) > 1e-6:
                self.x_buffer *= self.global_multiplier
            for index, (bbox, cond, uncond, _) in enumerate(self.custom_bboxes):
                # unpack sigma_in, x_in, image_cond
                image_cond = None
                if c_in is not None and isinstance(cond, dict):
                    image_cond = cond['c_concat'][0]
                if self.is_kdiff:
                    x_tile = x_in[-self.batch_size:, :, bbox[1]:bbox[3], bbox[0]:bbox[2]]
                    if image_cond is not None:
                        if image_cond.shape[2] == self.h and image_cond.shape[3] == self.w:
                            image_cond = image_cond[-self.batch_size:, :, :, :]
                        else:
                            image_cond = image_cond[-self.batch_size:, :]
                        c_tile = {'c_concat': [image_cond], 'c_crossattn': [cond['c_crossattn'][0]]}
                else:
                    x_tile = x_in[:self.batch_size, :, bbox[1]:bbox[3], bbox[0]:bbox[2]]
                    if image_cond is not None:
                        if image_cond.shape[2] == self.h and image_cond.shape[3] == self.w:
                            image_cond = image_cond[:self.batch_size, :, :, :]
                        else:
                            image_cond = image_cond[:self.batch_size, :]
                        c_tile = {'c_concat': [image_cond], 'c_crossattn': [cond['c_crossattn'][0]]}
                x_tile_out = self.custom_apply_model(x_tile, t_in, c_tile, index, bbox, cond, uncond)
                x_tile_out *= self.custom_weights[index]
                self.x_buffer[:, :, bbox[1]:bbox[3],bbox[0]:bbox[2]] += x_tile_out
                self.update_pbar()

        x_out = torch.where(self.weights > 0, self.x_buffer /
                            self.weights, self.x_buffer)
        return x_out
