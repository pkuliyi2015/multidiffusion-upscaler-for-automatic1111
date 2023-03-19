from methods.abstractdiffusion import TiledDiffusion
import torch

from modules import shared
from modules.shared import state


class MultiDiffusion(TiledDiffusion):
    """
    MultiDiffusion Implementation
    Hijack the sampler for latent image tiling and fusion
    """
    def __init__(self, sampler, sampler_name, *args, **kwargs):
        super().__init__("MultiDiffusion", sampler, sampler_name, *args, **kwargs)
        # record the steps for progress bar
        # hook the sampler
        assert sampler_name is not 'UniPC', 'MultiDiffusion is not compatible with UniPC, please use other samplers instead.'
        if self.is_kdiff:
            # For K-Diffusion sampler with uniform prompt, we hijack into the inner model for simplicity
            # Otherwise, the masked-redraw will break due to the init_latent
            self.sampler_func = self.sampler.inner_model.forward
            self.sampler.inner_model.forward = self.kdiff_repeat
        else:
            self.sampler_func = sampler.orig_p_sample_ddim
            self.sampler.orig_p_sample_ddim = self.ddim_repeat
        # For ddim sampler we need to cache the pred_x0
        self.x_buffer_pred = None

    def repeat_cond_dict(self, cond_input, bboxes):
        cond = cond_input['c_crossattn'][0]
        # repeat the condition on its first dim
        cond_shape = cond.shape
        cond = cond.repeat((len(bboxes),) + (1,) * (len(cond_shape) - 1))
        image_cond = cond_input['c_concat'][0]
        if image_cond.shape[2] == self.h and image_cond.shape[3] == self.w:
            image_cond_list = []
            for bbox in bboxes:
                image_cond_list.append(
                    image_cond[:, :, bbox[1]:bbox[3], bbox[0]:bbox[2]])
            image_cond_tile = torch.cat(image_cond_list, dim=0)
        else:
            image_cond_shape = image_cond.shape
            image_cond_tile = image_cond.repeat(
                (len(bboxes),) + (1,) * (len(image_cond_shape) - 1))
        return {"c_crossattn": [cond], "c_concat": [image_cond_tile]}
    
    def add_weights(self, input):
        input += 1.0

    def prepare_custom_bbox(self, prompts, negative_prompts, global_multiplier, bbox_control_states):
        super().prepare_custom_bbox(prompts, negative_prompts, global_multiplier, bbox_control_states)
        for bbox, _, _, m in self.custom_bboxes:
            self.weights[:, :, bbox[1]:bbox[3], bbox[0]:bbox[2]] += m

    @torch.no_grad()
    def kdiff_repeat(self, x_in, sigma_in, cond):
        def repeat_func(x_tile, bboxes):
            # For kdiff sampler, the dim 0 of input x_in is batch_size * (num_AND + 1) if it is not an edit model;
            # otherwise, it is batch_size * (num_AND + 2)
            sigma_in_tile = sigma_in.repeat(len(bboxes))
            new_cond = self.repeat_cond_dict(cond, bboxes)
            x_tile_out = self.sampler_func(
                x_tile, sigma_in_tile, cond=new_cond)
            return x_tile_out
        
        custom_func = lambda x, custom_cond, uncond, bbox: self.kdiff_custom_forward(
            x, cond, custom_cond, uncond, bbox, sigma_in, self.is_edit_model, self.sampler_func
        )

        return self.compute_x_tile(x_in, repeat_func, custom_func)

    @torch.no_grad()
    def ddim_repeat(self, x_in, cond_in, ts, unconditional_conditioning, *args, **kwargs):

        def repeat_func(x_tile, bboxes):
            if isinstance(cond_in, dict):
                ts_tile = ts.repeat(len(bboxes))
                cond_tile = self.repeat_cond_dict(cond_in, bboxes)
                ucond_tile = self.repeat_cond_dict(
                    unconditional_conditioning, bboxes)
            else:
                ts_tile = ts.repeat(len(bboxes))
                cond_shape = cond_in.shape
                cond_tile = cond_in.repeat(
                    (len(bboxes),) + (1,) * (len(cond_shape) - 1))
                ucond_shape = unconditional_conditioning.shape
                ucond_tile = unconditional_conditioning.repeat(
                    (len(bboxes),) + (1,) * (len(ucond_shape) - 1))
            x_tile_out, x_pred = self.sampler_func(
                x_tile, cond_tile, ts_tile, unconditional_conditioning=ucond_tile, *args, **kwargs)
            return x_tile_out, x_pred
        
        custom_func = lambda x, cond, uncond, bbox: self.ddim_custom_forward(x, cond_in, cond, uncond, bbox, ts, self.sampler_func, *args, **kwargs)
        return self.compute_x_tile(x_in, repeat_func, custom_func)

    def compute_x_tile(self, x_in, func, custom_func):
        N, C, H, W = x_in.shape
        assert H == self.h and W == self.w

        self.init_pbar()

        # Kdiff 'AND' support and image editing model support
        if len(self.custom_bboxes) > 0 and self.is_kdiff and not hasattr(self, 'is_edit_model'):
            self.is_edit_model = shared.sd_model.cond_stage_key == "edit" and self.sampler.image_cfg_scale is not None and self.sampler.image_cfg_scale != 1.0

        # ControlNet support
        self.prepare_control_tensors(N)

        if self.x_buffer is None:
            self.x_buffer = torch.zeros_like(x_in, device=x_in.device)
        else:
            self.x_buffer.zero_()
        if not self.is_kdiff:
            if self.x_buffer_pred is None:
                self.x_buffer_pred = torch.zeros_like(x_in, device=x_in.device)
            else:
                self.x_buffer_pred.zero_()
        
        # Global sampling
        if self.global_multiplier > 0:
            for batch_id, bboxes in enumerate(self.batched_bboxes):
                if state.interrupted:
                    return x_in
                x_tile_list = []
                for bbox in bboxes:
                    x_tile_list.append(
                        x_in[:, :, bbox[1]:bbox[3], bbox[0]:bbox[2]])
                x_tile = torch.cat(x_tile_list, dim=0)
                # controlnet tiling
                self.switch_controlnet_tensors(batch_id, x_tile)
                # compute tiles
                if self.is_kdiff:
                    x_tile_out = func(x_tile, bboxes)
                    for i, bbox in enumerate(bboxes):
                        self.x_buffer[:, :, bbox[1]:bbox[3], bbox[0]:bbox[2]] += x_tile_out[i*N:(i+1)*N, :, :, :]
                else:
                    x_tile_out, x_tile_pred = func(x_tile, bboxes)
                    for i, bbox in enumerate(bboxes):
                        self.x_buffer[:, :, bbox[1]:bbox[3], bbox[0]:bbox[2]] += x_tile_out[i*N:(i+1)*N, :, :, :]
                        self.x_buffer_pred[:, :, bbox[1]:bbox[3], bbox[0]:bbox[2]] += x_tile_pred[i*N:(i+1)*N, :, :, :]
                # update progress bar
                self.update_pbar()

        # Custom region sampling
        if len(self.custom_bboxes) > 0:
            if self.global_multiplier > 0 and abs(self.global_multiplier - 1.0) > 1e-6:
                self.x_buffer *= self.global_multiplier
            if not self.is_kdiff:
                self.x_buffer_pred *= self.global_multiplier
            for index, bbox, cond, uncond, multiplier in enumerate(self.custom_bboxes):
                self.switch_custom_controlnet_tensors(index, x_in)
                if self.is_kdiff:
                    # retrieve original x_in from construncted input
                    # kdiff last batch is always the correct original input
                    x_tile = x_in[-self.batch_size:, :,
                                  bbox[1]:bbox[3], bbox[0]:bbox[2]]
                    x_tile_out = custom_func(x_tile, cond, uncond, bbox)
                    x_tile_out *= multiplier
                    self.x_buffer[:, :, bbox[1]:bbox[3],
                                  bbox[0]:bbox[2]] += x_tile_out
                else:
                    x_tile = x_in[:self.batch_size, :, bbox[1]:bbox[3], bbox[0]:bbox[2]]
                    x_tile_out, x_tile_pred = custom_func(
                        x_tile, cond, uncond, bbox)
                    x_tile_out *= multiplier
                    x_tile_pred *= multiplier
                    self.x_buffer[:, :, bbox[1]:bbox[3],
                                  bbox[0]:bbox[2]] += x_tile_out
                    self.x_buffer_pred[:, :, bbox[1]:bbox[3],
                                       bbox[0]:bbox[2]] += x_tile_pred
                self.update_pbar()

        x_out = torch.where(self.weights > 1, self.x_buffer /
                            self.weights, self.x_buffer)
        if not self.is_kdiff:
            x_pred = torch.where(
                self.weights > 1, self.x_buffer_pred / self.weights, self.x_buffer_pred)
            return x_out, x_pred
        return x_out