import torch
import k_diffusion as K

from tqdm import trange
from modules import devices, shared, extra_networks
from modules.shared import state

from tile_methods.abstractdiffusion import TiledDiffusion
from tile_utils.utils import *
from tile_utils.typing import *


class MixtureOfDiffusers(TiledDiffusion):
    """
        Mixture-of-Diffusers Implementation
        https://github.com/albarji/mixture-of-diffusers
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # weights for custom bboxes
        self.custom_weights: List[Tensor] = []
        self.get_weight = gaussian_weights

    def hook(self):
        if not hasattr(shared.sd_model, 'apply_model_original_md'):
            shared.sd_model.apply_model_original_md = shared.sd_model.apply_model
        shared.sd_model.apply_model = self.apply_model_hijack

    @staticmethod
    def unhook():
        if hasattr(shared.sd_model, 'apply_model_original_md'):
            shared.sd_model.apply_model = shared.sd_model.apply_model_original_md
            del shared.sd_model.apply_model_original_md

    def enable_noise_inverse(self, steps: int, randomness: float):
        super().enable_noise_inverse(steps, randomness)
        self.get_weight = lambda w, h: torch.ones((h, w), device=devices.device, dtype=torch.float32)

    def init_done(self):
        super().init_done()
        # The original gaussian weights can be extremely small, so we rescale them for numerical stability
        self.rescale_factor = 1 / self.weights
        # Meanwhile, we rescale the custom weights in advance to save time of slicing
        for bbox_id, bbox in enumerate(self.custom_bboxes):
            if bbox.blend_mode == BlendMode.BACKGROUND:
                self.custom_weights[bbox_id] *= self.rescale_factor[bbox.slicer]

    @grid_bbox
    def get_tile_weights(self) -> Tensor:
        # weights for grid bboxes
        if not hasattr(self, 'tile_weights'):
            self.tile_weights = self.get_weight(self.tile_w, self.tile_h)
        return self.tile_weights

    @custom_bbox
    def init_custom_bbox(self, *args):
        super().init_custom_bbox(*args)

        for bbox in self.custom_bboxes:
            if bbox.blend_mode == BlendMode.BACKGROUND:
                custom_weights = self.get_weight(bbox.w, bbox.h)
                self.weights[bbox.slicer] += custom_weights
                self.custom_weights.append(custom_weights.unsqueeze(0).unsqueeze(0))
            else:
                self.custom_weights.append(None)

    ''' ↓↓↓ kernel hijacks ↓↓↓ '''

    def custom_apply_model(self, x_in, t_in, c_in, bbox_id, bbox) -> Tensor:
        if self.is_kdiff:
            return self.kdiff_custom_forward(x_in, t_in, c_in, bbox_id, bbox, forward_func=shared.sd_model.apply_model_original_md)
        else:
            def forward_func(x, c, ts, unconditional_conditioning, *args, **kwargs) -> Tensor:
                # copy from p_sample_ddim in ddim.py
                c_in = dict()
                for k in c:
                    if isinstance(c[k], list):
                        c_in[k] = [torch.cat([unconditional_conditioning[k][i], c[k][i]]) for i in range(len(c[k]))]
                    else:
                        c_in[k] = torch.cat([unconditional_conditioning[k], c[k]])
                self.set_controlnet_tensors(bbox_id, x.shape[0])
                return shared.sd_model.apply_model_original_md(x, ts, c_in)
            return self.ddim_custom_forward(x_in, c_in, bbox, ts=t_in, forward_func=forward_func)

    @torch.no_grad()
    @keep_signature
    def apply_model_hijack(self, x_in:Tensor, t_in:Tensor, cond:CondDict):
        assert LatentDiffusion.apply_model

        # KDiffusion Compatibility
        c_in = cond
        N, C, H, W = x_in.shape
        if H != self.h or W != self.w:
            # We don't tile highres, let's just use the original apply_model
            self.reset_controlnet_tensors()
            return shared.sd_model.apply_model_original_md(x_in, t_in, c_in)

        self.reset_buffer(x_in)

        # Global sampling
        if self.draw_background:
            for batch_id, bboxes in enumerate(self.batched_bboxes):     # batch_id is the `Latent tile batch size`
                if state.interrupted: return x_in

                # batching
                x_tile_list     = []
                t_tile_list     = []
                attn_tile_list  = []
                image_cond_list = []
                for bbox in bboxes:
                    x_tile_list.append(x_in[bbox.slicer])
                    t_tile_list.append(t_in)
                    if c_in is not None and isinstance(c_in, dict):
                        image_cond = c_in['c_concat'][0]        # dummy for txt2img, latent mask for img2img
                        if image_cond.shape[2] == self.h and image_cond.shape[3] == self.w:
                            image_cond = image_cond[bbox.slicer]
                        image_cond_list.append(image_cond)
                        attn_tile = c_in['c_crossattn'][0]      # cond, [1, 77, 768]
                        attn_tile_list.append(attn_tile)
                x_tile          = torch.cat(x_tile_list,     dim=0)     # differs each
                t_tile          = torch.cat(t_tile_list,     dim=0)     # just repeat
                attn_tile       = torch.cat(attn_tile_list,  dim=0)     # just repeat
                image_cond_tile = torch.cat(image_cond_list, dim=0)     # differs each
                c_tile = {'c_concat': [image_cond_tile], 'c_crossattn': [attn_tile]}

                # controlnet
                self.switch_controlnet_tensors(batch_id, N, len(bboxes), is_denoise=True)
                
                # denoising: here the x is the noise
                x_tile_out = shared.sd_model.apply_model_original_md(x_tile, t_tile, c_tile)

                # de-batching
                for i, bbox in enumerate(bboxes):
                    # This weights can be calcluated in advance, but will cost a lot of vram 
                    # when you have many tiles. So we calculate it here.
                    w = self.tile_weights * self.rescale_factor[bbox.slicer]
                    self.x_buffer[bbox.slicer] += x_tile_out[i*N:(i+1)*N, :, :, :] * w

                self.update_pbar()
        
        # Custom region sampling
        x_feather_buffer = None
        x_feather_mask   = None
        x_feather_count  = None
        if len(self.custom_bboxes) > 0:
            for bbox_id, bbox in enumerate(self.custom_bboxes):
                if not self.p.disable_extra_networks:
                    with devices.autocast():
                        extra_networks.activate(self.p, bbox.extra_network_data)

                x_tile = x_in[bbox.slicer]
                x_tile_out = self.custom_apply_model(x_tile, t_in, c_in, bbox_id, bbox)

                if bbox.blend_mode == BlendMode.BACKGROUND:
                    self.x_buffer[bbox.slicer] += x_tile_out * self.custom_weights[bbox_id]
                elif bbox.blend_mode == BlendMode.FOREGROUND:
                    if x_feather_buffer is None:
                        x_feather_buffer = torch.zeros_like(self.x_buffer)
                        x_feather_mask   = torch.zeros_like(self.x_buffer)
                        x_feather_count  = torch.zeros_like(self.x_buffer)
                    x_feather_buffer[bbox.slicer] += x_tile_out
                    x_feather_mask  [bbox.slicer] += bbox.feather_mask
                    x_feather_count [bbox.slicer] += 1

                self.update_pbar()

                if not self.p.disable_extra_networks:
                    with devices.autocast():
                        extra_networks.deactivate(self.p, bbox.extra_network_data)

        x_out = self.x_buffer
        if x_feather_buffer is not None:
            # Average overlapping feathered regions
            x_feather_buffer = torch.where(x_feather_count > 1, x_feather_buffer / x_feather_count, x_feather_buffer)
            x_feather_mask   = torch.where(x_feather_count > 1, x_feather_mask   / x_feather_count, x_feather_mask)
            # Weighted average with original x_buffer
            x_out = torch.where(x_feather_count > 0, x_out * (1 - x_feather_mask) + x_feather_buffer * x_feather_mask, x_out)

        return x_out

    @torch.no_grad()
    def find_noise_for_image_sigma_adjustment(self, cond_basis, uncond_basis, cfg_scale, steps) -> Tensor:
        '''
        Migrate from the built-in script img2imgalt.py
        Tiled noise inverse for better image upscaling
        '''
        assert self.p.sampler_name == 'Euler'
        x = self.p.init_latent

        s_in = x.new_ones([x.shape[0]])
        if shared.sd_model.parameterization == "v":
            dnw = K.external.CompVisVDenoiser(shared.sd_model)
            skip = 1
        else:
            dnw = K.external.CompVisDenoiser(shared.sd_model)
            skip = 0
        sigmas = dnw.get_sigmas(steps).flip(0)
        shared.state.sampling_steps = len(sigmas)
        print("Tiled noise inverse...")
        for i in trange(1, len(sigmas)):
            if shared.state.interrupted:
                return x
            shared.state.sampling_step += 1
            cond = Condition.reconstruct_cond(cond_basis, i)
            uncond = Condition.reconstruct_uncond(uncond_basis, i)

            x_in = torch.cat([x] * 2)
            sigma_in = torch.cat([sigmas[i - 1] * s_in] * 2)
            cond_in = torch.cat([uncond, cond])

            image_conditioning = torch.cat([self.p.image_conditioning] * 2)
            cond_in = {"c_concat": [image_conditioning], "c_crossattn": [cond_in]}

            c_out, c_in = [K.utils.append_dims(k, x_in.ndim) for k in dnw.get_scalings(sigma_in)[skip:]]

            if i == 1:
                t = dnw.sigma_to_t(torch.cat([sigmas[i] * s_in] * 2))
            else:
                t = dnw.sigma_to_t(sigma_in)

            eps = self.apply_model_hijack(x_in * c_in, t, cond=cond_in)
            denoised_uncond, denoised_cond = (x_in + eps * c_out).chunk(2)

            denoised = denoised_uncond + (denoised_cond - denoised_uncond) * cfg_scale

            if i == 1:
                d = (x - denoised) / (2 * sigmas[i])
            else:
                d = (x - denoised) / sigmas[i - 1]

            dt = sigmas[i] - sigmas[i - 1]
            x = x + d * dt

            sd_samplers_common.store_latent(x)

            # This shouldn't be necessary, but solved some VRAM issues
            del x_in, sigma_in, cond_in, c_out, c_in, t,
            del eps, denoised_uncond, denoised_cond, denoised, d, dt
        
        shared.state.nextjob()
        return x / sigmas[-1]