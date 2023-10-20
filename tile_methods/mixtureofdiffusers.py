from tile_methods.abstractdiffusion import AbstractDiffusion
from tile_utils.utils import *


class MixtureOfDiffusers(AbstractDiffusion):
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

    @torch.no_grad()
    @keep_signature
    def apply_model_hijack(self, x_in:Tensor, t_in:Tensor, cond:CondDict, noise_inverse_step:int=-1):
        assert LatentDiffusion.apply_model

        # KDiffusion Compatibility for naming
        c_in: CondDict = cond

        N, C, H, W = x_in.shape
        if (H, W) != (self.h, self.w):
            # We don't tile highres, let's just use the original apply_model
            self.reset_controlnet_tensors()
            return shared.sd_model.apply_model_original_md(x_in, t_in, c_in)

        # clear buffer canvas
        self.reset_buffer(x_in)

        # Global sampling
        if self.draw_background:
            for batch_id, bboxes in enumerate(self.batched_bboxes):     # batch_id is the `Latent tile batch size`
                if state.interrupted: return x_in

                # batching
                x_tile_list     = []
                t_tile_list     = []
                tcond_tile_list = []
                icond_tile_list = []
                vcond_tile_list = []
                for bbox in bboxes:
                    x_tile_list.append(x_in[bbox.slicer])
                    t_tile_list.append(t_in)
                    if isinstance(c_in, dict):
                        # tcond
                        tcond_tile = self.get_tcond(c_in)      # cond, [1, 77, 768]
                        tcond_tile_list.append(tcond_tile)
                        # icond: might be dummy for txt2img, latent mask for img2img
                        icond = self.get_icond(c_in)
                        if icond.shape[2:] == (self.h, self.w):
                            icond = icond[bbox.slicer]
                        icond_tile_list.append(icond)
                        # vcond:
                        vcond = self.get_vcond(c_in)
                        vcond_tile_list.append(vcond)
                    else:
                        print('>> [WARN] not supported, make an issue on github!!')
                x_tile     = torch.cat(x_tile_list,     dim=0)  # differs each
                t_tile     = torch.cat(t_tile_list,     dim=0)  # just repeat
                tcond_tile = torch.cat(tcond_tile_list, dim=0)  # just repeat
                icond_tile = torch.cat(icond_tile_list, dim=0)  # differs each
                vcond_tile = torch.cat(vcond_tile_list, dim=0) if None not in vcond_tile_list else None # just repeat

                c_tile = self.make_cond_dict(c_in, tcond_tile, icond_tile, vcond_tile)

                # controlnet
                self.switch_controlnet_tensors(batch_id, N, len(bboxes), is_denoise=True)
                
                # stablesr
                self.switch_stablesr_tensors(batch_id)

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
                if noise_inverse_step < 0:
                    x_tile_out = self.custom_apply_model(x_tile, t_in, c_in, bbox_id, bbox)
                else:
                    tcond = Condition.reconstruct_cond(bbox.cond, noise_inverse_step)
                    icond = self.get_icond(c_in)
                    if icond.shape[2:] == (self.h, self.w):
                        icond = icond[bbox.slicer]
                    vcond = self.get_vcond(c_in)
                    c_out = self.make_cond_dict(c_in, tcond, icond, vcond)
                    x_tile_out = shared.sd_model.apply_model(x_tile, t_in, cond=c_out)

                if bbox.blend_mode == BlendMode.BACKGROUND:
                    self.x_buffer[bbox.slicer] += x_tile_out * self.custom_weights[bbox_id]
                elif bbox.blend_mode == BlendMode.FOREGROUND:
                    if x_feather_buffer is None:
                        x_feather_buffer = torch.zeros_like(self.x_buffer)
                        x_feather_mask   = torch.zeros((1, 1, H, W), device=self.x_buffer.device)
                        x_feather_count  = torch.zeros((1, 1, H, W), device=self.x_buffer.device)
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

        # For mixture of diffusers, we cannot fill the not denoised area.
        # So we just leave it as it is.
        return x_out

    def custom_apply_model(self, x_in, t_in, c_in, bbox_id, bbox) -> Tensor:
        if self.is_kdiff:
            return self.kdiff_custom_forward(x_in, t_in, c_in, bbox_id, bbox, forward_func=shared.sd_model.apply_model_original_md)
        else:
            def forward_func(x, c, ts, unconditional_conditioning, *args, **kwargs) -> Tensor:
                # copy from p_sample_ddim in ddim.py
                c_in: CondDict = dict()
                for k in c:
                    if isinstance(c[k], list):
                        c_in[k] = [torch.cat([unconditional_conditioning[k][i], c[k][i]]) for i in range(len(c[k]))]
                    else:
                        c_in[k] = torch.cat([unconditional_conditioning[k], c[k]])
                self.set_custom_controlnet_tensors(bbox_id, x.shape[0])
                self.set_custom_stablesr_tensors(bbox_id)
                return shared.sd_model.apply_model_original_md(x, ts, c_in)
            return self.ddim_custom_forward(x_in, c_in, bbox, ts=t_in, forward_func=forward_func)

    @torch.no_grad()
    def get_noise(self, x_in:Tensor, sigma_in:Tensor, cond_in:Dict[str, Tensor], step:int) -> Tensor:
        return self.apply_model_hijack(x_in, sigma_in, cond=cond_in, noise_inverse_step=step)
