# ------------------------------------------------------------------------
#
#   Tiled Diffusion for Automatic1111 WebUI
#
#   Introducing a revolutionary large image drawing method 
#       MultiDiffusion & Mixture of Diffusers!
#
#   Techniques is not originally proposed by me, please refer to
#
#   MultiDiffusion: https://multidiffusion.github.io
#   Mixture of Diffusers: https://github.com/albarji/mixture-of-diffusers
#
#   The script contains a few optimizations including:
#       - symmetric tiling bboxes
#       - cached tiling weights
#       - batched denoising
#       - prompt control for each tile (in progress)
#
# ------------------------------------------------------------------------
#
#   This script hooks into the original sampler and decomposes the latent
#   image, sampled separately and run weighted average to merge them back.
#
#   Advantages:
#   - Allows for super large resolutions (2k~8k) for both txt2img and img2img.
#   - The merged output is completely seamless without any post-processing.
#   - Training free. No need to train a new model, and you can control the
#       text prompt for each tile.
#
#   Drawbacks:
#   - Depending on your parameter settings, the process can be very slow,
#       especially when overlap is relatively large.
#   - The gradient calculation is not compatible with this hack. It
#       will break any backward() or torch.autograd.grad() that passes UNet.
#
#   How it works (insanely simple!)
#   1) The latent image x_t is split into tiles
#   2) The tiles are denoised by original sampler to get x_t-1
#   3) The tiles are added together, but divided by how many times each pixel
#       is added.
#
#   Enjoy!
#
#   @author: LI YI @ Nanyang Technological University - Singapore
#   @date: 2023-03-03
#   @license: MIT License
#
#   Please give me a star if you like this project!
#
# -------------------------------------------------------------------------


from abc import ABC, abstractmethod
import math
import numpy as np
from numpy import pi, exp, sqrt
import torch
from tqdm import tqdm
import gradio as gr

from modules import sd_samplers, images, devices, shared, scripts, prompt_parser
from modules.shared import opts, state
from modules.ui import gr_show

from modules.processing import StableDiffusionProcessing


BBOX_MAX_NUM = min(shared.cmd_opts.md_max_regions if hasattr(
    shared.cmd_opts, "md_max_regions") else 8, 16)

class TiledDiffusion(ABC):
    def __init__(self, iters, batch_size, steps, 
                 w, h, tile_w=64, tile_h=64, overlap=32, tile_batch_size=1,
                 controlnet_script=None, control_tensor_cpu=False) -> None:
        self.batch_size = batch_size
        self.steps = steps
        self.iters = iters
                # initialize the tile bboxes and weights
        self.w, self.h = w//8, h//8
        if tile_w > self.w:
            tile_w = self.w
        if tile_h > self.h:
            tile_h = self.h
        min_tile_size = min(tile_w, tile_h)
        if overlap >= min_tile_size:
            overlap = min_tile_size - 4
        if overlap < 0:
            overlap = 0
        self.tile_w = tile_w
        self.tile_h = tile_h
        bboxes, weights = self.split_views(tile_w, tile_h, overlap)
        self.batched_bboxes = []
        self.num_batches = math.ceil(len(bboxes) / tile_batch_size)
        optimal_batch_size = math.ceil(len(bboxes) / self.num_batches)
        self.tile_batch_size = optimal_batch_size
        for i in range(self.num_batches):
            start = i * tile_batch_size
            end = min((i + 1) * tile_batch_size, len(bboxes))
            self.batched_bboxes.append(bboxes[start:end])
        self.weights = weights.unsqueeze(0).unsqueeze(0)

        # Avoid the overhead of creating a new tensor for each batch
        # And avoid the overhead of weight summing
        self.x_buffer = None
        # Region prompt control
        self.custom_bboxes = []
        self.global_multiplier = 1.0

        # For controlnet
        self.controlnet_script = controlnet_script
        self.control_tensor_batch = None
        self.control_params = None
        self.control_tensor_cpu = control_tensor_cpu

        # Progress bar
        self.pbar = None
    
    def split_views(self, tile_w, tile_h, overlap):
        non_overlap_width = tile_w - overlap
        non_overlap_height = tile_h - overlap
        w, h = self.w, self.h
        cols = math.ceil((w - overlap) / non_overlap_width)
        rows = math.ceil((h - overlap) / non_overlap_height)

        dx = (w - tile_w) / (cols - 1) if cols > 1 else 0
        dy = (h - tile_h) / (rows - 1) if rows > 1 else 0

        bbox = []
        count = torch.zeros((h, w), device=devices.device)
        for row in range(rows):
            y = int(row * dy)
            if y + tile_h >= h:
                y = h - tile_h
            for col in range(cols):
                x = int(col * dx)
                if x + tile_w >= w:
                    x = w - tile_w
                bbox.append([x, y, x + tile_w, y + tile_h])
                self.add_weights(count[y:y+tile_h, x:x+tile_w])
        return bbox, count
    
    @abstractmethod
    def add_weights(self, input):
        pass

    @staticmethod
    def splitable(w, h, tile_w, tile_h, overlap):
        w, h = w//8, h//8
        min_tile_size = min(tile_w, tile_h)
        if overlap >= min_tile_size:
            overlap = min_tile_size - 4
        non_overlap_width = tile_w - overlap
        non_overlap_height = tile_h - overlap
        cols = math.ceil((w - overlap) / non_overlap_width)
        rows = math.ceil((h - overlap) / non_overlap_height)
        return cols > 1 or rows > 1
    
    def prepare_custom_bbox(self, prompts, negative_prompts, global_multiplier, bbox_control_states):
        '''
        Prepare custom bboxes for region prompt
        '''
        self.global_multiplier = global_multiplier
        c_weights = torch.zeros_like(self.weights)
        for i in range(0, len(bbox_control_states) - 8, 8):
            e, m, x, y, w, h, p, neg = bbox_control_states[i:i+8]
            if not e or m < 1 or w <= 0 or h <= 0 or p == '':
                continue
            bbox = [int(x * self.w), int(y * self.h),
                    int((x + w) * self.w), int((y + h) * self.h)]
            c_weights[:, :, bbox[1]:bbox[3], bbox[0]:bbox[2]] += m
            c_prompt = [prompt + ', ' + p for prompt in prompts]
            if neg != '':
                c_negative_prompt = [prompt + ', ' +
                                     neg for prompt in negative_prompts]
            else:
                c_negative_prompt = negative_prompts
            c_prompt = prompt_parser.get_multicond_learned_conditioning(
                shared.sd_model, c_prompt, self.steps)
            c_negative_prompt = prompt_parser.get_learned_conditioning(
                shared.sd_model, c_negative_prompt, self.steps)
            self.custom_bboxes.append((bbox, c_prompt, c_negative_prompt, m))

        if len(self.custom_bboxes) > 0:
            self.weights = self.weights * global_multiplier + c_weights
    
    def prepare_control_tensors(self, batch_size):
        '''
        Crop the control tensor into tiles and cache them
        '''
        if self.control_tensor_batch is not None:
            return
        if self.controlnet_script is None or self.control_params is not None:
            return
        latest_network = self.controlnet_script.latest_network
        if latest_network is None or not hasattr(latest_network, 'control_params'):
            return
        self.control_params = latest_network.control_params
        tensors = [param.hint_cond for param in latest_network.control_params]
        if len(tensors) == 0:
            return
        self.control_tensor_batch = []
        for i in range(len(tensors)):
            control_tile_list = []
            control_tensor = tensors[i]
            for bboxes in self.batched_bboxes:
                single_batch_tensors = []
                for bbox in bboxes:
                    if len(control_tensor.shape) == 3:
                        control_tile = control_tensor[:, bbox[1] *
                                                      8:bbox[3]*8, bbox[0]*8:bbox[2]*8].unsqueeze(0)
                    else:
                        control_tile = control_tensor[:, :,
                                                      bbox[1]*8:bbox[3]*8, bbox[0]*8:bbox[2]*8]
                    single_batch_tensors.append(control_tile)
                if self.is_kdiff:
                    control_tile = torch.cat(
                        [t for t in single_batch_tensors for _ in range(batch_size)], dim=0)
                else:
                    control_tile = torch.cat(
                        single_batch_tensors*batch_size, dim=0)
                if self.control_tensor_cpu:
                    control_tile = control_tile.cpu()
                control_tile_list.append(control_tile)
            self.control_tensor_batch.append(control_tile_list)


class MultiDiffusion(TiledDiffusion):
    """
    MultiDiffusion Implementation
    Hijack the sampler for latent image tiling and fusion
    """
    def __init__(self, sampler, sampler_name, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # record the steps for progress bar
        # hook the sampler
        self.is_kdiff = sampler_name not in ['DDIM', 'PLMS', 'UniPC']
        assert sampler_name is not 'UniPC', 'MultiDiffusion is not compatible with UniPC, please use other samplers instead.'
        if self.is_kdiff:
            # For K-Diffusion sampler with uniform prompt, we hijack into the inner model for simplicity
            # Otherwise, the masked-redraw will break due to the init_latent
            self.sampler = sampler.model_wrap_cfg
            self.sampler_func = self.sampler.inner_model.forward
            self.sampler.inner_model.forward = self.kdiff_repeat
        else:
            self.iters = self.steps
            self.sampler = sampler
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

    def kdiff_repeat(self, x_in, sigma_in, cond):
        def repeat_func(x_tile, bboxes):
            # For kdiff sampler, the dim 0 of input x_in is batch_size * (num_AND + 1) if it is not an edit model;
            # otherwise, it is batch_size * (num_AND + 2)
            sigma_in_tile = sigma_in.repeat(len(bboxes))
            new_cond = self.repeat_cond_dict(cond, bboxes)
            x_tile_out = self.sampler_func(
                x_tile, sigma_in_tile, cond=new_cond)
            return x_tile_out

        def custom_func(x, custom_cond, uncond, bbox):
            '''
            Code migrate from modules/sd_samplers_kdiffusion.py
            '''
            is_edit_model = self.is_edit_model
            conds_list, tensor = prompt_parser.reconstruct_multicond_batch(
                custom_cond, self.sampler.step)
            uncond = prompt_parser.reconstruct_cond_batch(
                uncond, self.sampler.step)
            image_cond = cond['c_concat'][0]
            if image_cond.shape[2] == self.h and image_cond.shape[3] == self.w:
                image_cond = image_cond[:, :, bbox[1]:bbox[3], bbox[0]:bbox[2]]
            batch_size = len(conds_list)
            repeats = [len(conds_list[i]) for i in range(batch_size)]
            if not is_edit_model:
                x_in = torch.cat([torch.stack([x[i] for _ in range(n)])
                                 for i, n in enumerate(repeats)] + [x])
                image_cond_in = torch.cat([torch.stack([image_cond[i] for _ in range(
                    n)]) for i, n in enumerate(repeats)] + [image_cond])
            else:
                x_in = torch.cat([torch.stack([x[i] for _ in range(n)])
                                 for i, n in enumerate(repeats)] + [x] + [x])
                image_cond_in = torch.cat([torch.stack([image_cond[i] for _ in range(n)]) for i, n in enumerate(
                    repeats)] + [image_cond] + [torch.zeros_like(self.sampler.init_latent)])

            if tensor.shape[1] == uncond.shape[1]:
                if not is_edit_model:
                    cond_in = torch.cat([tensor, uncond])
                else:
                    cond_in = torch.cat([tensor, uncond, uncond])

                if shared.batch_cond_uncond:
                    x_out = self.sampler_func(x_in, sigma_in, cond={"c_crossattn": [
                                              cond_in], "c_concat": [image_cond_in]})
                else:
                    x_out = torch.zeros_like(x_in)
                    for batch_offset in range(0, x_out.shape[0], batch_size):
                        a = batch_offset
                        b = a + batch_size
                        x_out[a:b] = self.sampler_func(x_in[a:b], sigma_in[a:b], cond={"c_crossattn": [
                                                       cond_in[a:b]], "c_concat": [image_cond_in[a:b]]})
            else:
                x_out = torch.zeros_like(x_in)
                batch_size = batch_size*2 if shared.batch_cond_uncond else batch_size
                for batch_offset in range(0, tensor.shape[0], batch_size):
                    a = batch_offset
                    b = min(a + batch_size, tensor.shape[0])

                    if not is_edit_model:
                        c_crossattn = [tensor[a:b]]
                    else:
                        c_crossattn = torch.cat([tensor[a:b]], uncond)

                    x_out[a:b] = self.sampler_func(x_in[a:b], sigma_in[a:b], cond={
                                                   "c_crossattn": c_crossattn, "c_concat": [image_cond_in[a:b]]})
                x_out[-uncond.shape[0]:] = self.sampler_func(x_in[-uncond.shape[0]:], sigma_in[-uncond.shape[0]:], cond={
                                                             "c_crossattn": [uncond], "c_concat": [image_cond_in[-uncond.shape[0]:]]})
            return x_out

        return self.compute_x_tile(x_in, repeat_func, custom_func)

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

        def custom_func(x, cond, uncond, bbox):
            image_conditioning = None
            if isinstance(cond_in, dict):
                image_cond = cond_in['c_concat'][0]
                if image_cond.shape[2] == self.h and image_cond.shape[3] == self.w:
                    image_cond = image_cond[:, :,
                                            bbox[1]:bbox[3], bbox[0]:bbox[2]]
                image_conditioning = image_cond

            conds_list, tensor = prompt_parser.reconstruct_multicond_batch(
                cond, self.sampler.step)
            uncond = prompt_parser.reconstruct_cond_batch(
                uncond, self.sampler.step)

            assert all([len(conds) == 1 for conds in conds_list]
                       ), 'composition via AND is not supported for DDIM/PLMS samplers'
            cond = tensor

            # for DDIM, shapes must match, we can't just process cond and uncond independently;
            # filling uncond with repeats of the last vector to match length is
            # not 100% correct but should work well enough
            if uncond.shape[1] < cond.shape[1]:
                last_vector = uncond[:, -1:]
                last_vector_repeated = last_vector.repeat(
                    [1, cond.shape[1] - uncond.shape[1], 1])
                uncond = torch.hstack(
                    [uncond, last_vector_repeated])
            elif uncond.shape[1] > cond.shape[1]:
                uncond = uncond[:,:cond.shape[1]]

            if self.sampler.mask is not None:
                img_orig = self.sampler.sampler.model.q_sample(
                    self.init_latent, ts)
                x = img_orig * self.sampler.mask + self.sampler.nmask * x

            # Wrap the image conditioning back up since the DDIM code can accept the dict directly.
            # Note that they need to be lists because it just concatenates them later.
            if image_conditioning is not None:
                cond = {"c_concat": [image_conditioning],
                        "c_crossattn": [cond]}
                uncond = {"c_concat": [
                    image_conditioning], "c_crossattn": [uncond]}

            x_tile_out, x_pred = self.sampler_func(
                x, cond, ts, unconditional_conditioning=uncond, *args, **kwargs)
            return x_tile_out, x_pred

        return self.compute_x_tile(x_in, repeat_func, custom_func)

    def compute_x_tile(self, x_in, func, custom_func):
        N, C, H, W = x_in.shape
        assert H == self.h and W == self.w

        if self.pbar is None:
            self.pbar = tqdm(total=(self.num_batches+len(self.custom_bboxes))
                             * self.iters, desc="MultiDiffusion Sampling: ")

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
        for batch_id, bboxes in enumerate(self.batched_bboxes):
            if state.interrupted:
                return x_in
            x_tile_list = []
            for bbox in bboxes:
                x_tile_list.append(
                    x_in[:, :, bbox[1]:bbox[3], bbox[0]:bbox[2]])
            x_tile = torch.cat(x_tile_list, dim=0)
            # controlnet tiling
            if self.control_tensor_batch is not None:
                for i in range(len(self.control_params)):
                    new_control = self.control_tensor_batch[i][batch_id]
                    if new_control.shape[0] != x_tile.shape[0]:
                        new_control = new_control[:x_tile.shape[0], :, :, :]
                    self.control_params[i].hint_cond = new_control.to(
                        x_in.device)
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
            if self.pbar.n >= self.pbar.total:
                self.pbar.close()
            else:
                self.pbar.update()

        if len(self.custom_bboxes) > 0:
            if self.global_multiplier > 1:
                self.x_buffer *= self.global_multiplier
            if not self.is_kdiff:
                self.x_buffer_pred *= self.global_multiplier
            for bbox, cond, uncond, multiplier in self.custom_bboxes:
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

        x_out = torch.where(self.weights > 1, self.x_buffer /
                            self.weights, self.x_buffer)
        if not self.is_kdiff:
            x_pred = torch.where(
                self.weights > 1, self.x_buffer_pred / self.weights, self.x_buffer_pred)
            return x_out, x_pred
        return x_out



class MixtureOfDiffusers(TiledDiffusion):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def _gaussian_weights(self, tile_w=None, tile_h=None):
        '''
        Gaussian weights to smooth the noise of each tile
        '''
        if tile_w is None:
            tile_w = self.tile_w
        if tile_h is None:
            tile_h = self.tile_h
        var = 0.01
        midpoint = (tile_w - 1) / 2  # -1 because index goes from 0 to latent_width - 1
        x_probs = [exp(-(x-midpoint)*(x-midpoint)/(tile_w*tile_w)/(2*var)) / sqrt(2*pi*var) for x in range(tile_w)]
        midpoint = tile_h / 2
        y_probs = [exp(-(y-midpoint)*(y-midpoint)/(tile_h*tile_h)/(2*var)) / sqrt(2*pi*var) for y in range(tile_h)]
        weights = torch.from_numpy(np.outer(y_probs, x_probs)).to(devices.device)
        return weights
    
    def add_weights(self, input):
        if not hasattr(self, 'per_tile_weights'):
            self.per_tile_weights = self._gaussian_weights().to(input.device)
        input += self.per_tile_weights
    
    def hook(self):
        if not hasattr(shared.sd_model, 'md_org_apply_model'):
            shared.sd_model.md_org_apply_model = shared.sd_model.apply_model
            shared.sd_model.apply_model = self.apply_model
    
    @staticmethod
    def unhook():
        if hasattr(shared.sd_model, 'md_org_apply_model'):
            shared.sd_model.apply_model = shared.sd_model.md_org_apply_model
            del shared.sd_model.md_org_apply_model

    def apply_model(self, x_in, t_in, c_in):
        '''
        Hook to UNet when predicting noise
        '''
        N, C, H, W = x_in.shape
        assert H == self.h and W == self.w

        if self.pbar is None:
            self.pbar = tqdm(total=(self.num_batches+len(self.custom_bboxes))
                             * self.iters, desc="MixtureofDiffusers Sampling: ")
        # ControlNet support
        self.prepare_control_tensors(N)

        if self.x_buffer is None:
            self.x_buffer = torch.zeros_like(x_in, device=x_in.device)
        else:
            self.x_buffer.zero_()
        for batch_id, bboxes in enumerate(self.batched_bboxes):
            if state.interrupted:
                return x_in
            x_tile_list = []
            t_tile_list = []
            c_tile_list = []
            for bbox in bboxes:
                x_tile_list.append(
                    x_in[:, :, bbox[1]:bbox[3], bbox[0]:bbox[2]])
                t_tile_list.append(t_in)
                if c_in is not None:
                    c_tile_list.append(c_in[:, :, bbox[1]:bbox[3], bbox[0]:bbox[2]])
            x_tile = torch.cat(x_tile_list, dim=0)
            t_tile = torch.cat(t_tile_list, dim=0)
            c_tile = torch.cat(c_tile_list, dim=0) if c_in is not None else None
            # controlnet tiling
            if self.control_tensor_batch is not None:
                for i in range(len(self.control_params)):
                    new_control = self.control_tensor_batch[i][batch_id]
                    if new_control.shape[0] != x_tile.shape[0]:
                        new_control = new_control[:x_tile.shape[0], :, :, :]
                    self.control_params[i].hint_cond = new_control.to(
                        x_in.device)
            x_tile_out = shared.sd_model.md_org_apply_model(
                x_tile, t_tile, c_tile) # here the x is the noise
            
            for i, bbox in enumerate(bboxes):
                self.x_buffer[:, :, bbox[1]:bbox[3], bbox[0]:bbox[2]] += x_tile_out[i*N:(i+1)*N, :, :, :] * self.per_tile_weights
            
            # update progress bar
            if self.pbar.n >= self.pbar.total:
                self.pbar.close()
            else:
                self.pbar.update()
                
        x_out = self.x_buffer / self.weights
        return x_out


class Script(scripts.Script):

    def title(self):
        return "Tiled Diffusion"

    def show(self, is_img2img):
        return scripts.AlwaysVisible

    def ui(self, is_img2img):
        tab = 't2i' if not is_img2img else 'i2i'
        is_t2i = 'true' if not is_img2img else 'false'
        with gr.Accordion('Tiled Diffusion', open=False):
            with gr.Row(variant='compact'):
                enabled = gr.Checkbox(label='Enable', value=False)
                method = gr.Dropdown(label='Method', choices=[
                                   'MultiDiffusion', 'Mixture of Diffusers'], value='MultiDiffusion')
                
            with gr.Row(variant='compact', visible=False) as tab_size:
                image_width = gr.Slider(minimum=256, maximum=16384, step=16, label='Image width', value=1024,
                                        elem_id=f'MD-overwrite-width-{tab}')
                image_height = gr.Slider(minimum=256, maximum=16384, step=16, label='Image height', value=1024,
                                         elem_id=f'MD-overwrite-height-{tab}')

            with gr.Group():
                with gr.Row(variant='compact'):
                    tile_width = gr.Slider(minimum=16, maximum=256, step=16, label='Latent tile width', value=96,
                                           elem_id=self.elem_id("latent_tile_width"))
                    tile_height = gr.Slider(minimum=16, maximum=256, step=16, label='Latent tile height', value=96,
                                            elem_id=self.elem_id("latent_tile_height"))

                with gr.Row(variant='compact'):
                    overlap = gr.Slider(minimum=0, maximum=256, step=4, label='Latent tile overlap', value=48,
                                        elem_id=self.elem_id("latent_overlap"))
                    batch_size = gr.Slider(
                        minimum=1, maximum=8, step=1, label='Latent tile batch size', value=1)

            with gr.Row(variant='compact', visible=is_img2img):
                upscaler_index = gr.Dropdown(label='Upscaler', choices=[x.name for x in shared.sd_upscalers], value="None",
                                             elem_id='MD-upscaler-index')
                scale_factor = gr.Slider(minimum=1.0, maximum=8.0, step=0.05, label='Scale Factor', value=2.0,
                                         elem_id='MD-upscaler-factor')

            with gr.Row(variant='compact'):
                overwrite_image_size = gr.Checkbox(
                    label='Overwrite image size', value=False, visible=(not is_img2img))
                keep_input_size = gr.Checkbox(
                    label='Keep input image size', value=True, visible=(is_img2img))
                if not is_img2img:
                    overwrite_image_size.change(fn=lambda x: gr_show(
                        x), inputs=overwrite_image_size, outputs=tab_size)
                control_tensor_cpu = gr.Checkbox(
                    label='Move ControlNet images to CPU (if applicable)', value=False)

            # The control includes txt2img and img2img, we use t2i and i2i to distinguish them
            with gr.Group(variant='panel', elem_id=f'MD-bbox-control-{tab}'):
                with gr.Accordion('Region Prompt Control', open=False):

                    with gr.Row(variant='compact'):
                        enable_bbox_control = gr.Checkbox(
                            label='Enable', value=False)
                        global_multiplier = gr.Slider(
                            minimum=1, maximum=32, step=0.1, label='Global Multiplier', value=1, interactive=True)
                    with gr.Row(variant='compact'):
                        create_button = gr.Button(
                            value="Create txt2img canvas" if not is_img2img else "From img2img")

                    bbox_controls = []  # control set for each bbox
                    with gr.Row(variant='compact'):
                        ref_image = gr.Image(
                            label='Ref image (for conviently locate regions)', image_mode=None, elem_id=f'MD-bbox-ref-{tab}')
                        if not is_img2img:
                            # gradio has a serious bug: it cannot accept multiple inputs when you use both js and fn.
                            # to workaround this, we concat the inputs into a single string and parse it in js
                            def create_t2i_ref(string):
                                w, h = [int(x) for x in string.split('x')]
                                if w < 8:
                                    w = 8
                                if h < 8:
                                    h = 8
                                return np.zeros(shape=(h, w, 3), dtype=np.uint8) + 255
                            create_button.click(fn=create_t2i_ref, inputs=[
                                                overwrite_image_size], outputs=ref_image, _js=f'(o)=>onCreateT2IRefClick(o)')
                        else:
                            create_button.click(
                                fn=None, inputs=[], outputs=ref_image, _js=f'onCreateI2IRefClick')
                    for i in range(BBOX_MAX_NUM):
                        with gr.Accordion(f'Region {i+1}', open=False):
                            with gr.Row(variant='compact'):
                                e = gr.Checkbox(
                                    label=f'Enable', value=False, elem_id=f'MD-enable-{i}')
                                e.change(fn=None, inputs=[e], outputs=[
                                         e], _js=f'(e)=>onBoxEnableClick({is_t2i},{i}, e)')
                                m = gr.Slider(label=f'Multiplier', value=1, minimum=1,
                                              maximum=32, step=0.1, interactive=True, elem_id=f'MD-mt-{i}')
                            with gr.Row(variant='compact'):
                                x = gr.Slider(label=f'x', value=0.4, minimum=0.0, maximum=1.0,
                                              step=0.01, interactive=True, elem_id=f'MD-{tab}-{i}-x')
                                y = gr.Slider(label=f'y', value=0.4, minimum=0.0, maximum=1.0,
                                              step=0.01, interactive=True, elem_id=f'MD-{tab}-{i}-y')
                                w = gr.Slider(label=f'w', value=0.2, minimum=0.0, maximum=1.0,
                                              step=0.01, interactive=True, elem_id=f'MD-{tab}-{i}-w')
                                h = gr.Slider(label=f'h', value=0.2, minimum=0.0, maximum=1.0,
                                              step=0.01, interactive=True, elem_id=f'MD-{tab}-{i}-h')

                                x.change(fn=None, inputs=[x], outputs=[
                                         x], _js=f'(v)=>onBoxChange({is_t2i}, {i}, \'x\', v)')
                                y.change(fn=None, inputs=[y], outputs=[
                                         y], _js=f'(v)=>onBoxChange({is_t2i}, {i}, \'y\', v)')
                                w.change(fn=None, inputs=[w], outputs=[
                                         w], _js=f'(v)=>onBoxChange({is_t2i}, {i}, \'w\', v)')
                                h.change(fn=None, inputs=[h], outputs=[
                                         h], _js=f'(v)=>onBoxChange({is_t2i}, {i}, \'h\', v)')

                            p = gr.Text(
                                show_label=False, placeholder=f'Prompt, will be appended to your {tab} prompt)', max_lines=2, elem_id=f'MD-p-{i}')
                            neg = gr.Text(
                                show_label=False, placeholder=f'Negative Prompt, will be appended too.', max_lines=1, elem_id=f'MD-p-{i}')

                        bbox_controls.append((e, m, x, y, w, h, p, neg))

        controls = [
            enabled, method,
            overwrite_image_size, keep_input_size, image_width, image_height,
            tile_width, tile_height, overlap, batch_size,
            upscaler_index, scale_factor,
            control_tensor_cpu,
            enable_bbox_control,
            global_multiplier
        ]
        for i in range(BBOX_MAX_NUM):
            controls.extend(bbox_controls[i])
        return controls

    def process(self, p: StableDiffusionProcessing,
                enabled: bool, method: str,
                overwrite_image_size: bool, keep_input_size: bool, image_width: int, image_height: int,
                tile_width: int, tile_height: int, overlap: int, tile_batch_size: int,
                upscaler_index: str, scale_factor: float,
                control_tensor_cpu: bool,
                enable_bbox_control: bool, global_multiplier: int, *bbox_control_states
                ):

        if not enabled:
            MixtureOfDiffusers.unhook()
            return

        ''' upscale '''
        if hasattr(p, "init_images") and len(p.init_images) > 0:    # img2img
            upscaler_name = [x.name for x in shared.sd_upscalers].index(
                upscaler_index)

            init_img = p.init_images[0]
            init_img = images.flatten(init_img, opts.img2img_background_color)
            upscaler = shared.sd_upscalers[upscaler_name]
            if upscaler.name != "None":
                print(
                    f"[Tiled Diffusion] upscaling image with {upscaler.name}...")
                image = upscaler.scaler.upscale(
                    init_img, scale_factor, upscaler.data_path)
                p.extra_generation_params["Tiled Diffusion upscaler"] = upscaler.name
                p.extra_generation_params["Tiled Diffusion scale factor"] = scale_factor
            else:
                image = init_img
            p.init_images[0] = image

            if keep_input_size:
                p.width = image.width
                p.height = image.height
            elif upscaler.name != "None":
                p.width *= scale_factor
                p.height *= scale_factor
        elif overwrite_image_size:       # txt2img
            p.width = image_width
            p.height = image_height

        ''' sanitiy check '''
        if not TiledDiffusion.splitable(p.width, p.height, tile_width, tile_height, overlap):
            print(
                "[Tiled Diffusion] ignore due to image too small or tile size too large.")
            return
        p.extra_generation_params["Tiled Diffusion tile width"] = tile_width
        p.extra_generation_params["Tiled Diffusion tile height"] = tile_height
        p.extra_generation_params["Tiled Diffusion overlap"] = overlap

        ''' ControlNet hackin '''
        # try to hook into controlnet tensors
        controlnet_script = None
        try:
            from scripts.cldm import ControlNet
            # fix controlnet multi-batch issue

            def align(self, hint, h, w):
                if (len(hint.shape) == 3):
                    hint = hint.unsqueeze(0)
                _, _, h1, w1 = hint.shape
                if h != h1 or w != w1:
                    hint = torch.nn.functional.interpolate(
                        hint, size=(h, w), mode="nearest")
                return hint
            ControlNet.align = align
            for script in p.scripts.scripts + p.scripts.alwayson_scripts:
                if hasattr(script, "latest_network") and script.title().lower() == "controlnet":
                    controlnet_script = script
                    print(
                        "[Tiled Diffusion] ControlNet found, MultiDiffusion-ControlNet support is enabled.")
                    break
        except ImportError:
            pass

        ''' sampler hijack '''
        # hack the create_sampler function to get the created sampler
        old_create_sampler = sd_samplers.create_sampler

        def create_sampler(name, model):
            # create the sampler with the original function
            sampler = old_create_sampler(name, model)
            # unhook the create_sampler function
            sd_samplers.create_sampler = old_create_sampler
            if name in ['DDIM', 'UniPC', 'PLMS']:
                iters = p.steps
            else:
                iters = len(sampler.get_sigmas(p, p.steps))
            if method == 'MultiDiffusion':
                delegate = MultiDiffusion(
                    sampler, p.sampler_name, 
                    iters,p.batch_size, p.steps, p.width, p.height,
                    tile_width, tile_height, overlap, tile_batch_size,
                    controlnet_script=controlnet_script,
                    control_tensor_cpu=control_tensor_cpu
                )
            elif method == 'Mixture of Diffusers':
                delegate = MixtureOfDiffusers(
                    iters, p.batch_size, p.steps, p.width, p.height,
                    tile_width, tile_height, overlap, tile_batch_size,
                    controlnet_script=controlnet_script,
                    control_tensor_cpu=control_tensor_cpu
                )
                delegate.hook()
            if (enable_bbox_control):
                delegate.prepare_custom_bbox(
                    p.all_prompts, p.all_negative_prompts, global_multiplier, bbox_control_states)

            print(f"{method} hooked into {p.sampler_name} sampler. " +
                  f"Tile size: {tile_width}x{tile_height}, " +
                  f"Tile batches: {len(delegate.batched_bboxes)}, " +
                  f"Batch size:", tile_batch_size)
            return sampler
        
        sd_samplers.create_sampler = create_sampler
