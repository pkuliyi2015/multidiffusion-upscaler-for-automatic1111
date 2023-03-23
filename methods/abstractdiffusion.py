import math
import torch
import numpy as np


from enum import Enum
from abc import ABC, abstractmethod
from typing import List
from tqdm import tqdm

from modules import devices, shared, prompt_parser, extra_networks
from modules.shared import state
from modules.processing import StableDiffusionProcessing

class BlendMode(Enum):
    MIX = 'Mix'
    FEATHER = 'Feather'

class CustomBBox:
    def __init__(self, x, y ,w, h, prompt, neg_prompt, blend_mode, feather_radius, multiplier, extra_network_data):
        self.x = x
        self.y = y
        self.w = w
        self.h = h
        self.box = [x, y, x+w, y+h]
        self.prompt = prompt
        self.neg_prompt = neg_prompt
        self.multiplier = multiplier
        self.blend_mode = blend_mode
        self.feather_ratio = min(max(feather_radius, 0), min(w, h)/2)
        self.extra_network_data = extra_network_data
        if self.blend_mode == BlendMode.FEATHER:
            self.feather_mask = self.generate_feather_mask()

    def __getitem__(self, index):
        return self.box[index]
    
    def generate_feather_mask(self):
        '''Generate a feather mask for the bbox'''
        w, h = self.w, self.h
        mask = np.zeros((h, w), dtype=np.float32)

        feather_radius = int(min(w, h) * self.feather_ratio)
        # Generate the mask via gaussian weights
        # adjust the weight near the edge. the closer to the edge, the lower the weight
        for i in range(feather_radius):
            weight = math.exp(-0.5 * (1 - i/feather_radius)**2)
            mask[:, i] = weight
            mask[:, w-i-1] = weight
            mask[i, :] = weight
            mask[h-i-1, :] = weight
        return torch.from_numpy(mask).to(devices.device, dtype=torch.float32)




class TiledDiffusion(ABC):

    def __init__(self, method:str, sampler, p:StableDiffusionProcessing, tile_w=64, tile_h=64, overlap=32, tile_batch_size=1,
                 controlnet_script=None, control_tensor_cpu=False):
        self.p = p
        self.is_kdiff = p.sampler_name not in ['DDIM', 'PLMS', 'UniPC']
        if self.is_kdiff:
            # The sampler is CFGDenoiser
            self.sampler = sampler.model_wrap_cfg
        else:
            self.sampler = sampler

        self.method = method
        self.batch_size = p.batch_size
        self.steps = p.steps

        # initialize the tile bboxes and weights
        self.w, self.h = p.width//8, p.height//8
        if tile_w > self.w: tile_w = self.w
        if tile_h > self.h: tile_h = self.h
        min_tile_size = min(tile_w, tile_h)
        if overlap >= min_tile_size: overlap = min_tile_size - 4
        if overlap < 0: overlap = 0
        self.tile_w = tile_w
        self.tile_h = tile_h
        self.batched_bboxes = []

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
        self.control_tensor_custom = []

        # Progress bar
        self.pbar = None
        # For kdiff sampler, the step counting is extremely tricky
        self.step_count = 0
        self.inner_loop_count = 0
        self.kdiff_step = -1

        # Split the latent into tiles
        bboxes, weights = self.split_views(tile_w, tile_h, overlap)
        self.num_batches = math.ceil(len(bboxes) / tile_batch_size)
        optimal_batch_size = math.ceil(len(bboxes) / self.num_batches)
        self.tile_batch_size = optimal_batch_size
        for i in range(self.num_batches):
            start = i * tile_batch_size
            end = min((i + 1) * tile_batch_size, len(bboxes))
            self.batched_bboxes.append(bboxes[start:end])
        self.weights = weights.unsqueeze(0).unsqueeze(0)

    def init(self, x_in):
        # Kdiff 'AND' support and image editing model support
        if self.is_kdiff and not hasattr(self, 'is_edit_model'):
            self.is_edit_model = shared.sd_model.cond_stage_key == "edit" \
                and self.sampler.image_cfg_scale is not None \
                and self.sampler.image_cfg_scale != 1.0

        if self.pbar is None:
            self.total_bboxes = (self.num_batches if self.global_multiplier > 0 else 0) + len(self.custom_bboxes)
            assert self.total_bboxes > 0, "No bbox to sample! global_multiplier is 0 and no custom bboxes provided."
            self.pbar = tqdm(total=(self.total_bboxes) * state.sampling_steps, desc=f"{self.method} Sampling: ")

        if self.x_buffer is None:
            self.weights = self.weights.to(x_in.device)
            self.x_buffer = torch.zeros_like(x_in, device=x_in.device)
        else:
            self.x_buffer.zero_()
        
        # ControlNet support
        self.prepare_control_tensors()

    def update_pbar(self):
        if self.pbar.n >= self.pbar.total:
            self.pbar.close()
        else:
            if self.step_count == state.sampling_step:
                self.inner_loop_count += 1
                if self.inner_loop_count < (self.total_bboxes):
                    self.pbar.update()
            else:
                self.step_count = state.sampling_step
                self.inner_loop_count = 0

    def split_views(self, tile_w, tile_h, overlap):
        non_overlap_width = tile_w - overlap
        non_overlap_height = tile_h - overlap
        w, h = self.w, self.h
        cols = math.ceil((w - overlap) / non_overlap_width)
        rows = math.ceil((h - overlap) / non_overlap_height)

        dx = (w - tile_w) / (cols - 1) if cols > 1 else 0
        dy = (h - tile_h) / (rows - 1) if rows > 1 else 0

        bbox = []
        count = torch.zeros((h, w), device=devices.device, dtype=torch.float32)
        for row in range(rows):
            y = int(row * dy)
            if y + tile_h >= h:
                y = h - tile_h
            for col in range(cols):
                x = int(col * dx)
                if x + tile_w >= w:
                    x = w - tile_w
                bbox.append([x, y, x + tile_w, y + tile_h])
                count[y:y+tile_h, x:x+tile_w] += self.get_global_weights()
        return bbox, count

    @abstractmethod
    def get_global_weights(self):
        pass

    def init_custom_bbox(self, global_multiplier, bbox_control_states):
        '''
        Prepare custom bboxes for region prompt
        '''
        for i in range(0, len(bbox_control_states) - 10, 10):
            enable, x, y ,w, h, p, neg, blend_mode, feather_radius, multiplier = bbox_control_states[i:i+10]
            if not enable or x >= 1 or y>=1 or w <= 0 or h <= 0 or p == '': continue
            if blend_mode == 'Mix' and multiplier <= 0: continue
            x = int(x * self.w)
            y = int(y * self.h)
            w = math.ceil(w * self.w)
            h = math.ceil(h * self.h)
            x = max(0, x)
            y = max(0, y)
            w = min(self.w - x, w)
            h = min(self.h - y, h)
            bbox_obj = CustomBBox(x, y, w, h, None, None, BlendMode(blend_mode), feather_radius, multiplier, None)
            self.custom_bboxes.append(bbox_obj)

        if len(self.custom_bboxes) == 0: return
        self.global_multiplier = max(global_multiplier, 0.0)
        if self.global_multiplier >= 0 and abs(self.global_multiplier - 1.0) > 1e-5:
            self.weights *= self.global_multiplier

    def hook(self, sampler):
        if self.is_kdiff:
            # The sampler is CFGDenoiser
            self.sampler = sampler.model_wrap_cfg
        else:
            self.sampler = sampler

        self.reset_controlnet_tensors()

    def refresh_prompt(self, bbox_control_states, prompts:List[str], neg_prompts:List[str]):
        index = 0
        for i in range(0, len(bbox_control_states) - 10, 10):
            enable, x, y ,w, h, p, neg, blend_mode, feather_radius, multiplier = bbox_control_states[i:i+10]
            if not enable or x >= 1 or y>=1 or w <= 0 or h <= 0 or p == '': continue
            if blend_mode == 'Mix' and multiplier <= 0: continue
            c_prompt = [shared.prompt_styles.apply_styles_to_prompt(prompt + ', ' + p, self.p.styles) for prompt in prompts]
            c_prompt, extra_network_data = extra_networks.parse_prompts(c_prompt)
            if neg != '': c_negative_prompt = [shared.prompt_styles.apply_styles_to_prompt(prompt + ', ' + neg, self.p.styles) for prompt in neg_prompts]
            else:         c_negative_prompt = neg_prompts
            c_prompt = prompt_parser.get_multicond_learned_conditioning(shared.sd_model, c_prompt, self.steps)
            c_negative_prompt = prompt_parser.get_learned_conditioning(shared.sd_model, c_negative_prompt, self.steps)
            old_bbox_obj:CustomBBox = self.custom_bboxes[index]
            old_bbox_obj.prompt = c_prompt
            old_bbox_obj.neg_prompt = c_negative_prompt
            old_bbox_obj.extra_network_data = extra_network_data
            index += 1
        self.all_pos_cond = prompt_parser.get_multicond_learned_conditioning(shared.sd_model, prompts, self.steps)
        self.all_neg_cond = prompt_parser.get_learned_conditioning(shared.sd_model, neg_prompts, self.steps)

    def reconstruct_custom_cond(self, org_cond, custom_cond, custom_uncond, bbox):
        image_conditioning = None
        if isinstance(org_cond, dict):
            image_cond = org_cond['c_concat'][0]
            if image_cond.shape[2] == self.h and image_cond.shape[3] == self.w:
                image_cond = image_cond[:, :, bbox[1]:bbox[3], bbox[0]:bbox[2]]
            image_conditioning = image_cond

        conds_list, tensor = prompt_parser.reconstruct_multicond_batch(custom_cond, self.sampler.step)
        custom_uncond = prompt_parser.reconstruct_cond_batch(custom_uncond, self.sampler.step)

        return conds_list, tensor, custom_uncond, image_conditioning
    
    def kdiff_custom_forward(self, x_tile, original_cond, custom_cond, uncond, bbox_id, bbox, sigma_in, forward_func):
        '''
        # The inner kdiff noise prediction is usually batched.
        # We need to unwrap the inside loop to simulate the batched behavior.
        # This can be extremely tricky.
        '''
        if self.kdiff_step != self.sampler.step:
            self.kdiff_step = self.sampler.step
            self.kdiff_step_bbox = [-1 for _ in range(len(self.custom_bboxes))]
            self.tensor = {}
            self.uncond = {}
            self.image_cond_in = {}
            # Initialize global prompts just for estimate the behavior of kdiff
            _, self.real_tensor = prompt_parser.reconstruct_multicond_batch(self.all_pos_cond, self.sampler.step)
            self.real_uncond = prompt_parser.reconstruct_cond_batch(self.all_neg_cond, self.sampler.step)
            # reset the progress for all bboxes
            self.a = [0 for _ in range(len(self.custom_bboxes))]

        if self.kdiff_step_bbox[bbox_id] != self.sampler.step:
            # When a new step starts for a bbox, we need to judge whether the tensor is batched.
            self.kdiff_step_bbox[bbox_id] = self.sampler.step

            _, tensor, uncond, image_cond_in = self.reconstruct_custom_cond(original_cond, custom_cond, uncond, bbox)

            if self.real_tensor.shape[1] == self.real_uncond.shape[1]:
                # when the real tensor is with equal length, all information is contained in x_tile.
                # we simulate the batched behavior and compute all the tensors in one go.
                if tensor.shape[1] == uncond.shape[1] and shared.batch_cond_uncond:
                    if not self.is_edit_model:
                        cond = torch.cat([tensor, uncond])
                    else:
                        cond = torch.cat([tensor, uncond, uncond])
                    self.set_control_tensor(bbox_id, x_tile.shape[0])
                    return forward_func(x_tile, sigma_in, cond={"c_crossattn": [cond], "c_concat": [image_cond_in]})
                else:
                    x_out = torch.zeros_like(x_tile)
                    cond_size = tensor.shape[0]
                    self.set_control_tensor(bbox_id, cond_size)
                    cond_out = forward_func(
                        x_tile  [:cond_size], 
                        sigma_in[:cond_size], 
                        cond={
                            "c_crossattn": [tensor], 
                            "c_concat": [image_cond_in[:cond_size]]
                        })
                    uncond_size = uncond.shape[0]
                    self.set_control_tensor(bbox_id, uncond_size)
                    uncond_out = forward_func(
                        x_tile  [cond_size:cond_size+uncond_size], 
                        sigma_in[cond_size:cond_size+uncond_size], 
                        cond={
                            "c_crossattn": [uncond], 
                            "c_concat": [image_cond_in[cond_size:cond_size+uncond_size]]
                        })
                    x_out[:cond_size] = cond_out
                    x_out[cond_size:cond_size+uncond_size] = uncond_out
                    if self.is_edit_model:
                        x_out[cond_size+uncond_size:] = uncond_out
                    return x_out
                
            # otherwise, the x_tile is only a partial batch. We have to denoise in different runs.
            # initialize the state variables for current bbox
            self.tensor[bbox_id] = tensor
            self.uncond[bbox_id] = uncond
            self.image_cond_in[bbox_id] = image_cond_in

        # get current condition and uncondition
        tensor = self.tensor[bbox_id]
        uncond = self.uncond[bbox_id]
        batch_size = x_tile.shape[0]
        # get the start and end index of the current batch
        a = self.a[bbox_id]
        b = a + batch_size
        self.a[bbox_id] += batch_size
        # Judge the progress of batched processing cond and uncond for each bbox.
        # NOTE: The end condition is a rather than b.
        if a < tensor.shape[0]:
            if not self.is_edit_model:
                c_crossattn = [tensor[a:b]]
            else:
                c_crossattn = torch.cat([tensor[a:b]], uncond)
            self.set_control_tensor(bbox_id, x_tile.shape[0])
            # complete this batch.
            return forward_func(
                x_tile, 
                sigma_in, 
                cond={
                    "c_crossattn": c_crossattn, 
                    "c_concat": [self.image_cond_in[bbox_id]]
                })
        else:
            # if the cond is finished, we need to process the uncond.
            self.set_control_tensor(bbox_id, uncond.shape[0])
            return forward_func(
                x_tile, 
                sigma_in, 
                cond={
                    "c_crossattn": [uncond], 
                    "c_concat": [self.image_cond_in[bbox_id]]
                })
    
    def ddim_custom_forward(self, x, cond_in, cond, uncond, bbox, ts, forward_func, *args, **kwargs):
        conds_list, tensor, uncond, image_conditioning = self.reconstruct_custom_cond(cond_in, cond, uncond, bbox)
        assert all([len(conds) == 1 for conds in conds_list]), \
            'composition via AND is not supported for DDIM/PLMS samplers'

        cond = tensor
        # for DDIM, shapes definitely match. So we dont need to do the same thing as in the KDIFF sampler.
        if uncond.shape[1] < cond.shape[1]:
            last_vector = uncond[:, -1:]
            last_vector_repeated = last_vector.repeat([1, cond.shape[1] - uncond.shape[1], 1])
            uncond = torch.hstack([uncond, last_vector_repeated])
        elif uncond.shape[1] > cond.shape[1]:
            uncond = uncond[:, :cond.shape[1]]

        # Wrap the image conditioning back up since the DDIM code can accept the dict directly.
        # Note that they need to be lists because it just concatenates them later.
        if image_conditioning is not None:
            cond   = {"c_concat": [image_conditioning], "c_crossattn": [cond]}
            uncond = {"c_concat": [image_conditioning], "c_crossattn": [uncond]}
        
        # We cannot determine the batch size here for different methods, so delay it to the forward_func.
        return forward_func(x, cond, ts, unconditional_conditioning=uncond, *args, **kwargs)
    
    def prepare_control_tensors(self):
        '''
        Crop the control tensor into tiles and cache them
        '''
        if self.control_tensor_batch is not None: return
        if self.controlnet_script is None or self.control_params is not None: return
        latest_network = self.controlnet_script.latest_network
        if latest_network is None or not hasattr(latest_network, 'control_params'): return
        self.control_params = latest_network.control_params
        tensors = [param.hint_cond for param in latest_network.control_params]
        self.org_control_tensor_batch = tensors
        if len(tensors) == 0: return

        self.control_tensor_batch = []
        for i in range(len(tensors)):
            control_tile_list = []
            control_tensor = tensors[i]
            for bboxes in self.batched_bboxes:
                single_batch_tensors = []
                for bbox in bboxes:
                    if len(control_tensor.shape) == 3:
                        control_tile = control_tensor[:, bbox[1] * 8:bbox[3]*8, bbox[0]*8:bbox[2]*8].unsqueeze(0)
                    else:
                        control_tile = control_tensor[:, :, bbox[1]*8:bbox[3]*8, bbox[0]*8:bbox[2]*8]
                    single_batch_tensors.append(control_tile)
                control_tile = torch.cat(single_batch_tensors, dim=0)
                if self.control_tensor_cpu:
                    control_tile = control_tile.cpu()
                control_tile_list.append(control_tile)
            self.control_tensor_batch.append(control_tile_list)

            if len(self.custom_bboxes) > 0:
                custom_control_tile_list = []
                for bbox in self.custom_bboxes:
                    if len(control_tensor.shape) == 3:
                        control_tile = control_tensor[:, bbox[1]*8:bbox[3]*8, bbox[0]*8:bbox[2]*8].unsqueeze(0)
                    else:
                        control_tile = control_tensor[:, :, bbox[1]*8:bbox[3]*8, bbox[0]*8:bbox[2]*8]
                    if self.control_tensor_cpu:
                        control_tile = control_tile.cpu()
                    custom_control_tile_list.append(control_tile)
                self.control_tensor_custom.append(custom_control_tile_list)

    def switch_controlnet_tensors(self, batch_id, x_batch_size, tile_batch_size, is_denoise=False):
        if self.control_tensor_batch is not None:
            for param_id in range(len(self.control_params)):
                control_tile = self.control_tensor_batch[param_id][batch_id]
                if self.is_kdiff:
                    all_control_tile = []
                    for i in range(tile_batch_size):
                        this_control_tile = [control_tile[i].unsqueeze(0)] * x_batch_size
                        all_control_tile.append(torch.cat(this_control_tile, dim=0))
                    control_tile = torch.cat(all_control_tile, dim=0)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 
                else:
                    control_tile = control_tile.repeat([x_batch_size if is_denoise else x_batch_size * 2, 1, 1, 1])
                self.control_params[param_id].hint_cond = control_tile.to(devices.device)

    def set_control_tensor(self, bbox_id, repeat_size):
        if len(self.control_tensor_custom) > 0:
            for param_id in range(len(self.control_params)):
                control_tensor = self.control_tensor_custom[param_id][bbox_id].to(devices.device)
                self.control_params[param_id].hint_cond = control_tensor.repeat((repeat_size, 1, 1, 1))


    def reset_controlnet_tensors(self):
        if self.control_tensor_batch is not None:
            for param_id in range(len(self.control_params)):
                self.control_params[param_id].hint_cond = self.org_control_tensor_batch[param_id]