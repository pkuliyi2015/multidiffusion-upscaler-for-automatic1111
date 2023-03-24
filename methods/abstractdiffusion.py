import math
from enum import Enum
from typing import List, Dict, Any

import torch
from torch import Tensor
from tqdm import tqdm

from modules import devices, shared, prompt_parser, extra_networks
from modules.shared import state
from modules.processing import opt_f, StableDiffusionProcessing
from modules.prompt_parser import MulticondLearnedConditioning, ScheduledPromptConditioning

from typing import Union
from modules.sd_samplers_compvis import VanillaStableDiffusionSampler
from modules.sd_samplers_kdiffusion import KDiffusionSampler, CFGDenoiser

from methods.utils import BBox, feather_mask, split_bboxes, controlnet


class Method(Enum):
    MULTI_DIFF = 'MultiDiffusion'
    MIX_DIFF   = 'Mixture of Diffusers'


class BlendMode(Enum):      # LayerType
    FOREGROUND = 'Foreground'
    BACKGROUND = 'Background'


class CustomBBox(BBox):

    def __init__(self, x:int, y:int, w:int, h:int, prompt:str, neg_prompt:str, blend_mode:BlendMode, feather_radio:float):
        super().__init__(x, y, w, h)

        self.prompt_cache = prompt      # main-prompts, will be appended to sub-prompts
        self.neg_prompt_cache = neg_prompt or ''
        self.cond: MulticondLearnedConditioning = None
        self.uncond: List[ScheduledPromptConditioning] = None
        self.blend_mode = blend_mode
        self.feather_ratio = max(min(feather_radio,1),0)
        self.extra_network_data = None
        self.feather_mask = feather_mask(self.w, self.h, self.feather_ratio) if self.blend_mode == BlendMode.FOREGROUND else None

    def prepare_cond(self, prompts:List[str], neg_prompts:List[str], styles, steps:int):
        ''' each CustomBBox is a relative independent subplot, so prepare for its own set of prompts (cond/uncond) '''
        c_prompt = [shared.prompt_styles.apply_styles_to_prompt(prompt + ', ' + self.prompt_cache, styles) for prompt in prompts]
        c_prompt, self.extra_network_data = extra_networks.parse_prompts(c_prompt)
        if self.neg_prompt_cache != '': 
            c_negative_prompt = [shared.prompt_styles.apply_styles_to_prompt(prompt + ', ' + self.neg_prompt_cache, styles)  for prompt in neg_prompts]
        else:
            c_negative_prompt = neg_prompts
        self.cond = prompt_parser.get_multicond_learned_conditioning(shared.sd_model, c_prompt, steps)
        self.uncond = prompt_parser.get_learned_conditioning(shared.sd_model, c_negative_prompt, steps)


class TiledDiffusion:

    def __init__(self, 
        p:StableDiffusionProcessing, 
        sampler:Union[KDiffusionSampler, VanillaStableDiffusionSampler], 
        tile_w=64, tile_h=64, overlap=32, tile_batch_size=1,
        controlnet_script=None, control_tensor_cpu=False, 
        causal_layers=False,
    ):
        self.method = self.__class__.__name__

        self.is_kdiff = isinstance(sampler, KDiffusionSampler)
        self.sampler_raw = sampler
        if self.is_kdiff:
            self.sampler: CFGDenoiser = sampler.model_wrap_cfg
        else:
            self.sampler: VanillaStableDiffusionSampler = sampler

        self.p = p
        self.batch_size = p.batch_size
        self.steps = p.steps        # total sampling steps

        # initialize the grid tile bboxes and weights
        self.w, self.h = p.width // opt_f, p.height // opt_f
        if tile_w > self.w: tile_w = self.w
        if tile_h > self.h: tile_h = self.h
        min_tile_size = min(tile_w, tile_h)
        if overlap >= min_tile_size: overlap = min_tile_size - 4
        overlap = max(0, overlap)
        self.tile_w = tile_w
        self.tile_h = tile_h

        # Split the latent into tiles
        bboxes, weights = split_bboxes(self.w, self.h, tile_w, tile_h, overlap, self.init_tile_weights)
        self.weights = weights.to(devices.device)
        self.num_batches     = math.ceil(len(bboxes) / tile_batch_size)
        self.tile_batch_size = math.ceil(len(bboxes) / self.num_batches)    # optimal_batch_size
        self.batched_bboxes: List[List[CustomBBox]] = []    # grid tile bboxes
        for i in range(self.num_batches):
            start = i * self.tile_batch_size
            end = min((i + 1) * self.tile_batch_size, len(bboxes))
            self.batched_bboxes.append(bboxes[start:end])

        # Avoid the overhead of creating a new tensor for each batch and weight summing
        self.x_buffer = None        # current result, [B, C=4, H, W]

        # Region prompt control
        self.custom_bboxes = []     # custom tile bboxes
        self.draw_background = True     # this must be True
        self.causal_layers = causal_layers

        # ControlNet
        self.controlnet_script = controlnet_script
        self.control_tensor_batch = None
        self.control_params = None
        self.control_tensor_cpu = control_tensor_cpu
        self.control_tensor_custom = []

        # Kdiff 'AND' support and image editing model support
        if self.is_kdiff and not hasattr(self, 'is_edit_model'):
            self.is_edit_model = shared.sd_model.cond_stage_key == "edit" \
                and self.sampler.image_cfg_scale is not None \
                and self.sampler.image_cfg_scale != 1.0

        # For kdiff sampler, the step counting is extremely tricky
        self.step_count = 0
        self.inner_loop_count = 0
        self.kdiff_step = -1

        # Progress bar
        self.total_bboxes = (self.num_batches if self.draw_background else 0) + len(self.custom_bboxes)
        assert self.total_bboxes > 0, "No region to sample! no background to draw and no custom bboxes were provided."
        self.pbar = tqdm(total=(self.total_bboxes) * state.sampling_steps, desc=f"{self.method} Sampling: ")

        # ControlNet support
        self.reset_controlnet_tensors()

    @property
    def init_tile_weights(self) -> Union[Tensor, float]:
        return 1.0

    def hook(self):
        raise NotImplementedError
        
    @staticmethod
    def unhook():
        raise NotImplementedError

    def reset_buffer(self, x_in:Tensor):
        # reset result canvas
        if self.x_buffer is None:
            self.x_buffer = torch.zeros_like(x_in, device=x_in.device)
        else:
            self.x_buffer.zero_()

        # ControlNet support
        self.prepare_controlnet_tensors()

    def init_custom_bbox(self, draw_background:bool, bbox_control_states:List[Any]):
        ''' Prepare custom bboxes for region prompt, init weight of each tile '''

<<<<<<< HEAD
        self.custom_bboxes: List[CustomBBox] = []
        for i in range(0, len(bbox_control_states) - 9, 9):
            e, x, y, w, h, p, n, blend_mode, feather_ratio = bbox_control_states[i:i+9]
            if not e or x > 1.0 or y > 1.0 or w <= 0.0 or h <= 0.0: continue

=======
    def split_views(self, tile_w, tile_h, overlap):
        non_overlap_width = tile_w - overlap
        non_overlap_height = tile_h - overlap
        w, h = self.w, self.h
        cols = math.ceil((w - overlap) / non_overlap_width)
        rows = math.ceil((h - overlap) / non_overlap_height)

        dx = (w - tile_w) / (cols - 1) if cols > 1 else 0
        dy = (h - tile_h) / (rows - 1) if rows > 1 else 0

        bbox_list = []
        count = torch.zeros((1, 1, h, w), device=devices.device, dtype=torch.float32)
        for row in range(rows):
            y = int(row * dy)
            if y + tile_h >= h:
                y = h - tile_h
            for col in range(cols):
                x = int(col * dx)
                if x + tile_w >= w:
                    x = w - tile_w
                bbox = BBox(x, y, tile_w, tile_h)
                bbox_list.append(bbox)
                count[bbox.slice] += self.get_global_weights()
        return bbox_list, count

    @abstractmethod
    def get_global_weights(self):
        pass

    def init_custom_bbox(self, draw_background, bbox_control_states):
        '''
        Prepare custom bboxes for region prompt
        '''
        self.custom_bboxes = []
        for i in range(0, len(bbox_control_states), 9):
            enable, x, y ,w, h, p, neg, blend_mode, feather_ratio = bbox_control_states[i:i+9]
            if not enable or x >= 1 or y>=1 or w <= 0 or h <= 0: continue
>>>>>>> main
            x = int(x * self.w)
            y = int(y * self.h)
            w = math.ceil(w * self.w)
            h = math.ceil(h * self.h)
            x = max(0, x)
            y = max(0, y)
            w = min(self.w - x, w)
            h = min(self.h - y, h)
            cbbox = CustomBBox(x, y, w, h, p, n, BlendMode(blend_mode), feather_ratio)
            self.custom_bboxes.append(cbbox)
        if len(self.custom_bboxes) == 0: return

        self.draw_background = draw_background
        if not draw_background: self.weights.zero_()

        for bbox in self.custom_bboxes:
            bbox.prepare_cond(self.p.all_prompts, self.p.all_negative_prompts, self.p.styles, self.steps)
        # prepare_cond
        self.all_pos_cond: MulticondLearnedConditioning = prompt_parser.get_multicond_learned_conditioning(shared.sd_model, self.p.all_prompts, self.steps)
        self.all_neg_cond: List[ScheduledPromptConditioning] = prompt_parser.get_learned_conditioning(shared.sd_model, self.p.all_negative_prompts, self.steps)

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

    def kdiff_custom_forward(self, 
            x_tile:Tensor, sigma_in:Tensor, 
            original_cond:Dict[str, Tensor], 
            custom_cond:MulticondLearnedConditioning, 
            uncond:ScheduledPromptConditioning, 
            bbox_id:int, bbox:CustomBBox, forward_func,
        ):
        ''' draw custom bbox '''
        '''
        # The inner kdiff noise prediction is usually batched.
        # We need to unwrap the inside loop to simulate the batched behavior.
        # This can be extremely tricky.
        '''
        # x_tile: [1, 4, 13, 15]
        # original_cond: {'c_crossattn': Tensor[1, 77, 768], 'c_concat': Tensor[1, 5, 1, 1]}
        # custom_cond: MulticondLearnedConditioning
        # uncond: Tensor[1, 231, 768]
        # bbox: CustomBBox
        # sigma_in: Tensor[1]
        # forward_func: CFGDenoiser.forward
        if self.kdiff_step != self.sampler.step:
            self.kdiff_step = self.sampler.step
            self.kdiff_step_bbox = [-1 for _ in range(len(self.custom_bboxes))]
            self.tensor = {}        # {int: Tensor[cond]}
            self.uncond = {}        # {int: Tensor[cond]}
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
                    self.set_controlnet_tensors(bbox_id, x_tile.shape[0])
                    return forward_func(x_tile, sigma_in, cond={"c_crossattn": [cond], "c_concat": [image_cond_in]})
                else:
                    x_out = torch.zeros_like(x_tile)
                    cond_size = tensor.shape[0]
                    self.set_controlnet_tensors(bbox_id, cond_size)
                    cond_out = forward_func(
                        x_tile  [:cond_size], 
                        sigma_in[:cond_size], 
                        cond={
                            "c_crossattn": [tensor], 
                            "c_concat": [image_cond_in[:cond_size]]
                        })
                    uncond_size = uncond.shape[0]
                    self.set_controlnet_tensors(bbox_id, uncond_size)
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
            self.set_controlnet_tensors(bbox_id, x_tile.shape[0])
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
            self.set_controlnet_tensors(bbox_id, uncond.shape[0])
            return forward_func(
                x_tile, 
                sigma_in, 
                cond={
                    "c_crossattn": [uncond], 
                    "c_concat": [self.image_cond_in[bbox_id]]
                })
    
    def ddim_custom_forward(self,
            x:Tensor, 
            cond_in:Dict[str, Tensor], 
            cond:MulticondLearnedConditioning, 
            uncond:ScheduledPromptConditioning, 
            bbox:CustomBBox, ts, forward_func, 
            *args, **kwargs
        ):
        ''' draw custom bbox '''

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
    
    def update_pbar(self):
        if self.pbar.n >= self.pbar.total:
            self.pbar.close()
        else:
            if self.step_count == state.sampling_step:
                self.inner_loop_count += 1
                if self.inner_loop_count < self.total_bboxes:
                    self.pbar.update()
            else:
                self.step_count = state.sampling_step
                self.inner_loop_count = 0

    @controlnet
    def prepare_controlnet_tensors(self):
        ''' Crop the control tensor into tiles and cache them '''

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
                        control_tile = control_tensor[:, bbox[1]* 8:bbox[3]*8, bbox[0]*8:bbox[2]*8].unsqueeze(0)
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

    @controlnet
    def switch_controlnet_tensors(self, batch_id:int, x_batch_size:int, tile_batch_size:int, is_denoise=False):
        if self.control_tensor_batch is None: return

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

    @controlnet
    def set_controlnet_tensors(self, bbox_id:int, repeat_size:int):
        if not len(self.control_tensor_custom): return
        
        for param_id in range(len(self.control_params)):
            control_tensor = self.control_tensor_custom[param_id][bbox_id].to(devices.device)
            self.control_params[param_id].hint_cond = control_tensor.repeat((repeat_size, 1, 1, 1))

    @controlnet
    def reset_controlnet_tensors(self):
        if self.control_tensor_batch is None: return

        for param_id in range(len(self.control_params)):
            self.control_params[param_id].hint_cond = self.org_control_tensor_batch[param_id]
