
import math
from enum import Enum

import torch
import numpy as np

from modules import devices, shared, prompt_parser, extra_networks
from modules.processing import opt_f

from tile_utils.typing import *


class Method(Enum):
    MULTI_DIFF = 'MultiDiffusion'
    MIX_DIFF   = 'Mixture of Diffusers'

    def __eq__(self, __value: object) -> bool:
        if isinstance(__value, str):
            return self.value == __value
        elif isinstance(__value, Method):
            return self.value == __value.value
        else:
            raise TypeError(f'unsupported type: {type(__value)}')    

class BlendMode(Enum):  # i.e. LayerType
    FOREGROUND = 'Foreground'
    BACKGROUND = 'Background'
    
    def __eq__(self, __value: object) -> bool:
        if isinstance(__value, str):
            return self.value == __value
        elif isinstance(__value, BlendMode):
            return self.value == __value.value
        else:
            raise TypeError(f'unsupported type: {type(__value)}')

class BBox:

    ''' grid bbox '''

    def __init__(self, x:int, y:int, w:int, h:int):
        self.x = x
        self.y = y
        self.w = w
        self.h = h
        self.box = [x, y, x+w, y+h]
        self.slicer = slice(None), slice(None), slice(y, y+h), slice(x, x+w)

    def __getitem__(self, idx:int) -> int:
        return self.box[idx]

class CustomBBox(BBox):

    ''' region control bbox '''

    def __init__(self, x:int, y:int, w:int, h:int, prompt:str, neg_prompt:str, blend_mode:str, feather_radio:float):
        super().__init__(x, y, w, h)

        self.prompt = prompt
        self.neg_prompt = neg_prompt
        self.blend_mode = BlendMode(blend_mode)
        self.feather_ratio = max(min(feather_radio, 1.0), 0.0)
        self.feather_mask = feather_mask(self.w, self.h, self.feather_ratio) if self.blend_mode == BlendMode.FOREGROUND else None
        self.cond: MulticondLearnedConditioning = None
        self.extra_network_data: DefaultDict[List[ExtraNetworkParams]] = None
        self.uncond: List[List[ScheduledPromptConditioning]] = None


class Prompt:

    ''' prompts handler '''

    @staticmethod
    def apply_styles(prompts:List[str], styles=None) -> List[str]:
        if not styles: return prompts
        return [shared.prompt_styles.apply_styles_to_prompt(p, styles) for p in prompts]

    @staticmethod
    def append_prompt(prompts:List[str], prompt:str='') -> List[str]:
        if not prompt: return prompts
        return [f'{p}, {prompt}' for p in prompts]

class Condition:

    ''' CLIP cond handler '''

    @staticmethod
    def get_cond(prompts:List[str], steps:int, styles=None) -> Tuple[Cond, ExtraNetworkData]:
        prompts = Prompt.apply_styles(prompts, styles)
        prompts, extra_network_data = extra_networks.parse_prompts(prompts)
        cond = prompt_parser.get_multicond_learned_conditioning(shared.sd_model, prompts, steps)
        return cond, extra_network_data 

    @staticmethod
    def get_uncond(neg_prompts:List[str], steps:int, styles=None) -> Uncond:
        neg_prompts = Prompt.apply_styles(neg_prompts, styles)
        uncond = prompt_parser.get_learned_conditioning(shared.sd_model, neg_prompts, steps)
        return uncond

    @staticmethod
    def reconstruct_cond(cond:Cond, step:int) -> Tuple[List, Tensor]:
        list_of_what, tensor = prompt_parser.reconstruct_multicond_batch(cond, step)
        return tensor

    def reconstruct_uncond(uncond:Uncond, step:int):
        tensor = prompt_parser.reconstruct_cond_batch(uncond, step)
        return tensor


def splitable(w:int, h:int, tile_w:int, tile_h:int, overlap:int=16) -> bool:
    w, h = w // opt_f, h // opt_f
    min_tile_size = min(tile_w, tile_h)
    if overlap >= min_tile_size:
        overlap = min_tile_size - 4
    cols = math.ceil((w - overlap) / (tile_w - overlap))
    rows = math.ceil((h - overlap) / (tile_h - overlap))
    return cols > 1 or rows > 1

def split_bboxes(w:int, h:int, tile_w:int, tile_h:int, overlap:int=16, init_weight:Union[Tensor, float]=1.0) -> Tuple[List[BBox], Tensor]:
    cols = math.ceil((w - overlap) / (tile_w - overlap))
    rows = math.ceil((h - overlap) / (tile_h - overlap))
    dx = (w - tile_w) / (cols - 1) if cols > 1 else 0
    dy = (h - tile_h) / (rows - 1) if rows > 1 else 0

    bbox_list: List[BBox] = []
    weight = torch.zeros((1, 1, h, w), device=devices.device, dtype=torch.float32)
    for row in range(rows):
        y = min(int(row * dy), h - tile_h)
        for col in range(cols):
            x = min(int(col * dx), w - tile_w)

            bbox = BBox(x, y, tile_w, tile_h)
            bbox_list.append(bbox)
            weight[bbox.slicer] += init_weight

    return bbox_list, weight


def gaussian_weights(tile_w:int, tile_h:int) -> Tensor:
    '''
    Copy from the original implementation of Mixture of Diffusers
    https://github.com/albarji/mixture-of-diffusers/blob/master/mixdiff/tiling.py
    This generates gaussian weights to smooth the noise of each tile.
    This is critical for this method to work.
    '''
    from numpy import pi, exp, sqrt
    
    f = lambda x, midpoint, var=0.01: exp(-(x-midpoint)*(x-midpoint) / (tile_w*tile_w) / (2*var)) / sqrt(2*pi*var)
    x_probs = [f(x, (tile_w - 1) / 2) for x in range(tile_w)]   # -1 because index goes from 0 to latent_width - 1
    y_probs = [f(y,  tile_h      / 2) for y in range(tile_h)]

    w = np.outer(y_probs, x_probs)
    return torch.from_numpy(w).to(devices.device, dtype=torch.float32)

def feather_mask(w:int, h:int, ratio:float) -> Tensor:
    '''Generate a feather mask for the bbox'''

    mask = np.ones((h, w), dtype=np.float32)
    feather_radius = int(min(w//2, h//2) * ratio)
    # Generate the mask via gaussian weights
    # adjust the weight near the edge. the closer to the edge, the lower the weight
    # weight = ( dist / feather_radius) ** 2
    for i in range(h//2):
        for j in range(w//2):
            dist = min(i, j)
            if dist >= feather_radius: continue
            weight = (dist / feather_radius) ** 2
            mask[i, j] = weight
            mask[i, w-j-1] = weight
            mask[h-i-1, j] = weight
            mask[h-i-1, w-j-1] = weight

    return torch.from_numpy(mask).to(devices.device, dtype=torch.float32)


def null_decorator(fn):
    def wrapper(*args, **kwargs):
        return fn(*args, **kwargs)
    return wrapper

keep_signature = null_decorator
controlnet     = null_decorator
grid_bbox      = null_decorator
custom_bbox    = null_decorator
