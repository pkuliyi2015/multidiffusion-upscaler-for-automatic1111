import math
from typing import List, Tuple, Union

import torch
from torch import Tensor
import numpy as np

from modules import devices
from modules.processing import opt_f


class BBox:

    def __init__(self, x:int, y:int, w:int, h:int):
        self.x = x
        self.y = y
        self.w = w
        self.h = h
        self.box = [x, y, x+w, y+h]
        self.slicer = slice(None), slice(None), slice(y, y+h), slice(x, x+w)

    def __getitem__(self, idx:int) -> int:
        return self.box[idx]


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
        y = int(row * dy)
        if y + tile_h >= h:
            y = h - tile_h
        for col in range(cols):
            x = int(col * dx)
            if x + tile_w >= w:
                x = w - tile_w

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

controlnet = null_decorator
