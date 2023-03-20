'''
   Export all methods
'''
from methods.multidiffusion import MultiDiffusion
from methods.mixtureofdiffusers import MixtureOfDiffusers
from methods.abstractdiffusion import TiledDiffusion
import math

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