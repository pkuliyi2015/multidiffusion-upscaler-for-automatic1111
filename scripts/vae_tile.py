#!/usr/bin/env python3
# Author: Armit
# Create Time: 2023/03/05

import math
from time import time
from traceback import print_exc
import gc

import torch
import torch.nn.functional as F
from tqdm import tqdm
import gradio as gr

import modules.devices as devices
from modules.scripts import Script, AlwaysVisible
from modules.shared import state
from modules.processing import opt_f

from torch import Tensor
from torch.nn import GroupNorm
from modules.processing import StableDiffusionProcessing
from ldm.models.autoencoder import AutoencoderKL
from ldm.modules.diffusionmodules.model import Encoder, Decoder, ResnetBlock, AttnBlock


if 'global const':
    def get_default_encoder_tile_size():
        if torch.cuda.is_available():
            total_memory = torch.cuda.get_device_properties(devices.device).total_memory // 2**20
            if total_memory > 16*1000:
                ENCODER_TILE_SIZE = 3072
            elif total_memory > 12*1000:
                ENCODER_TILE_SIZE = 2048
            elif total_memory > 8*1000:
                ENCODER_TILE_SIZE = 1536
            else:
                ENCODER_TILE_SIZE = 960
        else:
            ENCODER_TILE_SIZE = 512
        return ENCODER_TILE_SIZE

    def get_default_decoder_tile_size():
        if torch.cuda.is_available():
            total_memory = torch.cuda.get_device_properties(devices.device).total_memory // 2**20
            if   total_memory > 30*1000:
                DECODER_TILE_SIZE = 256
            elif total_memory > 16*1000:
                DECODER_TILE_SIZE = 192
            elif total_memory > 12*1000:
                DECODER_TILE_SIZE = 128
            elif total_memory >  8*1000:
                DECODER_TILE_SIZE = 96
            else:
                DECODER_TILE_SIZE = 64
        else:
            DECODER_TILE_SIZE = 64
        return DECODER_TILE_SIZE

    DEFAULT_ENABLED = False
    DEFAULT_PAD_SIZE = 2
    DEFAULT_ENCODER_TILE_SIZE = get_default_encoder_tile_size()
    DEFAULT_DECODER_TILE_SIZE = get_default_decoder_tile_size()

    DEBUG_SHAPE = False


# ↓↓↓ copied from 'vae_optimize.py' ↓↓↓

def get_var_mean(input, num_groups, eps=1e-6):
    """
    Get mean and var for group norm
    """
    b, c = input.size(0), input.size(1)
    channel_in_group = int(c/num_groups)
    input_reshaped = input.contiguous().view(1, int(b * num_groups), channel_in_group, *input.size()[2:])
    var, mean = torch.var_mean(input_reshaped, dim=[0, 2, 3, 4], unbiased=False)
    return var, mean

def custom_group_norm(input, num_groups, mean, var, weight=None, bias=None, eps=1e-6):
    """
    Custom group norm with fixed mean and var

    @param input: input tensor
    @param num_groups: number of groups. by default, num_groups = 32
    @param mean: mean, must be pre-calculated by get_var_mean
    @param var: var, must be pre-calculated by get_var_mean
    @param weight: weight, should be fetched from the original group norm
    @param bias: bias, should be fetched from the original group norm
    @param eps: epsilon, by default, eps = 1e-6 to match the original group norm

    @return: normalized tensor
    """
    b, c = input.size(0), input.size(1)
    channel_in_group = int(c/num_groups)
    input_reshaped = input.contiguous().view(1, int(b * num_groups), channel_in_group, *input.size()[2:])

    out = F.batch_norm(input_reshaped, mean, var, weight=None, bias=None, training=False, momentum=0, eps=eps)
    out = out.view(b, c, *input.size()[2:])

    try:
        # post affine transform
        if weight is not None:
            out *= weight.view(1, -1, 1, 1)
        if bias is not None:
            out += bias.view(1, -1, 1, 1)
    except:
        breakpoint()
    
    return out

# ↑↑↑ copied from 'vae_optimize.py' ↑↑↑


# ↓↓↓ modified from 'ldm/modules/diffusionmodules/model.py' ↓↓↓

def nonlinearity(x):
    return F.silu(x, inplace=True)

def Resblock_forward(self:ResnetBlock, x:Tensor):   # yield-3
    x = x.cpu()
    
    h = x.clone() ; yield self.norm1, h ; h = h.to(devices.device)
    h = nonlinearity(h)
    h = self.conv1(h)

    h = h.cpu() ; yield self.norm2, h ; h = h.to(devices.device)
    h = nonlinearity(h)
    #h = self.dropout(h)
    h = self.conv2(h)

    if self.in_channels != self.out_channels:
        x = x.to(devices.device)
        if self.use_conv_shortcut:
            x = self.conv_shortcut(x)
        else:
            x = self.nin_shortcut(x)

    yield x.to(devices.device) + h

def AttnBlock_forward(self:AttnBlock, x:Tensor):    # yield-2
    x = x.cpu()

    h = x.clone() ; yield self.norm, h ; h = h.to(devices.device)
    q = self.q(h)
    k = self.k(h)
    v = self.v(h)

    # compute attention
    B, C, H, W = q.shape
    q = q.reshape(B, C, H * W)
    q = q.permute(0, 2, 1)         # b,hw,c
    k = k.reshape(B, C, H * W)     # b,c,hw
    w = torch.bmm(q, k)            # b,hw,hw    w[b,i,j]=sum_c q[b,i,c]k[b,c,j]
    w = w * (int(C)**(-0.5))
    w = torch.nn.functional.softmax(w, dim=2)

    # attend to values
    v = v.reshape(B, C, H * W)
    w = w.permute(0,2,1)          # b,hw,hw (first hw of k, second of q)
    h = torch.bmm(v, w)           # b, c,hw (hw of q) h_[b,c,j] = sum_i v[b,c,i] w_[b,i,j]
    h = h.reshape(B, C, H, W)

    h = self.proj_out(h)
    yield x.to(devices.device) + h

def Encoder_forward(self:Encoder, x:Tensor):        # yield-?
    # prenet
    x = self.conv_in(x)
    
    # downsampling
    for i_level in range(self.num_resolutions):
        for i_block in range(self.num_res_blocks):
            x = Resblock_forward(self.down[i_level].block[i_block], x)
            if len(self.down[i_level].attn) > 0:
                x = AttnBlock_forward(self.down[i_level].attn[i_block], x)
        if i_level != self.num_resolutions-1:
            x = self.down[i_level].downsample(x)

    # middle
    x = Resblock_forward(self.mid.block_1, x)
    x = AttnBlock_forward(self.mid.attn_1, x)
    x = Resblock_forward(self.mid.block_2, x)

    # end
    yield self.norm_out, x

    x = nonlinearity(x)
    x = self.conv_out(x)
    yield x

def Decoder_forward(self:Decoder, x:Tensor):        # yield-?
    # prenet
    x = self.conv_in(x)     # [B, C=4, H, W] => [B, C=512, H, W]
    if DEBUG_SHAPE: print('conv_in:', x.shape)

    # middle
    for item in Resblock_forward(self.mid.block_1, x):
        if isinstance(item, Tensor): x = item
        else: yield item
    if DEBUG_SHAPE: print('block_1:', x.shape)
    for item in AttnBlock_forward(self.mid.attn_1, x):
        if isinstance(item, Tensor): x = item
        else: yield item
    if DEBUG_SHAPE: print('attn_1:', x.shape)
    for item in Resblock_forward(self.mid.block_2, x):
        if isinstance(item, Tensor): x = item
        else: yield item
    if DEBUG_SHAPE: print('block_2:', x.shape)

    # upsampling
    for i_level in reversed(range(self.num_resolutions)):
        for i_block in range(self.num_res_blocks+1):
            for item in Resblock_forward(self.up[i_level].block[i_block], x):
                if isinstance(item, Tensor): x = item
                else: yield item
            if DEBUG_SHAPE: print(f'up[{i_level}].block[{i_block}]:', x.shape)
            if len(self.up[i_level].attn) > 0:      # assert empty
                for item in AttnBlock_forward(self.up[i_level].attn[i_block], x):
                    if isinstance(item, Tensor): x = item
                    else: yield item
        if i_level != 0:
            x = self.up[i_level].upsample(x)
            if DEBUG_SHAPE: print(f'up[{i_level}].upsample:', x.shape)

    # end
    if self.give_pre_end: yield x.cpu()

    x = x.cpu() ; yield self.norm_out, x ; x = x.to(devices.device)
    x = nonlinearity(x)
    x = self.conv_out(x)
    if DEBUG_SHAPE: print(f'conv_out:', x.shape)
    if self.tanh_out: x = torch.tanh(x)
    yield x.cpu()

# ↑↑↑ modified from 'ldm/modules/diffusionmodules/model.py' ↑↑↑


def perfcount(fn):
    def wrapper(*args, **kwargs):
        ts = time()

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats(devices.device)
        devices.torch_gc()
        gc.collect()

        ret = fn(*args, **kwargs)
        
        devices.torch_gc()
        gc.collect()
        if torch.cuda.is_available():
            vram = torch.cuda.max_memory_allocated(devices.device) / 2**20
            torch.cuda.reset_peak_memory_stats(devices.device)
            print(f'Done in {time() - ts:.3f}s, max VRAM alloc {vram:.3f} MB')
        else:
            print(f'Done in {time() - ts:.3f}s')

        return ret
    return wrapper

@perfcount
def Encoder_forward_tile(self:Decoder, x:Tensor, tile_size:int, pad_size:int):
    z = Encoder_forward(self, x)

    return z

@perfcount
def Decoder_forward_tile(self:Decoder, z:Tensor, tile_size:int, pad_size:int):
    B, C, H, W = z.shape

    if 'estimate max tensor shape':
        shape = torch.Size((B, 512//2, H*opt_f, W*opt_f))
        size_t = 2 if self.conv_in.weight.dtype == torch.float16 else 4
        print(f'>> max tensor shape: {tuple(shape)}, memsize: {shape.numel() * size_t / 2**20:.3f} MB')

    n_tiles_H = math.ceil(H / tile_size)
    n_tiles_W = math.ceil(W / tile_size)
    print(f'>> split to {n_tiles_H}x{n_tiles_W} = {n_tiles_H*n_tiles_W} tiles')

    if pad_size != 0: z = F.pad(z, (pad_size, pad_size, pad_size, pad_size), mode='reflect')     # [B, C, H+2*pad, W+2*pad]

    bbox_inputs  = []
    bbox_outputs = []
    x = 0
    for _ in range(n_tiles_H):
        y = 0
        for _ in range(n_tiles_W):
            bbox_inputs.append((
                (x, min(x + tile_size, H) + 2 * pad_size),
                (y, min(y + tile_size, W) + 2 * pad_size),
            ))
            bbox_outputs.append((
                (x * opt_f, min(x + tile_size, H) * opt_f),
                (y * opt_f, min(y + tile_size, W) * opt_f),
            ))
            y += tile_size
        x += tile_size
    if DEBUG_SHAPE:
        print('bbox_inputs:')
        print(bbox_inputs)
        print('bbox_outputs:')
        print(bbox_outputs)

    workers = []
    for bbox in bbox_inputs:
        (Hs, He), (Ws, We) = bbox
        tile = z[:, :, Hs:He, Ws:We]
        workers.append(Decoder_forward(self, tile))

    result = z[:, :3, :, :]     # very cheap tmp result

    interrupted = False
    pbar = tqdm(total=31, desc='VAE tile decoding...')
    while True:
        if state.interrupted or interrupted: break

        try:
            outputs = [ next(worker) for worker in workers ]
            ret_types = { type(o) for o in outputs }
        except StopIteration:
            print_exc()
            raise ValueError('Error: workers stopped early !!')

        if   ret_types == { tuple }:    # GroupNorm sync barrier
            gns = { gn for gn, _ in outputs }
            if len(gns) > 1:
                print(f'group_norms: {gns}')
                raise ValueError('Error: workers progressing states not synchronized !!')

            gn: GroupNorm = list(gns)[0]
            num_groups = gn.num_groups
            weight     = gn.weight                      # 'cuda'
            bias       = gn.bias
            eps        = gn.eps
            del gns

            tiles = [ tile for _, tile in outputs ]     # 'cpu'
            if DEBUG_SHAPE: print('tile.shape:', tiles[0].shape)

            dtype = None
            var_list, mean_list = [], []
            for tile in tiles:
                if state.interrupted: interrupted = True ; break

                dtype = tile.dtype
                var, mean = get_var_mean(tile.float().to(devices.device), num_groups, eps)
                var_list.append(var)
                mean_list.append(mean)
            var  = torch.stack(var_list,  dim=0).mean(dim=0).to(devices.device)     # [NG=32], float32
            mean = torch.stack(mean_list, dim=0).mean(dim=0).to(devices.device)
            del var_list, mean_list

            for tile in tiles:
                if state.interrupted: interrupted = True ; break

                tile_n = custom_group_norm(tile.float().to(devices.device), num_groups, mean, var, weight, bias, eps)
                tile_n = tile_n.to(dtype).cpu()
                tile.data = tile_n
            del tiles

            devices.torch_gc()
            gc.collect()
        
        elif ret_types == { Tensor }:   # final Tensor splits
            if DEBUG_SHAPE: print('output.shape:', outputs[0].shape)    # 'cpu'
            assert len(bbox_outputs) == len(outputs), 'n_tiles != n_bbox_outputs'

            result = torch.zeros([B, 3, H*opt_f, W*opt_f], dtype=outputs[0].dtype)
            count  = torch.zeros([B, 1, H*opt_f, W*opt_f], dtype=torch.uint8)

            def crop_pad(x:Tensor, size:int):
                if size == 0: return x
                return x[:, :, size:-size, size:-size]

            for i, bbox in enumerate(bbox_outputs):
                (Hs, He), (Ws, We) = bbox
                result[:, :, Hs:He, Ws:We] += crop_pad(outputs[i], pad_size * opt_f)
                count [:, :, Hs:He, Ws:We] += 1
            del outputs

            count = count.clamp_(min=1)
            result /= count
            break       # we're done!

        else:
            print(f'ret_types: {ret_types}')
            raise ValueError('Error: workers progressing states not synchronized !!')

        pbar.update()

    # Done!
    pbar.close()

    return result


def Encoder_forward_hijack(self:Decoder, x:Tensor, tile_size:int, pad_size:int):
    B, C, H, W = x.shape
    if max(H, W) < tile_size: return self.original_forward(x)
    else: return Encoder_forward_tile(self, x, tile_size, pad_size)

def Decoder_forward_hijack(self:Decoder, x:Tensor, tile_size:int, pad_size:int):
    B, C, H, W = x.shape
    if max(H, W) <= tile_size: return self.original_forward(x)
    else: return Decoder_forward_tile(self, x, tile_size, pad_size)


class Script(Script):

    def title(self):
        return "VAE Tiling"

    def show(self, is_img2img):
        return AlwaysVisible

    def ui(self, is_img2img):
        with gr.Accordion('Yet Another VAE Tiling', open=True):
            with gr.Row():
                enabled = gr.Checkbox(label='Enabled', value=lambda: DEFAULT_ENABLED)
                pad_size = gr.Slider(label='Pad size', minimum=0, maximum=12, step=1, value=lambda: DEFAULT_PAD_SIZE)
                reset = gr.Button(value='Reset defaults')

            with gr.Row():
                encoder_tile_size = gr.Slider(label='Encoder tile size', minimum=32, maximum=256, step=8, value=lambda: DEFAULT_ENCODER_TILE_SIZE)
                decoder_tile_size = gr.Slider(label='Decoder tile size', minimum=32, maximum=256, step=8, value=lambda: DEFAULT_DECODER_TILE_SIZE)

            reset.click(fn=lambda: [DEFAULT_ENCODER_TILE_SIZE, DEFAULT_DECODER_TILE_SIZE, DEFAULT_PAD_SIZE], outputs=[encoder_tile_size, decoder_tile_size, pad_size])
        
        return enabled, pad_size, encoder_tile_size, decoder_tile_size

    def process(self, p:StableDiffusionProcessing, enabled:bool, pad_size:int, encoder_tile_size:int, decoder_tile_size:int):
        vae: AutoencoderKL = p.sd_model.first_stage_model
        if vae.device == torch.device('cpu'): return

        # for shorthand
        encoder: Encoder = vae.encoder
        decoder: Decoder = vae.decoder

        # save original forward (only once)
        if not hasattr(encoder, 'original_forward'): encoder.original_forward = encoder.forward
        if not hasattr(decoder, 'original_forward'): decoder.original_forward = decoder.forward

        if not enabled:
            # undo hijack
            encoder.forward = encoder.original_forward
            decoder.forward = decoder.original_forward
        else:
            # apply hijack
            encoder.forward = lambda x: Encoder_forward_hijack(encoder, x, encoder_tile_size, pad_size)
            decoder.forward = lambda x: Decoder_forward_hijack(decoder, x, decoder_tile_size, pad_size)
