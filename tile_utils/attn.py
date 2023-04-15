'''
    This file is modified from the sd_hijack_optimizations.py to remove the residual and norm part,
    So that the Tiled VAE can support other types of attention.
'''
import math
import torch

from modules import shared, devices, errors, sd_hijack
from modules.shared import cmd_opts
from einops import rearrange
from modules.sub_quadratic_attention import efficient_dot_product_attention
from modules.sd_hijack_optimizations import get_available_vram


try:
    import xformers
    import xformers.ops
except ImportError:
    pass


def get_attn_func():
    method = sd_hijack.model_hijack.optimization_method
    if method is None:
        return attn_forward
    method = method.lower()
    # The method should be one of the following:
    # ['none', 'sdp-no-mem', 'sdp', 'xformers', ''sub-quadratic', 'v1', 'invokeai', 'doggettx']
    if method not in ['none', 'sdp-no-mem', 'sdp', 'xformers', 'sub-quadratic', 'v1', 'invokeai', 'doggettx']:
        print(f"[Tiled VAE] Warning: Unknown attention optimization method {method}. Please try to update the extension.")
        return attn_forward
    
    if method == 'none':
        return attn_forward
    elif method == 'xformers':
        return xformers_attnblock_forward
    elif method == 'sdp-no-mem':
        return sdp_no_mem_attnblock_forward
    elif method == 'sdp':
        return sdp_attnblock_forward
    elif method == 'sub-quadratic':
        return sub_quad_attnblock_forward
    elif method == 'doggettx':
        return cross_attention_attnblock_forward
    
    return attn_forward

# The following functions are all copied from modules.sd_hijack_optimizations
# However, the residual & normalization are removed and computed later.


def attn_forward(self, h_):
    q = self.q(h_)
    k = self.k(h_)
    v = self.v(h_)

    # compute attention
    b, c, h, w = q.shape
    q = q.reshape(b, c, h*w)
    q = q.permute(0, 2, 1)   # b,hw,c
    k = k.reshape(b, c, h*w)  # b,c,hw
    w_ = torch.bmm(q, k)     # b,hw,hw    w[b,i,j]=sum_c q[b,i,c]k[b,c,j]
    w_ = w_ * (int(c)**(-0.5))
    w_ = torch.nn.functional.softmax(w_, dim=2)

    # attend to values
    v = v.reshape(b, c, h*w)
    w_ = w_.permute(0, 2, 1)   # b,hw,hw (first hw of k, second of q)
    # b, c,hw (hw of q) h_[b,c,j] = sum_i v[b,c,i] w_[b,i,j]
    h_ = torch.bmm(v, w_)
    h_ = h_.reshape(b, c, h, w)

    h_ = self.proj_out(h_)

    return h_


def xformers_attnblock_forward(self, h_):
    try:
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)
        b, c, h, w = q.shape
        q, k, v = map(lambda t: rearrange(t, 'b c h w -> b (h w) c'), (q, k, v))
        dtype = q.dtype
        if shared.opts.upcast_attn:
            q, k = q.float(), k.float()
        q = q.contiguous()
        k = k.contiguous()
        v = v.contiguous()
        out = xformers.ops.memory_efficient_attention(q, k, v, op=get_xformers_flash_attention_op(q, k, v))
        out = out.to(dtype)
        out = rearrange(out, 'b (h w) c -> b c h w', h=h)
        out = self.proj_out(out)
        return out
    except NotImplementedError:
        return cross_attention_attnblock_forward(self, h_)


def cross_attention_attnblock_forward(self, h_):
        q1 = self.q(h_)
        k1 = self.k(h_)
        v = self.v(h_)

        # compute attention
        b, c, h, w = q1.shape

        q2 = q1.reshape(b, c, h*w)
        del q1

        q = q2.permute(0, 2, 1)   # b,hw,c
        del q2

        k = k1.reshape(b, c, h*w) # b,c,hw
        del k1

        h_ = torch.zeros_like(k, device=q.device)

        mem_free_total = get_available_vram()

        tensor_size = q.shape[0] * q.shape[1] * k.shape[2] * q.element_size()
        mem_required = tensor_size * 2.5
        steps = 1

        if mem_required > mem_free_total:
            steps = 2**(math.ceil(math.log(mem_required / mem_free_total, 2)))

        slice_size = q.shape[1] // steps if (q.shape[1] % steps) == 0 else q.shape[1]
        for i in range(0, q.shape[1], slice_size):
            end = i + slice_size

            w1 = torch.bmm(q[:, i:end], k)     # b,hw,hw    w[b,i,j]=sum_c q[b,i,c]k[b,c,j]
            w2 = w1 * (int(c)**(-0.5))
            del w1
            w3 = torch.nn.functional.softmax(w2, dim=2, dtype=q.dtype)
            del w2

            # attend to values
            v1 = v.reshape(b, c, h*w)
            w4 = w3.permute(0, 2, 1)   # b,hw,hw (first hw of k, second of q)
            del w3

            h_[:, :, i:end] = torch.bmm(v1, w4)     # b, c,hw (hw of q) h_[b,c,j] = sum_i v[b,c,i] w_[b,i,j]
            del v1, w4

        h2 = h_.reshape(b, c, h, w)
        del h_

        h3 = self.proj_out(h2)
        del h2

        return h3


def sdp_no_mem_attnblock_forward(self, x):
    with torch.backends.cuda.sdp_kernel(enable_flash=True, enable_math=True, enable_mem_efficient=False):
        return sdp_attnblock_forward(self, x)
    

def sdp_attnblock_forward(self, h_):
    q = self.q(h_)
    k = self.k(h_)
    v = self.v(h_)
    b, c, h, w = q.shape
    q, k, v = map(lambda t: rearrange(t, 'b c h w -> b (h w) c'), (q, k, v))
    dtype = q.dtype
    if shared.opts.upcast_attn:
        q, k = q.float(), k.float()
    q = q.contiguous()
    k = k.contiguous()
    v = v.contiguous()
    out = torch.nn.functional.scaled_dot_product_attention(q, k, v, dropout_p=0.0, is_causal=False)
    out = out.to(dtype)
    out = rearrange(out, 'b (h w) c -> b c h w', h=h)
    out = self.proj_out(out)
    return out

def sub_quad_attnblock_forward(self, h_):
    q = self.q(h_)
    k = self.k(h_)
    v = self.v(h_)
    b, c, h, w = q.shape
    q, k, v = map(lambda t: rearrange(t, 'b c h w -> b (h w) c'), (q, k, v))
    q = q.contiguous()
    k = k.contiguous()
    v = v.contiguous()
    out = sub_quad_attention(q, k, v, q_chunk_size=shared.cmd_opts.sub_quad_q_chunk_size, kv_chunk_size=shared.cmd_opts.sub_quad_kv_chunk_size, chunk_threshold=shared.cmd_opts.sub_quad_chunk_threshold, use_checkpoint=self.training)
    out = rearrange(out, 'b (h w) c -> b c h w', h=h)
    out = self.proj_out(out)
    return out


def sub_quad_attention(q, k, v, q_chunk_size=1024, kv_chunk_size=None, kv_chunk_size_min=None, chunk_threshold=None, use_checkpoint=True):
    bytes_per_token = torch.finfo(q.dtype).bits//8
    batch_x_heads, q_tokens, _ = q.shape
    _, k_tokens, _ = k.shape
    qk_matmul_size_bytes = batch_x_heads * bytes_per_token * q_tokens * k_tokens

    if chunk_threshold is None:
        chunk_threshold_bytes = int(get_available_vram() * 0.9) if q.device.type == 'mps' else int(get_available_vram() * 0.7)
    elif chunk_threshold == 0:
        chunk_threshold_bytes = None
    else:
        chunk_threshold_bytes = int(0.01 * chunk_threshold * get_available_vram())

    if kv_chunk_size_min is None and chunk_threshold_bytes is not None:
        kv_chunk_size_min = chunk_threshold_bytes // (batch_x_heads * bytes_per_token * (k.shape[2] + v.shape[2]))
    elif kv_chunk_size_min == 0:
        kv_chunk_size_min = None

    if chunk_threshold_bytes is not None and qk_matmul_size_bytes <= chunk_threshold_bytes:
        # the big matmul fits into our memory limit; do everything in 1 chunk,
        # i.e. send it down the unchunked fast-path
        query_chunk_size = q_tokens
        kv_chunk_size = k_tokens

    with devices.without_autocast(disable=q.dtype == v.dtype):
        return efficient_dot_product_attention(
            q,
            k,
            v,
            query_chunk_size=q_chunk_size,
            kv_chunk_size=kv_chunk_size,
            kv_chunk_size_min = kv_chunk_size_min,
            use_checkpoint=use_checkpoint,
        )


def get_xformers_flash_attention_op(q, k, v):
    if not shared.cmd_opts.xformers_flash_attention:
        return None

    try:
        flash_attention_op = xformers.ops.MemoryEfficientAttentionFlashAttentionOp
        fw, bw = flash_attention_op
        if fw.supports(xformers.ops.fmha.Inputs(query=q, key=k, value=v, attn_bias=None)):
            return flash_attention_op
    except Exception as e:
        errors.display_once(e, "enabling flash attention")

    return None