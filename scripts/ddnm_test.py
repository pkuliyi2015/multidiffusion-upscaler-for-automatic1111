import os
import torch
import gradio as gr
import numpy as np

import modules.scripts as scripts
from tqdm import tqdm
from modules import devices
import modules.sd_samplers as sd_samplers

from modules.devices import NansException

"""
This script is used to test the DDNM sampler
"""

# copied from https://github.com/wyhuai/DDNM/blob/main/guided_diffusion/diffusion.py
def get_gaussian_noisy_img(img, noise_level):
    return img + torch.randn_like(img).cuda() * noise_level

def upsample(x, scale):
    n, c, h, w = x.shape
    out = torch.zeros(n, c, h, scale, w, scale).to(x.device) + x.view(n,c,h,1,w,1)
    out = out.view(n, c, scale*h, scale*w)
    return out

def color2gray(x):
    coef=1/3
    x = x[:,0,:,:] * coef + x[:,1,:,:]*coef +  x[:,2,:,:]*coef
    return x.repeat(1,3,1,1)

def gray2color(x):
    x = x[:,0,:,:]
    coef=1/3
    base = coef**2 + coef**2 + coef**2
    return torch.stack((x*coef/base, x*coef/base, x*coef/base), 1)    

def get_beta_schedule(beta_schedule, *, beta_start, beta_end, num_diffusion_timesteps):
    def sigmoid(x):
        return 1 / (np.exp(-x) + 1)

    if beta_schedule == "quad":
        betas = (
            np.linspace(
                beta_start ** 0.5,
                beta_end ** 0.5,
                num_diffusion_timesteps,
                dtype=np.float64,
            )
            ** 2
        )
    elif beta_schedule == "linear":
        betas = np.linspace(
            beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64
        )
    elif beta_schedule == "const":
        betas = beta_end * np.ones(num_diffusion_timesteps, dtype=np.float64)
    elif beta_schedule == "jsd":  
        betas = 1.0 / np.linspace(
            num_diffusion_timesteps, 1, num_diffusion_timesteps, dtype=np.float64
        )
    elif beta_schedule == "sigmoid":
        betas = np.linspace(-6, 6, num_diffusion_timesteps)
        betas = sigmoid(betas) * (beta_end - beta_start) + beta_start
    else:
        raise NotImplementedError(beta_schedule)
    assert betas.shape == (num_diffusion_timesteps,)
    return betas


# modified from ldm.models.diffusion.ddim.py and DDNM official code
@torch.no_grad()
def ddnm_sampling(self, cond, shape,
                    x_T=None, mode='scale', scale_factor=1, ddim_use_original_steps=False,
                    callback=None, timesteps=None, quantize_denoised=False,
                    mask=None, x0=None, img_callback=None, log_every_t=100,
                    temperature=1., noise_dropout=0., score_corrector=None, corrector_kwargs=None,
                    unconditional_guidance_scale=1., unconditional_conditioning=None, dynamic_threshold=None,
                    ucg_schedule=None):
    device = self.model.betas.device
    b = shape[0]
    assert x_T is not None
    if mode == 'scale':
        scale_factor = int(min(scale_factor,1))
        img = torch.randn((shape[0], shape[1], int(shape[2] * scale_factor), shape[3] * scale_factor), device=device)
    else:
        img = torch.randn(shape, device=device)

    if timesteps is None:
        timesteps = self.ddpm_num_timesteps if ddim_use_original_steps else self.ddim_timesteps
    elif timesteps is not None and not ddim_use_original_steps:
        subset_end = int(min(timesteps / self.ddim_timesteps.shape[0], 1) * self.ddim_timesteps.shape[0]) - 1
        timesteps = self.ddim_timesteps[:subset_end]

    intermediates = {'x_inter': [img], 'pred_x0': [img]}
    time_range = reversed(range(0,timesteps)) if ddim_use_original_steps else np.flip(timesteps)
    total_steps = timesteps if ddim_use_original_steps else timesteps.shape[0]
    print(f"Running DDIM Sampling with {total_steps} timesteps")

    iterator = tqdm(time_range, desc='DDIM Sampler', total=total_steps)

    for i, step in enumerate(iterator):
        index = total_steps - i - 1
        ts = torch.full((b,), step, device=device, dtype=torch.long)

        if mask is not None:
            assert x0 is not None
            img_orig = self.model.q_sample(x0, ts)  # TODO: deterministic forward pass?
            img = img_orig * mask + (1. - mask) * img

        if ucg_schedule is not None:
            assert len(ucg_schedule) == len(time_range)
            unconditional_guidance_scale = ucg_schedule[i]

        outs = self.p_sample_ddim(img, cond, ts, index=index, use_original_steps=ddim_use_original_steps,
                                    quantize_denoised=quantize_denoised, temperature=temperature,
                                    noise_dropout=noise_dropout, score_corrector=score_corrector,
                                    corrector_kwargs=corrector_kwargs,
                                    unconditional_guidance_scale=unconditional_guidance_scale,
                                    unconditional_conditioning=unconditional_conditioning,
                                    dynamic_threshold=dynamic_threshold)
        img, pred_x0 = outs
        if callback: callback(i)
        if img_callback: img_callback(pred_x0, i)

        if index % log_every_t == 0 or index == total_steps - 1:
            intermediates['x_inter'].append(img)
            intermediates['pred_x0'].append(pred_x0)

    return img, intermediates

@torch.no_grad()
def p_sample_ddim(self, x, c, t, index, repeat_noise=False, use_original_steps=False, quantize_denoised=False,
                    temperature=1., noise_dropout=0., score_corrector=None, corrector_kwargs=None,
                    unconditional_guidance_scale=1., unconditional_conditioning=None,
                    dynamic_threshold=None):
    b, *_, device = *x.shape, x.device

    if unconditional_conditioning is None or unconditional_guidance_scale == 1.:
        model_output = self.model.apply_model(x, t, c)
    else:
        x_in = torch.cat([x] * 2)
        t_in = torch.cat([t] * 2)
        if isinstance(c, dict):
            assert isinstance(unconditional_conditioning, dict)
            c_in = dict()
            for k in c:
                if isinstance(c[k], list):
                    c_in[k] = [torch.cat([
                        unconditional_conditioning[k][i],
                        c[k][i]]) for i in range(len(c[k]))]
                else:
                    c_in[k] = torch.cat([
                            unconditional_conditioning[k],
                            c[k]])
        elif isinstance(c, list):
            c_in = list()
            assert isinstance(unconditional_conditioning, list)
            for i in range(len(c)):
                c_in.append(torch.cat([unconditional_conditioning[i], c[i]]))
        else:
            c_in = torch.cat([unconditional_conditioning, c])
        model_uncond, model_t = self.model.apply_model(x_in, t_in, c_in).chunk(2)
        model_output = model_uncond + unconditional_guidance_scale * (model_t - model_uncond)

    if self.model.parameterization == "v":
        e_t = self.model.predict_eps_from_z_and_v(x, t, model_output)
    else:
        e_t = model_output

    if score_corrector is not None:
        assert self.model.parameterization == "eps", 'not implemented'
        e_t = score_corrector.modify_score(self.model, e_t, x, t, c, **corrector_kwargs)

    alphas = self.model.alphas_cumprod if use_original_steps else self.ddim_alphas
    alphas_prev = self.model.alphas_cumprod_prev if use_original_steps else self.ddim_alphas_prev
    sqrt_one_minus_alphas = self.model.sqrt_one_minus_alphas_cumprod if use_original_steps else self.ddim_sqrt_one_minus_alphas
    sigmas = self.model.ddim_sigmas_for_original_num_steps if use_original_steps else self.ddim_sigmas
    # select parameters corresponding to the currently considered timestep
    a_t = torch.full((b, 1, 1, 1), alphas[index], device=device)
    a_prev = torch.full((b, 1, 1, 1), alphas_prev[index], device=device)
    sigma_t = torch.full((b, 1, 1, 1), sigmas[index], device=device)
    sqrt_one_minus_at = torch.full((b, 1, 1, 1), sqrt_one_minus_alphas[index],device=device)

    # current prediction for x_0
    if self.model.parameterization != "v":
        pred_x0 = (x - sqrt_one_minus_at * e_t) / a_t.sqrt()
    else:
        pred_x0 = self.model.predict_start_from_z_and_v(x, t, model_output)

    if quantize_denoised:
        pred_x0, _, *_ = self.model.first_stage_model.quantize(pred_x0)

    if dynamic_threshold is not None:
        raise NotImplementedError()

    # direction pointing to x_t
    dir_xt = (1. - a_prev - sigma_t**2).sqrt() * e_t
    noise = sigma_t * noise_like(x.shape, device, repeat_noise) * temperature
    if noise_dropout > 0.:
        noise = torch.nn.functional.dropout(noise, p=noise_dropout)
    x_prev = a_prev.sqrt() * pred_x0 + dir_xt + noise
    return x_prev, pred_x0


class Scripts(scripts.Script):
    def __init__(self) -> None:
        model_rootpath = os.path.join(scripts.basedir(), 'models')
        if not os.path.exists(model_rootpath):
            os.makedirs(model_rootpath)
        # Cache the original sampler creator.

    def title(self):
        return "DDNM"

    def show(self, is_img2img):
        return is_img2img

    def ui(self, is_img2img):
        if not is_img2img:
            return
        # show in both modes
        with gr.Group():
            with gr.Row():
                enabled = gr.Checkbox(label="Enable", value=False)
                scale = gr.Slider(label="Scale Factor", min=2, max=16, step=1, value=2)
        
        return[enabled, scale]

    def process(self, p, enabled, scale):
        if not enabled:
            # restore the sampling process
            sd_samplers.create_sampler = self.org_sampler
            return

        input_image = p.init_images[0]

        return p
