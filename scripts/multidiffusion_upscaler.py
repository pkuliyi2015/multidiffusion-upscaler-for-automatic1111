import math
import types

import torch
from scripts.vae_optimize import vae_tile_decode, vae_tile_encode
from tqdm import tqdm

import modules.scripts as scripts
import gradio as gr

from modules import processing, shared, sd_samplers, images, devices, prompt_parser
from modules.shared import opts, state
import numpy as np


"""
    The code is largely based on the original SD Upscale script, 
    but it uses the MultiDiffusion and merge the latent instead of post-processing the image.
    Currently only works with the DDIM sampler.
"""


class Script(scripts.Script):

    def title(self):
        return "MultiDiffusion Upscale"

    def show(self, is_img2img):
        return is_img2img

    def ui(self, is_img2img):
        info = gr.HTML(
            "<p style=\"margin-bottom:0.75em\">Will upscale the image by the selected scale factor; use width and height sliders to set tile size</p>")

        with gr.Row():
            tile_width = gr.Slider(minimum=16, maximum=128, step=16, label='Latent tile width', value=64,
                                   elem_id=self.elem_id("latent_tile_width"))
            tile_height = gr.Slider(minimum=16, maximum=128, step=16, label='Latent tile height', value=64,
                                    elem_id=self.elem_id("latent_tile_height"))
        with gr.Row():
            scale_factor = gr.Slider(minimum=1.0, maximum=8.0, step=0.05, label='Scale Factor', value=2.0,
                                     elem_id=self.elem_id("scale_factor"))
            overlap = gr.Slider(minimum=2, maximum=128, step=4, label='Latent tile overlap', value=48,
                                elem_id=self.elem_id("latent_overlap"))
        upscaler_index = gr.Radio(label='Upscaler', choices=[x.name for x in shared.sd_upscalers],
                                  value=shared.sd_upscalers[0].name, type="index",
                                  elem_id=self.elem_id("upscaler_index"))

        return [info, tile_width, tile_height, overlap, upscaler_index, scale_factor]

    """
        Generates crops from a 2-D image with given height, width, window height, window width, and stride. 
        The implementation ensures that the crops are symmetrically cropped with minimum overlap
        The stride will be modified when the window size is not divisible by the stride
        Modified from Automatic1111's implementation
    """

    def split_grid(self, w, h, tile_w=64, tile_h=64, overlap=8):

        non_overlap_width = tile_w - overlap
        non_overlap_height = tile_h - overlap

        cols = math.ceil((w - overlap) / non_overlap_width)
        rows = math.ceil((h - overlap) / non_overlap_height)

        dx = (w - tile_w) / (cols - 1) if cols > 1 else 0
        dy = (h - tile_h) / (rows - 1) if rows > 1 else 0

        views = []
        for row in range(rows):

            y = int(row * dy)

            if y + tile_h >= h:
                y = h - tile_h

            for col in range(cols):
                x = int(col * dx)

                if x + tile_w >= w:
                    x = w - tile_w

                views.append([x, y, x + tile_w, y + tile_h])

        return views

    def run(self, p, _, tile_width, tile_height, overlap, upscaler_index, scale_factor):

        if isinstance(upscaler_index, str):
            upscaler_index = [x.name.lower() for x in shared.sd_upscalers].index(upscaler_index.lower())
        processing.fix_seed(p)
        upscaler = shared.sd_upscalers[upscaler_index]

        p.extra_generation_params["SD upscale tile width"] = tile_width
        p.extra_generation_params["SD upscale tile height"] = tile_height
        p.extra_generation_params["SD upscale overlap"] = overlap
        p.extra_generation_params["SD upscale upscaler"] = upscaler.name

        initial_info = None
        seed = p.seed

        init_img = p.init_images[0]
        init_img = images.flatten(init_img, opts.img2img_background_color)

        # upscale the image if needed
        if upscaler.name != "None":
            print(f"Upscaling image with {upscaler.name}...")
            image = upscaler.scaler.upscale(init_img, scale_factor, upscaler.data_path)
        else:
            image = init_img

        print(f"Upscaled image size: {image.width}x{image.height}")

        p.do_not_save_grid = True
        p.init_images = [image]
        p.width = image.width
        p.height = image.height

        devices.torch_gc()

        # calculate latent tile bbox.
        # code is from the MultiDiffusion repo
        latent_width = image.width // 8
        latent_height = image.height // 8

        # ensure the tile size is less than the latent size
        tile_width = min(latent_width, tile_width)
        tile_height = min(latent_height, tile_height)

        min_tile_size = min(tile_height, tile_width)
        if overlap >= min_tile_size:
            overlap = min_tile_size - 8

        views = self.split_grid(latent_width, latent_height, tile_width, tile_height, overlap)

        batch_size = p.batch_size
        p.batch_size = 1
        batch_views = []
        num_batches = math.ceil(len(views) / batch_size)

        for i in range(num_batches):
            batch_views.append(views[i * batch_size: min((i + 1) * batch_size, len(views))])

        print(f"MultiDiffusion upscaling will process a total of {len(views)} latent tiles.")

        # custom ddim sampler, which merges the latent instead of post-processing the image

        @torch.no_grad()
        def multidiffusion_decode(self, x_latent, cond, t_start, unconditional_guidance_scale=1.0,
                                  unconditional_conditioning=None,
                                  use_original_steps=False, callback=None):
            # In test mode, the ddim sampling will get called only once.

            timesteps = np.arange(self.ddpm_num_timesteps) if use_original_steps else self.ddim_timesteps
            timesteps = timesteps[:t_start]

            time_range = np.flip(timesteps)
            total_steps = timesteps.shape[0]
            print(f"Running MultiDiffusion DDIM Sampling with {total_steps} timesteps")
            pbar = tqdm(desc='Decoding image', total=total_steps*len(batch_views))
            x_dec = x_latent
            new_dec = torch.zeros_like(x_dec, device=x_latent.device)
            # save memory via broadcasting
            fusion_count = torch.zeros((x_dec.shape[0], 1, x_dec.shape[2], x_dec.shape[3]), device=x_latent.device)
            uc_batch = prompt_parser.get_learned_conditioning(shared.sd_model, p.all_negative_prompts * batch_size,
                                                              p.steps)
            c_batch = prompt_parser.get_multicond_learned_conditioning(shared.sd_model, p.all_prompts * batch_size,
                                                                       p.steps)
            if len(batch_views[-1]) < batch_size:
                uc_batch_tail = prompt_parser.get_learned_conditioning(shared.sd_model,
                                                                       p.all_negative_prompts * len(batch_views[-1]),
                                                                       p.steps)
                c_batch_tail = prompt_parser.get_multicond_learned_conditioning(shared.sd_model,
                                                                                p.all_prompts * len(batch_views[-1]),
                                                                                p.steps)
            else:
                uc_batch_tail = uc_batch
                c_batch_tail = c_batch
            for i, step in enumerate(time_range):
                index = total_steps - i - 1
                new_dec.zero_()
                fusion_count.zero_()
                for views_in_batch in batch_views:
                    tiles = []
                    for view in views_in_batch:
                        x1, y1, x2, y2 = view
                        tile = x_dec[:, :, y1:y2, x1:x2]
                        tiles.append(tile)
                    tiles = torch.cat(tiles, dim=0)
                    ts = torch.full((tiles.shape[0],), step, device=x_latent.device, dtype=torch.long)
                    if tiles.shape[0] == batch_size:
                        c = c_batch
                        uc = uc_batch
                    else:
                        c = c_batch_tail
                        uc = uc_batch_tail
                    # TODO: use tile-wise text prompt
                    tiles, _ = self.p_sample_ddim(tiles, c, ts, index=index, use_original_steps=use_original_steps,
                                                  unconditional_guidance_scale=unconditional_guidance_scale,
                                                  unconditional_conditioning=uc)

                    for j, view in enumerate(views_in_batch):
                        x1, y1, x2, y2 = view
                        new_dec[:, :, y1:y2, x1:x2] += tiles[j, :, :, :]
                        fusion_count[:, :, y1:y2, x1:x2] += 1
                    pbar.update(1)
                x_dec = torch.where(fusion_count > 1, new_dec / fusion_count, new_dec)
                if callback: callback(i)

            org_vae_decoder_forward = self.model.first_stage_model.decoder.forward
            # hijack the vae to save vram
            if x_latent.shape[2] > 192 or x_latent.shape[3] > 192:
                print("The latent is larger than 192x192. Hijack VAE decoding...")
                def delegate_decode(self, x):
                    try:
                        return vae_tile_decode(self, x)
                    finally:
                        self.forward = org_vae_decoder_forward
                self.model.first_stage_model.decoder.forward = types.MethodType(delegate_decode,self.model.first_stage_model.decoder)
            return x_dec

        org_sampler = sd_samplers.create_sampler
        custom_sampler = org_sampler('DDIM', p.sd_model)
        custom_sampler.sampler.decode = types.MethodType(multidiffusion_decode, custom_sampler.sampler)
        sd_samplers.create_sampler = lambda name, model: custom_sampler
        org_vae_encoder_forward = p.sd_model.first_stage_model.encoder.forward
        p.sd_model.first_stage_model.encoder.forward = types.MethodType(vae_tile_encode, p.sd_model.first_stage_model.encoder)

        try:
            return processing.process_images(p)
        finally:
            sd_samplers.create_sampler = org_sampler
            p.sd_model.first_stage_model.encoder.forward = org_vae_encoder_forward

