# ------------------------------------------------------------------------
#
#   MultiDiffusion for Automatic1111 WebUI
#
#   Introducing a revolutionary large image drawing method - MultiDiffusion!
#
#   Techniques is not originally proposed by me, please refer to
#   Original Project: https://multidiffusion.github.io
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
import math

import torch
from tqdm import tqdm
import gradio as gr

from modules import sd_samplers, images, devices, shared, scripts, prompt_parser, sd_samplers_common
from modules.shared import opts, state
from modules.script_callbacks import cfg_denoiser_callback
from modules.sd_samplers_kdiffusion import CFGDenoiserParams


class MultiDiffusionDelegate(object):
    """
    Hijack the original sampler into MultiDiffusion samplers
    """

    def __init__(self, sampler, is_kdiff, steps, w, h, tile_w=64, tile_h=64, overlap=32, tile_batch_size=1, tile_prompt=False, prompt=[], neg_prompt=[]):
 
        self.steps = steps
        # record the steps for progress bar
        # hook the sampler
        if is_kdiff:
            self.sampler = sampler.model_wrap_cfg
            if tile_prompt:
                self.sampler_func = self.sampler.forward
                self.sampler.forward = self.kdiff_tile_prompt
                raise NotImplementedError("Tile prompt is not supported yet")
            else:
                # For K-Diffusion sampler with uniform prompt, we hijack into the inner model for simplicity
                # Otherwise, the masked-redraw will break due to the init_latent
                self.sampler_func = self.sampler.inner_model.forward
                self.sampler.inner_model.forward = self.kdiff_repeat
        else:
            if tile_prompt:
                raise NotImplementedError("Tile prompt is not supported yet")
            else:
                self.sampler_func = sampler.orig_p_sample_ddim
                self.sampler.orig_p_sample_ddim = self.ddim_repeat

        # initialize the tile bboxes and weights
        self.w, self.h = w//8, h//8
        min_tile_size = min(tile_w, tile_h)
        if overlap >= min_tile_size:
            overlap = min_tile_size - 4
        if overlap == 0:
            raise ValueError("Overlap must be greater than 0")
        self.tile_w = tile_w
        self.tile_h = tile_h
        bboxes, weights = self.split_views(tile_w, tile_h, overlap)
        self.batched_bboxes = []
        self.batched_conds = []
        self.batched_unconds = []
        self.num_batches = math.ceil(len(bboxes) / tile_batch_size)
        self.tile_prompt = tile_prompt
        for i in range(self.num_batches):
            start = i * tile_batch_size
            end = min((i + 1) * tile_batch_size, len(bboxes))
            self.batched_bboxes.append(bboxes[start:end])
            # TODO: deal with per tile prompt
            if tile_prompt:
                self.batched_conds.append(prompt_parser.get_multicond_learned_conditioning(shared.sd_model, prompt[start:end], self.steps))
                self.batched_unconds.append(prompt_parser.get_learned_conditioning(shared.sd_model, neg_prompt[start:end], self.steps))
        
        # Avoid the overhead of creating a new tensor for each batch
        # And avoid the overhead of weight summing
        self.weights = weights.unsqueeze(0).unsqueeze(0)
        self.x_buffer = None
        self.pbar = None


    @staticmethod
    def splitable(w, h, tile_w, tile_h, overlap):
        w, h = w//8, h//8
        min_tile_size = min(tile_w, tile_h)
        if overlap >= min_tile_size:
            overlap = min_tile_size - 4
        if overlap == 0:
            raise ValueError("Overlap must be greater than 0")
        non_overlap_width = tile_w - overlap
        non_overlap_height = tile_h - overlap
        cols = math.ceil((w - overlap) / non_overlap_width)
        rows = math.ceil((h - overlap) / non_overlap_height)
        return cols > 1 or rows > 1

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
                bbox.append((row, col, [x, y, x + tile_w, y + tile_h]))
                count[y:y+tile_h, x:x+tile_w] += 1
        return bbox, count
    
    def repeat_con_dict(self, cond_input, bboxes):
        cond = cond_input['c_crossattn'][0]
        # repeat the condition on its first dim
        cond_shape = cond.shape
        cond = cond.repeat((len(bboxes),) + (1,) * (len(cond_shape) - 1))
        image_cond = cond_input['c_concat'][0]
        if image_cond.shape[2] == self.h and image_cond.shape[3] == self.w:
            image_cond_list = []
            for _, _, bbox in bboxes:
                image_cond_list.append(image_cond[:, :, bbox[1]:bbox[3], bbox[0]:bbox[2]])
            image_cond_tile = torch.cat(image_cond_list, dim=0)
        else:
            image_cond_shape = image_cond.shape
            image_cond_tile = image_cond.repeat((len(bboxes),) + (1,) * (len(image_cond_shape) - 1))
        return {"c_crossattn": [cond], "c_concat": [image_cond_tile]}
    
    def kdiff_repeat(self, x_in, sigma_in, cond):
        def func(x_tile, bboxes):
            sigma_in_tile = sigma_in.repeat(len(bboxes))
            new_cond = self.repeat_con_dict(cond, bboxes)
            x_tile_out = self.sampler_func(x_tile, sigma_in_tile, cond=new_cond)
            return x_tile_out
        return self.compute_x_tile(x_in, func)
                
    def ddim_repeat(self, x_in, cond_in, ts, unconditional_conditioning, *args, **kwargs):
        def func(x_tile, bboxes):
            if isinstance(cond_in, dict):
                cond_in_tile = self.repeat_con_dict(cond_in, bboxes)
                ucond_tile = self.repeat_con_dict(unconditional_conditioning, bboxes)
            else:
                cond_shape = cond_in.shape
                cond_in_tile = cond_in.repeat((len(bboxes),) + (1,) * (len(cond_shape) - 1))
                ucond_shape = unconditional_conditioning.shape
                ucond_tile = unconditional_conditioning.repeat((len(bboxes),) + (1,) * (len(ucond_shape) - 1))
            x_tile_out = self.sampler_func(x_tile, cond_in_tile, ts, unconditional_conditioning=ucond_tile, *args, **kwargs)
            return x_tile_out
        return self.compute_x_tile(x_in, func)
    
    def compute_x_tile(self, x_in, func):
        N, C, H, W = x_in.shape
        assert H == self.h and W == self.w
        if self.x_buffer is None:
            self.x_buffer = torch.zeros_like(x_in, device=x_in.device)
        else:
            self.x_buffer.zero_()

        if self.pbar is None:
            self.pbar = tqdm(total=self.num_batches * ((state.job_count * state.sampling_steps) * 2 - 1), desc="MultiDiffusion Sampling: ")
        
        for bboxes in self.batched_bboxes:
            if state.interrupted:
                return x_in
            x_tile_list = []
            for _, _, bbox in bboxes:
                x_tile_list.append(x_in[:, :, bbox[1]:bbox[3], bbox[0]:bbox[2]])
            x_tile = torch.cat(x_tile_list, dim=0)
            x_tile_out = func(x_tile, bboxes)
            for i, (_, _, bbox) in enumerate(bboxes):
                self.x_buffer[:, :, bbox[1]:bbox[3], bbox[0]:bbox[2]] += x_tile_out[i*N:(i+1)*N, :, :, :]
            if self.pbar.n >= self.pbar.total:
                self.pbar.close()
            else:
                self.pbar.update()
        x_out = torch.where(self.weights > 1, self.x_buffer / self.weights, self.x_buffer)
        return x_out
    
    def kdiff_tile_prompt(delegate_self, self, x, sigma, uncond, cond, cond_scale, image_cond):
        '''
            Hijack into the CFGDenoiser forward function to support per tile prompt control
            This is because the K-Diffusion sampler may deal with prompt differently
            Also, we want to eliminate the overhead of reconstructing the useless overall prompt
        '''

        if state.interrupted or state.skipped:
            raise sd_samplers_common.InterruptedException

        conds_list, tensor = prompt_parser.reconstruct_multicond_batch(cond, self.step)
        uncond = prompt_parser.reconstruct_cond_batch(uncond, self.step)

        batch_size = len(conds_list)
        repeats = [len(conds_list[i]) for i in range(batch_size)]

        x_in = torch.cat([torch.stack([x[i] for _ in range(n)]) for i, n in enumerate(repeats)] + [x])
        image_cond_in = torch.cat([torch.stack([image_cond[i] for _ in range(n)]) for i, n in enumerate(repeats)] + [image_cond])
        sigma_in = torch.cat([torch.stack([sigma[i] for _ in range(n)]) for i, n in enumerate(repeats)] + [sigma])

        denoiser_params = CFGDenoiserParams(x_in, image_cond_in, sigma_in, state.sampling_step, state.sampling_steps)
        cfg_denoiser_callback(denoiser_params)
        x_in = denoiser_params.x
        image_cond_in = denoiser_params.image_cond
        sigma_in = denoiser_params.sigma

        # TODO: start multi tile processing here

        if tensor.shape[1] == uncond.shape[1]:
            cond_in = torch.cat([tensor, uncond])

            if shared.batch_cond_uncond:
                x_out = self.inner_model(x_in, sigma_in, cond={"c_crossattn": [cond_in], "c_concat": [image_cond_in]})
            else:
                x_out = torch.zeros_like(x_in)
                for batch_offset in range(0, x_out.shape[0], batch_size):
                    a = batch_offset
                    b = a + batch_size
                    x_out[a:b] = self.inner_model(x_in[a:b], sigma_in[a:b], cond={"c_crossattn": [cond_in[a:b]], "c_concat": [image_cond_in[a:b]]})
        else:
            x_out = torch.zeros_like(x_in)
            batch_size = batch_size*2 if shared.batch_cond_uncond else batch_size
            for batch_offset in range(0, tensor.shape[0], batch_size):
                a = batch_offset
                b = min(a + batch_size, tensor.shape[0])
                x_out[a:b] = self.inner_model(x_in[a:b], sigma_in[a:b], cond={"c_crossattn": [tensor[a:b]], "c_concat": [image_cond_in[a:b]]})

            x_out[-uncond.shape[0]:] = self.inner_model(x_in[-uncond.shape[0]:], sigma_in[-uncond.shape[0]:], cond={"c_crossattn": [uncond], "c_concat": [image_cond_in[-uncond.shape[0]:]]})

        # TODO: end multi tile processing here

        devices.test_for_nans(x_out, "unet")

        if opts.live_preview_content == "Prompt":
            sd_samplers_common.store_latent(x_out[0:uncond.shape[0]])
        elif opts.live_preview_content == "Negative prompt":
            sd_samplers_common.store_latent(x_out[-uncond.shape[0]:])

        denoised = self.combine_denoised(x_out, conds_list, uncond, cond_scale)

        if self.mask is not None:
            denoised = self.init_latent * self.mask + self.nmask * denoised

        self.step += 1

        return denoised


class Script(scripts.Script):

    def title(self):
        return "MultiDiffusion"

    def show(self, is_img2img):
        return scripts.AlwaysVisible

    def ui(self, is_img2img):
        with gr.Group():
            with gr.Accordion('MultiDiffusion', open=True):
                with gr.Row():
                    enabled = gr.Checkbox(
                        label='Enable MultiDiffusion', value=False)
                    override_image_size = gr.Checkbox(
                        label='Overwrite image size', value=False, visible=(not is_img2img))

                with gr.Row():
                    image_width = gr.Slider(minimum=256, maximum=8192, step=16, label='Image width', value=1024,
                                            elem_id=self.elem_id("image_width"), visible=False)
                    image_height = gr.Slider(minimum=256, maximum=8192, step=16, label='Image height', value=1024,
                                             elem_id=self.elem_id("image_height"), visible=False)
                with gr.Group():
                    with gr.Row():
                        tile_width = gr.Slider(minimum=16, maximum=128, step=16, label='Latent tile width', value=64,
                                               elem_id=self.elem_id("latent_tile_width"))
                        tile_height = gr.Slider(minimum=16, maximum=128, step=16, label='Latent tile height', value=64,
                                                elem_id=self.elem_id("latent_tile_height"))

                    with gr.Row():
                        overlap = gr.Slider(minimum=2, maximum=128, step=4, label='Latent tile overlap', value=48,
                                            elem_id=self.elem_id("latent_overlap"))
                        batch_size = gr.Slider(
                            minimum=1, maximum=8, step=1, label='Latent tile batch size', value=1)
                with gr.Group():
                    with gr.Row():
                        upscaler_index = gr.Dropdown(label='Upscaler', choices=[x.name for x in shared.sd_upscalers],
                                                     value="None", elem_id=self.elem_id("upscaler_index"))
                        scale_factor = gr.Slider(minimum=1.0, maximum=8.0, step=0.05, label='Scale Factor', value=2.0,
                                                 elem_id=self.elem_id("scale_factor"))
        if not is_img2img:
            def on_override_image_size(value):
                if value:
                    return gr.update(visible=True, interactive=True), gr.update(visible=True, interactive=True)
                else:
                    return gr.update(visible=False, interactive=False), gr.update(visible=False, interactive=False)

            override_image_size.change(fn=on_override_image_size, inputs=[
                                       override_image_size], outputs=[image_width, image_height])

        return [enabled, override_image_size, image_width, image_height, tile_width, tile_height, overlap, batch_size, upscaler_index, scale_factor]

    def process(self, p, enabled, override_image_size, image_width, image_height, tile_width, tile_height, overlap, tile_batch_size, upscaler_index, scale_factor):
        if not enabled:
            return False
        if isinstance(upscaler_index, str):
            upscaler_index = [x.name.lower() for x in shared.sd_upscalers].index(
                upscaler_index.lower())
        if len(p.init_images) > 0:
            init_img = p.init_images[0]
            init_img = images.flatten(init_img, opts.img2img_background_color)
            upscaler = shared.sd_upscalers[upscaler_index]
            if upscaler.name != "None":
                print(f"Upscaling image with {upscaler.name}...")
                image = upscaler.scaler.upscale(
                    init_img, scale_factor, upscaler.data_path)
            else:
                image = init_img
            p.init_images[0] = image
            p.width = image.width
            p.height = image.height
        elif override_image_size:
            p.width = image_width
            p.height = image_height
        if not MultiDiffusionDelegate.splitable(p.width, p.height, tile_width, tile_height, overlap):
            print(
                "MultiDiffusion is disabled because the image is too small or the tile size is too large.")
            return p
        p.extra_generation_params["MultiDiffusion tile width"] = tile_width
        p.extra_generation_params["MultiDiffusion tile height"] = tile_height
        p.extra_generation_params["MultiDiffusion overlap"] = overlap
        p.extra_generation_params["MultiDiffusion upscaler"] = upscaler.name
        p.extra_generation_params["MultiDiffusion scale factor"] = scale_factor
        # hack the create_sampler function to get the created sampler
        old_create_sampler = sd_samplers.create_sampler
        def create_sampler(name, model):
            # create the sampler with the original function
            sampler = old_create_sampler(name, model)
            # unhook the create_sampler function
            sd_samplers.create_sampler = old_create_sampler
            print("MultiDiffusion hooked into", p.sampler_name, "sampler.")
            MultiDiffusionDelegate(sampler, p.sampler_name not in [
                                   'DDIM', 'PLMS'], p.steps, p.width, p.height, tile_width, tile_height, overlap, tile_batch_size, False)
            return sampler
        sd_samplers.create_sampler = create_sampler
        
        return p
