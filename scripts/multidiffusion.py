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
import torch.nn.functional as F
from tqdm import tqdm
import gradio as gr

from modules import sd_samplers, images, devices, shared, scripts, prompt_parser, sd_samplers_common
from modules.shared import opts, state
from modules.sd_samplers_kdiffusion import CFGDenoiserParams
from modules.ui import gr_show

from gradio import Slider, Text, Group
from typing import Tuple, List
from modules.processing import StableDiffusionProcessing


if 'global const':
    BBOX_MAX_NUM = 8



class MultiDiffusionDelegate(object):
    """
    Hijack the original sampler into MultiDiffusion samplers
    """

    def __init__(self, sampler, sampler_name, steps, 
                 w, h, tile_w=64, tile_h=64, overlap=32, tile_batch_size=1, 
                 tile_prompt=False, prompt=[], neg_prompt=[], 
                 controlnet_script=None, control_tensor_cpu=False):

        self.steps = steps
        # record the steps for progress bar
        # hook the sampler
        self.is_kdiff = sampler_name not in ['DDIM', 'PLMS', 'UniPC']
        if self.is_kdiff:
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
            self.sampler = sampler
            if tile_prompt:
                raise NotImplementedError("Tile prompt is not supported yet")
            else:
                self.sampler_func = sampler.orig_p_sample_ddim
                self.sampler.orig_p_sample_ddim = self.ddim_repeat

        # initialize the tile bboxes and weights
        self.w, self.h = w//8, h//8
        if tile_w > self.w:
            tile_w = self.w
        if tile_h > self.h:
            tile_h = self.h
        min_tile_size = min(tile_w, tile_h)
        if overlap >= min_tile_size:
            overlap = min_tile_size - 4
        if overlap < 0:
            overlap = 0
        self.tile_w = tile_w
        self.tile_h = tile_h
        bboxes, weights = self.split_views(tile_w, tile_h, overlap)
        self.batched_bboxes = []
        self.batched_conds = []
        self.batched_unconds = []
        self.num_batches = math.ceil(len(bboxes) / tile_batch_size)
        optimal_batch_size = math.ceil(len(bboxes) / self.num_batches)
        self.tile_batch_size = optimal_batch_size
        self.tile_prompt = tile_prompt
        for i in range(self.num_batches):
            start = i * tile_batch_size
            end = min((i + 1) * tile_batch_size, len(bboxes))
            self.batched_bboxes.append(bboxes[start:end])
            # TODO: deal with per tile prompt
            if tile_prompt:
                self.batched_conds.append(prompt_parser.get_multicond_learned_conditioning(
                    shared.sd_model, prompt[start:end], self.steps))
                self.batched_unconds.append(prompt_parser.get_learned_conditioning(
                    shared.sd_model, neg_prompt[start:end], self.steps))

        # Avoid the overhead of creating a new tensor for each batch
        # And avoid the overhead of weight summing
        self.weights = weights.unsqueeze(0).unsqueeze(0)
        self.x_buffer = None
        # For ddim sampler we need to cache the pred_x0
        self.x_buffer_pred = None
        self.pbar = None

        # For controlnet
        self.controlnet_script = controlnet_script
        self.control_tensor_batch = None
        self.control_params = None
        self.control_tensor_cpu = control_tensor_cpu

    @staticmethod
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
                image_cond_list.append(
                    image_cond[:, :, bbox[1]:bbox[3], bbox[0]:bbox[2]])
            image_cond_tile = torch.cat(image_cond_list, dim=0)
        else:
            image_cond_shape = image_cond.shape
            image_cond_tile = image_cond.repeat(
                (len(bboxes),) + (1,) * (len(image_cond_shape) - 1))
        return {"c_crossattn": [cond], "c_concat": [image_cond_tile]}

    def kdiff_repeat(self, x_in, sigma_in, cond):
        def func(x_tile, bboxes):
            sigma_in_tile = sigma_in.repeat(len(bboxes))
            new_cond = self.repeat_con_dict(cond, bboxes)
            x_tile_out = self.sampler_func(
                x_tile, sigma_in_tile, cond=new_cond)
            return x_tile_out
        return self.compute_x_tile(x_in, func)

    def ddim_repeat(self, x_in, cond_in, ts, unconditional_conditioning, *args, **kwargs):
        def func(x_tile, bboxes):
            if isinstance(cond_in, dict):
                ts_tile = ts.repeat(len(bboxes))
                cond_tile = self.repeat_con_dict(cond_in, bboxes)
                ucond_tile = self.repeat_con_dict(
                    unconditional_conditioning, bboxes)
            else:
                ts_tile = ts.repeat(len(bboxes))
                cond_shape = cond_in.shape
                cond_tile = cond_in.repeat(
                    (len(bboxes),) + (1,) * (len(cond_shape) - 1))
                ucond_shape = unconditional_conditioning.shape
                ucond_tile = unconditional_conditioning.repeat(
                    (len(bboxes),) + (1,) * (len(ucond_shape) - 1))
            x_tile_out, x_pred = self.sampler_func(
                x_tile, cond_tile, ts_tile, unconditional_conditioning=ucond_tile, *args, **kwargs)
            return x_tile_out, x_pred
        return self.compute_x_tile(x_in, func)
    
    def prepare_control_tensors(self):
        """
        Crop the control tensor into tiles and cache them
        """
        if self.control_tensor_batch is not None: return
        if self.controlnet_script is None or self.control_params is not None: return
        latest_network = self.controlnet_script.latest_network
        if latest_network is None or not hasattr(latest_network, 'control_params'): return
        self.control_params = latest_network.control_params
        tensors = [param.hint_cond for param in latest_network.control_params]
        if len(tensors) == 0: return
        self.control_tensor_batch = []
        for bboxes in self.batched_bboxes:
            single_batch_tensors = []
            for i in range(len(tensors)):
                control_tile_list = []
                control_tensor = tensors[i]
                for _, _, bbox in bboxes:
                    if len(control_tensor.shape) == 3:
                        control_tile = control_tensor[:, bbox[1] *
                                                    8:bbox[3]*8, bbox[0]*8:bbox[2]*8].unsqueeze(0)
                    else:
                        control_tile = control_tensor[:, :,
                                                    bbox[1]*8:bbox[3]*8, bbox[0]*8:bbox[2]*8]
                    control_tile_list.append(control_tile)
                if self.is_kdiff:
                    control_tile = torch.cat(
                        [t for t in control_tile_list for _ in range(2)], dim=0)
                else:
                    control_tile = torch.cat(control_tile_list*2, dim=0)
                if self.control_tensor_cpu:
                    control_tile = control_tile.cpu()
                single_batch_tensors.append(control_tile)
            self.control_tensor_batch.append(single_batch_tensors)

    def compute_x_tile(self, x_in, func):
        N, C, H, W = x_in.shape
        assert H == self.h and W == self.w
        
        # ControlNet support
        self.prepare_control_tensors()

        if self.x_buffer is None:
            self.x_buffer = torch.zeros_like(x_in, device=x_in.device)
        else:
            self.x_buffer.zero_()
        if not self.is_kdiff:
            if self.x_buffer_pred is None:
                self.x_buffer_pred = torch.zeros_like(x_in, device=x_in.device)
            else:
                self.x_buffer_pred.zero_()
        if self.pbar is None:
            self.pbar = tqdm(total=self.num_batches * (state.job_count *
                             state.sampling_steps), desc="MultiDiffusion Sampling: ")
        for batch_id, bboxes in enumerate(self.batched_bboxes):
            if state.interrupted:
                return x_in
            x_tile_list = []
            for _, _, bbox in bboxes:
                x_tile_list.append(
                    x_in[:, :, bbox[1]:bbox[3], bbox[0]:bbox[2]])
            x_tile = torch.cat(x_tile_list, dim=0)
            # controlnet tiling
            if self.control_tensor_batch is not None:
                single_batch_tensors = self.control_tensor_batch[batch_id]
                for i in range(len(single_batch_tensors)):
                    self.control_params[i].hint_cond = single_batch_tensors[i].to(x_in.device)
            # compute tiles
            if self.is_kdiff:
                x_tile_out = func(x_tile, bboxes)
                for i, (_, _, bbox) in enumerate(bboxes):
                    self.x_buffer[:, :, bbox[1]:bbox[3], bbox[0]:bbox[2]] += x_tile_out[i*N:(i+1)*N, :, :, :]
            else:
                x_tile_out, x_tile_pred = func(x_tile, bboxes)
                for i, (_, _, bbox) in enumerate(bboxes):
                    self.x_buffer[:, :, bbox[1]:bbox[3], bbox[0]:bbox[2]] += x_tile_out[i*N:(i+1)*N, :, :, :]
                    self.x_buffer_pred[:, :, bbox[1]:bbox[3], bbox[0]:bbox[2]] += x_tile_pred[i*N:(i+1)*N, :, :, :]
            # update progress bar
            if self.pbar.n >= self.pbar.total:
                self.pbar.close()
            else:
                self.pbar.update()
        x_out = torch.where(self.weights > 1, self.x_buffer / self.weights, self.x_buffer)
        if not self.is_kdiff:
            x_pred = torch.where(self.weights > 1, self.x_buffer_pred / self.weights, self.x_buffer_pred)
            return x_out, x_pred
        return x_out


class Script(scripts.Script):

    def title(self):
        return "MultiDiffusion"

    def show(self, is_img2img):
        return scripts.AlwaysVisible

    def ui(self, is_img2img):
        with gr.Accordion('MultiDiffusion', open=False):
            with gr.Row(variant='compact'):
                enabled = gr.Checkbox(label='Enable MultiDiffusion', value=False)
                override_image_size = gr.Checkbox(label='Overwrite image size', value=False, visible=(not is_img2img))
                keep_input_size = gr.Checkbox(label='Keep input image size', value=True, visible=(is_img2img))

                enable_bbox_control = gr.Checkbox(label='Draw bboxes', value=False, visible=is_img2img)
                btn_bbox_new = gr.Button(value='+', variant='tool', visible=False)
                btn_bbox_del = gr.Button(value='-', variant='tool', visible=False)

            with gr.Row(visible=False) as tab_size:
                image_width = gr.Slider(minimum=256, maximum=16384, step=16, label='Image width', value=1024, 
                                        elem_id=self.elem_id("image_width"))
                image_height = gr.Slider(minimum=256, maximum=16384, step=16, label='Image height', value=1024, 
                                         elem_id=self.elem_id("image_height"))
            if not is_img2img:
                override_image_size.change(fn=lambda x: gr_show(x), inputs=override_image_size, outputs=tab_size)

            with gr.Group():
                with gr.Row():
                    tile_width = gr.Slider(minimum=16, maximum=256, step=16, label='Latent tile width', value=64,
                                            elem_id=self.elem_id("latent_tile_width"))
                    tile_height = gr.Slider(minimum=16, maximum=256, step=16, label='Latent tile height', value=64,
                                            elem_id=self.elem_id("latent_tile_height"))

                with gr.Row():
                    overlap = gr.Slider(minimum=0, maximum=256, step=4, label='Latent tile overlap', value=32,
                                        elem_id=self.elem_id("latent_overlap"))
                    batch_size = gr.Slider(minimum=1, maximum=8, step=1, label='Latent tile batch size', value=1)

            with gr.Row(visible=is_img2img):
                upscaler_index = gr.Dropdown(label='Upscaler', choices=[x.name for x in shared.sd_upscalers], value="None", 
                                             elem_id=self.elem_id("upscaler_index"))
                scale_factor = gr.Slider(minimum=1.0, maximum=8.0, step=0.05, label='Scale Factor', value=2.0,
                                         elem_id=self.elem_id("scale_factor"))

            if is_img2img and 'bbox_control':
                with gr.Group(visible=False, variant='panel', elem_id='MD-bbox-control') as tab_bbox:
                    bbox_ip = gr.State(value=0, elem_id='MD-bbox-ip')  # counter of the activated bboxes
                    bbox_controls = gr.State([])  # control set for each bbox
                    bbox_control_groups = gr.State([])  # ui show/hide a group

                    gr.Image(label='Ref image (for conviently deciding bbox)', image_mode=None, elem_id='MD-bbox-ref')

                    for i in range(BBOX_MAX_NUM):
                        with gr.Group(visible=i==0) as tab_bbox_grp:
                            with gr.Row(variant='compact'):
                                x = gr.Slider(label=f'x{i}', value=lambda: 0.4, minimum=0.0, maximum=1.0, step=0.01, interactive=True, elem_id=f'MD-x-{i}')
                                y = gr.Slider(label=f'y{i}', value=lambda: 0.4, minimum=0.0, maximum=1.0, step=0.01, interactive=True, elem_id=f'MD-y-{i}')
                                w = gr.Slider(label=f'w{i}', value=lambda: 0.2, minimum=0.0, maximum=1.0, step=0.01, interactive=True, elem_id=f'MD-w-{i}')
                                h = gr.Slider(label=f'h{i}', value=lambda: 0.2, minimum=0.0, maximum=1.0, step=0.01, interactive=True, elem_id=f'MD-h-{i}')
                            with gr.Row(variant='compact'):
                                m = gr.Slider(label=f'weight{i}', value=lambda: 1.0, minimum=0.0, maximum=1.0, step=0.01, interactive=True, elem_id=f'MD-wt-{i}')
                                t = gr.Text(show_label=False, placeholder=f'prompt{i}', max_lines=1, elem_id=f'MD-p-{i}')
                        bbox_controls.append((x, y, w, h, m, t))
                        bbox_control_groups.append(tab_bbox_grp)

                    def bbox_new_click(bbox_ip):
                        if bbox_ip < BBOX_MAX_NUM - 1: bbox_ip += 1
                        return [ gr_show(i<=bbox_ip) for i in range(len(bbox_control_groups)) ]

                    def bbox_del_click(bbox_ip):
                        if bbox_ip > 0: bbox_ip -= 1
                        return [ gr_show(i<=bbox_ip) for i in range(len(bbox_control_groups)) ]

                    btn_bbox_new.click(fn=bbox_new_click, inputs=[bbox_ip], outputs=bbox_control_groups)
                    btn_bbox_del.click(fn=bbox_del_click, inputs=[bbox_ip], outputs=bbox_control_groups, _js='btn_bbox_del_click')

                enable_bbox_control.change(
                    fn=lambda x: [gr_show(x), gr_show(x), gr_show(x)], 
                    inputs=enable_bbox_control, 
                    outputs=[tab_bbox, btn_bbox_new, btn_bbox_del],
                    _js='enable_bbox_control_change',
                )

            control_tensor_cpu = gr.Checkbox(label='Move ControlNet images to CPU (if applicable)', value=False)

        return [
            enabled, 
            override_image_size, keep_input_size, image_width, image_height, 
            tile_width, tile_height, overlap, batch_size, 
            upscaler_index, scale_factor,
            control_tensor_cpu,
            enable_bbox_control,
        ]

    def process(self, p:StableDiffusionProcessing, 
            enabled:bool, 
            override_image_size:bool, keep_input_size:bool, image_width:int, image_height:int, 
            tile_width:int, tile_height:int, overlap:int, tile_batch_size:int, 
            upscaler_index:str, scale_factor:float,
            control_tensor_cpu:bool,
            enable_user_bbox:bool,
        ):

        if not enabled: return

        ''' upscale '''
        if hasattr(p, "init_images") and len(p.init_images) > 0:    # img2img
            upscaler_name = [x.name for x in shared.sd_upscalers].index(upscaler_index)

            init_img = p.init_images[0]
            init_img = images.flatten(init_img, opts.img2img_background_color)
            upscaler = shared.sd_upscalers[upscaler_name]
            if upscaler.name != "None":
                print(f"[MultiDiffusion] upscaling image with {upscaler.name}...")
                image = upscaler.scaler.upscale(init_img, scale_factor, upscaler.data_path)
                p.extra_generation_params["MultiDiffusion upscaler"] = upscaler.name
                p.extra_generation_params["MultiDiffusion scale factor"] = scale_factor
            else:
                image = init_img
            p.init_images[0] = image

            if keep_input_size:
                p.width = image.width
                p.height = image.height
            elif upscaler.name != "None":
                p.width *= scale_factor
                p.height *= scale_factor
        elif override_image_size:       # txt2img
            p.width = image_width
            p.height = image_height

        ''' sanitiy check '''
        if not MultiDiffusionDelegate.splitable(p.width, p.height, tile_width, tile_height, overlap):
            print("[MultiDiffusion] ignore due to image too small or tile size too large.")
            return
        p.extra_generation_params["MultiDiffusion tile width"] = tile_width
        p.extra_generation_params["MultiDiffusion tile height"] = tile_height
        p.extra_generation_params["MultiDiffusion overlap"] = overlap

        ''' ControlNet hackin '''
        # try to hook into controlnet tensors
        controlnet_script = None
        try:
            from scripts.cldm import ControlNet
            # fix controlnet multi-batch issue

            def align(self, hint, h, w):
                if (len(hint.shape) == 3):
                    hint = hint.unsqueeze(0)
                _, _, h1, w1 = hint.shape
                if h != h1 or w != w1:
                    hint = torch.nn.functional.interpolate(
                        hint, size=(h, w), mode="nearest")
                return hint
            ControlNet.align = align
            for script in p.scripts.scripts + p.scripts.alwayson_scripts:
                if hasattr(script, "latest_network") and script.title().lower() == "controlnet":
                    controlnet_script = script
                    print("[MultiDiffusion] ControlNet found, MultiDiffusion-ControlNet support is enabled.")
                    break
        except ImportError:
            pass

        ''' sampler hijack '''
        # hack the create_sampler function to get the created sampler
        old_create_sampler = sd_samplers.create_sampler
        def create_sampler(name, model):
            # create the sampler with the original function
            sampler = old_create_sampler(name, model)
            # unhook the create_sampler function
            sd_samplers.create_sampler = old_create_sampler
            delegate = MultiDiffusionDelegate(
                sampler, 
                p.sampler_name, p.steps, p.width, p.height, 
                tile_width, tile_height, overlap, tile_batch_size, 
                tile_prompt=False, 
                controlnet_script=controlnet_script, 
                control_tensor_cpu=control_tensor_cpu
            )
            print(f"[MultiDiffusion] hooked into {p.sampler_name} sampler. " + 
                  f"Tile size: {tile_width}x{tile_height}, " + 
                  f"Tile batches: {len(delegate.batched_bboxes)}, " +
                  f"Batch size:", tile_batch_size)
            return sampler
        sd_samplers.create_sampler = create_sampler