'''
# ------------------------------------------------------------------------
#
#   Tiled Diffusion for Automatic1111 WebUI
#
#   Introducing revolutionary large image drawing methods:
#       MultiDiffusion and Mixture of Diffusers!
#
#   Techniques is not originally proposed by me, please refer to
#
#   MultiDiffusion: https://multidiffusion.github.io
#   Mixture of Diffusers: https://github.com/albarji/mixture-of-diffusers
#
#   The script contains a few optimizations including:
#       - symmetric tiling bboxes
#       - cached tiling weights
#       - batched denoising
#       - advanced prompt control for each tile
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
# ------------------------------------------------------------------------
'''

import torch
import numpy as np
from enum import Enum
import gradio as gr

from modules import sd_samplers, images, shared, scripts
from modules.shared import opts
from modules.ui import gr_show
from modules.processing import StableDiffusionProcessing

from methods import MultiDiffusion, MixtureOfDiffusers, splitable, BlendMode


BBOX_MAX_NUM = min(shared.cmd_opts.md_max_regions if hasattr(
    shared.cmd_opts, "md_max_regions") else 8, 16)

class Method(Enum):
    MULTI_DIFF = 'MultiDiffusion'
    MIX_DIFF   = 'Mixture of Diffusers'


class Script(scripts.Script):

    def __init__(self):
        self.controlnet_script = None
        self.delegate = None

    def title(self):
        return "Tiled Diffusion"

    def show(self, is_img2img):
        return scripts.AlwaysVisible

    def ui(self, is_img2img):
        tab = 't2i' if not is_img2img else 'i2i'
        is_t2i = 'true' if not is_img2img else 'false'

        with gr.Accordion('Tiled Diffusion', open=False):
            with gr.Row(variant='compact'):
                enabled = gr.Checkbox(label='Enable', value=False)
                method = gr.Dropdown(label='Method', choices=[e.value for e in Method], value=Method.MULTI_DIFF.value)

            with gr.Row(variant='compact', visible=False) as tab_size:
                image_width = gr.Slider(minimum=256, maximum=16384, step=16, label='Image width', value=1024,
                                        elem_id=f'MD-overwrite-width-{tab}')
                image_height = gr.Slider(minimum=256, maximum=16384, step=16, label='Image height', value=1024,
                                         elem_id=f'MD-overwrite-height-{tab}')

            with gr.Group():
                with gr.Row(variant='compact'):
                    tile_width = gr.Slider(minimum=16, maximum=256, step=16, label='Latent tile width', value=96,
                                           elem_id=self.elem_id("latent_tile_width"))
                    tile_height = gr.Slider(minimum=16, maximum=256, step=16, label='Latent tile height', value=96,
                                            elem_id=self.elem_id("latent_tile_height"))

                with gr.Row(variant='compact'):
                    overlap = gr.Slider(minimum=0, maximum=256, step=4, label='Latent tile overlap', value=48,
                                        elem_id=self.elem_id("latent_overlap"))
                    batch_size = gr.Slider(
                        minimum=1, maximum=8, step=1, label='Latent tile batch size', value=1)

            with gr.Row(variant='compact', visible=is_img2img):
                upscaler_index = gr.Dropdown(label='Upscaler', choices=[x.name for x in shared.sd_upscalers], value="None",
                                             elem_id='MD-upscaler-index')
                scale_factor = gr.Slider(minimum=1.0, maximum=8.0, step=0.05, label='Scale Factor', value=2.0,
                                         elem_id='MD-upscaler-factor')

            with gr.Row(variant='compact'):
                overwrite_image_size = gr.Checkbox(
                    label='Overwrite image size', value=False, visible=(not is_img2img))
                keep_input_size = gr.Checkbox(
                    label='Keep input image size', value=True, visible=(is_img2img))
                if not is_img2img:
                    overwrite_image_size.change(fn=lambda x: gr_show(
                        x), inputs=overwrite_image_size, outputs=tab_size)
                control_tensor_cpu = gr.Checkbox(
                    label='Move ControlNet images to CPU (if applicable)', value=False)

            # The control includes txt2img and img2img, we use t2i and i2i to distinguish them
            with gr.Group(variant='panel', elem_id=f'MD-bbox-control-{tab}'):
                with gr.Accordion('Region Prompt Control', open=False):

                    with gr.Row(variant='compact'):
                        enable_bbox_control = gr.Checkbox(
                            label='Enable', value=False)
                        draw_background = gr.Checkbox(
                            label='Draw tiles as background (SLOW but save VRAM) ', value=False)
                    with gr.Row(variant='compact'):
                        create_button = gr.Button(
                            value="Create txt2img canvas" if not is_img2img else "From img2img")

                    bbox_controls = []  # control set for each bbox
                    with gr.Row(variant='compact'):
                        ref_image = gr.Image(label='Ref image (for conviently locate regions)', image_mode=None, 
                                             elem_id=f'MD-bbox-ref-{tab}', interactive=True)
                        if not is_img2img:
                            # gradio has a serious bug: it cannot accept multiple inputs when you use both js and fn.
                            # to workaround this, we concat the inputs into a single string and parse it in js
                            def create_t2i_ref(string):
                                w, h = [int(x) for x in string.split('x')]
                                w = max(w, 8)
                                h = max(h, 8)
                                return np.zeros(shape=(h, w, 3), dtype=np.uint8) + 255
                            create_button.click(
                                fn=create_t2i_ref, 
                                inputs=overwrite_image_size, 
                                outputs=ref_image, 
                                _js='onCreateT2IRefClick')
                        else:
                            create_button.click(fn=None, outputs=ref_image, _js='onCreateI2IRefClick')

                    for i in range(BBOX_MAX_NUM):
                        with gr.Accordion(f'Region {i+1}', open=False):
                            with gr.Row(variant='compact'):
                                e = gr.Checkbox(label='Enable', value=False, elem_id=f'MD-enable-{tab}-{i}')
                                e.change(fn=None, inputs=e, outputs=e, _js=f'e => onBoxEnableClick({is_t2i}, {i}, e)')
                                
                                blend_mode = gr.Dropdown(label='Type', choices=[e.value for e in BlendMode], value=BlendMode.BACKGROUND.value, 
                                                         elem_id=f'MD-blend-{tab}-{i}', visible=True)
                                
                                feather_ratio = gr.Slider(label='Feather', value=0.2, minimum=0, maximum=1, step=0.05, 
                                            interactive=True, elem_id=f'MD-feather-{tab}-{i}', visible=blend_mode.value == BlendMode.FOREGROUND.value)
                                
                                def on_blend_mode_change(x): 
                                    return gr_show(x==BlendMode.FOREGROUND.value)
                                blend_mode.change(fn=on_blend_mode_change, inputs=blend_mode, outputs=[feather_ratio])
                                

                            with gr.Row(variant='compact'):
                                x = gr.Slider(label='x', value=0.4, minimum=0.0, maximum=1.0, step=0.01, 
                                              interactive=True, elem_id=f'MD-{tab}-{i}-x')
                                y = gr.Slider(label='y', value=0.4, minimum=0.0, maximum=1.0, step=0.01, 
                                              interactive=True, elem_id=f'MD-{tab}-{i}-y')
                                w = gr.Slider(label='w', value=0.2, minimum=0.0, maximum=1.0, step=0.01, 
                                              interactive=True, elem_id=f'MD-{tab}-{i}-w')
                                h = gr.Slider(label='h', value=0.2, minimum=0.0, maximum=1.0, step=0.01, 
                                              interactive=True, elem_id=f'MD-{tab}-{i}-h')

                                x.change(fn=None, inputs=x, outputs=x, _js=f'(v) => onBoxChange({is_t2i}, {i}, "x", v)')
                                y.change(fn=None, inputs=y, outputs=y, _js=f'(v) => onBoxChange({is_t2i}, {i}, "y", v)')
                                w.change(fn=None, inputs=w, outputs=w, _js=f'(v) => onBoxChange({is_t2i}, {i}, "w", v)')
                                h.change(fn=None, inputs=h, outputs=h, _js=f'(v) => onBoxChange({is_t2i}, {i}, "h", v)')

                            prompt   = gr.Text(show_label=False, placeholder=f'Prompt, will be appended to your {tab} prompt', 
                                          max_lines=2, elem_id=f'MD-p-{tab}-{i}')
                            neg_prompt = gr.Text(show_label=False, placeholder=f'Negative Prompt, will be appended to your {tab} negative prompt',         
                                          max_lines=1, elem_id=f'MD-n-{tab}-{i}')

                        bbox_controls.append([e, x, y ,w, h, prompt, neg_prompt, blend_mode, feather_ratio])

        controls = [
            enabled, method,
            overwrite_image_size, keep_input_size, image_width, image_height,
            tile_width, tile_height, overlap, batch_size,
            upscaler_index, scale_factor,
            control_tensor_cpu,
            enable_bbox_control,
            draw_background
        ]
        for i in range(BBOX_MAX_NUM): controls.extend(bbox_controls[i])
        return controls
    
    def custom_create_sampler(self, name, model, method, p, tile_width, tile_height, overlap, tile_batch_size, control_tensor_cpu, enable_bbox_control, draw_background, bbox_control_states, prompts, n):
        # create the sampler with the original function
        method: Method = Method(method) 
        sampler = sd_samplers.md_org_create_sampler(name, model)
        if self.delegate is None:
            if  method == Method.MULTI_DIFF:
                self.delegate = MultiDiffusion(
                    sampler, p,
                    tile_width, tile_height, overlap, tile_batch_size,
                    controlnet_script=self.controlnet_script,
                    control_tensor_cpu=control_tensor_cpu
                )
            elif method == Method.MIX_DIFF:
                self.delegate = MixtureOfDiffusers(
                    sampler, p,
                    tile_width, tile_height, overlap, tile_batch_size,
                    controlnet_script=self.controlnet_script,
                    control_tensor_cpu=control_tensor_cpu
                )
            else:
                raise NotImplementedError(f"Method {method} not implemented.")
            if enable_bbox_control:
                self.delegate.init_custom_bbox(draw_background, bbox_control_states)
    
        self.delegate.hook(sampler)

        if len(self.delegate.custom_bboxes) > 0:
            neg_prompts = p.all_negative_prompts[n * p.batch_size:(n + 1) * p.batch_size]
            self.delegate.refresh_prompt(prompts, neg_prompts)

        print(f"{method.value} hooked into {p.sampler_name} sampler. " +
                f"Tile size: {tile_width}x{tile_height}, " +
                f"Tile batches: {len(self.delegate.batched_bboxes)}, " +
                f"Batch size:", tile_batch_size)
        
        return sampler

    def process(self, p: StableDiffusionProcessing,
            enabled: bool, method: str,
            overwrite_image_size: bool, keep_input_size: bool, image_width: int, image_height: int,
            tile_width: int, tile_height: int, overlap: int, tile_batch_size: int,
            upscaler_index: str, scale_factor: float,
            control_tensor_cpu: bool, enable_bbox_control: bool, draw_background: bool,
            *bbox_control_states
        ):

        if hasattr(sd_samplers, "md_org_create_sampler"):
            sd_samplers.create_sampler = sd_samplers.md_org_create_sampler
            del sd_samplers.md_org_create_sampler

        MixtureOfDiffusers.unhook()
        self.delegate = None

        if not enabled: return

        ''' upscale '''
        if hasattr(p, "init_images") and len(p.init_images) > 0:    # img2img
            upscaler_name = [x.name for x in shared.sd_upscalers].index(upscaler_index)

            init_img = p.init_images[0]
            init_img = images.flatten(init_img, opts.img2img_background_color)
            upscaler = shared.sd_upscalers[upscaler_name]
            if upscaler.name != "None":
                print(f"[Tiled Diffusion] upscaling image with {upscaler.name}...")
                image = upscaler.scaler.upscale(init_img, scale_factor, upscaler.data_path)
                p.extra_generation_params["Tiled Diffusion upscaler"] = upscaler.name
                p.extra_generation_params["Tiled Diffusion scale factor"] = scale_factor
            else:
                image = init_img
            p.init_images[0] = image

            if keep_input_size:
                p.width = image.width
                p.height = image.height
            elif upscaler.name != "None":
                if not hasattr(p, "md_original_width"):
                    p.md_original_width = p.width
                    p.md_original_height = p.height
                p.width = scale_factor * p.md_original_width
                p.height = scale_factor * p.md_original_height
        elif overwrite_image_size:       # txt2img
            p.width = image_width
            p.height = image_height

        ''' sanitiy check '''
        if not enable_bbox_control and not splitable(p.width, p.height, tile_width, tile_height, overlap):
            print("[Tiled Diffusion] ignore due to image too small or tile size too large.")
            return

        p.extra_generation_params["Tiled Diffusion method"] = method
        p.extra_generation_params["Tiled Diffusion tile width"] = tile_width
        p.extra_generation_params["Tiled Diffusion tile height"] = tile_height
        p.extra_generation_params["Tiled Diffusion overlap"] = overlap
        p.extra_generation_params["Tiled Diffusion batch size"] = tile_batch_size

        ''' ControlNet hackin '''
        # try to hook into controlnet tensors
        if self.controlnet_script is None:
            try:
                from scripts.cldm import ControlNet
                # fix controlnet multi-batch issue

                def align(self, hint, h, w):
                    if (len(hint.shape) == 3):
                        hint = hint.unsqueeze(0)
                    _, _, h1, w1 = hint.shape
                    if h != h1 or w != w1:
                        hint = torch.nn.functional.interpolate(hint, size=(h, w), mode="nearest")
                    return hint
                
                ControlNet.align = align

                for script in p.scripts.scripts + p.scripts.alwayson_scripts:
                    if hasattr(script, "latest_network") and script.title().lower() == "controlnet":
                        self.controlnet_script = script
                        print("[Tiled Diffusion] ControlNet found, support is enabled.")
                        break

            except ImportError:
                pass

        
        create_sampler = lambda name, model : self.custom_create_sampler(
            name, model, method, p, tile_width, tile_height, overlap, tile_batch_size, control_tensor_cpu, enable_bbox_control, draw_background, bbox_control_states, p.all_prompts[:p.batch_size], 0)

        if not hasattr(sd_samplers, "md_org_create_sampler"):
            setattr(sd_samplers, "md_org_create_sampler", sd_samplers.create_sampler)
        sd_samplers.create_sampler = create_sampler


        
    def process_batch(self, p: StableDiffusionProcessing,
            enabled: bool, method: str,
            overwrite_image_size: bool, keep_input_size: bool, image_width: int, image_height: int,
            tile_width: int, tile_height: int, overlap: int, tile_batch_size: int,
            upscaler_index: str, scale_factor: float,
            control_tensor_cpu: bool, enable_bbox_control: bool, draw_background: float, 
            *bbox_control_states, batch_number, prompts, seeds, subseeds):
        '''
        compatible with the webui batch processing
        '''
        if not enabled: return
        n = batch_number

        ''' sampler hijack '''
        
        # hack the create_sampler function to get the created sampler
        create_sampler = lambda name, model : self.custom_create_sampler(
            name, model, method, p, tile_width, tile_height, overlap, tile_batch_size, control_tensor_cpu, enable_bbox_control, draw_background, bbox_control_states, prompts, n)

        if not hasattr(sd_samplers, "md_org_create_sampler"):
            setattr(sd_samplers, "md_org_create_sampler", sd_samplers.create_sampler)
        sd_samplers.create_sampler = create_sampler

    
    def postprocess(self, p, processed, *args):
        self.delegate = None
        if hasattr(sd_samplers, "md_org_create_sampler"):
            sd_samplers.create_sampler = sd_samplers.md_org_create_sampler
            del sd_samplers.md_org_create_sampler
        MixtureOfDiffusers.unhook()

