import os
import json
import torch
import torch.nn.functional as F
import numpy as np
import gradio as gr

from modules import sd_samplers, images, shared, devices, processing, scripts, sd_samplers_common, rng
from modules.shared import opts
from modules.processing import opt_f, get_fixed_seed
from modules.ui import gr_show

from tile_methods.abstractdiffusion import AbstractDiffusion
from tile_methods.demofusion import DemoFusion
from tile_utils.utils import *
from modules.sd_samplers_common import InterruptedException
# import k_diffusion.sampling


CFG_PATH = os.path.join(scripts.basedir(), 'region_configs')
BBOX_MAX_NUM = min(getattr(shared.cmd_opts, 'md_max_regions', 8), 16)


def create_infotext_hijack(p, all_prompts, all_seeds, all_subseeds, comments=None, iteration=0, position_in_batch=0, use_main_prompt=False, index=-1, all_negative_prompts=None):
    idx = index
    if index == -1:
        idx = None
    text = processing.create_infotext_ori(p, all_prompts, all_seeds, all_subseeds, comments, iteration, position_in_batch, use_main_prompt, idx, all_negative_prompts)
    start_index = text.find("Size")
    if start_index != -1:
        r_text = f"Size:{p.width_list[index]}x{p.height_list[index]}"
        end_index = text.find(",", start_index) 
        if end_index != -1:
            replaced_string = text[:start_index] + r_text + text[end_index:]
            return replaced_string
    return text 

class Script(scripts.Script):
    def __init__(self):
        self.controlnet_script: ModuleType = None
        self.stablesr_script: ModuleType = None
        self.delegate: AbstractDiffusion = None
        self.noise_inverse_cache: NoiseInverseCache = None

    def title(self):
        return 'demofusion'

    def show(self, is_img2img):
        return scripts.AlwaysVisible

    def ui(self, is_img2img):
        ext_id = 'demofusion'
        tab    = f'{ext_id}-t2i'  if not is_img2img else f'{ext_id}-i2i'
        is_t2i = 'true' if not is_img2img else 'false'
        uid = lambda name: f'MD-{tab}-{name}'

        with gr.Accordion('DemoFusion', open=False, elem_id=f'MD-{tab}'):
            with gr.Row(variant='compact') as tab_enable:
                enabled = gr.Checkbox(label='Enable DemoFusion(Dont open with tilediffusion)', value=False,  elem_id=uid('enabled'))
                random_jitter = gr.Checkbox(label='Random Jitter', value = True, elem_id=uid('random-jitter'))
                keep_input_size = gr.Checkbox(label='Keep input-image size', value=False,visible=is_img2img, elem_id=uid('keep-input-size'))
                mixture_mode = gr.Checkbox(label='Mixture mode', value=False,elem_id=uid('mixture-mode'))

                gaussian_filter = gr.Checkbox(label='Gaussian Filter', value=True, visible=False, elem_id=uid('gaussian'))


            with gr.Row(variant='compact') as tab_param:
                method = gr.Dropdown(label='Method', choices=[Method_2.DEMO_FU.value], value=Method_2.DEMO_FU.value, visible= False, elem_id=uid('method'))
                control_tensor_cpu = gr.Checkbox(label='Move ControlNet tensor to CPU (if applicable)', value=False, elem_id=uid('control-tensor-cpu'))
                reset_status = gr.Button(value='Free GPU', variant='tool')
                reset_status.click(fn=self.reset_and_gc, show_progress=False)

            with gr.Group() as tab_tile:
                with gr.Row(variant='compact'):
                    window_size = gr.Slider(minimum=16, maximum=256, step=16, label='Latent window size', value=128, elem_id=uid('latent-window-size'))

                with gr.Row(variant='compact'):
                    overlap = gr.Slider(minimum=0, maximum=256, step=4, label='Latent window overlap', value=64, elem_id=uid('latent-tile-overlap'))
                    batch_size = gr.Slider(minimum=1, maximum=8, step=1, label='Latent window batch size', value=4, elem_id=uid('latent-tile-batch-size'))
                    batch_size_g = gr.Slider(minimum=1, maximum=8, step=1, label='Global window batch size', value=4, elem_id=uid('Global-tile-batch-size'))
                with gr.Row(variant='compact', visible=True) as tab_c:
                    c1  = gr.Slider(minimum=0, maximum=5, step=0.01, label='Cosine Scale 1',  value=3, elem_id=f'C1-{tab}')
                    c2 = gr.Slider(minimum=0, maximum=5, step=0.01, label='Cosine Scale 2', value=1, elem_id=f'C2-{tab}')
                    c3 = gr.Slider(minimum=0, maximum=5, step=0.01, label='Cosine Scale 3', value=1, elem_id=f'C3-{tab}')
                    sigma = gr.Slider(minimum=0, maximum=2, step=0.01, label='Sigma', value=0.6, elem_id=f'Sigma-{tab}')
            with gr.Group() as tab_denoise:
                strength  = gr.Slider(minimum=0, maximum=1, step=0.01, value = 0.85,label='Denoising Strength for Substage',visible=not is_img2img, elem_id=f'strength-{tab}')
            with gr.Row(variant='compact') as tab_upscale:
                scale_factor = gr.Slider(minimum=1.0, maximum=8.0, step=1, label='Scale Factor', value=2.0, elem_id=uid('upscaler-factor'))


            with gr.Accordion('Noise Inversion', open=True, visible=is_img2img) as tab_noise_inv:
                with gr.Row(variant='compact'):
                    noise_inverse = gr.Checkbox(label='Enable Noise Inversion', value=False, elem_id=uid('noise-inverse'))
                    noise_inverse_steps = gr.Slider(minimum=1, maximum=200, step=1, label='Inversion steps', value=10, elem_id=uid('noise-inverse-steps'))
                    gr.HTML('<p>Please test on small images before actual upscale. Default params require denoise <= 0.6</p>')
                with gr.Row(variant='compact'):
                    noise_inverse_retouch = gr.Slider(minimum=1, maximum=100, step=0.1, label='Retouch', value=1, elem_id=uid('noise-inverse-retouch'))
                    noise_inverse_renoise_strength = gr.Slider(minimum=0, maximum=2, step=0.01, label='Renoise strength', value=1, elem_id=uid('noise-inverse-renoise-strength'))
                    noise_inverse_renoise_kernel = gr.Slider(minimum=2, maximum=512, step=1, label='Renoise kernel size', value=64, elem_id=uid('noise-inverse-renoise-kernel'))

            # The control includes txt2img and img2img, we use t2i and i2i to distinguish them

        return [
            enabled, method,
            keep_input_size,
            window_size, overlap, batch_size,
            scale_factor,
            noise_inverse, noise_inverse_steps, noise_inverse_retouch, noise_inverse_renoise_strength, noise_inverse_renoise_kernel,
            control_tensor_cpu,
            random_jitter,
            c1,c2,c3,gaussian_filter,strength,sigma,batch_size_g,mixture_mode
        ]


    def process(self, p: Processing,
            enabled: bool, method: str,
             keep_input_size: bool,
            window_size:int, overlap: int, tile_batch_size: int,
            scale_factor: float,
            noise_inverse: bool, noise_inverse_steps: int, noise_inverse_retouch: float, noise_inverse_renoise_strength: float, noise_inverse_renoise_kernel: int,
            control_tensor_cpu: bool,
            random_jitter:bool,
            c1,c2,c3,gaussian_filter,strength,sigma,batch_size_g,mixture_mode
        ):

        # unhijack & unhook, in case it broke at last time
        self.reset()
        p.mixture = mixture_mode
        if not mixture_mode:
            sigma = sigma/2
        if not enabled: return

        ''' upscale '''
        # store canvas size settings
        if hasattr(p, "init_images"):
            p.init_images_original_md = [img.copy() for img in p.init_images]

        p.width_original_md  = p.width
        p.height_original_md = p.height
        p.current_scale_num = 1
        p.gaussian_filter = gaussian_filter
        p.scale_factor = int(scale_factor)

        is_img2img = hasattr(p, "init_images") and len(p.init_images) > 0
        if is_img2img:
            init_img = p.init_images[0]
            init_img = images.flatten(init_img, opts.img2img_background_color)
            image = init_img
            if keep_input_size:
                p.width  = image.width
                p.height = image.height
                p.width_original_md  = p.width
                p.height_original_md = p.height
            else: #XXX:To adapt to noise inversion, we do not multiply the scale factor here
                p.width  = p.width_original_md
                p.height = p.height_original_md
        else:  # txt2img
            p.width  = p.width_original_md
            p.height = p.height_original_md

        if 'png info':
            info = {}
            p.extra_generation_params["Tiled Diffusion"] = info

            info['Method']           = method
            info['Window Size']  = window_size
            info['Tile Overlap']     = overlap
            info['Tile batch size']  = tile_batch_size
            info["Global batch size"] = batch_size_g

            if is_img2img:
                info['Upscale factor'] = scale_factor
                if keep_input_size:
                    info['Keep input size'] = keep_input_size
                if noise_inverse:
                    info['NoiseInv']                  = noise_inverse
                    info['NoiseInv Steps']            = noise_inverse_steps
                    info['NoiseInv Retouch']          = noise_inverse_retouch
                    info['NoiseInv Renoise strength'] = noise_inverse_renoise_strength
                    info['NoiseInv Kernel size']      = noise_inverse_renoise_kernel

        ''' ControlNet hackin '''
        try:
            from scripts.cldm import ControlNet

            for script in p.scripts.scripts + p.scripts.alwayson_scripts:
                if hasattr(script, "latest_network") and script.title().lower() == "controlnet":
                    self.controlnet_script = script
                    print("[Demo Fusion] ControlNet found, support is enabled.")
                    break
        except ImportError:
            pass

        ''' StableSR hackin '''
        for script in p.scripts.scripts:
            if hasattr(script, "stablesr_model") and script.title().lower() == "stablesr":
                if script.stablesr_model is not None:
                    self.stablesr_script = script
                    print("[Demo Fusion] StableSR found, support is enabled.")
                    break

        ''' hijack inner APIs, see unhijack in reset() '''
        Script.create_sampler_original_md = sd_samplers.create_sampler

        sd_samplers.create_sampler = lambda name, model: self.create_sampler_hijack(
            name, model, p, Method_2(method), control_tensor_cpu,window_size, noise_inverse, noise_inverse_steps, noise_inverse_retouch,
            noise_inverse_renoise_strength, noise_inverse_renoise_kernel, overlap, tile_batch_size,random_jitter,batch_size_g
        )


        p.sample = lambda conditioning, unconditional_conditioning,seeds, subseeds, subseed_strength, prompts: self.sample_hijack(
        conditioning, unconditional_conditioning, seeds, subseeds, subseed_strength, prompts,p, is_img2img,
        window_size, overlap, tile_batch_size,random_jitter,c1,c2,c3,strength,sigma,batch_size_g)

        processing.create_infotext_ori = processing.create_infotext

        p.width_list = [p.height]
        p.height_list = [p.height]

        processing.create_infotext = create_infotext_hijack
        ## end


    def postprocess_batch(self, p: Processing, enabled, *args, **kwargs):
        if not enabled: return

        if self.delegate is not None: self.delegate.reset_controlnet_tensors()

    def postprocess_batch_list(self, p, pp, *args, **kwargs):
        for idx,image in enumerate(pp.images):
            idx_b = idx//p.batch_size
            pp.images[idx] = image[:,:image.shape[1]//(p.scale_factor)*(idx_b+1),:image.shape[2]//(p.scale_factor)*(idx_b+1)]
        p.seeds =  [item for _ in range(p.scale_factor) for item in p.seeds]
        p.prompts = [item for _ in range(p.scale_factor) for item in p.prompts]
        p.all_negative_prompts = [item for _ in range(p.scale_factor) for item in p.all_negative_prompts]
        p.negative_prompts = [item  for _ in range(p.scale_factor) for item in p.negative_prompts]
        if p.color_corrections != None:
           p.color_corrections = [item for _ in range(p.scale_factor) for item in p.color_corrections]
        p.width_list = [item*(idx+1) for idx in range(p.scale_factor) for item in [p.width for _ in range(p.batch_size)]]
        p.height_list = [item*(idx+1) for idx in range(p.scale_factor) for item in [p.height for _ in range(p.batch_size)]]
        return

    def postprocess(self, p: Processing, processed, enabled, *args):
        if not enabled: return
        # unhijack & unhook
        self.reset()

        # restore canvas size settings
        if hasattr(p, 'init_images') and hasattr(p, 'init_images_original_md'):
            p.init_images.clear()       # NOTE: do NOT change the list object, compatible with shallow copy of XYZ-plot
            p.init_images.extend(p.init_images_original_md)
            del p.init_images_original_md
        p.width  = p.width_original_md  ; del p.width_original_md
        p.height = p.height_original_md ; del p.height_original_md

        # clean up noise inverse latent for folder-based processing
        if hasattr(p, 'noise_inverse_latent'):
            del p.noise_inverse_latent

    ''' ↓↓↓ inner API hijack ↓↓↓ '''
    @torch.no_grad()
    def sample_hijack(self, conditioning, unconditional_conditioning,seeds, subseeds, subseed_strength, prompts,p,image_ori,window_size, overlap, tile_batch_size,random_jitter,c1,c2,c3,strength,sigma,batch_size_g):
        ################################################## Phase Initialization ######################################################

        if not image_ori:
            p.current_step = 0
            p.denoising_strength = strength
            # p.sampler = sd_samplers.create_sampler(p.sampler_name, p.sd_model) #NOTE:Wrong but very useful. If corrected, please replace with the content with the following lines
            # latents = p.rng.next()

            p.sampler = Script.create_sampler_original_md(p.sampler_name, p.sd_model) #scale
            x = p.rng.next()
            print("### Phase 1 Denoising ###")
            latents = p.sampler.sample(p, x, conditioning, unconditional_conditioning, image_conditioning=p.txt2img_image_conditioning(x))
            latents_ = F.pad(latents, (0, latents.shape[3]*(p.scale_factor-1), 0, latents.shape[2]*(p.scale_factor-1)))
            res = latents_
            del x
            p.sampler = sd_samplers.create_sampler(p.sampler_name, p.sd_model)
            starting_scale = 2
        else: # img2img
            print("### Encoding Real Image ###")
            latents = p.init_latent
            starting_scale = 1


        anchor_mean = latents.mean()
        anchor_std = latents.std()

        devices.torch_gc()

        ####################################################### Phase Upscaling #####################################################
        p.cosine_scale_1 = c1
        p.cosine_scale_2 = c2
        p.cosine_scale_3 = c3
        self.delegate.sig = sigma
        p.latents = latents
        for current_scale_num in range(starting_scale, p.scale_factor+1):
            p.current_scale_num = current_scale_num
            print("### Phase {} Denoising ###".format(current_scale_num))
            p.current_height = p.height_original_md * current_scale_num
            p.current_width = p.width_original_md * current_scale_num


            p.latents = F.interpolate(p.latents, size=(int(p.current_height / opt_f), int(p.current_width / opt_f)), mode='bicubic')
            p.rng = rng.ImageRNG(p.latents.shape[1:], p.seeds, subseeds=p.subseeds, subseed_strength=p.subseed_strength, seed_resize_from_h=p.seed_resize_from_h, seed_resize_from_w=p.seed_resize_from_w)


            self.delegate.w = int(p.current_width  / opt_f)
            self.delegate.h = int(p.current_height / opt_f)
            self.delegate.get_views(overlap, tile_batch_size,batch_size_g)

            info = ', '.join([
                # f"{method.value} hooked into {name!r} sampler",
                f"Tile size: {self.delegate.window_size}",
                f"Tile count: {self.delegate.num_tiles}",
                f"Batch size: {self.delegate.tile_bs}",
                f"Tile batches: {len(self.delegate.batched_bboxes)}",
                f"Global batch size: {self.delegate.global_tile_bs}",
                f"Global batches: {len(self.delegate.global_batched_bboxes)}",
            ])

            print(info)

            noise = p.rng.next()
            if hasattr(p,'initial_noise_multiplier'):
                if p.initial_noise_multiplier != 1.0:
                    p.extra_generation_params["Noise multiplier"] = p.initial_noise_multiplier
                    noise *= p.initial_noise_multiplier
            else:
                p.image_conditioning = p.txt2img_image_conditioning(noise)

            p.noise = noise
            p.x = p.latents.clone()
            p.current_step=0

            p.latents = p.sampler.sample_img2img(p,p.latents, noise , conditioning, unconditional_conditioning, image_conditioning=p.image_conditioning)
            if self.flag_noise_inverse:
                self.delegate.sampler_raw.sample_img2img = self.delegate.sample_img2img_original
                self.flag_noise_inverse = False

            p.latents = (p.latents - p.latents.mean()) / p.latents.std() * anchor_std + anchor_mean
            latents_ = F.pad(p.latents, (0, p.latents.shape[3]//current_scale_num*(p.scale_factor-current_scale_num), 0, p.latents.shape[2]//current_scale_num*(p.scale_factor-current_scale_num)))
            if current_scale_num==1:
                res = latents_
            else:
                res = torch.concatenate((res,latents_),axis=0)

        #########################################################################################################################################

        return res

    @staticmethod
    def callback_hijack(self_sampler,d,p):
        p.current_step = d['i']

        if self_sampler.stop_at is not None and p.current_step > self_sampler.stop_at:
            raise InterruptedException

        state.sampling_step = p.current_step
        shared.total_tqdm.update()
        p.current_step += 1


    def create_sampler_hijack(
            self, name: str, model: LatentDiffusion, p: Processing, method: Method_2, control_tensor_cpu:bool,window_size, noise_inverse: bool, noise_inverse_steps: int, noise_inverse_retouch:float,
            noise_inverse_renoise_strength: float, noise_inverse_renoise_kernel: int, overlap:int, tile_batch_size:int, random_jitter:bool,batch_size_g:int
        ):
        if self.delegate is not None:
            # samplers are stateless, we reuse it if possible
            if self.delegate.sampler_name == name:
                # before we reuse the sampler, we refresh the control tensor
                # so that we are compatible with ControlNet batch processing
                if self.controlnet_script:
                    self.delegate.prepare_controlnet_tensors(refresh=True)
                return self.delegate.sampler_raw
            else:
                self.reset()
        sd_samplers_common.Sampler.callback_ori = sd_samplers_common.Sampler.callback_state
        sd_samplers_common.Sampler.callback_state = lambda self_sampler,d:Script.callback_hijack(self_sampler,d,p)

        self.flag_noise_inverse = hasattr(p, "init_images") and len(p.init_images) > 0 and noise_inverse
        flag_noise_inverse = self.flag_noise_inverse
        if flag_noise_inverse:
            print('warn: noise inversion only supports the "Euler" sampler, switch to it sliently...')
            name = 'Euler'
            p.sampler_name = 'Euler'
        if name is None: print('>> name is empty')
        if model is None: print('>> model is empty')
        sampler = Script.create_sampler_original_md(name, model)
        if method ==Method_2.DEMO_FU: delegate_cls = DemoFusion
        else: raise NotImplementedError(f"Method {method} not implemented.")

        delegate = delegate_cls(p, sampler)
        delegate.window_size = min(min(window_size,p.width//8),p.height//8)
        p.random_jitter = random_jitter

        if flag_noise_inverse:
            get_cache_callback = self.noise_inverse_get_cache
            set_cache_callback = lambda x0, xt, prompts: self.noise_inverse_set_cache(p, x0, xt, prompts, noise_inverse_steps, noise_inverse_retouch)
            delegate.init_noise_inverse(noise_inverse_steps, noise_inverse_retouch, get_cache_callback, set_cache_callback, noise_inverse_renoise_strength, noise_inverse_renoise_kernel)

        # delegate.get_views(overlap,tile_batch_size,batch_size_g)
        if self.controlnet_script:
            delegate.init_controlnet(self.controlnet_script, control_tensor_cpu)
        if self.stablesr_script:
            delegate.init_stablesr(self.stablesr_script)

        # init everything done, perform sanity check & pre-computations
        # hijack the behaviours
        delegate.hook()

        self.delegate = delegate

        exts = [
            "ContrlNet"  if self.controlnet_script else None,
            "StableSR"   if self.stablesr_script   else None,
        ]
        ext_info = ', '.join([e for e in exts if e])
        if ext_info: ext_info = f' (ext: {ext_info})'
        print(ext_info)

        return delegate.sampler_raw

    def create_random_tensors_hijack(
            self, bbox_settings: Dict, region_info: Dict,
            shape, seeds, subseeds=None, subseed_strength=0.0, seed_resize_from_h=0, seed_resize_from_w=0, p=None,
        ):
        org_random_tensors = Script.create_random_tensors_original_md(shape, seeds, subseeds, subseed_strength, seed_resize_from_h, seed_resize_from_w, p)
        height, width = shape[1], shape[2]
        background_noise = torch.zeros_like(org_random_tensors)
        background_noise_count = torch.zeros((1, 1, height, width), device=org_random_tensors.device)
        foreground_noise = torch.zeros_like(org_random_tensors)
        foreground_noise_count = torch.zeros((1, 1, height, width), device=org_random_tensors.device)

        for i, v in bbox_settings.items():
            seed = get_fixed_seed(v.seed)
            x, y, w, h = v.x, v.y, v.w, v.h
            # convert to pixel
            x = int(x * width)
            y = int(y * height)
            w = math.ceil(w * width)
            h = math.ceil(h * height)
            # clamp
            x = max(0, x)
            y = max(0, y)
            w = min(width  - x, w)
            h = min(height - y, h)
            # create random tensor
            torch.manual_seed(seed)
            rand_tensor = torch.randn((1, org_random_tensors.shape[1], h, w), device=devices.cpu)
            if BlendMode(v.blend_mode) == BlendMode.BACKGROUND:
                background_noise      [:, :, y:y+h, x:x+w] += rand_tensor.to(background_noise.device)
                background_noise_count[:, :, y:y+h, x:x+w] += 1
            elif BlendMode(v.blend_mode) == BlendMode.FOREGROUND:
                foreground_noise      [:, :, y:y+h, x:x+w] += rand_tensor.to(foreground_noise.device)
                foreground_noise_count[:, :, y:y+h, x:x+w] += 1
            else:
                raise NotImplementedError
            region_info['Region ' + str(i+1)]['seed'] = seed

        # average
        background_noise = torch.where(background_noise_count > 1, background_noise / background_noise_count, background_noise)
        foreground_noise = torch.where(foreground_noise_count > 1, foreground_noise / foreground_noise_count, foreground_noise)
        # paste two layers to original random tensor
        org_random_tensors = torch.where(background_noise_count > 0, background_noise, org_random_tensors)
        org_random_tensors = torch.where(foreground_noise_count > 0, foreground_noise, org_random_tensors)
        return org_random_tensors

    ''' ↓↓↓ helper methods ↓↓↓ '''

    def dump_regions(self, cfg_name, *bbox_controls):
        if not cfg_name: return gr_value(f'<span style="color:red">Config file name cannot be empty.</span>', visible=True)

        bbox_settings = build_bbox_settings(bbox_controls)
        data = {'bbox_controls': [v._asdict() for v in bbox_settings.values()]}

        if not os.path.exists(CFG_PATH): os.makedirs(CFG_PATH)
        fp = os.path.join(CFG_PATH, cfg_name)
        with open(fp, 'w', encoding='utf-8') as fh:
            json.dump(data, fh, indent=2, ensure_ascii=False)

        return gr_value(f'Config saved to {fp}.', visible=True)

    def load_regions(self, ref_image, cfg_name, *bbox_controls):
        if ref_image is None:
            return [gr_value(v) for v in bbox_controls] + [gr_value(f'<span style="color:red">Please create or upload a ref image first.</span>', visible=True)]
        fp = os.path.join(CFG_PATH, cfg_name)
        if not os.path.exists(fp): 
            return [gr_value(v) for v in bbox_controls] + [gr_value(f'<span style="color:red">Config {fp} not found.</span>', visible=True)]

        try:
            with open(fp, 'r', encoding='utf-8') as fh:
                data = json.load(fh)
        except Exception as e:
            return [gr_value(v) for v in bbox_controls] + [gr_value(f'<span style="color:red">Failed to load config {fp}: {e}</span>', visible=True)]

        num_boxes = len(data['bbox_controls'])
        data_list = []
        for i in range(BBOX_MAX_NUM):
            if i < num_boxes:
                for k in BBoxSettings._fields:
                    if k in data['bbox_controls'][i]:
                        data_list.append(data['bbox_controls'][i][k])
                    else:
                        data_list.append(None)
            else:
                data_list.extend(DEFAULT_BBOX_SETTINGS)

        return [gr_value(v) for v in data_list] + [gr_value(f'Config loaded from {fp}.', visible=True)]


    def noise_inverse_set_cache(self, p: ProcessingImg2Img, x0: Tensor, xt: Tensor, prompts: List[str], steps: int, retouch:float):
        self.noise_inverse_cache = NoiseInverseCache(p.sd_model.sd_model_hash, x0,  xt, steps, retouch, prompts)

    def noise_inverse_get_cache(self):
        return self.noise_inverse_cache


    def reset(self):
        ''' unhijack inner APIs, see hijack in process() '''
        if hasattr(Script, "create_sampler_original_md"):
            sd_samplers.create_sampler = Script.create_sampler_original_md
            del Script.create_sampler_original_md
        if hasattr(Script, "create_random_tensors_original_md"):
            processing.create_random_tensors = Script.create_random_tensors_original_md
            del Script.create_random_tensors_original_md
        if hasattr(sd_samplers_common.Sampler, "callback_ori"):
            sd_samplers_common.Sampler.callback_state = sd_samplers_common.Sampler.callback_ori
            del sd_samplers_common.Sampler.callback_ori
        if hasattr(processing, "create_infotext_ori"):
            processing.create_infotext = processing.create_infotext_ori
            del processing.create_infotext_ori
        DemoFusion.unhook()
        self.delegate = None

    def reset_and_gc(self):
        self.reset()
        self.noise_inverse_cache = None

        import gc; gc.collect()
        devices.torch_gc()

        try:
            import os
            import psutil
            mem = psutil.Process(os.getpid()).memory_info()
            print(f'[Mem] rss: {mem.rss/2**30:.3f} GB, vms: {mem.vms/2**30:.3f} GB')
            from modules.shared import mem_mon as vram_mon
            from modules.memmon import MemUsageMonitor
            vram_mon: MemUsageMonitor
            free, total = vram_mon.cuda_mem_get_info()
            print(f'[VRAM] free: {free/2**30:.3f} GB, total: {total/2**30:.3f} GB')
        except:
            pass
