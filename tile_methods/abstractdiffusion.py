import math
import torch
from tqdm import tqdm

from modules import devices, shared
from modules.shared import state
from modules.processing import opt_f

from tile_utils.utils import *
from tile_utils.typing import *


class TiledDiffusion:

    def __init__(self, p:StableDiffusionProcessing, sampler:Sampler):
        self.method = self.__class__.__name__
        self.p = p
        self.pbar = None

        # sampler
        self.sampler_name = p.sampler_name
        self.sampler_raw = sampler
        if self.is_kdiff: self.sampler: CFGDenoiser = sampler.model_wrap_cfg
        else: self.sampler: VanillaStableDiffusionSampler = sampler

        # fix. Kdiff 'AND' support and image editing model support
        if self.is_kdiff and not hasattr(self, 'is_edit_model'):
            self.is_edit_model = (shared.sd_model.cond_stage_key == "edit"      # "txt"
                and self.sampler.image_cfg_scale is not None 
                and self.sampler.image_cfg_scale != 1.0)

        # cache. final result of current sampling step, [B, C=4, H//8, W//8]
        # avoiding overhead of creating new tensors and weight summing
        self.x_buffer: Tensor = None
        self.w: int = int(self.p.width  // opt_f)       # latent size
        self.h: int = int(self.p.height // opt_f)
        # weights for background & grid bboxes
        self.weights: Tensor = torch.zeros((1, 1, self.h, self.w), device=devices.device, dtype=torch.float32)

        # FIXME: I'm trying to count the step correctly but it's not working
        self.step_count = 0         
        self.inner_loop_count = 0  
        self.kdiff_step = -1

        # ext. Grid tiling painting (grid bbox)
        self.enable_grid_bbox: bool = False
        self.tile_w: int = None
        self.tile_h: int = None
        self.num_batches: int = None
        self.batched_bboxes: List[List[BBox]] = []

        # ext. Region Prompt Control (custom bbox)
        self.enable_custom_bbox: bool = False
        self.custom_bboxes: List[CustomBBox] = []
        self.cond_basis: Cond = None
        self.uncond_basis: Uncond = None
        self.draw_background: bool = True       # by default we draw major prompts in grid tiles
        self.causal_layers: bool = None

        # ext. ControlNet
        self.enable_controlnet: bool = False
        self.controlnet_script: ModuleType = None
        self.control_tensor_batch: List[List[Tensor]] = []
        self.control_params: Dict[str, Tensor] = {}
        self.control_tensor_cpu: bool = None
        self.control_tensor_custom: List[List[Tensor]] = []

    @property
    def is_kdiff(self):
        return isinstance(self.sampler_raw, KDiffusionSampler)

    @property
    def is_ddim(self):
        return isinstance(self.sampler_raw, VanillaStableDiffusionSampler)

    def update_pbar(self):
        if self.pbar.n >= self.pbar.total:
            self.pbar.close()
        else:
            if self.step_count == state.sampling_step:
                self.inner_loop_count += 1
                if self.inner_loop_count < self.total_bboxes:
                    self.pbar.update()
            else:
                self.step_count = state.sampling_step
                self.inner_loop_count = 0

    def reset_buffer(self, x_in:Tensor):
        if self.x_buffer is None:
            self.x_buffer = torch.zeros_like(x_in, device=x_in.device, dtype=x_in.dtype)
        else:
            self.x_buffer.zero_()

    def init_done(self):
        '''
          Call this after all `init_*`, settings are done, now perform:
            - settings sanity check 
            - pre-computations, cache init
            - anything thing needed before denoising starts
        '''

        self.total_bboxes = 0
        if self.enable_grid_bbox:   self.total_bboxes += self.num_batches
        if self.enable_custom_bbox: self.total_bboxes += len(self.custom_bboxes)
        assert self.total_bboxes > 0, "Nothing to paint! No background to draw and no custom bboxes were provided."

        self.pbar = tqdm(total=(self.total_bboxes) * state.sampling_steps, desc=f"{self.method} Sampling: ")

    ''' ↓↓↓ extensive functionality ↓↓↓ '''

    @grid_bbox
    def init_grid_bbox(self, tile_w:int, tile_h:int, overlap:int, tile_bs:int):
        self.enable_grid_bbox = True

        self.tile_w = min(tile_w, self.w)
        self.tile_h = min(tile_h, self.h)
        overlap = max(0, min(overlap, min(tile_w, tile_h) - 4))
        # split the latent into overlapped tiles, then batching
        # weights basically indicate how many times a pixel is painted
        bboxes, weights = split_bboxes(self.w, self.h, self.tile_w, self.tile_h, overlap, self.get_tile_weights())
        self.weights += weights
        self.num_batches = math.ceil(len(bboxes) / tile_bs)
        BS = math.ceil(len(bboxes) / self.num_batches)          # optimal_batch_size
        self.batched_bboxes = [bboxes[i*BS:(i+1)*BS] for i in range(self.num_batches)]

    @grid_bbox
    def get_tile_weights(self) -> Union[Tensor, float]:
        return 1.0


    @custom_bbox
    def init_custom_bbox(self, bbox_control_states:BBoxControls, draw_background:bool, causal_layers:bool):
        self.enable_custom_bbox = True

        self.causal_layers = causal_layers
        self.draw_background = draw_background
        if not draw_background:
            self.enable_grid_bbox = False
            self.weights.zero_()

        # The number parameters needed to initialize the CustomBBox is 9 currently.
        # Need to be the same as the number of parameters in the `bbox_control_states` list.
        n_controls = 9
        assert len(bbox_control_states) % n_controls == 0, '`control_states` count misalign, please check ui()'
        self.custom_bboxes: List[CustomBBox] = []
        for i in range(0, len(bbox_control_states), n_controls):
            e, x, y, w, h, p, n, blend_mode, feather_ratio = bbox_control_states[i:i+n_controls]
            if not e or x > 1.0 or y > 1.0 or w <= 0.0 or h <= 0.0: continue

            x = int(x * self.w)
            y = int(y * self.h)
            w = math.ceil(w * self.w)
            h = math.ceil(h * self.h)
            x = max(0, x)
            y = max(0, y)
            w = min(self.w - x, w)
            h = min(self.h - y, h)
            self.custom_bboxes.append(CustomBBox(x, y, w, h, p, n, blend_mode, feather_ratio))

        if len(self.custom_bboxes) == 0: return

        # prepare cond
        p = self.p
        prompts = p.all_prompts[:p.batch_size]
        neg_prompts = p.all_negative_prompts[:p.batch_size]
        for bbox in self.custom_bboxes:
            bbox.cond, bbox.extra_network_data = Condition.get_cond(Prompt.append_prompt(prompts, bbox.prompt), p.steps, p.styles)
            bbox.uncond = Condition.get_uncond(Prompt.append_prompt(neg_prompts, bbox.neg_prompt), p.steps, p.styles)
        self.cond_basis, _ = Condition.get_cond(prompts, p.steps)
        self.uncond_basis = Condition.get_uncond(neg_prompts, p.steps)

    @custom_bbox
    def reconstruct_custom_cond(self, org_cond:CondDict, custom_cond:Cond, custom_uncond:Uncond, bbox:CustomBBox) -> Tuple[List, Tensor, Uncond, Tensor]:
        image_conditioning = None
        if isinstance(org_cond, dict):
            image_cond = org_cond['c_concat'][0]
            if image_cond.shape[2:] == (self.h, self.w):    # img2img
                image_cond = image_cond[bbox.slicer]
            image_conditioning = image_cond

        tensor = Condition.reconstruct_cond(custom_cond, self.sampler.step)
        custom_uncond = Condition.reconstruct_uncond(custom_uncond, self.sampler.step)
        return tensor, custom_uncond, image_conditioning

    @custom_bbox
    def kdiff_custom_forward(self, x_tile:Tensor, sigma_in:Tensor, original_cond:CondDict, bbox_id:int, bbox:CustomBBox, forward_func:Callable) -> Tensor:
        '''
        The inner kdiff noise prediction is usually batched.
        We need to unwrap the inside loop to simulate the batched behavior.
        This can be extremely tricky.
        '''

        if self.kdiff_step != self.sampler.step:
            self.kdiff_step = self.sampler.step
            self.kdiff_step_bbox = [-1 for _ in range(len(self.custom_bboxes))]
            self.tensor = {}        # {int: Tensor[cond]}
            self.uncond = {}        # {int: Tensor[cond]}
            self.image_cond_in = {}
            # Initialize global prompts just for estimate the behavior of kdiff
            self.real_tensor = Condition.reconstruct_cond(self.cond_basis, self.sampler.step)
            self.real_uncond = Condition.reconstruct_uncond(self.uncond_basis, self.sampler.step)
            # reset the progress for all bboxes
            self.a = [0 for _ in range(len(self.custom_bboxes))]

        if self.kdiff_step_bbox[bbox_id] != self.sampler.step:
            # When a new step starts for a bbox, we need to judge whether the tensor is batched.
            self.kdiff_step_bbox[bbox_id] = self.sampler.step

            tensor, uncond, image_cond_in = self.reconstruct_custom_cond(original_cond, bbox.cond, bbox.uncond, bbox)

            if self.real_tensor.shape[1] == self.real_uncond.shape[1]:
                if shared.batch_cond_uncond:
                    # when the real tensor is with equal length, all information is contained in x_tile.
                    # we simulate the batched behavior and compute all the tensors in one go.
                    if tensor.shape[1] == uncond.shape[1]:
                        # When our prompt tensor is with equal length, we can directly their code.
                        if not self.is_edit_model:
                            cond = torch.cat([tensor, uncond])
                        else:
                            cond = torch.cat([tensor, uncond, uncond])
                        self.set_controlnet_tensors(bbox_id, x_tile.shape[0])
                        return forward_func(x_tile, sigma_in, cond={"c_crossattn": [cond], "c_concat": [image_cond_in]})
                    else:
                        # When not, we need to pass the tensor to UNet separately.
                        x_out = torch.zeros_like(x_tile)
                        cond_size = tensor.shape[0]
                        self.set_controlnet_tensors(bbox_id, cond_size)
                        cond_out = forward_func(
                            x_tile  [:cond_size], 
                            sigma_in[:cond_size], 
                            cond={
                                "c_crossattn": [tensor], 
                                "c_concat": [image_cond_in[:cond_size]]
                            })
                        uncond_size = uncond.shape[0]
                        self.set_controlnet_tensors(bbox_id, uncond_size)
                        uncond_out = forward_func(
                            x_tile  [cond_size:cond_size+uncond_size], 
                            sigma_in[cond_size:cond_size+uncond_size], 
                            cond={
                                "c_crossattn": [uncond], 
                                "c_concat": [image_cond_in[cond_size:cond_size+uncond_size]]
                            })
                        x_out[:cond_size] = cond_out
                        x_out[cond_size:cond_size+uncond_size] = uncond_out
                        if self.is_edit_model:
                            x_out[cond_size+uncond_size:] = uncond_out
                        return x_out
                
            # otherwise, the x_tile is only a partial batch. 
            # We have to denoise in different runs.
            # We store the prompt and neg_prompt tensors for current bbox
            self.tensor[bbox_id] = tensor
            self.uncond[bbox_id] = uncond
            self.image_cond_in[bbox_id] = image_cond_in

        # Now we get current batch of prompt and neg_prompt tensors
        tensor = self.tensor[bbox_id]
        uncond = self.uncond[bbox_id]
        batch_size = x_tile.shape[0]
        # get the start and end index of the current batch
        a = self.a[bbox_id]
        b = a + batch_size
        self.a[bbox_id] += batch_size

        if self.real_tensor.shape[1] == self.real_uncond.shape[1]:
            # When use --lowvram or --medvram, kdiff will slice the cond and uncond with [a:b]
            # So we need to slice our tensor and uncond with the same index as original kdiff.
            
            # --- original code in kdiff ---
            # if not self.is_edit_model:
            #     cond = torch.cat([tensor, uncond])
            # else:
            #     cond = torch.cat([tensor, uncond, uncond])
            # cond = cond[a:b]
            # ------------------------------
            
            # The original kdiff code is to concat and then slice, but this cannot apply to
            # our custom prompt tensor when tensor.shape[1] != uncond.shape[1]. So we adapt it.
            cond_in, uncond_in = None, None
            # Slice the [prompt, neg prompt, (possibly) neg prompt] with [a:b]
            if not self.is_edit_model:
                if   b <= tensor.shape[0]: cond_in = tensor[a:b]
                elif a >= tensor.shape[0]: cond_in = uncond[a-tensor.shape[0]:b-tensor.shape[0]]
                else:
                    cond_in = tensor[a:]
                    uncond_in = uncond[:b-tensor.shape[0]]
            else:
                if b <= tensor.shape[0]:
                    cond_in = tensor[a:b]
                elif b > tensor.shape[0] and b <= tensor.shape[0] + uncond.shape[0]:
                    if a>= tensor.shape[0]:
                        cond_in = uncond[a-tensor.shape[0]:b-tensor.shape[0]]
                    else:
                        cond_in = tensor[a:]
                        uncond_in = uncond[:b-tensor.shape[0]]
                else:
                    if a >= tensor.shape[0] + uncond.shape[0]:
                        cond_in = uncond[a-tensor.shape[0]-uncond.shape[0]:b-tensor.shape[0]-uncond.shape[0]]
                    elif a >= tensor.shape[0]:
                        cond_in = torch.cat([uncond[a-tensor.shape[0]:], uncond[:b-tensor.shape[0]-uncond.shape[0]]])

            if uncond_in is None or tensor.shape[1] == uncond.shape[1]:
                # If the tensor can be passed to UNet in one go, do it.
                if uncond_in is not None:
                    cond_in = torch.cat([cond_in, uncond_in])
                self.set_controlnet_tensors(bbox_id, x_tile.shape[0])
                return forward_func(
                    x_tile, 
                    sigma_in, 
                    cond={
                    "c_crossattn": [cond_in], 
                    "c_concat": [self.image_cond_in[bbox_id]],
                })
            else:
                # If not, we need to pass the tensor to UNet separately.
                x_out = torch.zeros_like(x_tile)
                cond_size = cond_in.shape[0]
                self.set_controlnet_tensors(bbox_id, cond_size)
                cond_out = forward_func(
                    x_tile  [:cond_size], 
                    sigma_in[:cond_size], 
                    cond={
                        "c_crossattn": [cond_in], 
                        "c_concat": [self.image_cond_in[bbox_id]],
                    })
                self.set_controlnet_tensors(bbox_id, uncond_in.shape[0])
                uncond_out = forward_func(
                    x_tile  [cond_size:], 
                    sigma_in[cond_size:], 
                    cond={
                        "c_crossattn": [uncond_in], 
                        "c_concat": [self.image_cond_in[bbox_id]],
                    })
                x_out[:cond_size] = cond_out
                x_out[cond_size:] = uncond_out
                return x_out

        # If the original prompt is with different length, 
        # kdiff will deal with the cond and uncond separately.
        # Hence we also deal with the tensor and uncond separately.
        # get the start and end index of the current batch

        if a < tensor.shape[0]:
            # Deal with custom prompt tensor
            if not self.is_edit_model:
                c_crossattn = [tensor[a:b]]
            else:
                c_crossattn = torch.cat([tensor[a:b]], uncond)
            self.set_controlnet_tensors(bbox_id, x_tile.shape[0])
            # complete this batch.
            return forward_func(
                x_tile, 
                sigma_in, 
                cond={
                    "c_crossattn": c_crossattn, 
                    "c_concat": [self.image_cond_in[bbox_id]]
                })
        else:
            # if the cond is finished, we need to process the uncond.
            self.set_controlnet_tensors(bbox_id, uncond.shape[0])
            return forward_func(
                x_tile, 
                sigma_in, 
                cond={
                    "c_crossattn": [uncond], 
                    "c_concat": [self.image_cond_in[bbox_id]]
                })

    @custom_bbox
    def ddim_custom_forward(self, x:Tensor, cond_in:CondDict, bbox:CustomBBox, ts:Tensor, forward_func:Callable, *args, **kwargs) -> Tensor:
        ''' draw custom bbox '''

        conds_list, tensor, uncond, image_conditioning = self.reconstruct_custom_cond(cond_in, bbox.cond, bbox.uncond, bbox)
        assert all([len(conds) == 1 for conds in conds_list]), 'composition via AND is not supported for DDIM/PLMS samplers'

        cond = tensor
        # for DDIM, shapes definitely match. So we dont need to do the same thing as in the KDIFF sampler.
        if uncond.shape[1] < cond.shape[1]:
            last_vector = uncond[:, -1:]
            last_vector_repeated = last_vector.repeat([1, cond.shape[1] - uncond.shape[1], 1])
            uncond = torch.hstack([uncond, last_vector_repeated])
        elif uncond.shape[1] > cond.shape[1]:
            uncond = uncond[:, :cond.shape[1]]

        # Wrap the image conditioning back up since the DDIM code can accept the dict directly.
        # Note that they need to be lists because it just concatenates them later.
        if image_conditioning is not None:
            cond   = {"c_concat": [image_conditioning], "c_crossattn": [cond]}
            uncond = {"c_concat": [image_conditioning], "c_crossattn": [uncond]}
        
        # We cannot determine the batch size here for different methods, so delay it to the forward_func.
        return forward_func(x, cond, ts, unconditional_conditioning=uncond, *args, **kwargs)


    @controlnet
    def init_controlnet(self, controlnet_script:ModuleType, control_tensor_cpu:bool):
        self.enable_controlnet = True

        self.controlnet_script = controlnet_script
        self.control_tensor_cpu = control_tensor_cpu
        self.control_tensor_batch = None
        self.control_params = None
        self.control_tensor_custom = []

        self.reset_controlnet_tensors()
        self.prepare_controlnet_tensors()

    @controlnet
    def reset_controlnet_tensors(self):
        if not self.enable_controlnet: return
        if self.control_tensor_batch is None: return

        for param_id in range(len(self.control_params)):
            self.control_params[param_id].hint_cond = self.org_control_tensor_batch[param_id]

    @controlnet
    def prepare_controlnet_tensors(self):
        ''' Crop the control tensor into tiles and cache them '''

        if not self.enable_controlnet: return
        if self.control_tensor_batch is not None: return
        if self.controlnet_script is None or self.control_params is not None: return
        latest_network = self.controlnet_script.latest_network
        if latest_network is None or not hasattr(latest_network, 'control_params'): return
        self.control_params = latest_network.control_params
        tensors = [param.hint_cond for param in latest_network.control_params]
        self.org_control_tensor_batch = tensors
        if len(tensors) == 0: return

        self.control_tensor_batch = []
        for i in range(len(tensors)):
            control_tile_list = []
            control_tensor = tensors[i]
            for bboxes in self.batched_bboxes:
                single_batch_tensors = []
                for bbox in bboxes:
                    if len(control_tensor.shape) == 3:
                        control_tensor.unsqueeze_(0)
                    control_tile = control_tensor[:, :, bbox[1]*opt_f:bbox[3]*opt_f, bbox[0]*opt_f:bbox[2]*opt_f]
                    single_batch_tensors.append(control_tile)
                control_tile = torch.cat(single_batch_tensors, dim=0)
                if self.control_tensor_cpu:
                    control_tile = control_tile.cpu()
                control_tile_list.append(control_tile)
            self.control_tensor_batch.append(control_tile_list)

            if len(self.custom_bboxes) > 0:
                custom_control_tile_list = []
                for bbox in self.custom_bboxes:
                    if len(control_tensor.shape) == 3:
                        control_tensor.unsqueeze_(0)
                    control_tile = control_tensor[:, :, bbox[1]*opt_f:bbox[3]*opt_f, bbox[0]*opt_f:bbox[2]*opt_f]
                    if self.control_tensor_cpu:
                        control_tile = control_tile.cpu()
                    custom_control_tile_list.append(control_tile)
                self.control_tensor_custom.append(custom_control_tile_list)

    @controlnet
    def switch_controlnet_tensors(self, batch_id:int, x_batch_size:int, tile_batch_size:int, is_denoise=False):
        if not self.enable_controlnet: return
        if self.control_tensor_batch is None: return

        for param_id in range(len(self.control_params)):
            control_tile = self.control_tensor_batch[param_id][batch_id]
            if self.is_kdiff:
                all_control_tile = []
                for i in range(tile_batch_size):
                    this_control_tile = [control_tile[i].unsqueeze(0)] * x_batch_size
                    all_control_tile.append(torch.cat(this_control_tile, dim=0))
                control_tile = torch.cat(all_control_tile, dim=0)                                           
            else:
                control_tile = control_tile.repeat([x_batch_size if is_denoise else x_batch_size * 2, 1, 1, 1])
            self.control_params[param_id].hint_cond = control_tile.to(devices.device)

    @controlnet
    def set_controlnet_tensors(self, bbox_id:int, repeat_size:int):
        if not self.enable_controlnet: return
        if not len(self.control_tensor_custom): return
        
        for param_id in range(len(self.control_params)):
            control_tensor = self.control_tensor_custom[param_id][bbox_id].to(devices.device)
            self.control_params[param_id].hint_cond = control_tensor.repeat((repeat_size, 1, 1, 1))
