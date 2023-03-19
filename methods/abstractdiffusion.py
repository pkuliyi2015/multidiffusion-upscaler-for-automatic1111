from abc import ABC, abstractmethod
import math
import torch
from tqdm import tqdm

from modules import devices, shared, prompt_parser


class TiledDiffusion(ABC):
    def __init__(self, method, sampler, sampler_name, iters, batch_size, steps, 
                 w, h, tile_w=64, tile_h=64, overlap=32, tile_batch_size=1,
                 controlnet_script=None, control_tensor_cpu=False) -> None:
        
        self.is_kdiff = sampler_name not in ['DDIM', 'PLMS', 'UniPC']
        if self.is_kdiff:
            # The sampler is CFGDenoiser
            self.sampler = sampler.model_wrap_cfg
        else:
            self.sampler = sampler
        
        self.method = method
        self.batch_size = batch_size
        self.steps = steps
        self.iters = iters
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
        self.num_batches = math.ceil(len(bboxes) / tile_batch_size)
        optimal_batch_size = math.ceil(len(bboxes) / self.num_batches)
        self.tile_batch_size = optimal_batch_size
        for i in range(self.num_batches):
            start = i * tile_batch_size
            end = min((i + 1) * tile_batch_size, len(bboxes))
            self.batched_bboxes.append(bboxes[start:end])
        self.weights = weights.unsqueeze(0).unsqueeze(0)

        # Avoid the overhead of creating a new tensor for each batch
        # And avoid the overhead of weight summing
        self.x_buffer = None
        # Region prompt control
        self.custom_bboxes = []
        self.global_multiplier = 1.0

        # For controlnet
        self.controlnet_script = controlnet_script
        self.control_tensor_batch = None
        self.control_params = None
        self.control_tensor_cpu = control_tensor_cpu
        self.control_tensor_custom = []

        # Progress bar
        self.pbar = None

    def init_pbar(self):
        if self.pbar is None:
            total_bboxes = (self.num_batches if self.global_multiplier > 0 else 0) + len(self.custom_bboxes)
            assert total_bboxes > 0, "No bboxes to sample! global_multiplier is 0 and no custom bboxes are provided."
            self.pbar = tqdm(total=(total_bboxes)* self.iters, desc=f"{self.method} Sampling: ")
    
    def update_pbar(self):
        if self.pbar.n >= self.pbar.total:
            self.pbar.close()
        else:
            self.pbar.update()
    
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
                bbox.append([x, y, x + tile_w, y + tile_h])
                self.add_weights(count[y:y+tile_h, x:x+tile_w])
        return bbox, count
    
    @abstractmethod
    def add_weights(self, input):
        pass
    
    def prepare_custom_bbox(self, prompts, negative_prompts, global_multiplier, bbox_control_states):
        '''
        Prepare custom bboxes for region prompt
        '''
        self.global_multiplier = max(global_multiplier, 0.0)
        for i in range(0, len(bbox_control_states) - 8, 8):
            e, m, x, y, w, h, p, neg = bbox_control_states[i:i+8]
            if not e or m < 1 or w <= 0 or h <= 0 or p == '':
                continue
            bbox = [int(x * self.w), int(y * self.h),
                    int((x + w) * self.w), int((y + h) * self.h)]
            c_prompt = [prompt + ', ' + p for prompt in prompts]
            if neg != '':
                c_negative_prompt = [prompt + ', ' +
                                     neg for prompt in negative_prompts]
            else:
                c_negative_prompt = negative_prompts
            c_prompt = prompt_parser.get_multicond_learned_conditioning(
                shared.sd_model, c_prompt, self.steps)
            c_negative_prompt = prompt_parser.get_learned_conditioning(
                shared.sd_model, c_negative_prompt, self.steps)
            self.custom_bboxes.append((bbox, c_prompt, c_negative_prompt, m))
        if self.global_multiplier > 0 and abs(self.global_multiplier - 1.0) < 1e-6:
            self.init_pbar()

    def prepare_custom_cond(self, cond_in, cond, uncond, bbox):
        image_conditioning = None
        if isinstance(cond_in, dict):
            image_cond = cond_in['c_concat'][0]
            if image_cond.shape[2] == self.h and image_cond.shape[3] == self.w:
                image_cond = image_cond[:, :,
                                        bbox[1]:bbox[3], bbox[0]:bbox[2]]
            image_conditioning = image_cond

        conds_list, tensor = prompt_parser.reconstruct_multicond_batch(
            cond, self.sampler.step)
        uncond = prompt_parser.reconstruct_cond_batch(
            uncond, self.sampler.step)
        return conds_list, tensor, uncond, image_conditioning
    

    def kdiff_custom_forward(self, x_tile, original_cond, custom_cond, uncond, bbox, sigma_in, is_edit_model, forward_func, batched=False):
        '''
        Code migrate from modules/sd_samplers_kdiffusion.py
        '''
        conds_list, tensor, uncond, image_cond = self.prepare_custom_cond(original_cond, custom_cond, uncond, bbox)
        repeats = [len(conds_list[i]) for i in range(batch_size)]
        if not is_edit_model:
            x_in = torch.cat([torch.stack([x_tile[i] for _ in range(n)])
                                for i, n in enumerate(repeats)] + [x_tile])
            image_cond_in = torch.cat([torch.stack([image_cond[i] for _ in range(
                n)]) for i, n in enumerate(repeats)] + [image_cond])
        else:
            x_in = torch.cat([torch.stack([x_tile[i] for _ in range(n)])
                                for i, n in enumerate(repeats)] + [x_tile] + [x_tile])
            image_cond_in = torch.cat([torch.stack([image_cond[i] for _ in range(n)]) for i, n in enumerate(
                repeats)] + [image_cond] + [torch.zeros_like(x_tile)])

        if tensor.shape[1] == uncond.shape[1]:
            if not is_edit_model:
                original_cond = torch.cat([tensor, uncond])
            else:
                original_cond = torch.cat([tensor, uncond, uncond])

            if shared.batch_cond_uncond or batched:
                x_out = forward_func(x_in, sigma_in, cond={"c_crossattn": [
                                            original_cond], "c_concat": [image_cond_in]})
            else:
                x_out = torch.zeros_like(x_in)
                for batch_offset in range(0, x_out.shape[0], batch_size):
                    a = batch_offset
                    b = a + batch_size
                    x_out[a:b] = forward_func(x_in[a:b], sigma_in[a:b], cond={"c_crossattn": [
                                                    original_cond[a:b]], "c_concat": [image_cond_in[a:b]]})
        else:
            x_out = torch.zeros_like(x_in)
            batch_size = batch_size*2 if shared.batch_cond_uncond or batched else batch_size
            for batch_offset in range(0, tensor.shape[0], batch_size):
                a = batch_offset
                b = min(a + batch_size, tensor.shape[0])

                if not is_edit_model:
                    c_crossattn = [tensor[a:b]]
                else:
                    c_crossattn = torch.cat([tensor[a:b]], uncond)

                x_out[a:b] = forward_func(x_in[a:b], sigma_in[a:b], cond={
                                                "c_crossattn": c_crossattn, "c_concat": [image_cond_in[a:b]]})
            x_out[-uncond.shape[0]:] = forward_func(x_in[-uncond.shape[0]:], sigma_in[-uncond.shape[0]:], cond={
                                                            "c_crossattn": [uncond], "c_concat": [image_cond_in[-uncond.shape[0]:]]})
        return x_out
    
    def ddim_custom_forward(self, x, cond_in, cond, uncond, bbox, ts, forward_func, *args, **kwargs):
        conds_list, tensor, uncond, image_conditioning = self.prepare_custom_cond(cond_in, cond, uncond, bbox)
        assert all([len(conds) == 1 for conds in conds_list]
                    ), 'composition via AND is not supported for DDIM/PLMS samplers'
        
        cond = tensor
        # for DDIM, shapes must match, we can't just process cond and uncond independently;
        # filling uncond with repeats of the last vector to match length is
        # not 100% correct but should work well enough
        if uncond.shape[1] < cond.shape[1]:
            last_vector = uncond[:, -1:]
            last_vector_repeated = last_vector.repeat(
                [1, cond.shape[1] - uncond.shape[1], 1])
            uncond = torch.hstack(
                [uncond, last_vector_repeated])
        elif uncond.shape[1] > cond.shape[1]:
            uncond = uncond[:,:cond.shape[1]]

        if self.sampler.mask is not None:
            if self.sampler.init_latent.shape[2] == self.h and self.sampler.init_latent.shape[3] == self.w:
                latent = self.sampler.init_latent[:, :, bbox[0]:bbox[2], bbox[1]:bbox[3]]
            img_orig = self.sampler.sampler.model.q_sample(
                latent, ts)
            x = img_orig * self.sampler.mask + self.sampler.nmask * x

        # Wrap the image conditioning back up since the DDIM code can accept the dict directly.
        # Note that they need to be lists because it just concatenates them later.
        if image_conditioning is not None:
            cond = {"c_concat": [image_conditioning],
                    "c_crossattn": [cond]}
            uncond = {"c_concat": [
                image_conditioning], "c_crossattn": [uncond]}

        x_tile_out, x_pred = forward_func(
            x, cond, ts, unconditional_conditioning=uncond, *args, **kwargs)
        return x_tile_out, x_pred

    
    def prepare_control_tensors(self, batch_size):
        '''
        Crop the control tensor into tiles and cache them
        '''
        if self.control_tensor_batch is not None:
            return
        if self.controlnet_script is None or self.control_params is not None:
            return
        latest_network = self.controlnet_script.latest_network
        if latest_network is None or not hasattr(latest_network, 'control_params'):
            return
        self.control_params = latest_network.control_params
        tensors = [param.hint_cond for param in latest_network.control_params]
        if len(tensors) == 0:
            return
        self.control_tensor_batch = []
        for i in range(len(tensors)):
            control_tile_list = []
            control_tensor = tensors[i]
            for bboxes in self.batched_bboxes:
                single_batch_tensors = []
                for bbox in bboxes:
                    if len(control_tensor.shape) == 3:
                        control_tile = control_tensor[:, bbox[1] *
                                                      8:bbox[3]*8, bbox[0]*8:bbox[2]*8].unsqueeze(0)
                    else:
                        control_tile = control_tensor[:, :,
                                                      bbox[1]*8:bbox[3]*8, bbox[0]*8:bbox[2]*8]
                    single_batch_tensors.append(control_tile)
                if self.is_kdiff:
                    control_tile = torch.cat(
                        [t for t in single_batch_tensors for _ in range(batch_size)], dim=0)
                else:
                    control_tile = torch.cat(
                        single_batch_tensors*batch_size, dim=0)
                if self.control_tensor_cpu:
                    control_tile = control_tile.cpu()
                control_tile_list.append(control_tile)
            self.control_tensor_batch.append(control_tile_list)
            
            if len(self.custom_bboxes) > 0:
                custom_control_tile_list = []
                for bbox, _, _, _ in self.custom_bboxes:
                    if len(control_tensor.shape) == 3:
                        control_tile = control_tensor[:, bbox[1] *
                                                      8:bbox[3]*8, bbox[0]*8:bbox[2]*8].unsqueeze(0)
                    else:
                        control_tile = control_tensor[:, :,
                                                      bbox[1]*8:bbox[3]*8, bbox[0]*8:bbox[2]*8]
                    if self.control_tensor_cpu:
                        control_tile = control_tile.cpu()
                    custom_control_tile_list.append(control_tile)
                self.control_tensor_custom.append(custom_control_tile_list)


    def switch_controlnet_tensors(self, batch_id, x_tile):
        if self.control_tensor_batch is not None:
            for i in range(len(self.control_params)):
                new_control = self.control_tensor_batch[i][batch_id]
                if new_control.shape[0] != x_tile.shape[0]:
                    new_control = new_control[:x_tile.shape[0], :, :, :]
                self.control_params[i].hint_cond = new_control.to(x_tile.device)

    def switch_custom_controlnet_tensors(self, index, x):
        if len(self.control_tensor_custom) > 0:
            for i in range(len(self.control_params)):
                self.control_params[i].hint_cond = self.control_tensor_custom[i][index].to(x.device)