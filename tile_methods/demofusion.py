from tile_methods.abstractdiffusion import AbstractDiffusion
from tile_utils.utils import *
import torch.nn.functional as F
import random
from copy import deepcopy
import inspect
from modules import sd_samplers_common


class DemoFusion(AbstractDiffusion):
    """
        DemoFusion Implementation
        https://arxiv.org/abs/2311.16973
    """

    def __init__(self, p:Processing, *args, **kwargs):
        super().__init__(p, *args, **kwargs)
        assert p.sampler_name != 'UniPC', 'Demofusion is not compatible with UniPC!'


    def hook(self):
        steps, self.t_enc = sd_samplers_common.setup_img2img_steps(self.p, None)

        self.sampler.model_wrap_cfg.forward_ori = self.sampler.model_wrap_cfg.forward
        self.sampler_forward = self.sampler.model_wrap_cfg.inner_model.forward
        self.sampler.model_wrap_cfg.forward = self.forward_one_step
        if self.is_kdiff:
            self.sampler: KDiffusionSampler
            self.sampler.model_wrap_cfg: CFGDenoiserKDiffusion
            self.sampler.model_wrap_cfg.inner_model: Union[CompVisDenoiser, CompVisVDenoiser]
        else:
            self.sampler: CompVisSampler
            self.sampler.model_wrap_cfg: CFGDenoiserTimesteps
            self.sampler.model_wrap_cfg.inner_model: Union[CompVisTimestepsDenoiser, CompVisTimestepsVDenoiser]
            self.timesteps = self.sampler.get_timesteps(self.p, steps)

    @staticmethod
    def unhook():
        if hasattr(shared.sd_model,  'apply_model_ori'):
            shared.sd_model.apply_model = shared.sd_model.apply_model_ori
            del shared.sd_model.apply_model_ori

    def reset_buffer(self, x_in:Tensor):
        super().reset_buffer(x_in)



    def repeat_tensor(self, x:Tensor, n:int) -> Tensor:
        ''' repeat the tensor on it's first dim '''
        if n == 1: return x
        B = x.shape[0]
        r_dims = len(x.shape) - 1
        if B == 1:      # batch_size = 1 (not `tile_batch_size`)
            shape = [n] + [-1] * r_dims     # [N, -1, ...]
            return x.expand(shape)          # `expand` is much lighter than `tile`
        else:
            shape = [n] + [1] * r_dims      # [N, 1, ...]
            return x.repeat(shape)

    def repeat_cond_dict(self, cond_in:CondDict, bboxes:List[CustomBBox]) -> CondDict:
        ''' repeat all tensors in cond_dict on it's first dim (for a batch of tiles), returns a new object '''
        # n_repeat
        n_rep = len(bboxes)
        # txt cond
        tcond = self.get_tcond(cond_in)           # [B=1, L, D] => [B*N, L, D]
        tcond = self.repeat_tensor(tcond, n_rep)
        # img cond
        icond = self.get_icond(cond_in)
        if icond.shape[2:] == (self.h, self.w):   # img2img, [B=1, C, H, W]
            icond = torch.cat([icond[bbox.slicer] for bbox in bboxes], dim=0)
        else:                                     # txt2img, [B=1, C=5, H=1, W=1]
            icond = self.repeat_tensor(icond, n_rep)
        # vec cond (SDXL)
        vcond = self.get_vcond(cond_in)           # [B=1, D]
        if vcond is not None:
            vcond = self.repeat_tensor(vcond, n_rep)  # [B*N, D]
        return self.make_cond_dict(cond_in, tcond, icond, vcond)


    def global_split_bboxes(self):
        cols = self.p.current_scale_num
        rows = cols

        bbox_list = []
        for row in range(rows):
            y = row
            for col in range(cols):
                x = col
                bbox = (x, y)
                bbox_list.append(bbox)

        return bbox_list

    def split_bboxes_jitter(self,w_l:int, h_l:int, tile_w:int, tile_h:int, overlap:int=16, init_weight:Union[Tensor, float]=1.0) -> Tuple[List[BBox], Tensor]:
        cols = math.ceil((w_l - overlap) / (tile_w - overlap))
        rows = math.ceil((h_l - overlap) / (tile_h - overlap))
        if rows==0:
            rows=1
        if cols == 0:
            cols=1
        dx = (w_l - tile_w) / (cols - 1) if cols > 1 else 0
        dy = (h_l - tile_h) / (rows - 1) if rows > 1 else 0
        bbox_list: List[BBox] = []
        self.jitter_range = 0
        for row in range(rows):
            for col in range(cols):
                h = min(int(row * dy), h_l - tile_h)
                w = min(int(col * dx), w_l - tile_w)
                if self.p.random_jitter:
                    self.jitter_range = min(max((min(self.w, self.h)-self.stride)//4,0),min(int(self.window_size/2),int(self.overlap/2)))
                    jitter_range = self.jitter_range
                    w_jitter = 0
                    h_jitter = 0
                    if (w != 0) and (w+tile_w != w_l):
                        w_jitter = random.randint(-jitter_range, jitter_range)
                    elif (w == 0) and (w + tile_w != w_l):
                        w_jitter = random.randint(-jitter_range, 0)
                    elif (w != 0) and (w + tile_w == w_l):
                        w_jitter = random.randint(0, jitter_range)
                    if (h != 0) and (h + tile_h != h_l):
                        h_jitter = random.randint(-jitter_range, jitter_range)
                    elif (h == 0) and (h + tile_h != h_l):
                        h_jitter = random.randint(-jitter_range, 0)
                    elif (h != 0) and (h + tile_h == h_l):
                        h_jitter = random.randint(0, jitter_range)
                    h +=(h_jitter + jitter_range)
                    w += (w_jitter + jitter_range)

                bbox = BBox(w, h, tile_w, tile_h)
                bbox_list.append(bbox)
        return bbox_list, None

    @grid_bbox
    def get_views(self, overlap:int, tile_bs:int):
        self.enable_grid_bbox = True
        self.tile_w = self.window_size
        self.tile_h = self.window_size

        self.overlap = max(0, min(overlap, self.window_size - 4))

        self.stride = max(4,self.window_size - self.overlap)

        # split the latent into overlapped tiles, then batching
        # weights basically indicate how many times a pixel is painted
        bboxes, _ = self.split_bboxes_jitter(self.w, self.h, self.tile_w, self.tile_h, self.overlap, self.get_tile_weights())
        self.num_tiles = len(bboxes)
        self.num_batches = math.ceil(self.num_tiles / tile_bs)
        self.tile_bs = math.ceil(len(bboxes) / self.num_batches)          # optimal_batch_size
        self.batched_bboxes = [bboxes[i*self.tile_bs:(i+1)*self.tile_bs] for i in range(self.num_batches)]

        global_bboxes = self.global_split_bboxes()
        self.global_num_tiles = len(global_bboxes)
        self.global_num_batches = math.ceil(self.global_num_tiles / tile_bs)
        self.global_tile_bs = math.ceil(len(global_bboxes) / self.global_num_batches)
        self.global_batched_bboxes = [global_bboxes[i*self.global_tile_bs:(i+1)*self.global_tile_bs] for i in range(self.global_num_batches)]

    def gaussian_kernel(self,kernel_size=3, sigma=1.0, channels=3):
        x_coord = torch.arange(kernel_size, device=devices.device)
        gaussian_1d = torch.exp(-(x_coord - (kernel_size - 1) / 2) ** 2 / (2 * sigma ** 2))
        gaussian_1d = gaussian_1d / gaussian_1d.sum()
        gaussian_2d = gaussian_1d[:, None] * gaussian_1d[None, :]
        kernel = gaussian_2d[None, None, :, :].repeat(channels, 1, 1, 1)

        return kernel

    def gaussian_filter(self,latents, kernel_size=3, sigma=1.0):
        channels = latents.shape[1]
        kernel = self.gaussian_kernel(kernel_size, sigma, channels).to(latents.device, latents.dtype)
        blurred_latents = F.conv2d(latents, kernel, padding=kernel_size//2, groups=channels)

        return blurred_latents
        


    ''' ↓↓↓ kernel hijacks ↓↓↓ '''
    @torch.no_grad()
    @keep_signature
    def forward_one_step(self, x_in, sigma, **kwarg):
        if self.is_kdiff:
            x_noisy = self.p.x + self.p.noise * sigma[0]
        else:
            alphas_cumprod = self.p.sd_model.alphas_cumprod
            sqrt_alpha_cumprod = torch.sqrt(alphas_cumprod[self.timesteps[self.t_enc-self.p.current_step]])
            sqrt_one_minus_alpha_cumprod = torch.sqrt(1 - alphas_cumprod[self.timesteps[self.t_enc-self.p.current_step]])
            x_noisy = self.p.x*sqrt_alpha_cumprod + self.p.noise * sqrt_one_minus_alpha_cumprod

        self.cosine_factor = 0.5 * (1 + torch.cos(torch.pi *torch.tensor(((self.p.current_step + 1) / (self.t_enc+1)))))

        c1 = self.cosine_factor ** self.p.cosine_scale_1

        x_in = x_in*(1 - c1) + x_noisy * c1

        if self.p.random_jitter:
            jitter_range = self.jitter_range
        else:
            jitter_range = 0
        x_in_ = F.pad(x_in,(jitter_range, jitter_range, jitter_range, jitter_range),'constant',value=0)
        _,_,H,W = x_in.shape

        self.sampler.model_wrap_cfg.inner_model.forward  = self.sample_one_step
        self.repeat_3 = False

        x_out = self.sampler.model_wrap_cfg.forward_ori(x_in_,sigma, **kwarg)
        self.sampler.model_wrap_cfg.inner_model.forward = self.sampler_forward
        x_out = x_out[:,:,jitter_range:jitter_range+H,jitter_range:jitter_range+W]

        return x_out


    @torch.no_grad()
    @keep_signature
    def sample_one_step(self, x_in, sigma, cond):
        assert LatentDiffusion.apply_model
        def repeat_func_1(x_tile:Tensor, bboxes:List[CustomBBox]) -> Tensor:
            sigma_tile = self.repeat_tensor(sigma, len(bboxes))
            cond_tile = self.repeat_cond_dict(cond, bboxes)
            return self.sampler_forward(x_tile, sigma_tile, cond=cond_tile)

        def repeat_func_2(x_tile:Tensor, bboxes:List[CustomBBox]) -> Tuple[Tensor, Tensor]:
            n_rep = len(bboxes)
            ts_tile = self.repeat_tensor(sigma, n_rep)
            if isinstance(cond, dict):   # FIXME: when will enter this branch?
                cond_tile = self.repeat_cond_dict(cond, bboxes)
            else:
                cond_tile = self.repeat_tensor(cond, n_rep)
            return self.sampler_forward(x_tile, ts_tile, cond=cond_tile)

        def repeat_func_3(x_tile:Tensor, bboxes:List[CustomBBox]):
            sigma_in_tile = sigma.repeat(len(bboxes))
            cond_out = self.repeat_cond_dict(cond, bboxes)
            x_tile_out = shared.sd_model.apply_model(x_tile, sigma_in_tile, cond=cond_out)
            return x_tile_out

        if self.repeat_3:
            repeat_func = repeat_func_3
            self.repeat_3 = False
        elif self.is_kdiff:
            repeat_func = repeat_func_1
        else:
            repeat_func = repeat_func_2
        N,_,_,_ = x_in.shape


        self.x_buffer = torch.zeros_like(x_in)
        self.weights = torch.zeros_like(x_in)

        for batch_id, bboxes in enumerate(self.batched_bboxes):
            if state.interrupted: return x_in
            x_tile = torch.cat([x_in[bbox.slicer] for bbox in bboxes], dim=0)
            x_tile_out = repeat_func(x_tile, bboxes)
            # de-batching
            for i, bbox in enumerate(bboxes):
                self.x_buffer[bbox.slicer] += x_tile_out[i*N:(i+1)*N, :, :, :]
                self.weights[bbox.slicer] += 1
        self.weights = torch.where(self.weights == 0, torch.tensor(1), self.weights) #Prevent NaN from appearing in random_jitter mode

        x_local = self.x_buffer/self.weights

        self.x_buffer = torch.zeros_like(self.x_buffer) 
        self.weights = torch.zeros_like(self.weights)

        std_, mean_ = x_in.std(), x_in.mean()
        c3 = 0.99 * self.cosine_factor ** self.p.cosine_scale_3 + 1e-2
        if self.p.gaussian_filter:
            x_in_g = self.gaussian_filter(x_in, kernel_size=(2*self.p.current_scale_num-1), sigma=self.sig*c3)
            x_in_g = (x_in_g - x_in_g.mean()) / x_in_g.std() * std_ + mean_

        if not hasattr(self.p.sd_model, 'apply_model_ori'):
            self.p.sd_model.apply_model_ori = self.p.sd_model.apply_model
        self.p.sd_model.apply_model = self.apply_model_hijack
        x_global = torch.zeros_like(x_local)
        jitter_range = self.jitter_range
        end = x_global.shape[3]-jitter_range

        for batch_id, bboxes in enumerate(self.global_batched_bboxes):
            for bbox in bboxes:
                w,h = bbox
                # self.x_out_list = []
                # self.x_out_idx = -1
                # self.flag = 1
                x_global_i0 = self.sampler_forward(x_in_g[:,:,h+jitter_range:end:self.p.current_scale_num,w+jitter_range:end:self.p.current_scale_num],sigma,cond = cond)
                # self.flag = 0
                x_global_i1 = self.sampler_forward(x_in[:,:,h+jitter_range:end:self.p.current_scale_num,w+jitter_range:end:self.p.current_scale_num],sigma,cond = cond) #NOTE According to the original execution process, it would be very strange to use the predicted noise of gaussian latents to predict the denoised data in non Gaussian latents. Why?
                self.x_buffer[:,:,h+jitter_range:end:self.p.current_scale_num,w+jitter_range:end:self.p.current_scale_num] +=  (x_global_i0 + x_global_i1)/2
                self.weights[:,:,h+jitter_range:end:self.p.current_scale_num,w+jitter_range:end:self.p.current_scale_num] += 1

        self.p.sd_model.apply_model = self.p.sd_model.apply_model_ori
        self.weights = torch.where(self.weights == 0, torch.tensor(1), self.weights) #Prevent NaN from appearing in random_jitter mode

        x_global = self.x_buffer/self.weights
        c2 = self.cosine_factor**self.p.cosine_scale_2
        self.x_buffer= x_local*(1-c2)+ x_global*c2

        return self.x_buffer



    @torch.no_grad()
    @keep_signature
    def apply_model_hijack(self, x_in:Tensor, t_in:Tensor, cond:CondDict):
        assert LatentDiffusion.apply_model

        x_tile_out = self.p.sd_model.apply_model_ori(x_in,t_in,cond)
        return x_tile_out
        # NOTE: Using Gaussian Latent to Predict Noise on the Original Latent
        # if self.flag == 1:
        #     x_tile_out = self.p.sd_model.apply_model_ori(x_in,t_in,cond)
        #     self.x_out_list.append(x_tile_out)
        #     return x_tile_out
        # else:
        #     self.x_out_idx += 1
        #     return self.x_out_list[self.x_out_idx]


    def get_noise(self, x_in:Tensor, sigma_in:Tensor, cond_in:Dict[str, Tensor], step:int) -> Tensor:
        # NOTE: The following code is analytically wrong but aesthetically beautiful
        cond_in_original = cond_in.copy()

        self.repeat_3 = True

        return self.sample_one_step_local(x_in, sigma_in, cond_in_original)
