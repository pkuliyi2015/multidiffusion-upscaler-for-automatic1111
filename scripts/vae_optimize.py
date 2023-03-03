# ------------------------------------------------------------------------
#
#   Ultimate VAE Tile Optimization
#
#   Introducing a revolutionary new optimization designed to make
#   the VAE work with giant images on limited VRAM!
#   Say goodbye to the frustration of OOM and hello to seamless output!
#
# ------------------------------------------------------------------------
#
#   This script is a wild hack that splits the image into tiles,
#   encodes each tile separately, and merges the result back together.
#
#   Advantages:
#   - The VAE can now work with giant images on limited VRAM
#       (~10 GB for 8K images!)
#   - The merged output is completely seamless without any post-processing.
#
#   Drawbacks:
#   - Giant RAM needed. To store the intermediate results for a 4096x4096
#       images, you need 32 GB RAM it consumes ~20GB); for 8192x8192
#       you need 128 GB RAM machine (it consumes ~100 GB)
#   - NaNs always appear in for 8k images when you use fp16 (half) VAE
#       You must use --no-half-vae to disable half VAE for that giant image.
#   - Slow speed. With default tile size, it takes around 50/200 seconds
#       to encode/decode a 4096x4096 image; and 200/900 seconds to encode/decode
#       a 8192x8192 image. (The speed is limited by both the GPU and the CPU.)
#   - The gradient calculation is not compatible with this hack. It
#       will break any backward() or torch.autograd.grad() that passes VAE.
#       (But you can still use the VAE to generate training data.)
#
#   How it works:
#   1) The image is split into tiles of 1024*1024 pixels (in real size).
#       To ensure perfect results, each tile is padded with 256 pixels
#       on each side, so that conv2d can produce identical results to
#       the original image without splitting.
#   2) The original forward is decomposed into a task queue and a task worker.
#       - The task queue is a list of functions that will be executed in order.
#       - The task worker is a loop that executes the tasks in the queue.
#   3) The task queue is executed for each tile.
#       - Current tile is sent to GPU.
#       - local operations are directly executed.
#       - Group norm calculation is temporarily suspended until the mean
#           and var of all tiles are calculated.
#       - The residual is pre-calculated and stored and addded back later.
#       - When need to go to the next tile, the current tile is send to cpu.
#   4) After all tiles are processed, tiles are merged on cpu and return.
#
#   Enjoy!
#
#   @author: LI YI @ Nanyang Technological University - Singapore
#   @date: 2023-03-02
#   @license: MIT License
#
#   Please give me a star if you like this project!
#
# -------------------------------------------------------------------------
import modules.devices as devices
import modules.shared as shared
from modules.script_callbacks import on_before_image_saved, remove_callbacks_for_function
import math
import torch
import torch.nn.functional as F
from tqdm import tqdm
import modules.scripts as scripts
import gradio as gr
import types

# inplace version of silu


def inplace_nonlinearity(x):
    # Test: fix for Nans
    return F.silu(x, inplace=True)


def resblock2task(queue, block):
    """
    Turn a ResNetBlock into a sequence of tasks and append to the task queue

    @param queue: the target task queue
    @param block: ResNetBlock

    """
    if block.in_channels != block.out_channels:
        if block.use_conv_shortcut:
            queue.append(('store_res', block.conv_shortcut))
        else:
            queue.append(('store_res', block.nin_shortcut))
    else:
        queue.append(('store_res', lambda x: x))
    queue.append(('pre_norm', block.norm1))
    queue.append(('silu', inplace_nonlinearity))
    queue.append(('conv1', block.conv1))
    queue.append(('temb', lambda h, temb: h +
                 block.temb_proj(inplace_nonlinearity(temb))[:, :, None, None]))
    queue.append(('pre_norm', block.norm2))
    queue.append(('silu', inplace_nonlinearity))
    queue.append(('conv2', block.conv2))
    queue.append(['add_res', None])


def build_sampling(task_queue, net, is_decoder):
    """
    Build the sampling part of a task queue
    @param task_queue: the target task queue
    @param net: the network
    @param is_decoder: currently building decoder or encoder
    """
    block_ids = net.num_res_blocks
    if is_decoder:
        resblock2task(task_queue, net.mid.block_1)
        resblock2task(task_queue, net.mid.block_2)
        module = net.up
        block_ids += 1
    else:
        module = net.down
    sample_condition = 0 if is_decoder else (net.num_resolutions - 1)
    resolution_iter = range(net.num_resolutions) if not is_decoder else reversed(
        range(net.num_resolutions))
    for i_level in resolution_iter:
        for i_block in range(block_ids):
            resblock2task(task_queue, module[i_level].block[i_block])
        if i_level != sample_condition:
            if is_decoder:
                task_queue.append(('upsample', module[i_level].upsample))
            else:
                task_queue.append(('downsample', module[i_level].downsample))
    if not is_decoder:
        resblock2task(task_queue, net.mid.block_1)
        resblock2task(task_queue, net.mid.block_2)


def build_task_queue(net, is_decoder):
    """
    Build a single task queue for the encoder or decoder
    @param net: the VAE decoder or encoder network
    @param is_decoder: currently building decoder or encoder
    @return: the task queue
    """
    task_queue = []
    task_queue.append(('conv_in', net.conv_in))

    # construct the sampling part of the task queue
    # because encoder and decoder share the same architecture, we extract the sampling part
    build_sampling(task_queue, net, is_decoder)

    if not is_decoder or not net.give_pre_end:
        task_queue.append(('pre_norm', net.norm_out))
        task_queue.append(('silu', inplace_nonlinearity))
        task_queue.append(('conv_out', net.conv_out))
        if is_decoder and net.tanh_out:
            task_queue.append(('tanh', torch.tanh))

    return task_queue


def get_var_mean(input, num_groups, eps=1e-6):
    """
    Get mean and var for group norm
    """
    b, c = input.size(0), input.size(1)
    channel_in_group = int(c/num_groups)
    input_reshaped = input.contiguous().view(
        1, int(b * num_groups), channel_in_group, *input.size()[2:])
    var, mean = torch.var_mean(
        input_reshaped, dim=[0, 2, 3, 4], unbiased=False)
    return var, mean


def custom_group_norm(input, num_groups, mean, var, weight=None, bias=None, eps=1e-6):
    """
    Custom group norm with fixed mean and var

    @param input: input tensor
    @param num_groups: number of groups. by default, num_groups = 32
    @param mean: mean, must be pre-calculated by get_var_mean
    @param var: var, must be pre-calculated by get_var_mean
    @param weight: weight, should be fetched from the original group norm
    @param bias: bias, should be fetched from the original group norm
    @param eps: epsilon, by default, eps = 1e-6 to match the original group norm

    @return: normalized tensor
    """
    b, c = input.size(0), input.size(1)
    channel_in_group = int(c/num_groups)
    input_reshaped = input.contiguous().view(
        1, int(b * num_groups), channel_in_group, *input.size()[2:])

    out = F.batch_norm(input_reshaped, mean, var, weight=None, bias=None,
                       training=False, momentum=0, eps=eps)

    out = out.view(b, c, *input.size()[2:])

    # post affine transform
    if weight is not None:
        out *= weight.view(1, -1, 1, 1)
    if bias is not None:
        out += bias.view(1, -1, 1, 1)
    return out


def crop_valid_region(x, input_bbox, target_bbox, scale):
    """
    Crop the valid region from the tile
    @param x: input tile
    @param input_bbox: original input bounding box
    @param target_bbox: output bounding box
    @param scale: scale factor
    @return: cropped tile
    """
    padded_bbox = [math.ceil(i*scale) for i in input_bbox]
    margin = [target_bbox[i] - padded_bbox[i] for i in range(4)]
    return x[:, :, margin[2]:x.size(2)+margin[3], margin[0]:x.size(3)+margin[1]]


def get_recommend_encoder_tile_size():
    """
    Get the recommended encoder tile size
    """
    if torch.cuda.is_available():
        total_memory = torch.cuda.get_device_properties(
            devices.device).total_memory//1024//1024
        if total_memory > 16*1000:
            ENCODER_TILE_SIZE = 3072
        elif total_memory > 12*1000:
            ENCODER_TILE_SIZE = 2048
        elif total_memory > 8*1000:
            ENCODER_TILE_SIZE = 1536
        else:
            ENCODER_TILE_SIZE = 960
    else:
        ENCODER_TILE_SIZE = 512
    return ENCODER_TILE_SIZE


def get_recommend_decoder_tile_size():
    if torch.cuda.is_available():
        total_memory = torch.cuda.get_device_properties(
            devices.device).total_memory//1024//1024
        if total_memory > 30*1000:
            DECODER_TILE_SIZE = 256
        elif total_memory > 16*1000:
            DECODER_TILE_SIZE = 192
        elif total_memory > 12*1000:
            DECODER_TILE_SIZE = 128
        elif total_memory > 8*1000:
            DECODER_TILE_SIZE = 96
        else:
            DECODER_TILE_SIZE = 64
    else:
        DECODER_TILE_SIZE = 64
    return DECODER_TILE_SIZE


@torch.inference_mode()
def vae_tile_forward(self, z, tile_size, is_decoder):
    """
    Decode a latent vector z into an image in a tiled manner.
    @param z: latent vector
    @return: image
    """
    height, width = z.shape[2], z.shape[3]
    self.last_z_shape = z.shape
    device = z.device
    # split the input into tiles and build a task queue for each tile

    tile_input_bboxes = []
    tile_output_bboxes = []

    pad = 32
    # the decoder contains 33 conv2d, some of which is nin short cut (kernel size 1)
    # we only need to pad 32 pixels to avoid seams for decoder (which is divisible by 8)
    num_height_tiles = math.ceil((height - 2 * pad) / tile_size)
    num_width_tiles = math.ceil((width - 2 * pad) / tile_size)

    for i in range(num_height_tiles):
        for j in range(num_width_tiles):
            # bbox: [x1, x2, y1, y2]
            # the padding is is unnessary for image borders. So we directly start from (32, 32)
            input_bbox = [pad + j * tile_size, min(pad + (j + 1) * tile_size, width),
                          pad + i * tile_size, min(pad + (i + 1) * tile_size, height)]

            # if the output bbox is close to the image boundary, we extend it to the image boundary
            output_bbox = [input_bbox[0] if input_bbox[0] > pad else 0,
                           input_bbox[1] if input_bbox[1] < width -
                           pad else width,
                           input_bbox[2] if input_bbox[2] > pad else 0,
                           input_bbox[3] if input_bbox[3] < height - pad else height]

            # scale to get the final output bbox
            scale_factor = 8 if is_decoder else 1/8
            output_bbox = [math.ceil(x * scale_factor) for x in output_bbox]
            tile_output_bboxes.append(output_bbox)

            # indistinguishable expand the input bbox by pad pixels
            input_bbox = [max(0, input_bbox[0] - pad), min(width, input_bbox[1] + pad),
                          max(0, input_bbox[2] - pad), min(height, input_bbox[3] + pad)]

            tile_input_bboxes.append(input_bbox)

    # Prepare tiles by split the input latents
    tiles = []
    for input_bbox in tile_input_bboxes:
        tile = z[:, :, input_bbox[2]:input_bbox[3],
                 input_bbox[0]:input_bbox[1]].cpu()
        tiles.append(tile)
        # DEBUG:
        # print('tile shape: ', tile.shape, 'input bbox: ', input_bbox, 'output bbox: ', tile_output_bboxes[len(completed_tiles)])
    num_tiles = len(tiles)
    completed = [None] * num_tiles
    num_completed = 0
    # Free memory of input latent tensor
    del z
    devices.torch_gc()

    # Build task queues
    task_queues = [build_task_queue(self, is_decoder).copy()
                   for _ in range(num_tiles)]
    # Task queue execution
    pbar = tqdm(total=num_tiles * len(task_queues[0]), desc='Executing Tiled VAE ' +
                ('Decoder' if is_decoder else 'Encoder') + ' Task Queue: ')
    # execute the task back and forth when switch tiles so that we always
    # keep one tile on the GPU to reduce unnecessary data transfer
    forward = True
    while True:
        group_norm_var_list = []
        group_norm_mean_list = []
        group_norm_pixel_list = []
        group_norm_tmp_weight = None
        group_norm_tmp_bias = None
        for i in range(num_tiles) if forward else reversed(range(num_tiles)):
            tile = tiles[i].to(device)
            input_bbox = tile_input_bboxes[i]
            task_queue = task_queues[i]
            while len(task_queue) > 0:
                # DEBUG: current task
                # print('Running task: ', task_queue[0][0], ' on tile ', i, '/', num_tiles, ' with shape ', tile.shape)
                task = task_queue.pop(0)
                if task[0] == 'pre_norm':
                    var, mean = get_var_mean(tile, 32)
                    # For giant images, the variance can be larger than max float16
                    # In this case we create a copy to float32
                    if var.dtype == torch.float16 and var.isinf().any():
                        fp32_tile = tile.float()
                        var, mean = get_var_mean(fp32_tile, 32)
                    # ============= DEBUG: test for infinite =============
                    # if torch.isinf(var).any():
                    #    print('var: ', var)
                    # if torch.isinf(mean).any():
                    #    print('mean: ', mean)
                    # if torch.isinf(var).any() or torch.isinf(mean).any():
                    #    print('Running task: ', task[0], ' on tile ', i, '/', num_tiles, ' with shape ', tile.shape)
                    #    print('pixel: ', (tile.shape[2]), 'x', (tile.shape[3]))
                    # ====================================================
                    group_norm_var_list.append(var)
                    group_norm_mean_list.append(mean)
                    group_norm_pixel_list.append(tile.shape[2]*tile.shape[3])
                    if hasattr(task[1], 'weight'):
                        group_norm_tmp_weight = task[1].weight
                        group_norm_tmp_bias = task[1].bias
                    else:
                        group_norm_tmp_weight = None
                        group_norm_tmp_bias = None
                    break
                elif task[0] == 'store_res':
                    task_id = 0
                    while task_queue[task_id][0] != 'add_res':
                        task_id += 1
                    task_queue[task_id][1] = task[1](tile).cpu()
                elif task[0] == 'add_res':
                    tile = tile + task[1].to(device)
                elif task[0] == 'temb':
                    pass
                else:
                    tile = task[1](tile)
                pbar.update(1)
            # check for NaNs in the tile.
            # If there are NaNs, we abort the process to save user's time
            try:
                devices.test_for_nans(tile, "vae")
            except:
                print(
                    "Detected NaNs in the VAE output. Please use VAE with proper weights.")
                raise
            if i == num_tiles - 1 and forward:
                forward = False
                tiles[i] = tile
            elif i == 0 and not forward:
                forward = True
                tiles[i] = tile
            else:
                tiles[i] = tile.cpu()
                del tile
                devices.torch_gc()
            if len(task_queue) == 0:
                completed[i] = tiles[i].cpu()
                num_completed += 1

        if num_completed == num_tiles:
            break
        # aggregate the mean and var for the group norm calculation
        if len(group_norm_mean_list) > 0:
            group_norm_var = torch.vstack(group_norm_var_list)
            group_norm_mean = torch.vstack(group_norm_mean_list)
            max_value = max(group_norm_pixel_list)
            group_norm_pixels = torch.tensor(
                group_norm_pixel_list, dtype=torch.float32, device=device)/max_value
            sum_group_norm_pixels = torch.sum(group_norm_pixels)
            group_norm_pixels = group_norm_pixels.unsqueeze(
                1) / sum_group_norm_pixels
            group_norm_var = torch.sum(
                group_norm_var * group_norm_pixels, dim=0)
            group_norm_mean = torch.sum(
                group_norm_mean * group_norm_pixels, dim=0)
            current_group_norm_weight = group_norm_tmp_weight
            current_group_norm_bias = group_norm_tmp_bias
            # insert the group norm task to the head of each task queue
            for i in range(num_tiles):
                task_queue = task_queues[i]
                task_queue.insert(0, ('apply_norm', lambda x: custom_group_norm(
                    x, 32, group_norm_mean, group_norm_var, current_group_norm_weight, current_group_norm_bias)))
            # cleanup group norm parameters
            group_norm_mean_list = []
            group_norm_var_list = []
            group_norm_pixel_list = []
            group_norm_tmp_weight = None
            group_norm_tmp_bias = None

    pbar.close()
    # crop tiles
    for i in range(num_tiles):
        completed[i] = crop_valid_region(
            completed[i], tile_input_bboxes[i], tile_output_bboxes[i], 8 if is_decoder else 1/8)
    # directly merge tiles on the cpu
    result = []
    for i in range(num_height_tiles):
        tile_w = []
        for j in range(num_width_tiles):
            tile_w.append(completed[i*num_width_tiles+j])
        result.append(torch.cat(tile_w, dim=3))
    result = torch.cat(result, dim=2)
    # if the model is a encoder, we send it back to the gpu
    if not is_decoder:
        result = result.to(device)
    if torch.cuda.is_available():
        print("Max memory allocated: ",
              torch.cuda.max_memory_allocated(device)/1024/1024, "MB")
        torch.cuda.reset_peak_memory_stats(device)
    return result


class Script(scripts.Script):

    def title(self):
        return "VAE Tiling"

    def show(self, is_img2img):
        return scripts.AlwaysVisible

    def ui(self, is_img2img):
        ctrls = []
        recommend_encoder_tile_size = get_recommend_encoder_tile_size()
        recommend_decoder_tile_size = get_recommend_decoder_tile_size()
        with gr.Group():
            with gr.Accordion('VAE Tiling', open=True):
                with gr.Row():
                    enabled = gr.Checkbox(label='Enable', value=True)
                    reset = gr.Button(value="Reset Tile Size")
                info = gr.HTML(
                    "<p style=\"margin-bottom:1.0em\">Please use smaller tile size when see CUDA error: out of memory.</p>")
                encoder_tile_size = gr.Slider(label='Encoder Tile Size', minimum=256,
                                              maximum=3088, step=16, value=recommend_encoder_tile_size, interactive=True)
                decoder_tile_size = gr.Slider(label='Decoder Tile Size', minimum=48,
                                              maximum=256, step=16, value=recommend_decoder_tile_size, interactive=True)

                def reset_tile_size():
                    return get_recommend_encoder_tile_size(), get_recommend_decoder_tile_size()
                reset.click(fn=reset_tile_size, inputs=[], outputs=[
                            encoder_tile_size, decoder_tile_size])
                ctrls.extend([enabled, encoder_tile_size, decoder_tile_size])
        return ctrls

    def process(self, p, enabled, encoder_tile_size, decoder_tile_size):
        if not enabled or p is None:
            return p
        if devices.device == torch.device('cpu'):
            print("VAE Tiling is not supported on CPU")
            return p
        image = p.init_images[0]
        width = image.width
        height = image.height
        # If the image is smaller than tile size + two sides' padding
        # we don't need to tile the VAE
        pad = 32
        hijack_encoder = width > encoder_tile_size + \
            2 * pad or height > encoder_tile_size + 2 * pad
        hijack_decoder = math.ceil(width/8) > decoder_tile_size + \
            2 * pad or math.ceil(height/8) > decoder_tile_size + 2 * pad
        if not hijack_encoder and not hijack_decoder:
            print("VAE Tiling is not needed for small images")
            return p
        print("VAE Tiling is hooked.")
        vae = p.sd_model.first_stage_model
        origin_encoder = vae.encoder
        origin_decoder = vae.decoder
        if hijack_encoder:
            def new_encoder_forward(self, x): return vae_tile_forward(
                self, x, encoder_tile_size, False)
            vae.encoder.forward = types.MethodType(
                new_encoder_forward, vae.encoder)
        if hijack_decoder:
            def new_decoder_forward(self, x): return vae_tile_forward(
                self, x, decoder_tile_size, True)
            vae.decoder.forward = types.MethodType(
                new_decoder_forward, vae.decoder)

        def recover(_):
            vae.encoder = origin_encoder
            vae.decoder = origin_decoder
            print("VAE Tiling is unhooked.")
            remove_callbacks_for_function(recover)

        on_before_image_saved(recover)
        return p
