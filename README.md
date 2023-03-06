# MultiDiffusion with Tiled VAE

English｜[中文](README_CN.md)

This repository contains two scripts that enable **ultra large image generation**.

- The MultiDiffusion comes from existing work. Please refer to their paper and GitHub page [MultiDiffusion](https://github.com/pkuliyi2015/multidiffusion-upscaler-for-automatic1111/blob/docs/multidiffusion.github.io)
- The Tiled VAE is my original algorithm, which is **very powerful** in VRAM optimization.
  - With the algorithm, **you no longer need --lowvram or --medvram** once you have >=6G GPU.

## MultiDiffusion

****

**Fast ultra-large images refinement (img2img)**

- **MultiDiffusion is especially good at adding details to upscaled images.**
  - **Nearly 2x faster than highres.fix ** with proper params
  - Much finer results than SD Upscaler & Ultimate SD Upscaler
  - You can control how many details you want to add, using **denoising strength from 0.1 - 0.6**
- Example: 1024 * 800 -> 4096 * 3200 image, denoise=0.4, steps=20, Sampler=DPM++ SDE Karras, Upscaler=RealESRGAN++
  - Before: 
  - ![lowres](https://github.com/pkuliyi2015/multidiffusion-upscaler-for-automatic1111/blob/docs/imgs/lowres.jpg?raw=true)
  - After: 4x upscale.
  - 2min30s on NVIDIA Testla V100. 1:00 used by MultiDiffusion + 1:30s used by Tiled VAE). 2x only needs 20s
  - ![highres](https://github.com/pkuliyi2015/multidiffusion-upscaler-for-automatic1111/blob/docs/imgs/highres.jpeg?raw=true)

****

**Wide Image Generation (txt2img)**

- txt2img panorama generation, as mentioned in MultiDiffusion.
  - All tiles share the same prompt currently.
  - **Please use simple positive prompts to get good results**, otherwise the result will be pool.
  - We are urgently working on the rectangular & fine-grained prompt control.

- Example - mastepiece, best quality, highres, city skyline, night.
- ![panorama](https://github.com/pkuliyi2015/multidiffusion-upscaler-for-automatic1111/blob/docs/imgs/city_panorama.jpeg?raw=true)

****

**It can cooperate with ControlNet** to produce wide images with controll.

- You cannot use complex positive prompt currently. However, you can use ControlNet.
- Canny edge seems to be the best as it provides sufficient local controls.
- Example: 22020 x 1080 ultra wide image conversion 
  - Masterpiece, best quality, highres, ultra detailed 8k unity wallpaper, bird's-eye view, trees, ancient architectures, stones, farms, crowd, pedestrians
  - Before: [click for raw image](https://github.com/pkuliyi2015/multidiffusion-upscaler-for-automatic1111/blob/docs/imgs/ancient_city_origin.jpeg)
  - ![ancient city origin](https://github.com/pkuliyi2015/multidiffusion-upscaler-for-automatic1111/blob/docs/imgs/ancient_city_origin.jpeg?raw=true)
  - After: [click for raw image](https://github.com/pkuliyi2015/multidiffusion-upscaler-for-automatic1111/blob/docs/imgs/ancient_city.jpeg)
  - ![ancient city](https://github.com/pkuliyi2015/multidiffusion-upscaler-for-automatic1111/blob/docs/imgs/ancient_city.jpeg?raw=true)
- Example: 2560 * 1280 large image drawing with controlnet
  - ControlNet canny edge
  - ![Your Name](https://github.com/pkuliyi2015/multidiffusion-upscaler-for-automatic1111/blob/docs/imgs/yourname_canny.jpeg?raw=true)
  - ![yourname](https://github.com/pkuliyi2015/multidiffusion-upscaler-for-automatic1111/blob/docs/imgs/yourname.jpeg?raw=true)

****

### Advantages

- Draw super large resolution (2k~8k) image in both txt2img and img2img
- Seamless output without any post-processing

### Drawbacks

- We haven't optimize it much, so it can be **slow especially for very large images** (8k) and with controlnet.
- **Prompt control is weak.** It will produce repeated patterns with strong positive prompt, and the result may not be usable.
- The gradient calculation is not compatible with this hack. It will break any backward() or torch.autograd.grad() that passes UNet.

### How it works (so simple!)

1. The latent image is split into tiles.
2. The tiles are denoised by the original sampler for one time step.
3. The tiles are added together, but divided by how many times each pixel is added.
4. Repeat 2-3 untile all timesteps completed.

****

## Tiled VAE

**This script is currently production-ready**

The `vae_optimize.py` script is a wild hack that splits the image into tiles, encodes each tile separately, and merges the result back together. This process allows the VAE to work with giant images on limited VRAM (~10 GB for 8K images!). 

Remove --lowvram and --medvram to enjoy!

### Advantages

- The tiled VAE work with giant images on limited VRAM (~12 GB for 8K images!), eliminate your need for --lowvram and --medvram.
- Unlike [my friend's implementation](https://github.com/Kahsolt/stable-diffusion-webui-vae-tile-infer) and huggingface diffuser's VAE tiling options that averages the tile borders, this VAE tiling removed attention blocks and use padding tricks.  The decoding results mathematically identical to that of not tiling, i.e., **it will not produce seams at all.**
- The script is extremely optimized with tons of tricks. Cannot be faster!

### Drawbacks

- Large RAM (e.g., 20 GB for a 4096*4096 image and 50GB for a 8k image) is still needed to store the intermediate results. If you use --no-half-vae the usage doubles.
- For >=8k images NaNs ocassionally appear in.  The 840000 VAE weights effectively solve most problems . You may use --no-half-vae to disable half VAE for that giant image. **We are figure out the root cause and trying to fix**
- The speed is limited by both your GPU and your CPU. So if any of them is not good, the speed will be affected.
- Similarly, the gradient calculation is not compatible with this hack. It will break any backward() or torch.autograd.grad() that passes VAE.

### How it works

1. The image is split into tiles and equiped with 11/32 pixels' padding in decoder/encoder.

2. The original VAE forward is decomposed into a task queue and a task worker. 

   - The task queue start to execute for one tile.

   - When GroupNorm is needed, it suspends, stores current GroupNorm mean and var, send everything to cpu, and turns to the next tile.
   - After all GroupNorm mean and var parameters are summarized, it applies group norm to tiles and continue. 
   - A zigzag execution order is used to reduce unnecessary data transfer.

3. After all tiles are processed, tiles are written to a result buffer and returned.

****

## Installation

- Open Automatic1111 WebUI -> Click Tab "Extensions" -> Click Tab "Install from URL" -> type in the link of this repo -> Click "Install" 
- ![installation](https://github.com/pkuliyi2015/multidiffusion-upscaler-for-automatic1111/blob/docs/imgs/installation.png?raw=true)
- After restart your WebUI, you shall see the following two tabs:
- ![Tabs](https://github.com/pkuliyi2015/multidiffusion-upscaler-for-automatic1111/blob/docs/imgs/Tabs.png?raw=true)

### MultiDiffusion Params

- Latent tile width & height: Basically, multidiffusion draws images tile by tile and each tile is a rectangle. Hence, this controls how large is the latent rectangle, each is 1/8 size of the actual image. Shouldn't be too large or too small (normally 64-128 is OK. but you can try other values.)
- Latent tile overlap: MultiDiffusion uses overlapping to prevent seams and fuses two latents. So this controls how long should two tiles be overlapped at one side. The larger this value is, the slower the process, but the result will contain less seams  and more natural.
- Latent tile batch size: allow UNet to process tiles in a batched manner. Larger values can speed up the UNet at the cost of more VRAM.

### Tiled VAE param

- The two params control how large tile should we split the image for VAE encoder and decoder.
  - Larger size, faster speed, but more VRAM use.

- You don't need to change the params when first time to use it.
  - It will recommend a set of parameters to you based on hand-crafted rules.
  - However, the recommended params may not be good to fit your device. 
  - Please adjust according to the GPU used in the console output. If you have more VRAM, turn it larger, or vice versus.


**Enjoy!**

****

## Current Progress

- Local prompt control is in progress.
- Automatic prompting is in plan.
- Video translation via MultiDiffusion frame interpolation is under proof-of-concept.

****

## License

These scripts are licensed under the MIT License. If you find them useful, please give the author a star.

