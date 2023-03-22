# Tiled Diffusion with Tiled VAE

English｜[中文](README_CN.md)

Introducing revolutionary **ultra-large image generation** via Tiled Diffusion & VAE!

- My optimized reimplementation of two state-of-the-art training-free compositional methods: [Mixture of Diffusers](https://github.com/albarji/mixture-of-diffusers) and [MultiDiffusion](https://multidiffusion.github.io)
- My original Tiled VAE algorithm, which is seam-free and **extremely powerful** in VRAM saving.


## Important Update on 2023.3.22
- **Fixed Logic erros in region prompt control**. Please update as previous versions cannot correctly deal with more than 1 custom region, it wrongly uses positive prompt as the negative condition so cannot draw meaningful objects.

## Important Update on 2023.3.20
- **Unprecedented regional prompt control with my super-convenient new UI**![region](https://github.com/pkuliyi2015/multidiffusion-img-demo/blob/master/Region.png?raw=true)
- **Add a new SOTA method: Mixture of Diffusers. Less seams & better quality!**
- Will update this README later. **If you find my extension powerful and are willing to help me improve this README, please feel free to PR!** 

## Tiled Diffusion

****

The following readme is out-of-date. Will update soon.

**Fast ultra-large images refinement (img2img)**

- **MultiDiffusion is especially good at adding details to upscaled images.**
  - **Faster than highres.fix** with proper params (see [here](https://github.com/pkuliyi2015/multidiffusion-upscaler-for-automatic1111/issues/3#issuecomment-1474541273) for performant params).
  - Much finer results than SD Upscaler & Ultimate SD Upscaler
- **How to use:**
  - **The checkpoint is crucial**. 
    - MultiDiffusion works very similar to highres.fix, so it highly relies on your checkpoint.
    - A checkpoint that good at drawing details (e.g., trained on high resolution images) can add amazing details to your image.
    - Some friends have found that using a **full checkpoint** instead of a pruned one yields much finer results.
  - **Don't include any concrete objects in your positive prompts.**  Otherwise the results get ruined.
      - Just use something like "highres, masterpiece, best quality, ultra-detailed unity 8k wallpaper, extremely clear".
  - You don't need too large tile size, large overlap and many denoising steps, or it can be slow.
    - Latent tile size=64 - 96, Overlap=32 - 48, and steps=20 - 25 are recommended. **If you find seams, please increase overlap.**
  - **CFG scale can significantly affect the details**, together with a proper sampler.
    - A large CFG scale (e.g., 14) gives you much more details. For samplers,I personally prefer Euler a and DPM++ SDE Karras.
  - You can control how much you want to change the original image with **denoising strength from 0.1 - 0.6**.
  - If your results are still not as satisfying as mine, [see our discussions here.](https://github.com/pkuliyi2015/multidiffusion-upscaler-for-automatic1111/issues/3)

- Example: 1024 * 800 -> 4096 * 3200 image, denoise=0.4, steps=20, Sampler=DPM++ SDE Karras, Upscaler=RealESRGAN++, Negative Prompts=EasyNegative
  - Before: 
  - ![lowres](https://github.com/pkuliyi2015/multidiffusion-img-demo/blob/master/lowres.jpg?raw=true)
  - After: 4x upscale.
  - 1min12s on NVIDIA Testla V100. (If 2x, it completes in 10s)
  - ![highres](https://github.com/pkuliyi2015/multidiffusion-img-demo/blob/master/highres.jpeg?raw=true)

****

**Wide Image Generation (txt2img)**

- txt2img panorama generation, as mentioned in MultiDiffusion.
  - All tiles share the same prompt currently.
  - **Please use simple positive prompts to get good results**, otherwise the result will be pool.
  - We are urgently working on the rectangular & fine-grained prompt control.

- Example - masterpiece, best quality, highres, city skyline, night.
- ![panorama](https://github.com/pkuliyi2015/multidiffusion-img-demo/blob/master/city_panorama.jpeg?raw=true)

****

**It can cooperate with ControlNet** to produce wide images with control.

- You cannot use complex positive prompts currently. However, you can use ControlNet.
- Canny edge seems to be the best as it provides sufficient local controls.
- Example: 22020 x 1080 ultra-wide image conversion 
  - Masterpiece, best quality, highres, ultra-detailed 8k unity wallpaper, bird's-eye view, trees, ancient architectures, stones, farms, crowd, pedestrians
  - Before: [click for the raw image](https://github.com/pkuliyi2015/multidiffusion-img-demo/blob/master/ancient_city_origin.jpeg)
  - ![ancient city origin](https://github.com/pkuliyi2015/multidiffusion-img-demo/blob/master/ancient_city_origin.jpeg?raw=true)
  - After: [click for the raw image](https://github.com/pkuliyi2015/multidiffusion-img-demo/blob/master/ancient_city.jpeg)
  - ![ancient city](https://github.com/pkuliyi2015/multidiffusion-img-demo/blob/master/ancient_city.jpeg?raw=true)
- Example: 2560 * 1280 large image drawing
  - ControlNet canny edge
  - ![Your Name](https://github.com/pkuliyi2015/multidiffusion-img-demo/blob/master/yourname_canny.jpeg?raw=true)
  - ![yourname](https://github.com/pkuliyi2015/multidiffusion-img-demo/blob/master/yourname.jpeg?raw=true)

****

### Advantages

- Draw super large resolution (2k~8k) image in both txt2img and img2img
- Seamless output without any post-processing

### Drawbacks

- We haven't optimized it much, so it can be **slow especially for very large images** (8k) and with ControlNet.
- **Prompt control is weak.** It will produce repeated patterns with strong positive prompts, and the result may not be usable.
- The gradient calculation is not compatible with this hack. It will break any backward() or torch.autograd.grad() that passes UNet.

### How it works (so simple!)

1. The latent image is split into tiles.
2. The tiles are denoised by the original sampler for one time step.
3. The tiles are added together but divided by how many times each pixel is added.
4. Repeat 2-3 until all timesteps are completed.

****

## Tiled VAE

**This script is currently production-ready**

The `vae_optimize.py` script is a wild hack that splits the image into tiles, encodes each tile separately, and merges the result back together. This process allows the VAE to work with giant images on limited VRAM (~10 GB for 8K images!). 

Remove --lowvram and --medvram to enjoy!

### Advantages

- The tiled VAE work with giant images on limited VRAM (~12 GB for 8K images!)
- Unlike [my friend's implementation](https://github.com/Kahsolt/stable-diffusion-webui-vae-tile-infer) and the HuggingFace diffuser's VAE tiling options that averages the tile borders, this VAE tiling removed attention blocks and use padding tricks.  The decoding results are mathematically identical to that of not tiling, i.e., **it will not produce seams at all.**
- The script is extremely optimized with tons of tricks. Cannot be faster!

### Drawbacks

- NaNs occassionally appear.  We are figuring out the root cause and trying to fix.
- Similarly, the gradient calculation is not compatible with this hack. It will break any backward() or torch.autograd.grad() that passes VAE.

### How it works

1. The image is split into tiles and padded with 11/32 pixels' in decoder/encoder.
2. When Fast Mode is disabled:
   1. The original VAE forward is decomposed into a task queue and a task worker, which start to process each tile.
   2. When GroupNorm is needed, it suspends, stores current GroupNorm mean and var, send everything to RAM, and turns to the next tile.
   3. After all GroupNorm mean and var parameters are summarized, it applies group norm to tiles and continues. 
   4. A zigzag execution order is used to reduce unnecessary data transfer.

3. When Fast Mode is enabled:
   1. The original input is downsampled and passed to a separate task queue.
   2. Its group norm parameters are recorded and used by all tiles' task queues.
   3. Each tile is separately processed without any RAM-VRAM data transfer.

4. After all tiles are processed, tiles are written to a result buffer and returned.

****

## Installation

- Open Automatic1111 WebUI -> Click Tab "Extensions" -> Click Tab "Install from URL" -> type in the link of this repo -> Click "Install" 
- ![installation](https://github.com/pkuliyi2015/multidiffusion-img-demo/blob/master/installation.png?raw=true)
- After restart your WebUI, you shall see the following two tabs:
- ![Tab](https://github.com/pkuliyi2015/multidiffusion-img-demo/blob/master/Tab.png?raw=true)

### MultiDiffusion Params

- Latent tile width & height: Basically, MultiDiffusion draws images tile by tile and each tile is a rectangle. Hence, this controls how large is the latent rectangle, each is 1/8 size of the actual image. Shouldn't be too large or too small (normally 64-128 is OK. but you can try other values.)
- Latent tile overlap: MultiDiffusion uses overlapping to prevent seams and fuses two latent images. So this controls how long should two tiles be overlapped at one side. The larger this value is, the slower the process, but the result will contain fewer seams  and be more natural.
- Latent tile batch size: allow UNet to process tiles in a batched manner. Larger values can speed up the UNet at the cost of more VRAM.

### Tiled VAE param

- **Move to GPU**: when you are running under --lowvram or medvram, this option will help to move the VAE to GPU temporarily and move it back later. It needs several seconds.
- **The two tile size params** control how large the tile should be we split for VAE encoder and decoder.
  - Basically, larger size brings faster speed at the cost of more VRAM usage. We will dynamicly shrink the size to make it faster.
  - You don't need to change the params at the first time of using. It will recommend a set of parameters based on hand-crafted rules. However, the recommended params may not be good to fit your device. 
  - Please adjust according to the GPU used in the console output. If you have more VRAM, turn it larger, or vice versus.
- **Fast Decoder**: We use a small latent image to estimate the decoder params and then comput very fast. By default, it is enabled. Not recommend to disable it. If you disable it, a large amount of CPU RAM and time will be consumed.
- **Fast Encoder**: We use a small image to estimate the encoder params; However, this is not accurate when your tile is very small and is further compressed by the encoder. Hence, it may do harm to your image's quality, especially colors.
- **Encoder Color Fix**: To fix the above problem, we provide a semi-fast mode that only estimate the params before downsampling. When you enable this, the slowest steps will be done in fast mode, and the remaining steps will run in legacy mode. Only enable this when you see visible color artifacts in pictures.

**Enjoy!**

****

## Current Progress

- Local prompt control is about to complete. 
- Automatic regional prompting is in consideration.
- Video translation via MultiDiffusion frame interpolation still need proof-of-concept.

****

## License

These scripts are licensed under the MIT License. If you find them useful, please give me a star.

Thank you!

