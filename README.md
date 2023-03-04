# MultiDiffusion with Tiled VAE

- This repository contains two scripts that enable the processing of large images using the MultiDiffusion and Tiled VAE
- The scripts are designed to work with the Automatic1111 WebUI and do not require any training of new models.

## MultiDiffusion

The `multidiffusion.py` script hooks into the original sampler and decomposes the latent image into tiles. The tiles are denoised and then merged back together using weighted average. This process allows for super large resolutions (2k~8k) for both txt2img and img2img. The merged output is completely seamless without any post-processing.

### Advantages

- Super large resolutions (2k~8k) for both txt2img and img2img
- Seamless output without any post-processing
- No need to train a new model
- You can control the text prompt for each tile

### Drawbacks

- Depending on your parameter settings, the process can be very slow, especially when overlap is relatively large.
- The gradient calculation is not compatible with this hack. It will break any backward() or torch.autograd.grad() that passes UNet.

### How it works

1. The latent image is split into tiles.
2. The tiles are denoised by the original sampler.
3. The tiles are added together, but divided by how many times each pixel is added.

## Ultimate VAE Tile Optimization

The `ultimate_vae_tile_optimization.py` script is a wild hack that splits the image into tiles, encodes each tile separately, and merges the result back together. This process allows the VAE to work with giant images on limited VRAM (~10 GB for 8K images!). The merged output is completely seamless without any post-processing.

### Advantages

- The VAE can now work with giant images on limited VRAM (~10 GB for 8K images!)
- The merged output is completely seamless without any post-processing.

### Drawbacks

- Giant RAM needed to store the intermediate results for large images
- NaNs always appear in for 8k images when you use fp16 (half) VAE. You must use --no-half-vae to disable half VAE for that giant image.
- Slow speed. The speed is limited by both the GPU and the CPU.
- The gradient calculation is not compatible with this hack. It will break any backward() or torch.autograd.grad() that passes VAE.

### How it works

1. The image is split into tiles.
2. The original forward is decomposed into a task queue and a task worker.
3. The task queue is executed for each tile.
4. After all tiles are processed, tiles are merged on CPU and returned.

## Examples

Here are some examples of images that have been processed using these scripts:

![Example 1](https://chat.openai.com/img/example1.png) ![Example 2](https://chat.openai.com/img/example2.png) ![Example 3](https://chat.openai.com/img/example3.png)

## License

These scripts are licensed under the MIT License. If you find them useful, please give the author a star.
