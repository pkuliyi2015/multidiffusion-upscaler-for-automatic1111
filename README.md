# MultiDiffusion Upscaler

This repository contains an upscaler that redraws ultra-high-resolution images using advanced techniques inspired by Multidiffusion.

## VAE Tile Optimization

In addition to the upscaler, this repository also includes a revolutionary VAE Tile Optimization technique designed to make the VAE work with giant images on limited VRAM. With this technique, you can say goodbye to the frustration of OOM and hello to seamless output! This script is a wild hack that splits the image into tiles, encodes each tile separately, and merges the result back together. Some of its advantages include:

The VAE can now work with giant images on limited VRAM (~10 GB for 8K images!)
The merged output is completely seamless without any post-processing.
However, there are also some drawbacks, such as the need for giant RAM and slower speeds. Please refer to the documentation for more information on using this technique.

## Upscaler

The upscaler in this repository uses the advanced techniques inspired by Multidiffusion to redraw ultra-high-resolution images. With the help of the VAE Tile Optimization, you can redraw images that were previously impossible to process. Please refer to the documentation for more information on using this upscaler.

## License

This project is licensed under the MIT License. Please give me stars if you like the project!
