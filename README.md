# Tiled Diffusion & VAE extension for sd-webui

[![CC BY-NC-SA 4.0][cc-by-nc-sa-shield]][cc-by-nc-sa]

This extension is licensed under [CC BY-NC-SA](https://creativecommons.org/licenses/by-nc-sa/4.0/), everyone is FREE of charge to access, use, modify and redistribute with the same license.  
**You cannot use versions after AOE 2023.3.28 for commercial sales (only refers to code of this repo, the derived artworks are NOT restricted).**

由于部分无良商家销售WebUI，捆绑本插件做卖点收取智商税，本仓库的许可证已修改为 [CC BY-NC-SA](https://creativecommons.org/licenses/by-nc-sa/4.0/)，任何人都可以自由获取、使用、修改、以相同协议重分发本插件。  
**自许可证修改之日(AOE 2023.3.28)起，之后的版本禁止用于商业贩售 (不可贩售本仓库代码，但衍生的艺术创作内容物不受此限制)。**

If you like the project, please give me a star! ⭐

[![ko-fi](https://ko-fi.com/img/githubbutton_sm.svg)](https://ko-fi.com/pkuliyi2015)

****


The extension helps you to **generate or upscale large images (≥2K) with limited VRAM (≤6GB)** via the following techniques:

- Reproduced SOTA Tiled Diffusion methods
  - [Mixture of Diffusers](https://github.com/albarji/mixture-of-diffusers)
  - [MultiDiffusion](https://multidiffusion.github.io)
  - [Demofusion](https://github.com/PRIS-CV/DemoFusion)
- Our original Tiled VAE method
- My original Tiled Noise Inversion method


### Features

- Core
  - [x] [Tiled VAE](#tiled-vae)
  - [x] [Tiled Diffusion: txt2img generation for ultra-large image](#tiled-diff-txt2img)
  - [x] [Tiled Diffusion: img2img upscaling for image detail enhancement](#tiled-diff-img2img)
  - [x] [Regional Prompt Control](#region-prompt-control)
  - [x] [Tiled Noise Inversion](#tiled-noise-inversion)
- Advanced
  - [x] [ControlNet support]()
  - [x] [StableSR support](https://github.com/pkuliyi2015/sd-webui-stablesr)
  - [x] [SDXL support](experimental)
  - [x] [Demofusion support]()

👉 在 [wiki](https://github.com/pkuliyi2015/multidiffusion-upscaler-for-automatic1111/wiki) 页面查看详细的文档和样例，以及由 [@PotatoBananaApple](https://github.com/pkuliyi2015/multidiffusion-upscaler-for-automatic1111/discussions/120) 制作的 [快速入门教程](https://civitai.com/models/34726)
👉 Find detailed documentation & examples at our [wiki](https://github.com/pkuliyi2015/multidiffusion-upscaler-for-automatic1111/wiki), and quickstart [Tutorial](https://civitai.com/models/34726) by [@PotatoBananaApple](https://github.com/pkuliyi2015/multidiffusion-upscaler-for-automatic1111/discussions/120) 🎉


### Examples

⚪ Txt2img: generating ultra-large images

`prompt: masterpiece, best quality, highres, city skyline, night.`

![panorama](https://github.com/pkuliyi2015/multidiffusion-img-demo/blob/master/city_panorama.jpeg?raw=true)

⚪ Img2img: upcaling for detail enhancement

| original | x4 upscale |
| :-: | :-: |
| ![lowres](https://github.com/pkuliyi2015/multidiffusion-img-demo/blob/master/lowres.jpg?raw=true) | ![highres](https://github.com/pkuliyi2015/multidiffusion-img-demo/blob/master/highres.jpeg?raw=true) |

⚪ Regional Prompt Control

| region setting | output1 | output2 |
| :-: | :-: | :-: |
| ![MultiCharacterRegions](https://github.com/pkuliyi2015/multidiffusion-img-demo/blob/master/multicharacter.png?raw=true) | ![MultiCharacter](https://github.com/pkuliyi2015/multidiffusion-img-demo/blob/master/multicharacter.jpeg?raw=true) | ![MultiCharacter](https://github.com/pkuliyi2015/multidiffusion-img-demo/blob/master/multicharacter2.jpeg?raw=true) |
| ![FullBodyRegions](https://github.com/pkuliyi2015/multidiffusion-img-demo/blob/master/fullbody_regions.png?raw=true) | ![FullBody](https://github.com/pkuliyi2015/multidiffusion-img-demo/blob/master/fullbody.jpeg?raw=true) | ![FullBody2](https://github.com/pkuliyi2015/multidiffusion-img-demo/blob/master/fullbody2.jpeg?raw=true) |

⚪ ControlNet support

| original | with canny |
| :-: | :-: |
| ![Your Name](https://github.com/pkuliyi2015/multidiffusion-img-demo/blob/master/yourname_canny.jpeg?raw=true) | ![Your Name](https://github.com/pkuliyi2015/multidiffusion-img-demo/blob/master/yourname.jpeg?raw=true)

|  | 重绘 “清明上河图” |
| :-: | :-: |
| original | ![ancient city](https://github.com/pkuliyi2015/multidiffusion-img-demo/blob/master/ancient_city_origin_compressed.jpeg?raw=true) |
| processed  | ![ancient city](https://github.com/pkuliyi2015/multidiffusion-img-demo/blob/master/ancient_city_compressed.jpeg?raw=true) |

⚪ DemoFusion support

| original | x3 upscale |
| :-: | :-: |
| ![demo-example](https://github.com/Jaylen-Lee/image-demo/blob/main/example.png?raw=true) | ![demo-result](https://github.com/Jaylen-Lee/image-demo/blob/main/3.png?raw=true) |


### License

Great thanks to all the contributors! 🎉🎉🎉  
This work is licensed under [Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License][cc-by-nc-sa].

[![CC BY-NC-SA 4.0][cc-by-nc-sa-image]][cc-by-nc-sa]
[![CC BY-NC-SA 4.0][cc-by-nc-sa-shield]][cc-by-nc-sa]

[cc-by-nc-sa]: http://creativecommons.org/licenses/by-nc-sa/4.0/
[cc-by-nc-sa-image]: https://licensebuttons.net/l/by-nc-sa/4.0/88x31.png
[cc-by-nc-sa-shield]: https://img.shields.io/badge/License-CC%20BY--NC--SA%204.0-lightgrey.svg
