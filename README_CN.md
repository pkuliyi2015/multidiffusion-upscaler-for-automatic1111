# WebUI的Demofusion插件

[![CC 署名-非商用-相同方式共享 4.0][cc-by-nc-sa-shield]][cc-by-nc-sa]

[English](README.md) | 中文

原项目地址https://github.com/pkuliyi2015/multidiffusion-upscaler-for-automatic1111

这个fork基于原项目的思路，为stable diffusion webui增添了基于k-diffusion实现的demofusion插件，使用方式与原项目相同。

注意，在使用过程中：

- 不要同时开启tilediffusion和demofusion
- 写实的画面会更适合demofusion
- 需要保持较高的denoising强度才可以得到较好的图片
- img2img模式下请尽可能用准确的text描述你的图片，如果图片本身就是txt2img生成的，建议用原来的随机种子、text以及生图模型
- 与原项目相同，兼容stablesr、controlnet，以及noise inversion

如果你喜欢这个项目，请给作者一个 star！⭐

 [![ko-fi](https://ko-fi.com/img/githubbutton_sm.svg)](https://ko-fi.com/pkuliyi2015)


