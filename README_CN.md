# MultiDiffusion + Tiled VAE

[English](readme.md)|中文

本存储库包含两个脚本，使用 [MultiDiffusion](multidiffusion.github.io) 和Tiled VAE（**原创方法**）处理**超大图片**。

- 第一个是前人已有的优秀工作。请参考他们的论文和网页。
- 第二个是我的原创算法。尽管原理上很简单，但在节省显存上非常强力，**让6G显存也不用开启低显存模式**

## MultiDiffusion

****

**快速超大图像细化（img2img）**

- **MultiDiffusion 特别擅长于大图像添加细节。**
  - **速度比Highres快一倍**，只要参数调整合适
  - 比SD Upscaler和Ultimate Upscaler产生更多的细节
  - 你可以通过去噪强度0.1-0.6控制细节数量。越低越接近原图，越高细节越丰富
- **与SD Upscaler和Ultimate SD Upscaler一样，请不要使用复杂的正面prompt**
- 示例：
  - 参数：masterpiece, best quality, highres, extremely detailed, clear background, 去噪=0.4，步数=20，采样器=DPM++ SDE Karras，放大器=RealESRGAN, Tile size=96, Overlap=48, Tile batch size=8.
  - 处理前
  - ![lowres](https://github.com/pkuliyi2015/multidiffusion-upscaler-for-automatic1111/blob/docs/imgs/lowres.jpg?raw=true)
  - 处理后：4x放大，NVIDIA Tesla V100,
    - 总耗时 2分11秒，其中1分钟用于VAE编解码。
    - 如果是2x放大仅需25秒
  - ![highres](https://github.com/pkuliyi2015/multidiffusion-upscaler-for-automatic1111/blob/docs/imgs/highres.jpeg?raw=true)

****

**宽图像生成（txt2img）**

- **MultiDiffusion适合生成宽图像**，例如韩国偶像团体大合照（雾）
- txt2img 全景生成，与 MultiDiffusion 中提到的相同。
  - 目前所有局部区域共享相同的prompt。
  - **因此，请使用简单的正prompt以获得良好的结果**，否则结果将很差。
  - 我们正在加急处理矩形和细粒度prompt控制。
- 示例 - mastepiece, best quality, highres, city skyline, night

- ![panorama](https://github.com/pkuliyi2015/multidiffusion-upscaler-for-automatic1111/blob/docs/imgs/city_panorama.jpeg?raw=true)

****

**与 ControlNet 配合**，产生具有受控内容的宽图像。

- 目前，虽然您不能使用复杂的prompt，但可以使用 ControlNet 完全控制内容。
- Canny edge似乎是最好用的，因为它提供足够的局部控制。
- 示例：22020 x 1080 超宽图像转换 - 清明上河图
  - Masterpiece, best quality, highres, ultra detailed 8k unity wallpaper, bird's-eye view, trees, ancient architectures, stones, farms, crowd, pedestrians
  - 转换前：[单击下载原始图像](https://github.com/pkuliyi2015/multidiffusion-upscaler-for-automatic1111/blob/docs/imgs/ancient_city_origin.jpeg)
  - ![ancient city origin](https://github.com/pkuliyi2015/multidiffusion-upscaler-for-automatic1111/blob/docs/imgs/ancient_city_origin.jpeg?raw=true)
  - 转换后：[单击下载原始图像](https://github.com/pkuliyi2015/multidiffusion-upscaler-for-automatic1111/blob/docs/imgs/ancient_city.jpeg)
  - ![ancient city](https://github.com/pkuliyi2015/multidiffusion-upscaler-for-automatic1111/blob/docs/imgs/ancient_city.jpeg?raw=true)
- 示例：2560 * 1280 大型图像绘制
  - ControlNet Canny 边缘
  - ![Your Name](https://github.com/pkuliyi2015/multidiffusion-upscaler-for-automatic1111/blob/docs/imgs/yourname_canny.jpeg?raw=true)
  - ![yourname](https://github.com/pkuliyi2015/multidiffusion-upscaler-for-automatic1111/blob/docs/imgs/yourname.jpeg?raw=true)

****

  ### 优点

  - 可以绘制超大分辨率（2k~8k）图，包括 txt2img 和 img2img
  - 无需进行任何后处理的无缝输出

  ### 缺点

  - **提示控制较弱。**你不能使用非常强烈的正面prompt，否则它将产生重复模式，结果可能无法使用。
  - 我们还没有进行过太多优化，因此对于非常大的图像（8k）和具有控制网络的图像，速度可能会比较慢。
  - 梯度计算不兼容。它将打破任何通过 UNet 的反向传播或自动梯度计算。

  ### 工作原理（非常简单！）

    1. 隐藏层图像被裁剪成小块
    2. 小块通过UNet并由原始采样器去噪一个时间步
    3. 小块被加在一起，但除以每个像素的累加次数（即加权平均）
    4. 重复2-3步直到走完所有时间步数

****

## Tiled VAE

**原创脚本**。**此算法目前已经可以投入生产**

`vae_optimize.py` 脚本是一个粗暴却精巧的 hack，将图像裁切成小块，单独对每个瓷砖进行编码，并将结果合并在一起，从而允许 VAE 在有限的显存上处理巨大的图像（~10 GB 用于 8K 图像！）。

- 你需要禁用低显存模式 （删除启动命令中的--lowvram 和 --medvram）来使用这个功能。
- 如果尝试成功，你从此以后就不用开启低显存模式，享受GPU编解码的高速度！

### 优点

- 在有限的显存上处理巨大的图像（6GB画2k，12GB画4k，16 GB 画8K），消除您对 --lowvram 和 --medvram 的需求。
- 与[我朋友的实现](https://github.com/Kahsolt/stable-diffusion-webui-vae-tile-infer) 以及Huggingface实现不同，它不会平均化裁切的小图边界，而是删除了attention并使用边缘扩张技巧。产生的解码结果在数学上与不平铺的结果完全相同，即它从根源上不会产生任何接缝。
- 脚本经过了极致的优化。不能更快了！

### 缺点

- 仍然需要大量的 RAM（例如 4096 * 4096 图像需要 20 GB，8k 图像需要 50GB）来存储中间结果。
  - 如果使用 --no-half-vae，则使用量会翻倍。

- 对于 >=8k 的图像，NaN 偶尔会出现。使用官方提供的840000 VAE 权重可以有效地解决大部分问题。
  - 您可以使用 --no-half-vae 禁用半 VAE，用于处理巨大的图像
  - **我们正在找出根本原因并努力解决问题**

- 速度同时受 GPU 和 CPU-GPU存储传输带宽两方面限制。因此，如果其中任何一个拉胯，速度都会变慢。
- 和MultiDiffusion一样，不兼容梯度传输。

### 工作原理

1. 图像被精巧地分成小块，并对于解码器 / 编码器各自进行了 11/32 像素的扩张。
2. 原始 VAE 前向传播被分解为任务队列。
   1. 任务队列在一个小块上开始执行。attention块被忽略
   2. 当需要做GroupNorm时，它会暂停，将GroupNorm所需参数和中间结果存储到 CPU内存，并切换到另一个小块。
   3. 汇总 GroupNorm 参数后，它执行GroupNorm并继续。
   4. 执行采用锯齿顺序以减少不必要的数据传输。

3. 处理完所有小块后，瓷砖被合并并返回。

****

## 安装

- 打开 Automatic1111 WebUI->单击选项卡“扩展”->单击选项卡“从 URL 安装”->输入此存储库的链接->单击“安装”
- ![installation](https://github.com/pkuliyi2015/multidiffusion-upscaler-for-automatic1111/blob/docs/imgs/installation.png?raw=true)
- 重启您的 WebUI 后，您应该会看到以下两个选项卡：
- ![Tabs](https://github.com/pkuliyi2015/multidiffusion-upscaler-for-automatic1111/blob/docs/imgs/Tabs.png?raw=true)

### MultiDiffusion 参数

- 覆盖原本尺寸（文生图）：
  - WebUI默认尺寸上限只有2048，对MultiDiffusion来说太小了
  - 开启这个选项可以把原本对尺寸覆盖掉
- 隐空间小图的宽度和高度：
  - multidiffusion 按小图绘制图像，每个小图都是一个矩形。
  - 因此，这两个参数控制着隐空间小图有多大，每个矩形的实际图像大小的 1/8。
  - 不应该太大或太小（通常 64-128 合适。但您可以尝试其他值。）
- 隐空间小图重叠：
  - MultiDiffusion使用重叠来消除接缝并融合两个潜在图像。
  - 此值越大，过程越慢，但结果将包含更少的接缝和更自然的结果。
- 隐空间图像批处理大小：
  - 允许 UNet 以批处理方式处理小图。
  - **能大大加快处理速度**，但会消耗更多的显存。


### Tiled VAE 参数

- 当输入和输出时，应该将图像裁成多大的小块。大尺寸速度快，但消耗更多显存。
  - 第一次使用的时候您不需要调整参数，脚本会根据一套简单的手工规则为你推荐参数
  - 但推荐的参数可能不适合您的设备。请根据控制台输出中使用的 GPU 使用进行调整。

- 如果GPU有很大程度没被利用，请把这个数字调大；反之如果显存爆炸，请把这个数字调小。

**尽情享受！**

****

## 当前进展

- 本地提示控制正在进行中
- 自动提示计划
- 通过 MultiDiffusion 插值进行视频转换正在进行概念验证

****

## 许可证

这些脚本是根据 MIT 许可证授权的。如果您觉得它们有用，请给作者一个star。