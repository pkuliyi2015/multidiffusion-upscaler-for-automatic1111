## Technical Part

    For those who want to know how this works.

### Tiled VAE

The core technique is to estimate GroupNorm params for a seamless generation.

1. The image is split into tiles, which are then padded with 11/32 pixels' in the decoder/encoder.
2. When Fast Mode is disabled:
    1. The original VAE forward is decomposed into a task queue and a task worker, which starts to process each tile.
    2. When GroupNorm is needed, it suspends, stores current GroupNorm mean and var, send everything to RAM, and turns to the next tile.
    3. After all GroupNorm means and vars are summarized, it applies group norm to tiles and continues. 
    4. A zigzag execution order is used to reduce unnecessary data transfer.
3. When Fast Mode is enabled:
    1. The original input is downsampled and passed to a separate task queue.
    2. Its group norm parameters are recorded and used by all tiles' task queues.
    3. Each tile is separately processed without any RAM-VRAM data transfer.
4. After all tiles are processed, tiles are written to a result buffer and returned.

ℹ Encoder color fix = only estimate GroupNorm before downsampling, i.e., run in a semi-fast mode.

****

### Tiled Diffusion

1. The latent image is split into tiles.
2. In MultiDiffusion:
    1. The UNet predicts the noise of each tile.
    2. The tiles are denoised by the original sampler for one time step.
    3. The tiles are added together but divided by how many times each pixel is added.
3. In Mixture of Diffusers:
    1. The UNet predicts the noise of each tile
    2. All noises are fused with a gaussian weight mask.
    3. The denoiser denoises the whole image for one time step using fused noises.
4. Repeat 2-3 until all timesteps are completed.

⚪ Advantages

- Draw super large resolution (2k~8k) images in limited VRAM
- Seamless output without any post-processing

⚪ Drawbacks

- It will be significantly slower than the usual generation.
- The gradient calculation is not compatible with this hack. It will break any backward() or torch.autograd.grad()

****


## 技术部分

    这部分内容是给想知道工作原理的人看的。

### Tiled VAE

核心技术是估算 GroupNorm 参数以实现无缝生成。

1. 图像被分成小块，然后在编码器 / 解码器中各进行了 11/32 像素的扩张。
2. 当禁用快速模式时：
    1. 原始的 VAE 前向传播被分解为任务队列和任务工作器，开始处理每个小块。
    2. 当需要 GroupNorm 时，它会暂停，存储当前的 GroupNorm 均值和方差，将所有内容发送到内存中，然后转到下一个小块。
    3. 在汇总所有 GroupNorm 均值和方差之后，将结果应用到小块中并继续。
    4. 使用锯齿形执行顺序以减少不必要的数据传输。
3. 当启用快速模式时：
    1. 原始输入被下采样并传递到单独的任务队列。
    2. 它的 GroupNorm 参数被记录并由所有小块的任务队列使用。
    3. 每个小块被单独处理，没有任何 内存 <-> 显存 的数据传输。
4. 处理完所有小块后，小块被写入结果缓冲区并返回。

ℹ 编码器颜色修复 = 仅在下采样之前估计 GroupNorm，即以半快速模式运行。

### Tiled Diffusion

1. 潜在图像被分成小块。
2. 在 MultiDiffusion 中：
    1. UNet 预测每个小块的噪声。
    2. 小块由原始采样器去噪一个时间步。
    3. 小块被加在一起，但除以每个像素的累加次数（即加权平均）。
3. 在 Mixture of Diffusers 中：
    1. UNet 预测每个小块的噪声。
    2. 所有噪声与高斯权重蒙版融合。
    3. 降噪器对整个图像使用融合的噪声去噪一个时间步。
4. 重复执行步骤 2-3，直到完成所有时间步长。

⚪ 优点

- 在有限的显存中绘制超大分辨率（2k~8k）图像
- 无需任何后处理即可实现无缝输出

⚪ 缺点

- 它将明显比通常的生成速度慢。
- 梯度计算与此技巧不兼容。它将破坏任何 backward() 或 torch.autograd.grad()。
