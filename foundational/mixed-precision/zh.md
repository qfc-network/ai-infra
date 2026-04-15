# 混合精度训练

- **作者 / 机构**：Paulius Micikevicius 等（NVIDIA / 百度）—— 原始 AMP 论文；NVIDIA Research —— Hopper/Blackwell 上的 FP8
- **发表时间**：2018-02（AMP 论文，arXiv:1710.03740）；持续演进至 FP8（2022–2024）
- **链接**：[AMP 论文 (arXiv:1710.03740)](https://arxiv.org/abs/1710.03740) · [NVIDIA H100 白皮书](https://resources.nvidia.com/en-us-tensor-core/gtc22-whitepaper-hopper) · [TransformerEngine 文档](https://docs.nvidia.com/deeplearning/transformer-engine/user-guide/index.html)

## TL;DR

混合精度训练不是一篇单独的论文，而是跨越三代硬件的十年演进。核心思路：在低精度数值格式（FP16、BF16、FP8）下运行前向和反向传播，利用窄格式张量核心提升吞吐量、降低激活值内存；同时保留全精度（FP32）的权重主副本和优化器状态，确保梯度累积的数值稳定性。

- **FP16（2018，V100）**：相比 FP32 节省 2× 内存；张量核心吞吐量提升 2–8×。问题：指数范围极窄（最大约 65,504），容易发生梯度溢出，需要配合损失缩放（loss scaling）使用。
- **BF16（2019，A100+）**：与 FP32 相同的 8 位指数——无溢出风险——但尾数仅 3 位。已成为通用默认格式。PyTorch `torch.autocast` 在 Ampere 及之后的硬件上默认使用 BF16。
- **FP8（2022–2024，H100/Blackwell）**：两种子格式——E4M3 用于权重和激活值（精度更高），E5M2 用于梯度（动态范围更大）。H100 在 FP8 下的峰值算力为 3,958 TFLOPS，BF16 下为 1,979 TFLOPS。挑战在于激活值异常值（outlier），若不采用逐张量或逐块缩放，会产生较大量化误差。

## 背景与动机

在 FP32 下训练大模型，内存和时间开销都极为高昂。对于含 N 个参数的模型：

- **权重**：4N 字节（FP32）
- **梯度**：4N 字节（FP32）
- **Adam 优化器状态**（m 和 v）：8N 字节（FP32）
- **合计**：最少 16N 字节，尚不含激活值

一个 70 亿参数的模型在激活值出现之前就需要 112 GB 显存，仅模型状态就需要 8 张 A100-80GB GPU。激活检查点（activation checkpointing）可以降低激活值内存，但无法减少参数内存。

混合精度的目标就是：在真正关键的地方保持数值精度，而在其他地方使用更短的格式（更少比特数），从而降低整体开销。

**FP16 为何不是最终答案。** FP16 的最大可表示值为 65,504。梯度——尤其在训练早期或条件较差的层中——常常超过这个范围。溢出的梯度会变成 inf 或 NaN，导致整次更新被污染。损失缩放的做法是：在调用 `backward()` 之前，将标量损失乘以一个大常数 S，然后在优化器 step 之前将所有梯度除以 S，使梯度值保持在 FP16 可表示范围内。动态损失缩放（PyTorch `GradScaler`）从较大的 S 值开始，每步检测到 inf/NaN 时将 S 减半，否则缓慢增大。这套机制可行，但每步都需要 CPU–GPU 同步来检查 NaN，增加了额外复杂度。

**BF16 如何解决溢出问题。** BF16 使用与 FP32 相同的指数字段（8 位），因而具有相同的 ±3.4 × 10³⁸ 动态范围。它牺牲的是尾数精度（3 位，而 FP32 为 23 位，FP16 为 10 位），但梯度累积对此容忍度较高——梯度只需大致准确，不需要精确到多位有效数字。BF16 彻底消除了现代硬件上对损失缩放的需求，成为所有认真训练任务的无条件默认选择。

**FP8 为何是下一步。** 即便使用 BF16，大型 Transformer 中的激活值张量——键/查询/值投影、注意力矩阵、FFN 中间层——仍会消耗数十 GB 显存。将这些张量从每元素 2 字节降至 1 字节，激活值内存直接减半，要么降低 GPU 内存占用，要么允许 batch size 翻倍。算力提升同样显著：H100 上的张量核心每秒 FP8 运算次数是 BF16 的两倍，因此计算密集型场景（大 batch、长序列）可以获得接近 2× 的加速。

## 核心方法

### 自动混合精度（AMP）

Micikevicius 等人提出的 AMP 协议在今天基本保持不变：

1. **维护 FP32 主副本**，存储所有可训练参数 θ_fp32。
2. **每次前向传播前转换为低精度**：θ_low = cast(θ_fp32)。这次转换代价很低——每个元素一次读、一次写。
3. **前向和反向传播全程使用低精度格式**。中间激活值、梯度张量以及输出损失均为 FP16 或 BF16。
4. **将梯度转回 FP32** 并累积到主副本：∂L/∂θ_fp32 = cast_up(∂L/∂θ_low)。
5. **用 FP32 优化器**（Adam、AdamW 等）更新主副本。

使用 FP16 时，步骤 3 需要**损失缩放**：在调用 `backward()` 之前将标量损失乘以缩放因子 S，然后在优化器 step 之前将所有梯度除以 S。动态损失缩放（PyTorch `GradScaler`）从较大的 S 值（例如 2¹⁶）开始，每步检测梯度中的 inf/NaN，并相应调整 S。

**内存核算。** 一个常见的误解是 AMP 将总内存减半。实际细分如下（N 为参数数量）：

- FP32 基线：4N（权重）+ 4N（梯度）+ 8N（Adam m+v）= **16N 字节**
- BF16 AMP：2N（BF16 权重）+ 2N（BF16 梯度）+ 4N（FP32 主副本）+ 8N（FP32 Adam）= **16N 字节**

参数相关的内存完全相同。真正的节省来自**激活值**——在标准 Transformer 中，激活值内存正比于 batch size × 序列长度 × 隐层维度。将这些张量从 FP32 改为 BF16 可将激活值内存减半，而激活值在大 batch size 下占主导地位。

### FP8 训练（TransformerEngine / DeepSeek V3）

FP8 引入了新的复杂性：指数位仅有 4 或 5 位，尾数仅有 3 或 2 位，可表示的范围极窄。一个幅值较大的激活值就可能占满整个可表示范围，导致其他较小的值被密集压缩在低端，产生严重的量化误差。

解决方案是**逐张量缩放（per-tensor scaling）**：在将张量转换为 FP8 之前，计算缩放因子

$$s = \frac{\max(|x|)}{x_{\max}^{\text{FP8}}}$$

其中 x_max^FP8 是所选子格式下最大可表示的 FP8 值。转换前将所有元素除以 s，FP8 矩阵乘法完成后再乘以 s 恢复原始尺度。这样可使值分布在整个 FP8 范围内。

**延迟缩放（delayed scaling）** 避免了每步同步计算 max(|x|) 的开销：缓存上一步的缩放因子，用于当前步。这打破了依赖链，否则每个张量每步都需要一次 GPU→CPU 往返。

**子格式选择：**
- **E4M3**（4 位指数，3 位尾数）：用于前向传播的权重和激活值。最大值：448。精度较高，适用于主计算路径。
- **E5M2**（5 位指数，2 位尾数）：用于反向传播的梯度。最大值：57,344。动态范围较大，可处理梯度张量中幅值分布更宽的情况。

**高精度累加：** H100 的 FP8 矩阵乘法单元在写回结果前以 FP32 累加中间和。这是强制性的——如果在 FP8 下累加，数值会立即饱和。TransformerEngine API 默认提供此行为。

**DeepSeek V3 的细粒度分块缩放：** 与其对整个张量使用一个缩放因子（会被最大元素主导），DeepSeek V3 采用逐块缩放——激活值使用 1×128 的 tile，权重使用 128×128 的 tile。这需要追踪更多缩放因子，但能大幅降低异常值对非异常值的影响，思路上与 SmoothQuant 在推理中的做法如出一辙。

**FP8 内存公式：**
- FP8 AMP（权重 + 激活值用 FP8，FP32 主副本 + Adam）：1N（FP8 权重）+ 1N（FP8 梯度）+ 4N（FP32 主副本）+ 8N（FP32 Adam）= **14N 字节**（参数部分），激活值内存相比 BF16 约减半

权重节省相对有限；激活值内存和计算吞吐量的提升才是主要收益。

## 工程权衡

| 决策 | 收益 | 代价 |
|---|---|---|
| FP16 vs FP32 训练 | 参数内存减半（在用到的地方）；张量核心吞吐量提升 2–8× | 梯度溢出风险；需要损失缩放；动态损失缩放每步增加 CPU–GPU 同步开销 |
| BF16 vs FP16 | 无梯度溢出（指数范围与 FP32 相同）；训练流程更简单，无需损失缩放 | 尾数精度略低（3 位 vs 10 位）；需要 Ampere 或更新的硬件；在对数值精度敏感的少数任务上略有差异 |
| FP8 vs BF16 | H100 上计算吞吐量约 2×；激活值内存减少约 50%；有效 batch size 更大 | 逐张量缩放开销；对激活值异常值敏感；调试 NaN/inf 更困难；需要 Hopper 或 Blackwell 硬件；TransformerEngine 内核级复杂性 |
| FP32 主权重副本（AMP） | 梯度累积数值稳定；全精度权重更新；收敛质量与 FP32 基线相当 | 相比只保留低精度权重，参数内存翻倍；FP32 Adam 状态是内存占用的主要来源 |
| 逐块 FP8 缩放（DeepSeek V3 风格） | 比逐张量缩放更好地处理异常值；异常值较多的层模型质量更高 | 需要存储和同步更多缩放因子；内核复杂度增加；需要自定义 CUDA/Triton 内核；难以用标准 TransformerEngine 复现 |

## 实验与结果

**AMP（2018）：** Micikevicius 等人在 ResNet-50、VGG 和 LSTM 语言模型上与 FP32 质量完全匹配，报告在 V100 GPU 上端到端训练速度提升 2–3×。FP16 需要损失缩放；该论文引入了动态损失缩放作为实用解决方案。

**BF16（2019 至今）：** 先由 Google（TPU）采用，随后 NVIDIA（A100）跟进，成为默认格式。在各类 Transformer 模型上均未观察到精度下降。推荐使用 PyTorch `torch.autocast(device_type='cuda', dtype=torch.bfloat16)`，无需 GradScaler。

**H100 上的 FP8：** NVIDIA 报告理论峰值为 FP8 下 3,958 TFLOPS vs BF16 下 1,979 TFLOPS（稠密，非稀疏），恰好 2× 的比例。实际加速比取决于模型架构：计算密集型工作负载（大 batch、长序列）收益最大；内存带宽受限的工作负载（小 batch 解码）收益较小。

**DeepSeek V3（2024）：** 对前向传播矩阵乘法使用 FP8，梯度和优化器状态使用 BF16。报告在 671B MoE 模型上稳定训练，相比 BF16 基线无质量下降。逐块缩放（激活值 1×128，权重 128×128）对稳定性至关重要；朴素的逐张量 FP8 在某些层上会明显降低质量。训练吞吐量：在 H800 GPU 集群上约为每秒 18 万 token，远超同等硬件预算下 BF16 的可达吞吐量。

**FlashAttention-3（2024）：** 充分利用 H100 的异步执行流水线和 FP8 支持，专门针对注意力内核，在长上下文工作负载上相比 FlashAttention-2 实现最高 1.5× 的加速。

## 复现说明

**BF16 AMP（PyTorch）：**
```python
with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
    loss = model(input)
loss.backward()
optimizer.step()
```
需要 PyTorch >= 1.10 和 Ampere 或更新的 GPU。无需 GradScaler。

**FP16 AMP + GradScaler（PyTorch）：**
```python
scaler = torch.cuda.amp.GradScaler()
with torch.autocast(device_type='cuda', dtype=torch.float16):
    loss = model(input)
scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
```

**FP8（NVIDIA TransformerEngine）：**
- 安装 `transformer-engine` 包；将 `torch.nn.Linear` 替换为 `transformer_engine.pytorch.Linear`
- 使用 `fp8_autocast` 上下文管理器；TransformerEngine 自动处理缩放、类型转换和高精度累加
- 需要 Hopper（H100）或 Blackwell 硬件

**DeepSeek V3 FP8 方案：** 详见技术报告（arXiv:2412.19437）。需要自定义 CUDA 内核实现逐块缩放；参考实现在撰写本文时尚未完全开源。

**关键硬件要求：**
- FP16/BF16 张量核心：Volta（V100）及以上支持 FP16；Ampere（A100）及以上支持 BF16
- FP8 张量核心：仅 Hopper（H100、H200）和 Blackwell（B100、B200）支持

## 评注

BF16 是现在的基本门槛。当今没有任何认真的训练任务会使用 FP32——硬件效率差距太大，而 BF16 几乎没有精度代价。FP32 仍然有意义的唯一场景是小规模调试，或对梯度精度敏感的极少数应用（在大型 Transformer 训练中极为罕见）。

FP8 是当前的技术前沿，工程挑战是真实存在的。激活值异常值问题——与 SmoothQuant 针对推理量化所应对的同一现象——在训练中同样突出。如果对某个异常值较多的层使用逐张量 FP8 缩放，可能导致整个训练不稳定。DeepSeek V3 的逐块缩放以内核复杂度为代价解决了这一问题，他们公开的结果表明，在正确实现的前提下，大规模 FP8 训练是可行的。

更宏观的规律是：精度降低始终是一场对抗异常值的战斗。无论是事后量化权重（AWQ、GPTQ）、以 INT8 提供服务（SmoothQuant），还是以 FP8 训练（TransformerEngine），模式都是一样的：找出占据可表示范围的大幅值元素，然后将量化负担从它们身上迁移走。BF16 通过保留指数位宽完全回避了这个问题；FP8 重新引入了这个问题，并需要明确的解决方案。

展望未来：Blackwell 宣布支持 FP4（针对 GB200）将进一步推动精度缩减曲线。在 FP4 下，激活值异常值和逐块缩放变得更加关键——这是 2025 年迈入的活跃研究方向。

## 参考文献

- [1] Micikevicius, P. 等（2018）。"Mixed Precision Training."《ICLR 2018》。arXiv:1710.03740。
- [2] NVIDIA（2022）。"NVIDIA H100 Tensor Core GPU Architecture Whitepaper." NVIDIA 技术报告。
- [3] NVIDIA（2023）。"Transformer Engine: FP8 Training for Large Language Models." NVIDIA 开发者文档。
- [4] DeepSeek-AI（2024）。"DeepSeek-V3 Technical Report." arXiv:2412.19437。
- [5] Shah, J. 等（2024）。"FlashAttention-3: Fast and Accurate Attention with Asynchrony and Low-precision." arXiv:2407.08608。
- [6] Xiao, G. 等（2023）。"SmoothQuant: Accurate and Efficient Post-Training Quantization for Large Language Models."《ICML 2023》。arXiv:2211.10438。
