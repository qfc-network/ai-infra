# 视觉 Transformer（ViT）

- **作者 / 机构**：Alexey Dosovitskiy、Lucas Beyer、Alexander Kolesnikov 等（Google Brain）
- **发表时间**：2020-10-22（ICLR 2021）
- **链接**：[arXiv:2010.11929](https://arxiv.org/abs/2010.11929) · [代码](https://github.com/google-research/vision_transformer)
- **涵盖变体**：ViT-B/16、ViT-L/14、ViT-H/14 · DeiT（Facebook，arXiv:2012.12877）

## 一句话总结

ViT 将标准 Transformer 编码器——与 NLP 领域完全一致——直接应用于固定尺寸图像 patch 的序列。每个 patch 经线性投影为嵌入向量，预置 [CLS] token，叠加位置嵌入，所得序列完全复用 BERT 式的多头自注意力层。在有充足预训练数据（JFT-300M）的情况下，ViT 以更低的计算量达到或超过 CNN。在 patch 尺寸 14、分辨率 224px 下，ViT-L/14 生成 256 个 patch token，成为 [CLIP](../clip/zh.md) 及后续所有视觉语言模型的标准视觉骨干。

## 背景与动机

到 2020 年，卷积网络已在视觉领域主导了十年。CNN 嵌入了强归纳偏置：局部性（每个滤波器只看局部区域）和平移等变性（同一滤波器在所有位置应用）。这些偏置在数据量小时很有用——模型不需要太多样本就能学到边缘特征与位置无关。但在网络规模下，归纳偏置反而成了约束，阻止模型学习数据中蕴含的任意长程依赖。

ViT 的核心问题是：**不带任何卷积先验的纯 Transformer 能否学会"看"？** 答案是肯定的——但前提是要有足够多的预训练数据。仅在 ImageNet（~120 万图像）上，ViT 因缺少局部性偏置、需要从零学习空间结构而不敌 ResNet。但在 JFT-300M（3 亿图像）上，ViT-L/14 以更少的推理 FLOPs 超越 EfficientNet。这种对数据量的高度依赖是重要的基础设施信号：ViT 是那些有能力负担预训练计算预算的大规模系统的首选骨干，而非小数据集微调的首选。

实际推论：[CLIP](../clip/zh.md) 选用 ViT 是因为 CLIP 在 4 亿网络图文对上训练——充足的数据完全克服了 ViT 的样本低效性——且 ViT 在深度和宽度上都能干净地扩展。

## 架构设计

### Patch 嵌入

分辨率为 H×W、通道数为 C 的图像被切分为 N 个不重叠的 P×P patch：

$$N = \frac{H \times W}{P^2}$$

每个 patch 的张量形状为 P×P×C，展平为长度 P²C 的向量，再线性投影到维度 D：

$$\mathbf{z}_i = \mathbf{E} \cdot \text{flatten}(\text{patch}_i) + \mathbf{p}_i$$

其中 E ∈ R^{D×P²C} 是可学习的投影矩阵，p_i 是位置嵌入。这个线性投影是 ViT 中唯一的图像特定组件，其余部分均为标准 Transformer。

常见配置：

| 模型 | Patch 尺寸 | 分辨率 | Patch token 数 N | 参数量 |
|---|---|---|---|---|
| ViT-B/16 | 16×16 | 224px | 196 | 86M |
| ViT-L/14 | 14×14 | 224px | 256 | 307M |
| ViT-L/14@336 | 14×14 | 336px | 576 | 307M |
| ViT-H/14 | 14×14 | 224px | 256 | 632M |

ViT-L/14@336px 意义重大：它是 CLIP 的图像编码器，随后也是 LLaVA 的骨干，每张图像产生 **576 个视觉 token**——这个数字主导了 VLM 预填充成本（详见 [LLaVA](../llava/zh.md)）。

### [CLS] Token 与序列结构

可学习的 [CLS] token 在 Transformer 前被插入 patch 序列头部，其最终隐藏状态作为图像级分类表示。完整输入序列长度为 N+1。

这与 BERT 架构完全对称。对于下游 VLM，传递给语言模型的是每个 patch 的 token（而非 [CLS] token）——每个 patch token 中的空间细节对于基于定位的任务比全局池化的 [CLS] 更具信息量。

### 位置嵌入

ViT 使用 **1D 可学习位置嵌入**，按 patch 位置（0 到 N）索引。这是最简单的实现方案。尽管图像本质上是 2D 的，栅格顺序的 1D 索引在实践中效果良好。

已探索的替代方案：2D 正弦嵌入（效果持平或略差）、相对位置编码。在比预训练更高的分辨率下微调时（如 CLIP 将 ViT-L/14 从 224px 微调至 336px），位置嵌入需要插值——N 从 256 增加到 576，中间位置通过 2D 双线性插值获得。

### Transformer 编码器

标准 Transformer 块：多头自注意力（MHSA）+ MLP，每个子层前施加 LayerNorm（Pre-LN 变体）。[FlashAttention](../../foundational/flash-attention/zh.md) 可直接应用于 ViT 的 MHSA，无需任何修改——QKV 注意力计算与语言模型注意力完全一致，只是序列更短。

对于 ViT-L（隐藏状态维度 1024，16 个注意力头）：每个注意力头的 d_k = 64。每一层每个头的注意力矩阵为 (N+1)×(N+1) = 577×577（ViT-L/14@336）——序列足够短，使 FlashAttention 分块开销最小，但其节省显存（无需 O(N²) 物化）的优势在处理大批量时仍然有价值。

### 关键扩展公式

自注意力 FLOPs 的缩放规律：

$$\text{FLOPs}_{\text{attn}} \approx 4 \cdot N^2 \cdot D$$

对于 ViT-L/14@336（N=576，D=1024）：每层注意力约 1.4 GFLOPs。24 层注意力总计约 34 GFLOPs。MLP 层（D→4D→D）贡献更多：约 48 GFLOPs。ViT-L/14@336 的完整前向传播总计约 190 GFLOPs——这是 VLM 流水线中任何 LLM 解码之前必须承担的成本。

## DeiT：数据高效 ViT

DeiT（Touvron et al., 2020）证明 ViT 可以通过**知识蒸馏**在 ImageNet-1k（120 万图像）上与 CNN 竞争性训练。关键改进：

- 增加蒸馏 token（类似 [CLS]），训练目标是匹配 CNN 教师的硬预测。
- 激进数据增强（RandAugment、MixUp、CutMix）替代大规模预训练数据。

DeiT 表明 ViT 的样本低效并非根本性缺陷——通过知识迁移和数据增强可以有效弥补。对于没有 JFT 规模数据的实践者，DeiT 是出发点。

## 与 FlashAttention 的兼容性

ViT 的注意力形式与 LLM 注意力完全相同，FlashAttention 无需修改即可应用。实际收益：

- **显存**：标准注意力需要为每个批次元素的每个注意力头物化 (N+1)² 的注意力矩阵。对于 64 张图像通过 ViT-L/14@336 的批量：64 × 577² × 16 头 × 2 字节 = 约 687 MB 仅用于注意力图。FlashAttention 彻底消除这一开销。
- **速度**：序列长度（≤577）较短，ViT 通常是算力瓶颈而非带宽瓶颈，但 FA 仍能降低显存压力，从而支持更大的批量。

## 工程 Tradeoff

| 决策 | 得到什么 | 放弃什么 |
|---|---|---|
| 无卷积先验 | 任意长程依赖；随数据量干净扩展 | 样本低效；需要 1 亿以上图像才能超越 CNN |
| 线性 patch 投影 | 简单；图像特定代码仅一次矩阵乘法 | 无多尺度特征；小于 P×P 的细粒度细节被均一化 |
| 1D 可学习位置嵌入 | 简单；实践中效果良好 | 无内置 2D 结构；分辨率变化时需要插值 |
| [CLS] token 用于分类 | 通过注意力干净池化；与 BERT 模式一致 | 将全局信息强压入单一向量；密集任务中空间细节丢失 |
| 较大 patch 尺寸（如 16 vs 14） | 更少 token → 更快注意力；更少 FLOPs | 更粗的空间分辨率；细粒度任务效果较差 |
| 为 VLM 保留每个 patch token | 丰富的空间特征；支持基于定位的问答 | 每图像 576 个 token 传递给 LLM；主导预填充成本 |

## 实验与结果

- ViT-L/16 在 JFT-300M 上 ImageNet top-1 达 87.8%，以约 4× 更少的 FLOPs 超越先前基于 ResNet 的 BiT。
- ViT-B/16 在 ImageNet-21k + 微调：85.5% top-1——与 EfficientNet-L2 持平。
- DeiT-B（86M 参数，仅 ImageNet-1k）：83.1% top-1——无需额外数据即与 CNN 基线竞争。
- 作为 CLIP 的图像编码器（ViT-L/14@336）时，产生的 576 个 token 驱动零样本 ImageNet top-1 达 76.2%。

## 个人评注

ViT 对基础设施体系的贡献在于其与 Transformer 生态的近乎完美的可组合性。由于 ViT 本质上是"应用于 patch 的 Transformer"，为语言 Transformer 开发的每一项优化——FlashAttention、混合精度训练、张量并行、量化——都无需适配即可直接迁移。这一点从架构角度被严重低估了：CNN 需要独立的优化路径（深度可分离卷积核、专用 NHWC 内存布局），而 ViT 直接搭上了 GPU 已经最大程度优化的 GEMM 运算。

ViT-L/14@336 的 576 token 成本是 VLM 服务中最核心的基础设施变量。当用户发送一张 1024×1024 的图像并被分割为四块 336px 的 tile 时，每块 tile 产生 576 个视觉 token，合计 2304 个 token 必须在生成第一个输出 token 之前完成预填充。这正是 VLM 需要[分块预填充](../../foundational/chunked-prefill/zh.md)的动机——若没有它，图像密集型请求将独占 GPU 并导致纯文本请求严重排队。

ViT 的弱点在于多尺度特征提取。CNN 天然构建特征金字塔（浅层细节、深层语义），ViT 缺乏这种层次结构——patch 尺寸 P 定义了唯一的固定空间尺度。DINOv2、Swin Transformer 等工作通过添加层次化池化或重叠窗口来解决这一问题，但标准 ViT-L/14 作为 CLIP 的用法仍是 VLM 骨干的主流选择，分辨率权衡（336px@P14 相对于理想的更高分辨率）是高分辨率分块策略（LLaVA-NeXT）部分补偿的已知局限。

## 参考文献

- [1] Dosovitskiy et al. _An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale._ arXiv:2010.11929, 2020.
- [2] Touvron et al. _Training data-efficient image transformers & distillation through attention (DeiT)._ arXiv:2012.12877, 2020.
- [3] Zhai et al. _Scaling Vision Transformers (ViT-22B)._ arXiv:2302.05442, 2022.
- [4] 相关词条：[FlashAttention](../../foundational/flash-attention/zh.md)、[CLIP](../clip/zh.md)、[LLaVA](../llava/zh.md)、[Llama 4](../../meta/llama4/zh.md)
