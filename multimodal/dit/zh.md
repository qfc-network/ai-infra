# DiT — 可扩展的扩散 Transformer

- **作者 / 机构**：William Peebles、Saining Xie（UC Berkeley / Meta AI）
- **发表时间**：2022-12-19（ICCV 2023）
- **链接**：[arXiv:2212.09748](https://arxiv.org/abs/2212.09748) · [代码](https://github.com/facebookresearch/DiT)

## 一句话总结

DiT 用纯 Transformer 替换了潜在扩散模型中的 U-Net 骨干，采用与 [ViT](../vit/zh.md) 相同的 patch 分词逻辑。VAE 将图像压缩到潜在空间；潜在表示被分块为 token；Transformer 通过自适应层归一化（adaLN）在扩散时间步和类别标签的条件下，对这些 token 进行 T 步去噪。DiT-XL/2（6.75 亿参数，patch 尺寸 2）在 ImageNet 256×256 类别条件生成上创下 FID 2.27 的新纪录，且扩展性优于任何 U-Net 变体。DiT 成为 Sora、Stable Diffusion 3 以及大多数后续视频生成系统的骨干架构。

## 背景与动机

U-Net 曾是扩散模型（DDPM、LDM、Stable Diffusion 1.x）的标准骨干。U-Net 具有强归纳偏置：编解码器阶段之间的跳跃连接提供多尺度特征复用，卷积层利用空间局部性。这些偏置在小规模时很有帮助，但也限制了 U-Net 从单纯扩大模型规模中获益的能力。

Transformer 扩展规律（Chinchilla、GPT-3）已证实 Transformer 随计算量的扩展比 CNN 更具可预测性。DiT 的问题是：若用 Transformer 替换 U-Net，在生成场景中是否也能获得同样干净的扩展？答案是肯定的——DiT 的 GFlops-FID 曲线单调改善，且 Transformer 架构允许驱动 LLM 的相同优化（FlashAttention、张量并行、FP8）。

DiT 的实际影响巨大。Sora（OpenAI，2024）使用基于 DiT 的"时空 patch" Transformer 进行视频生成。Stable Diffusion 3 和 FLUX 用多模态 DiT（MM-DiT）替换 U-Net，对图像和文本 token 进行联合注意力计算。生成侧的多模态 AI 现在与理解侧同样以 Transformer 为主导。

## 潜在扩散背景

DiT 在**潜在空间**而非像素空间中运行，遵循潜在扩散模型（LDM / Stable Diffusion）的范式：

1. **编码器**：VAE 编码器 f 将图像 x ∈ R^{H×W×3} 压缩为潜在表示 z ∈ R^{h×w×c}。典型压缩：8× 空间缩减（512px → 64×64 潜在），c=4 通道。像素空间计算量减少 64×。
2. **潜在空间扩散**：按噪声调度在 T 步内向 z 逐步添加高斯噪声。前向过程：z_t = √ᾱ_t z + √(1−ᾱ_t) ε，ε ~ N(0,I)。
3. **去噪**：神经网络预测噪声 ε̂ 或直接预测去噪后的潜在表示。DiT 就是这个去噪网络。
4. **解码器**：VAE 解码器 g 将去噪后的潜在表示映射回像素空间。

VAE 独立预训练并在 DiT 训练/推理期间保持冻结。训练和生成过程中唯一运行的组件是 DiT 去噪器。

## 架构设计

### 潜在表示的 Patch 分词

潜在表示 z ∈ R^{h×w×c} 采用与 ViT 相同的分词策略：

1. 将 z 切分为 p×p 大小的不重叠 patch。对于 patch 尺寸 2、32×32 潜在（来自 256px 图像，8× VAE）：N = (32/2)² = 256 个 patch。
2. 展平每个 patch：p×p×c → 长度 p²c 的向量。
3. 线性投影到维度 D。
4. 添加 2D 正弦位置嵌入。

这是 DiT 中唯一的图像特定代码，其余均为 Transformer 块。

对于更高分辨率或视频生成，token 数量随 (h×w)/p² 增长。Sora 的"时空 patch"对（时间、高度、宽度）联合分词：1 秒 720p、24fps 视频，patch 尺寸 (t=1, h=8, w=8)，N = 24 × (720/8) × (1280/8) = 24 × 90 × 160 = 345,600 个 token——这正是视频生成比图像生成昂贵得多的原因。

### DiT Transformer 块

在标准 Transformer 块基础上有两处修改：

**adaLN（自适应层归一化）**用于基于时间步 t 和类别标签 c 的条件化：

$$\text{adaLN}(x, t, c) = \gamma(t, c) \cdot \frac{x - \mu}{\sigma} + \beta(t, c)$$

缩放参数（γ）和偏移参数（β）不是固定值——它们是扩散时间步和类别嵌入之和经小型 MLP 计算的函数。这使 Transformer 能在不同噪声水平下表现不同（早期去噪步骤 vs. 后期精细化步骤），而无需独立的权重集。

论文中评测了四种条件化策略，adaLN-Zero 表现最优：γ 和 β 初始化为产生恒等变换（γ=1，β=0），使训练初期残差块如恒等函数一样工作，训练更加稳定。

**无因果掩码**：与语言模型不同，DiT 使用双向注意力——每个 patch token 可以关注所有其他 patch token。这是正确的，因为去噪不是自回归的；所有潜在 patch 在每个时间步同时去噪。

### DiT 变体

| 模型 | 层数 | D | 注意力头数 | Patch 尺寸 | GFLOPs | 参数量 |
|---|---|---|---|---|---|---|
| DiT-S/2 | 12 | 384 | 6 | 2 | 6.1 | 33M |
| DiT-B/4 | 12 | 768 | 12 | 4 | 5.6 | 130M |
| DiT-L/4 | 24 | 1024 | 16 | 4 | 19.7 | 458M |
| DiT-XL/2 | 28 | 1152 | 16 | 2 | 118.6 | 675M |

Patch 尺寸 2 对于 256px 图像生成 256 个 token——是 patch 尺寸 4 的 4 倍，因此注意力 FLOPs 增加 16×（N² 缩放）。DiT-XL/2 是性能最好的模型；其每次去噪步骤 119 GFLOPs 是单步成本，而非总推理成本。

## 推理：顺序去噪

DiT 推理从根本上不同于 LLM 推理：

- **无跨 token 的自回归循环**——所有 N 个潜在 token 在每一步并行去噪。
- **跨时间步的顺序循环**（T=250 用于 DDPM，T=50 用于 DDIM/DPM-Solver）。
- 每张图像的总 FLOPs：119 GFLOPs × 250 步 = DiT-XL/2 完整 DDPM 采样约 **29.75 TFLOPs**。

这是**算力瓶颈**，而非带宽瓶颈。与 LLM 解码（每个 token 需要反复读取模型权重，通常受带宽限制）不同，DiT 每个去噪步骤加载一次权重，并对所有 N 个 token 进行完全并行的前向传播。在 H100 上：

- 使用 DDIM（50 步）：119 GFLOPs × 50 ≈ 5.95 TFLOPs。H100 FP16 峰值 ≈ 989 TFLOPs/s → 100% 利用率下约 6ms/张。实际吞吐（含开销、VRAM 带宽等）：批量大小 1 时约 50–200ms/张。

在 VRAM 满载之前，增大批量大小可提高 GPU 利用率——批次内的图像之间没有顺序依赖。这与 LLM 解码正好相反，后者批量大小增大有助于提升吞吐量，但每个请求仍需等待各自的顺序解码。

## 核心公式——噪声调度与去噪目标

训练目标是最小化：

$$\mathcal{L} = \mathbb{E}_{z, \varepsilon, t} \left[ \| \varepsilon - \varepsilon_\theta(z_t, t, c) \|^2 \right]$$

其中 ε_θ 是预测噪声的 DiT 网络，z_t 是时间步 t 处的含噪潜在表示，c 是条件信号（类别标签或文本嵌入）。这是标准 DDPM ε-预测目标，与 U-Net 扩散完全相同——DiT 纯粹是骨干架构的替换。

## 工程 Tradeoff

| 决策 | 得到什么 | 放弃什么 |
|---|---|---|
| Transformer 替代 U-Net | 干净的扩展规律；复用 LLM 优化（FlashAttention、TP、FP8） | 无多尺度特征层次；必须通过注意力从全局到局部自行学习 |
| 潜在扩散（VAE + 潜在空间去噪） | 比像素空间便宜 64×；质量保持 | VAE 编解码增加延迟；VAE 重建上限约束输出质量 |
| adaLN 条件化 | 强时间步+类别条件化；零初始化训练稳定 | adaLN MLP 在每层运行；少量 FLOPs 开销 |
| 双向注意力 | 所有 patch 互相感知；生成图像全局一致性更好 | 无法流式输出；解码为像素前须完成完整潜在表示的生成 |
| 小 patch 尺寸（p=2） | 更高空间保真度；更多 token 捕获细节 | N² 注意力成本；比 p=4 多 16× 注意力 FLOPs |
| 顺序去噪（T 步） | 通过 T 灵活权衡质量与速度；渐进式精细化 | 无论图像复杂度如何，至少需要 T 步；实时使用需要少步蒸馏（一致性模型） |

## 实验与结果

- **DiT-XL/2**：类别条件 ImageNet 256×256 的 FID-50K = 2.27——超越所有先前扩散和 GAN 模型的新最优。
- **扩展性**：随 GFLOPs 增加 FID 单调改善（更小 patch 尺寸 + 更大模型）；扩展曲线平滑，与 U-Net 变体出现平台期不同。
- **无分类器引导**：引导尺度 4.0 时，DiT-XL/2 实现 FID=2.27，IS=278.24——生成质量与 DALL-E 2 竞争。
- **下游应用**：DiT-XL/2 骨干被 Sora、SD3、FLUX.1 和众多开源视频生成模型采用。

## 个人评注

DiT 的主要贡献不是新颖的注意力机制或更优的噪声调度——而是**对"Transformer 在扩散方面的扩展性优于 U-Net"这一命题的实证确认**。这对基础设施具有深远意义：整套 LLM 优化工具链（FlashAttention、量化、张量并行、适配交叉注意力的 KV 缓存技术）均可迁移到生成模型。基础模型部署的两大主要分支——理解与生成——现在都是 Transformer 形态，可以共享基础设施。

DiT 推理的计算特性（算力瓶颈、跨 token 并行、跨时间步顺序）需要与 LLM 推理不同的批处理策略。LLM 在解码阶段受延迟约束，受益于持续批处理；DiT 受益于大型并行图像批次处理。运行 DiT 的生产图像生成集群不需要 vLLM 部署的复杂请求调度器——它更像经典批量 ML 工作负载，将图像打包填满 GPU 显存并连续提交所有去噪步骤。

DiT 服务最值得关注的开放问题是**少步蒸馏**：一致性模型和流匹配（如 SD3 和 FLUX 中使用的）可以在 4–8 步而非 50–250 步内生成可接受的图像。这将 DiT 推理从约 100ms 降至约 10ms，推入与 VLM 文本生成相同的延迟级别。届时，结合理解与生成的多模态系统的服务基础设施将开始呈现单一统一流水线的形态——在基础设施层面，"生成"与"理解"模型之间的分界开始消融。参见 [Mamba-SSM](../../foundational/mamba-ssm/zh.md) 了解尝试降低每步计算量的混合 DiT-Mamba 架构。

## 参考文献

- [1] Peebles & Xie. _Scalable Diffusion Models with Transformers._ arXiv:2212.09748, 2022.
- [2] Rombach et al. _High-Resolution Image Synthesis with Latent Diffusion Models._ arXiv:2112.10752, 2021.
- [3] Ho et al. _Denoising Diffusion Probabilistic Models._ arXiv:2006.11239, 2020.
- [4] Song et al. _Consistency Models._ arXiv:2303.01469, 2023.
- [5] 相关词条：[ViT](../vit/zh.md)、[FlashAttention](../../foundational/flash-attention/zh.md)、[Mamba-SSM](../../foundational/mamba-ssm/zh.md)
