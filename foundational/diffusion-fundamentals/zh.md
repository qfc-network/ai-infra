# DDPM / DDIM / 无分类器引导 — 扩散模型基础

- **核心论文**：Ho 等（2020）、Song 等（2020、2022）、Ho & Salimans（2022）
- **发表时间**：DDPM — 2020-06 · DDIM — 2020-10 · CFG — 2022-07 · Improved DDPM — 2021-02
- **链接**：[DDPM (arXiv:2006.11239)](https://arxiv.org/abs/2006.11239) · [DDIM (arXiv:2010.02502)](https://arxiv.org/abs/2010.02502) · [CFG (arXiv:2207.12598)](https://arxiv.org/abs/2207.12598) · [Improved DDPM (arXiv:2102.09672)](https://arxiv.org/abs/2102.09672)

## 一句话总结

扩散模型定义了一个固定的前向过程，将数据逐步加入高斯噪声，再训练神经网络逐步逆转这一过程。DDPM 建立了理论框架和简洁的噪声预测训练目标；DDIM 证明同一个训练好的模型可以用更少步骤进行确定性采样；无分类器引导（CFG）则引入了轻量级条件控制机制，以牺牲多样性换取生成质量，且不需要独立的分类器。三者共同构成现代图像与视频生成系统的数学与工程基础，包括基于 DiT 的架构（参见 [../../multimodal/dit/zh.md](../../multimodal/dit/zh.md)）。

## 背景与动机

扩散模型出现之前，生成图像质量由 GAN（模式崩溃、训练不稳定）和 VAE（输出模糊、后验崩溃）主导。基于分数的生成模型与扩散概率模型在 2021 年被统一纳入分数匹配 / 去噪目标框架下。核心洞察是：预测添加到图像上的噪声，等价于在每个噪声水平上估计数据分布的对数密度梯度（分数），而这一目标的稳定性和可扩展性远优于 GAN 训练。

推动 DDIM 和 CFG 改进的实际问题是推理开销：DDPM 需要 1000 个顺序去噪步骤，每步都是一次完整的神经网络前向传播。对于任何大到能产生高质量输出的网络而言，1000 步都过于昂贵。DDIM 将每张图像的理论开销减少约 20 倍；CFG 在无需独立分类器的情况下实现了对文本或类别的条件生成，代价是每步需要两次前向传播。

## 前向过程

前向过程是一条固定的（不可学习的）马尔可夫链，在 $T$ 个时间步内向数据样本 $x_0$ 逐步加入高斯噪声：

$$q(x_t \mid x_{t-1}) = \mathcal{N}\!\left(x_t;\, \sqrt{1 - \beta_t}\, x_{t-1},\; \beta_t \mathbf{I}\right)$$

其中 $\beta_t \in (0, 1)$ 是噪声调度——控制信号被破坏速度的一组小正数。由于过程是高斯马尔可夫链，它存在闭合形式的边际分布：

$$q(x_t \mid x_0) = \mathcal{N}\!\left(x_t;\, \sqrt{\bar\alpha_t}\, x_0,\; (1 - \bar\alpha_t)\mathbf{I}\right)$$

其中 $\bar\alpha_t = \prod_{s=1}^t (1 - \beta_s)$。这意味着在任意时间步 $t$，含噪样本可以通过单次操作直接计算：$x_t = \sqrt{\bar\alpha_t}\, x_0 + \sqrt{1 - \bar\alpha_t}\, \varepsilon$，$\varepsilon \sim \mathcal{N}(0, \mathbf{I})$。这一闭合形式边际分布对高效训练至关重要——无需模拟完整马尔可夫链即可获得任意步骤的训练样本。

随着 $t \to T$，$\bar\alpha_t \to 0$，$x_T \approx \mathcal{N}(0, \mathbf{I})$，即近似纯各向同性噪声。前向过程无需学习任何参数，它定义了逆向过程需要学习的目标含噪分布。

## 噪声调度

调度序列 $\{\beta_t\}$ 决定信息被破坏的速度。

**线性调度（DDPM 原始版）**：$\beta_t$ 在 $T = 1000$ 步内从 $\beta_1 = 10^{-4}$ 线性增大至 $\beta_T = 0.02$。实现简单，但在高分辨率下早期步骤破坏信号过多，留下一段信噪比极差的时间步对学习贡献甚微。

**余弦调度（Improved DDPM，2021）**：直接将 $\bar\alpha_t$ 定义为余弦函数：$\bar\alpha_t = \cos^2\!\left(\frac{t/T + s}{1 + s} \cdot \frac{\pi}{2}\right)$。这给出更平滑的退化曲线，在中等噪声水平保留更多信号，避免线性调度末尾的近零信噪比平台。余弦调度已成为高分辨率图像生成的标准。

**二次与 sigmoid 调度**：进一步针对特定分辨率或潜在空间调整信噪比轨迹的变体。当去噪网络在 VAE 潜在表示（而非原始像素）上运行时（如 LDM / Stable Diffusion / DiT），潜在空间统计与像素不同，调度通常需要重新校准。

基础设施影响：调度选择不是自由变量。训练后更改调度，已学的去噪器将不再校准——$\varepsilon_\theta(x_t, t)$ 期望对应训练调度在步骤 $t$ 处 $\bar\alpha_t$ 的噪声统计分布。

## 逆向过程与 DDPM

逆向过程是一条从 $x_T \sim \mathcal{N}(0, \mathbf{I})$ 出发、逐步去噪回 $x_0$ 的学习马尔可夫链：

$$p_\theta(x_{t-1} \mid x_t) = \mathcal{N}\!\left(x_{t-1};\, \mu_\theta(x_t, t),\; \Sigma_\theta(x_t, t)\right)$$

DDPM 通过噪声预测对均值进行参数化：网络 $\varepsilon_\theta$ 预测将 $x_0$ 变成 $x_t$ 时加入的噪声 $\varepsilon$，再通过代数关系恢复均值。训练目标简化为：

$$\mathcal{L}_{\text{simple}} = \mathbb{E}_{x_0,\, \varepsilon,\, t}\!\left[\, \|\varepsilon - \varepsilon_\theta(x_t, t)\|^2 \,\right]$$

这是对数似然的加权证据下界（ELBO），权重设为简单常数。完整 VLB 目标包含逐时间步 KL 项，可提升对数似然但略降采样质量；$\mathcal{L}_{\text{simple}}$ 是视觉生成任务的首选。

**协方差参数化**：DDPM 固定 $\Sigma_\theta = \beta_t \mathbf{I}$（前向后验方差）。Improved DDPM 学习两个有效方差选择之间的线性插值（$\beta_t$ 与 $\tilde\beta_t = \frac{1-\bar\alpha_{t-1}}{1-\bar\alpha_t}\beta_t$），可提升对数似然分数。固定协方差更简单，感知质量上也已足够。

**采样（DDPM）**：从 $x_T \sim \mathcal{N}(0, \mathbf{I})$ 出发，运行 $T = 1000$ 步去噪链。每步需要对去噪网络进行一次前向传播。对于 256×256 分辨率使用 DiT-XL 主干，约为 119 GFLOPs × 1000 步 = 每张图像 119 TFLOPs。

## DDIM：用更少步骤实现确定性采样

DDIM（去噪扩散隐式模型）指出 DDPM 的随机采样器并非唯一有效的逆向过程。对于任意辅助变量方差 $\sigma_t \geq 0$，存在一个有效的非马尔可夫逆向过程，其边际分布与相同前向过程和相同已训练去噪器相容。DDIM 更新规则为：

$$x_{t-1} = \sqrt{\bar\alpha_{t-1}}\underbrace{\left(\frac{x_t - \sqrt{1-\bar\alpha_t}\,\varepsilon_\theta}{\sqrt{\bar\alpha_t}}\right)}_{\text{预测的 }x_0} + \sqrt{1 - \bar\alpha_{t-1} - \sigma_t^2}\,\varepsilon_\theta + \sigma_t\,\varepsilon_t$$

对所有 $t$ 设置 $\sigma_t = 0$ 给出完全确定性轨迹：相同的 $x_T$ 始终产生相同的 $x_0$。这一确定性使以下操作成为可能：

1. **步骤跳跃**：由于轨迹是一致的（非马尔可夫），可以在 $\{T, T-\Delta, T-2\Delta, \ldots, 0\}$ 的子集上评估去噪器，而不必逐整数步。使用 50 步而非 1000 步，DDIM 可产生与 DDPM 1000 步相当的质量，将函数评估次数（NFE）减少 20 倍。
2. **潜在空间插值**：两张图像可通过 DDIM 反转得到各自的 $x_T$ 表示并在噪声空间中插值，产生语义上平滑的过渡。
3. **可控编辑**：DDIM 反转后进行条件重采样，可实现无需重新训练的图像编辑。

**NFE 是主要延迟驱动因子**。对于任何扩散模型，推理墙钟时间近似为：

$$\text{延迟} \approx \text{NFE} \times t_{\text{step}}$$

其中 $t_{\text{step}}$ 是去噪网络单次前向传播的墙钟时间。NFE 在推理时直接可控；$t_{\text{step}}$ 由模型规模、硬件和批大小决定。DiT-XL/2 在批大小 1 时的 50 步 DDIM 推理，在 H100 上约为 50 × 4–6ms ≈ 200–300ms。批量推理将 $t_{\text{step}}$ 以次线性方式增大，直至显存饱和，因此对非交互式工作负载，批吞吐量优化几乎总是优于延迟优化。

$\sigma_t > 0$（随机）的情形在 DDPM（$\sigma_t = \sqrt{\tilde\beta_t}$）与确定性情形之间插值。在 $\sigma_t = 0$ 时，输出多样性完全依赖于 $x_T$ 起始点的多样性；采样过程中不注入随机性。

## 无分类器引导（CFG）

分类器引导（Dhariwal & Nichol，2021）通过在每个去噪步骤将去噪分数偏移已单独训练的分类器对数似然梯度，改善了感知质量，代价是多样性下降。这需要在每个去噪步骤都训练并运行一个独立的分类器——成本很高。

无分类器引导将分类器合并到去噪网络自身。训练时，以 $p_{\text{uncond}} \approx 0.1$–$0.2$ 的概率丢弃条件信号 $c$（文本嵌入、类别标签），替换为空标记 $\varnothing$。同一个网络因此同时学习条件与无条件分数函数。推理时，引导后的分数为：

$$\hat\varepsilon_\theta(x_t, t, c) = \varepsilon_\theta(x_t, t, \varnothing) + w \cdot \left(\varepsilon_\theta(x_t, t, c) - \varepsilon_\theta(x_t, t, \varnothing)\right)$$

其中 $w \geq 0$ 是引导强度。$w = 0$ 恢复无条件采样；$w = 1$ 恢复标准条件采样；$w > 1$ 放大条件方向，改善与条件的对齐，代价是样本多样性降低，有时出现饱和伪影。

**典型取值**：Stable Diffusion 中 $w = 7.5$，DiT-XL/2 评测中 $w = 4.0$–$6.0$。最优 $w$ 因数据集和任务而异，需要调优。

**CFG 的推理开销**：每个去噪步骤现在需要两次前向传播——一次条件，一次无条件。与无条件或标准条件采样相比，这使 $t_{\text{step}}$ 翻倍。对于 50 步 DDIM + CFG 流水线，真实 NFE 为 100。将条件和无条件输入拼接后通过单次 $2B$ 大小的批量前向传播，可以部分恢复这一开销。

**CFG 的显存影响**：标准实现通过拼接条件与无条件输入将批大小翻倍。对于 $B$ 张图像的批次，去噪器在每步处理 $2B$ 个样本。在大批大小时，这可能使峰值激活显存翻倍，将 OOM 前的最大可行批大小减半。

## 工程影响：NFE 是成本轴

扩散推理的核心基础设施洞察是：**NFE 主导延迟**，且 NFE 可通过采样算法独立于模型质量单独调优。这与语言模型推理给出了不同的优化空间：

- LLM 解码延迟由模型规模 × 输出长度 × 每 token 计算量决定，推理时无法自由调整。
- 扩散延迟由 NFE × 每步成本决定。NFE 可在无需重新训练基础模型的情况下从 1000 步减少到 50 步（DDIM）、20 步（DPM-Solver）、4–8 步（一致性模型、流匹配蒸馏），代价是逐级的质量损失。

**每步成本**由去噪主干决定。对于 UNet（Stable Diffusion 1.x/2.x），单步涉及空间激活的编码器-注意力-解码器通道。对于 DiT，是 $N$ 个 patch token 双向注意力的标准 Transformer 前向传播。DiT 的步骤是算力瓶颈，可对大批次干净地并行；UNet 步骤在小分辨率下因顺序编解码结构而更多受带宽限制。

**批量推理**：扩散模型批次内图像之间无顺序依赖（与自回归 LLM 不同），批量图像完美摊销权重加载。GPU 利用率随批大小线性增长直至显存耗尽——届时是 OOM 而非吞吐量软曲线。CFG 将有效批大小翻倍，实际 OOM 前最大批大小约为单次通道限制的一半。

**生产环境延迟 SLO**：交互式用例（消费级图像生成）通常需要端到端延迟低于 3–5 秒。以 DiT-XL 主干（119 GFLOPs/步）在 H100 上为例：
- DDPM 1000 步：批大小 1 时 > 100 秒——不可接受。
- DDIM 50 步 + CFG：约 0.5–1.5 秒——勉强可用。
- 流匹配 / 一致性蒸馏 4–8 步：约 100–400ms——可接受。

这正是为何一致性模型和流匹配（FLUX、SD3）不是可选优化，而是任何有延迟约束的生产图像生成系统的功能性需求。

## 工程 Tradeoff

| 方案 | 优势 | 劣势 | 适用场景 |
|---|---|---|---|
| DDPM（1000 步） | 最高对数似然；随机多样性；理论最干净 | 每张图像 1000 NFE；大多数场景延迟不可接受 | 离线研究、FID 基准测评、延迟无约束且追求质量上限时 |
| DDIM（50 步，确定性） | 减少 20× NFE；可反转（支持编辑）；无需重新训练 | 多样性略低于随机 DDPM；低于约 20 步质量下降 | 生产图像生成；离线批量作业；所有以 DDPM 为基础模型的场景 |
| DDIM + CFG（$w > 1$） | 强条件对齐；高感知质量 | NFE 为无条件的 2 倍；批大小翻倍导致显存峰值飙升；多样性降低 | 文生图、类别条件生成，提示词保真度比多样性更重要时 |
| 流匹配 / 蒸馏（4–8 步） | 延迟与 VLM 生成持平；可用于交互场景 | 需要额外训练（蒸馏或流匹配预训练）；存在一定质量损失 | 交互式应用、实时生成、移动端部署 |

## 与 DiT 的关联

上述数学与主干架构无关：无论去噪网络 $\varepsilon_\theta$ 是 UNet 还是 DiT，DDPM / DDIM / CFG 的运行方式完全相同。本仓库中的 DiT 词条（参见 [../../multimodal/dit/zh.md](../../multimodal/dit/zh.md)）使用的正是这里的训练目标——$\mathcal{L}_{\text{simple}}$ 方程——以 DiT 网络作为 $\varepsilon_\theta$。DiT 中的 adaLN 条件化是将时间步 $t$ 和类别标签 $c$ 注入网络的机制；通过在训练时将 $c$ 丢弃为一个学习得到的无条件嵌入，实现了 CFG 的空标记机制。

这种分离的实际意义是：可以在不改变扩散算法的情况下替换去噪主干，也可以在不重新训练主干的情况下替换采样算法（DDPM、DDIM、DPM-Solver、一致性模型）。这种模块性是框架的刻意设计属性，也是扩散文献能够独立于架构研究迭代采样算法的原因。

## 可复现性说明

三个核心算法均在开源库中完整实现：

- HuggingFace `diffusers`：`DDPMScheduler`、`DDIMScheduler`、`UniPCMultistepScheduler`，均支持 CFG。
- 参考实现（Ho 2020、Song 2020）代码量小，在单张 A100 上数小时内可完成类别条件 ImageNet 的复现。
- CFG 引导强度调优需要逐模型网格搜索；$w = 7.5$ 是文生图的合理起点，但并非通用值。
- 噪声调度变更需要重新训练或仔细的重缩放；训练与推理之间不要混用不同调度类型。

## 参考文献

- [1] Ho 等。_Denoising Diffusion Probabilistic Models._ arXiv:2006.11239，2020。
- [2] Song 等。_Denoising Diffusion Implicit Models._ arXiv:2010.02502，2020。
- [3] Nichol & Dhariwal。_Improved Denoising Diffusion Probabilistic Models._ arXiv:2102.09672，2021。
- [4] Dhariwal & Nichol。_Diffusion Models Beat GANs on Image Synthesis._ arXiv:2105.05233，2021。
- [5] Ho & Salimans。_Classifier-Free Diffusion Guidance._ arXiv:2207.12598，2022。
- [6] Song 等。_Score-Based Generative Modeling through Stochastic Differential Equations._ arXiv:2011.13456，2020。
- [7] Lipman 等。_Flow Matching for Generative Modeling._ arXiv:2210.02747，2022。
- [8] 相关词条：[DiT](../../multimodal/dit/zh.md)
