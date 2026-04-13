# Mamba 与状态空间模型

- **关键论文**：
  - **S4** —— Gu 等，Stanford。ICLR '22。
  - **Mamba** —— Gu & Dao，CMU + Princeton。2023-12。
  - **Mamba-2** —— Dao & Gu。ICML '24。
- **链接**：[S4 (arXiv:2111.00396)](https://arxiv.org/abs/2111.00396) · [Mamba (arXiv:2312.00752)](https://arxiv.org/abs/2312.00752) · [Mamba-2 (arXiv:2405.21060)](https://arxiv.org/abs/2405.21060) · [代码](https://github.com/state-spaces/mamba)

## 一句话总结

**状态空间模型（SSM）**是一种非 transformer 序列架构，**prefill 线性时间**、**decode 常量显存**——不要 KV cache。从经典控制理论以深度学习友好形式复活（S4，2022）之后，**Mamba**（2023）通过**输入相关（selective）的状态转移**让 SSM 在 ~7B 规模与 transformer 竞争，同时保留线性 scaling。**Mamba-2**（2024）证明 SSM 与 attention 密切相关（SSD 框架），解锁了快速硬件友好 kernel。infra 故事才是重点：SSM 有**根本不同的推理经济学**——长上下文便宜、decode 不随上下文增长——但在**一些任务上损失 attention 的上下文 recall**。纯 SSM 没能推翻 transformer，但**混合架构**（Jamba、Zamba、Samba、Nemotron-H）——混入 SSM 与 attention 层——已成真实部署选项，尤其对 transformer KV cache 吃力的长上下文 workload。

## 背景与动机

Transformer 有一个具体 infra 痛点：**attention 对序列长度的平方代价**（算力与显存）。FlashAttention 让它精确且 cache 友好，Ring Attention 让它并行，但两者都没改变复杂度。1M+ token 上下文下，transformer prefill 贵、decode 的 KV cache 巨大。

线性注意力变体（Performer、Linformer、Linear Attention）尝试改复杂度但都丢质量。问题是：是否存在**亚平方架构且保持 transformer 级质量**？

状态空间模型提供了一个有前景的方向。数学：SSM 按下式演化隐状态 `h_t`

```
h_t = A · h_{t-1} + B · x_t
y_t = C · h_t
```

`A, B, C` 为矩阵。选出足够结构化的 `A` 可得：

- **Prefill**：沿序列的卷积（可 FFT），线性时间。
- **Decode**：一次递推，每 token `O(hidden^2)`，**无 KV cache**。

问题是质量。经典 SSM 每层 `A, B, C` 固定（线性时不变 LTI）。能压缩*任何*序列但不能**选择性地关注正确部分**——这是 attention 天然擅长的技能。

## S4 —— 规模化高效 SSM

Gu 等 2022 有三个贡献：

1. **HiPPO 初始化** —— 一个有原理的 `A` 选取方式，让状态最优记忆历史。
2. **对角 + 低秩结构** —— `A` 分解成可高效计算的部分。
3. **基于 FFT 的并行扫描** 做 prefill。

S4 证明 SSM 在 Long Range Arena 上能匹敌 transformer。但它是 LTI——每 token 都用同一组 `A, B, C`。意味着无选择性 recall；在语言任务上质量落后于 transformer。

## Mamba —— selective SSM

Mamba 的主动作：**让 `B`、`C`（以及时间步 `Δ`）依赖输入**。给定输入 `x_t`：

```
B_t = W_B · x_t
C_t = W_C · x_t
Δ_t = softplus(W_Δ · x_t)
A_t = discretize(A, Δ_t)
```

SSM 不再 LTI。模型可以**基于输入内容动态决定存什么、忽略什么、状态衰减多快**。这与 attention 的 Q/K/V 路由概念相似——每个 token 可以选择从状态里读什么。

### 效率代价

输入相关 SSM 打破基于 FFT 的并行扫描。朴素回退到逐 token 递推（慢）。Mamba 的解法：**一个自定义 CUDA kernel 做"并行 selective scan"**——按硬件不同以 `O(N log N)` 或 `O(N)` 算扫描，用 work-efficient 并行前缀算法。

kernel 是手工调的（又是 Tri Dao——写 FlashAttention 的同一个人）。把状态放在 SRAM、最小化 HBM 流量。这就是让 Mamba 实用化的关键：**kernel 与数学同等重要**。

### 结果

- Mamba-2.8B 在常见 LM benchmark 上匹敌 Llama-1-3B，推理吞吐快 2–4×。
- Decode 延迟随上下文长度近乎常量。
- Prefill 随序列长度线性扩展（vs transformer 的平方）。
- 在 DNA、音频、长程合成任务上特别强。

## Mamba-2 —— State Space Duality

2024。核心洞察：**selective SSM 在数学上等价于一种特定形式的线性注意力（标量值 mask）**。作者称之为 **SSD 框架**。

含义：

- 来自高效注意力的很多技巧（FlashAttention 风格 kernel、tensor core 友好布局）直接迁移到 SSM。
- Mamba-2 kernel 更简单，相同 state size 下比 Mamba-1 快 2–8×。
- 打开**混合 SSM/attention kernel** 的大门——同一数学底层。

Mamba-2 是实务里混合架构会用的版本。

## 混合架构

纯 SSM 在需要精确上下文 recall（"找到 prompt 里这个 token 并使用"）的任务上撞天花板。attention 做这件事几乎免费；SSM 要把 prompt 压进固定状态，会失真。

**混合架构交错 SSM 层与 attention 层**——SSM 做多数层（便宜、线性），attention 做少数（精确 recall）。例子：

- **Jamba**（AI21，2024）—— 52B 参数 256K 上下文的生产部署混合模型。
- **Zamba**（Zyphra，2024）—— 混合 SSM + 共享 attention。
- **Samba** —— 另一种栈，滑窗 attention + SSM。
- **Nemotron-H**（NVIDIA，2025）—— 规模化混合。

比例很重要：常见每 6–8 层 SSM 配 1 层 attention。够处理依赖 recall 的任务，又能保住 infra 优势。

## 基础设施影响

### 推理

- **Decode 常量显存** —— 无 KV cache，只有固定大小的隐状态（每层每序列 ~MB 级）。长上下文下显存节省巨大。
- **长上下文 prefill** 线性，不是平方。
- **跨请求前缀缓存不可** —— 状态被不可逆压缩；你可以缓存前缀*状态*，但远不如 KV cache 可组合（拼接前缀并不平凡）。
- **PagedAttention 不适用** —— 没有 KV 块可分页。显存管理完全不同。

对 serving：SSM 推理是不同制度。Mooncake 式 KV cache 池不适用；前缀缓存需新思路；PD 解耦也不同，因为没有 KV 要传。

### 训练

- selective scan kernel 替代 attention kernel——主要算力 + 显存形状不同。
- 序列并行（Ring Attention）不直接迁移；SSM 需要自己的序列维并行故事。
- 模型并行其他方面类似（沿 hidden 维的 TP 可用）。

### 混合 serving

Jamba 式模型仍有 attention 层，所以 KV cache 对那些层仍存在——只是更小。混合 serving 栈需同时管理 **KV cache 块 + SSM 状态**。实现（如 vLLM Jamba 支持）把显存管理分摊到两者。

## 工程 Tradeoff

| 决策 | 得到什么 | 放弃什么 |
|---|---|---|
| 纯 SSM | 线性时间、decode 常量显存 | 上下文 recall 质量 |
| 输入相关 selectivity（Mamba） | 多数任务上质量追平 | 打破 FFT；需要自定义 scan kernel |
| SSD 框架（Mamba-2） | 与 attention 统一；kernel 更快 | 架构纯度下降 |
| 混合 SSM + attention | 兼得二者：多数层便宜、必要处精确 | serving 栈要处理两种 cache；混合比例要调 |
| 无 KV cache | 推理显存大幅节省 | 现有推理基础设施（paged attention、前缀缓存）不能直接用 |
| 手调 scan kernel | 相比朴素 2–4× | kernel 复杂度；与具体架构代绑定 |

## 实验与结果

- 纯 Mamba 能干净扩到 2.8B；纯形式下超过 7B 匹敌 transformer 变难。
- 混合模型在同大小通用 benchmark 上匹敌或超过 transformer；长上下文吞吐 benchmark 上显著更好。
- Jamba-1.5-Large（398B MoE + SSM 混合）是 2025 年最大的公开含 SSM 模型。

## 复现要点

- 参考实现开源。
- Mamba-2 是要用的版本；Mamba-1 已被取代。
- Hugging Face Transformers 支持 Mamba；vLLM 支持 Jamba serving。
- 规模训练有实际坑（状态初始化、归一化位置）——论文里的细节值得细读。

## 个人评注

SSM 故事是 infra 驱动的。数学自控制理论起就在那里；变化的是 (a) HiPPO 给了靠谱初始化，(b) Mamba 让 selectivity 可用，(c) Tri Dao 写了 kernel。没 kernel Mamba 不实用；有 kernel 架构才竞争得起。

诚实评价：**SSM 没取代 transformer，大概率也不会**。attention 的上下文 recall 太有用，transformer 的生态惯性太大。但 SSM **不需要赢** —— 占住一个有用生态位就够。那个生态位是**长上下文、便宜 decode、attention 稀疏的混合架构**。Jamba、Nemotron-H 以及下一代 Claude / Gemini 等都在悄悄包含 SSM 层，不是出于架构纯粹主义，而是**纯 attention 下 10M token 上下文很痛，SSM + 稀疏 attention 则容易**。

更广的启示：**推理经济学在规模下驱动架构选择**。transformer 赢在训练时代是因为 attention 训练效率好。SSM 与混合赢在长上下文推理生态位是因为 1M+ token KV cache 痛。下一次架构转换很可能被"哪种推理形状最贵"驱动——大概率是带深度、多样状态的 agentic workload。

2026 年做长上下文产品：**在默认纯 transformer 前先评估 SSM+attention 混合模型**。serving 成本结构差异足够大，以至于对长上下文重应用，混合在略小的质量代价下仍胜在成本。

## 参考

- [1] Gu et al. _Efficiently Modeling Long Sequences with Structured State Spaces._ ICLR '22 / arXiv:2111.00396.
- [2] Gu & Dao. _Mamba._ arXiv:2312.00752, 2023.
- [3] Dao & Gu. _Transformers are SSMs (Mamba-2)._ ICML '24 / arXiv:2405.21060.
- [4] Lieber et al. _Jamba._ arXiv:2403.19887, 2024.
- [5] state-spaces/mamba：https://github.com/state-spaces/mamba
