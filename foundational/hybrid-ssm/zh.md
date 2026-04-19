# Jamba / 混合 SSM-Transformer 架构

- **关键论文**：
  - **Jamba** — Lieber 等，AI21 Labs。2024-03。
  - **Nemotron-H** — NVIDIA。2025。
- **链接**：[Jamba (arXiv:2403.19887)](https://arxiv.org/abs/2403.19887) · [Nemotron-H (arXiv:2507.00509)](https://arxiv.org/abs/2507.00509) · [Jamba-1.5 (arXiv:2408.12570)](https://arxiv.org/abs/2408.12570) · [代码](https://huggingface.co/ai21labs)

## 一句话总结

纯 Mamba/SSM 的预填充复杂度是线性的，解码显存恒定，但在需要精确上下文检索的任务上表现不如注意力机制。纯注意力架构的预填充是二次复杂度，KV 缓存随上下文长度线性增长，在长上下文场景下成为主要显存开销。混合架构以固定比例交替叠加 SSM 层和注意力层——通常每 4–8 个 SSM 层配 1 个注意力层——利用 SSM 的效率覆盖大多数层，同时让注意力处理少数需要精确 token 召回的位置。**Jamba**（AI21 Labs，2024）是首个在生产中部署的大规模混合模型，将 Mamba 层、Transformer 注意力层和 MoE 组合进一个总参数 520 亿、激活参数 120 亿的模型，在 256K 上下文下实现了比 Mixtral-8×7B 高 3 倍的吞吐量。**Nemotron-H**（NVIDIA，2025）将该方案扩展至前沿规模，结合 FP8 和 TransformerEngine 进行 NVIDIA 硬件优化。基础设施层面的影响是具体可量化的：KV 缓存按注意力层比例缩减，SSM 主导模型的解码更快，预填充由较少的注意力层决定，服务栈必须同时管理两套独立的显存池。

## 背景：为何两种纯架构都无法完胜

### 纯注意力架构的问题

标准多头注意力在序列长度 `S`、模型维度 `d` 下的每层计算量为 `O(S²·d)`。FlashAttention 消除了 `O(S²)` 的显存物化并使内核 IO 高效，但并不改变 FLOPs——注意力的浮点运算量仍是二次方的。更关键的是推理时：每个注意力层都会累积 KV 缓存。对于拥有 `L_attn` 个注意力层、`H` 个注意力头、头维度 `d_h`、上下文长度 `S` 的模型，KV 缓存为：

```
KV_cache = S × L_attn × 2 × H × d_h × 每元素字节数

示例（32 层稠密 Transformer，S=256k，H=32，d_h=128，BF16）：
  = 256,000 × 32 × 2 × 32 × 128 × 2 字节
  ≈ 134 GB
```

在 256K 上下文下，这比许多前沿模型的权重本身还大。KV 缓存随上下文膨胀，压缩批大小，最终限制了一个部署能同时服务的长上下文会话数量。

### 纯 SSM 架构的问题

Mamba 及其衍生版本通过离散化循环在每层演进一个隐藏状态 `h_t ∈ R^d_state`。在稳态下，每层的显存开销为 `O(d_state)`，与序列长度无关——是常数，不会增长。预填充使用并行扫描（FLOPs 为 O(S·d_state)），不是二次操作。解码时每步每层推进状态的代价为 O(d_state)，没有不断增长的缓存。

代价是信息层面的：SSM 状态是所有先前上下文的**有损、固定大小压缩**。模型必须将完整的提示历史压缩进一个固定维度的向量。对于需要从长上下文中检索特定 token 或值的任务——大海捞针、联想召回、逐字复制——这种压缩会丢失信息。实验结果一致：在硬性召回基准上，纯 Mamba 模型相比同等规模的注意力模型表现出明显退化。注意力的 softmax 选择是精确的；给定正确的 key，它可以以零信息损失检索出对应 value。SSM 状态压缩对于任意召回目标无法做到这一点。

### 混合架构的洞察

解决方案是架构层面的分工：对大多数层使用 SSM（上下文压缩足以应付任务），对少数层使用注意力（需要精确检索时）。这带来：

- KV 缓存开销仅为 `L_attn / L_total` 比例的注意力层所产生的部分。
- 预填充由注意力层主导（二次方），但这些层很少——大多数层是线性的。
- 解码状态主要是固定大小的 SSM 状态，加上注意力层产生的小量有界 KV 缓存。
- 质量与纯注意力模型相当甚至更好，因为注意力层专门处理对检索敏感的位置。

## Jamba 架构

Jamba（2024）是这一混合设计在规模上的生产实现。

### 模块结构

Jamba 以固定循环模式交替叠加 Mamba 块、Transformer 注意力块和 MoE 层：

```
[Mamba, Mamba, Attention, MoE-FFN] × N_repeat

Jamba 52B 具体配置：
  总层数：  ~72
  注意力层：~18（每 4 层 1 个）
  Mamba 层：~54
  MoE 层：  位于部分 FFN 位置
  总参数量：52B
  激活参数：12B（MoE 稀疏性）
```

Mamba 层使用标准 Mamba-2 架构（选择性 SSM，d_state=16 或 64，取决于配置，B、C、Δ 参数输入相关）。注意力层使用带有 GQA 和 RoPE 位置编码的标准多头注意力。MoE 层在部分位置用稀疏专家 FFN（top-2 路由）替换稠密 FFN。

### KV 缓存缩减

只有注意力层累积 KV 缓存。Mamba 层维护固定大小的 SSM 状态：

```
每层 SSM 状态 = batch_size × d_state × d_model × 字节数

示例（d_state=16，d_model=4096，BF16，batch=1）：
  = 1 × 16 × 4096 × 2 = 131 KB（每层）
  × 54 个 Mamba 层 ≈ 7 MB 总 SSM 状态

18 个注意力层在 S=256k 上下文下的 KV 缓存（H=32，d_h=128，BF16）：
  = 256,000 × 18 × 2 × 32 × 128 × 2 字节
  ≈ 75 GB

对比等层数稠密 72 层 Transformer 的 KV 缓存：
  = 256,000 × 72 × 2 × 32 × 128 × 2 字节
  ≈ 301 GB
```

缩减比例为 `L_attn / L_total = 18/72 = 25%`——在相同上下文长度、相同层数的条件下，Jamba 的 KV 缓存约为稠密 Transformer 的 1/4。SSM 状态仅约 7 MB，与上下文长度无关，可忽略不计。

AI21 Labs 报告在 256K 上下文下吞吐量比 Mixtral-8×7B 高 3 倍。Mixtral 全部注意力层均需完整 KV 缓存；在该上下文长度下 KV 缓存撑满显存，批大小被迫降至 1。Jamba 因 KV 缓存更小可以批处理更多请求。

### 预填充效率

预填充阶段（处理完整提示），各层类型的贡献不同：

```
注意力层预填充代价：O(S² × d)      — 随序列长度二次方增长
Mamba 层预填充代价：O(S × d_state × d)  — 并行扫描，线性

总预填充：
  ≈ L_attn × S² × d  +  L_mamba × S × d_state × d
  = 18 × S² × d  +  54 × S × d_state × d

在 S=256k 时：S² = 6.5×10^10；S×d_state ≈ 256k×16 = 4.1×10^6
中等长度以上序列，注意力项主导；Mamba 层代价极低。
```

Mamba 预填充代价与注意力预填充代价持平的交叉点随 `S ~ d_state × L_mamba / L_attn` 缩放。对于 Jamba 的参数，这约等于 `S ~ 16 × 3 = 48` 个 token——对任何实际提示而言，Mamba 层的 FLOPs 始终被注意力层主导。

### 解码效率

自回归解码时，每个 token 推进模型状态的代价如下：

```
注意力层每步代价：O(S × d)    — 查询遍历完整 KV 缓存
Mamba 层每步代价：O(d_state × d)  — 循环状态推进，恒定

在 S=256k 时：
  注意力：每层 256,000 × d 次运算
  Mamba：  每层 16 × d 次运算（d_state=16 时）
  比值：在该上下文长度下，Mamba 比注意力便宜 16,000 倍
```

注意力层每步每层仍需 O(S·d) 的代价，因为必须查询完整 KV 缓存（FlashDecoding 并行化查询但不减少总工作量）。在 256K 上下文下，每个注意力层每个解码 token 的代价是相当可观的。相比之下，Mamba 层几乎是免费的，只需做一次简单的矩阵向量乘法推进状态。在 Jamba 72 层中有 18 个注意力层时，注意力层主导解码算力，但其比例（25%）意味着每步解码的总代价约为 `(18/72) × S×d + (54/72) × d_state×d`，远低于完整 72 层注意力模型。

## SSM 状态作为有损压缩

注意力与 SSM 在长上下文显存上的根本不对称需要精确阐述：

**注意力 KV 缓存**：在每个注意力层存储每个先前 token 的精确 key 和 value 投影。给定一个查询，答案从精确存储的表示中计算得出。先前 token 的信息不丢失（浮点精度范围内）。代价是 O(S × L_attn) 的显存，无界增长。

**SSM 隐藏状态**：一个固定大小的向量（每层 O(d_state)），通过可学习的递推汇总所有先前上下文。状态在每步被 `h_t = A·h_{t-1} + B·x_t` 更新；关于早期 token 的信息会被逐渐覆盖，除非可学习的 `A` 恰好保留了它。这种压缩是有损且不可逆的——无法从状态中重建原始序列。

这就是为什么混合架构在少数层中放置注意力：这些层可以精确检索其 KV 缓存覆盖的任何先前 token，而 SSM 层处理不需要精确召回的"背景上下文"。纯 SSM 模型的典型大海捞针失败，正是因为"针"的键值关联在输出需要引用它之前，已被后续信息覆盖在 SSM 状态中。

一个实际推论：**SSM 状态无法像 KV 缓存那样进行前缀缓存复合**。拼接两个 KV 缓存是精确的；扩展 SSM 状态需要从头或从保存的检查点重新运行递推。使用前缀缓存的服务栈（如 vLLM 的 RadixAttention）从 SSM 层获得的收益较低——可以在前缀处缓存 SSM 状态，但这些状态不具有组合性，一般情况下不能在前缀重叠但不完全相同的请求之间共享。

## Nemotron-H（NVIDIA，2025）

Nemotron-H 将混合 SSM-Transformer 方案扩展至前沿规模，并进行 NVIDIA 专项优化：

- 使用类似的 Mamba + 注意力交替叠加，针对 H100/H200 硬件调优。
- 对注意力层集成 **TransformerEngine FP8**，在 Hopper 上实现接近峰值的 Tensor Core 利用率。
- Mamba-2 内核受益于与 FA3 相同的 WGMMA / async TMA 基础设施。
- 训练栈基于 Megatron-Core，带有自定义 Mamba 序列并行扩展。

Nemotron-H 证明混合架构在前沿实验室竞争的规模下同样可行，而不仅仅是中等规模的尝试。FP8 注意力、线性扫描 SSM 层、以及 MoE（在适用处）的组合，在长上下文吞吐量基准上超越了纯 Transformer 基线。

## 服务栈影响

混合模型要求推理系统同时管理两套独立的显存池：

1. **KV 缓存**（注意力层）：分页或连续块，可使用 PagedAttention / RadixAttention 策略，随上下文增长。
2. **SSM 状态**（Mamba 层）：每序列固定大小，不随上下文增长，必须在序列被抢占时保存/恢复。

vLLM 的 Jamba 支持同时分配两个池并按序列复用。SSM 状态足够小（每序列 MB 级），通常不是约束瓶颈；即使减少了 4 倍，KV 缓存仍然是主要的显存消费者。

**批处理**：由于 KV 缓存更小，可同时批处理更多序列。这是长上下文下吞吐量提升的主要来源。在较短上下文（< 8K token）下，KV 缓存优势不明显，两种架构的服务吞吐量大致相当。

**序列并行**：对于超长序列，注意力层仍需序列并行（如 Ring Attention）来分布 O(S²) 的预填充算力。Mamba 层需要自己的序列并行扫描实现。Nemotron-H 的训练栈（Megatron-Core）同时实现了两者。

**抢占与分页**：与 KV 块不同，SSM 状态是每层每序列的单一稠密张量，无法在子序列粒度上分页。若一个序列被抢占并从显存中驱逐，完整的 SSM 状态栈（所有 Mamba 层的状态）必须写入主机内存并在恢复时重新加载——这是比一个 KV 页更大的原子单元。

## 工程 Tradeoff

| 架构 | 预填充复杂度 | 解码显存 | 召回精度 | 硬件效率 | 服务复杂度 |
|---|---|---|---|---|---|
| 纯 Mamba/SSM | O(S·d) 线性；并行扫描 | 恒定 O(d_state·L)；无 KV 缓存 | 硬性召回任务退化；有损压缩限制检索 | 吞吐量好；需自定义扫描内核；无成熟基础设施 | 较简单；无 KV 缓存管理；前缀缓存受限 |
| 纯 Transformer（稠密注意力） | O(S²·d) 二次方；FlashAttention 改善内核效率 | O(S·L_attn·d_h·H)；无界增长 | 精确召回；softmax 选择无损 | FlashAttention 接近峰值；长上下文 KV 缓存限制批大小 | 成熟；PagedAttention、前缀缓存、分块预填充均可用 |
| Jamba 混合（SSM + 注意力，1:3 比例） | 由注意力层主导；注意力算子比稠密少 4–8 倍 | 约为稠密 Transformer KV 缓存的 1/4 到 1/8；另加小量固定 SSM 状态 | 接近注意力水平；少量注意力层处理检索敏感位置 | 良好；256K 时吞吐是 Mixtral 的 3 倍 | 中等；两种缓存类型；抢占时需管理 SSM 状态 |
| 线性注意力变体（RetNet、GLA 等） | O(S·d) | 恒定 | 中等；优于纯 SSM，但难召回任务弱于 softmax 注意力 | 与 SSM 相当；部分变体无需自定义扫描内核 | 类似 SSM；前缀缓存受限 |

## 实践指导

**何时使用混合模型**：上下文长度 64K 以上、纯 Transformer KV 缓存成为显存约束瓶颈的工作负载。上下文较短时（< 8K），考虑到纯 Transformer 更成熟的服务基础设施，其延迟和质量特性通常更优。

**比例选择**：每 4 个 Mamba 层配 1 个注意力层是 Jamba 的默认设置，在当前基准上似乎接近质量-效率边界。更多注意力层提升召回质量但代价是更多 KV 缓存；更少注意力层减少 KV 缓存但召回退化。实验表明，每 8 个 SSM 层少于 1 个注意力层时，长文档任务的召回退化是可测量的。

**KV 缓存估算**：对于注意力层比例为 `f_attn` 的混合模型：
```
KV_cache = S × L_total × f_attn × 2 × H × d_h × 字节数
```
这与纯注意力情形完全一致，只是加了 `f_attn` 系数。在 `f_attn = 0.25`、`S=256K` 时，72 层混合模型使用的 KV 显存等同于 18 层纯 Transformer。

**SSM 状态管理**：确保服务框架在 KV 缓存池之外独立分配 SSM 状态，并正确处理抢占（将完整状态驱逐至主机内存）。忽略这一点会在序列抢占时导致静默的状态损坏。

## 交叉参考

- `../mamba-ssm/` — Mamba 架构；理解 SSM 状态、并行扫描和选择性的前置读物
- `../flash-attention/` — 混合模型中的注意力层仍使用 FlashAttention；KV 缓存计算参考 FA3 性能数据
- `../ring-attention/` — 混合模型的长上下文服务仍需对注意力层进行序列并行
- `../../nvidia/megatron-core/` — Nemotron-H 训练栈；Mamba 序列并行与 TransformerEngine FP8 集成

## 参考文献

- [1] Lieber et al. _Jamba: A Hybrid Transformer-Mamba Language Model._ arXiv:2403.19887, 2024.
- [2] Team AI21. _Jamba-1.5: Hybrid Transformer-Mamba Models at Scale._ arXiv:2408.12570, 2024.
- [3] NVIDIA. _Nemotron-H: A Family of Accurate and Efficient Hybrid Mamba-Transformer Models._ arXiv:2507.00509, 2025.
- [4] Gu & Dao. _Mamba: Linear-Time Sequence Modeling with Selective State Spaces._ arXiv:2312.00752, 2023.
- [5] Dao & Gu. _Transformers are SSMs: Generalized Models and Efficient Algorithms Through Structured State Space Duality._ ICML '24 / arXiv:2405.21060.
- [6] Dao et al. _FlashAttention-3: Fast and Accurate Attention with Asynchrony and Low-precision._ arXiv:2407.08608, 2024.
