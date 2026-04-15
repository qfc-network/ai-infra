# 分组查询注意力（GQA）

- **作者 / 机构**：Joshua Ainslie、James Lee-Thorp、Michiel de Jong、Yury Zemlyanskiy、Federico Lebrón、Sumit Sanghai（Google Research）
- **发表时间**：2023-05 / arXiv:2305.13245
- **链接**：[论文](https://arxiv.org/abs/2305.13245)

## 一句话总结

GQA 介于多头注意力（MHA）和多查询注意力（MQA）之间。MHA 为每个查询头分配一对独立的 K/V 头——质量最高，KV 缓存最大；MQA 将所有 K/V 头合并为一个——缓存最小，但质量明显下降。GQA 将 H 个查询头划分为 G 组，每组共享一对 K/V，KV 缓存缩小 H/G 倍，质量与 MHA 持平。关键在于：现有 MHA 检查点可以通过**少量步骤的再训练**（uptraining）转换为 GQA——只需对每组内的 K/V 头做均值池化初始化，无需从头训练。2023 年中期以后发布的所有主流开源大模型——Llama 2、Llama 3、Mistral、Gemma、DeepSeek——均采用 GQA，它已成为注意力机制的默认变体。

## 背景与动机

### KV 缓存瓶颈

Transformer 推理分为两个阶段：预填充（处理输入提示，计算密集型）和解码（逐 token 生成，内存带宽密集型）。解码阶段，模型每生成一个 token 都要读取完整的 KV 缓存。MHA 下，每个 token 每层的 KV 缓存大小为：

```
KV 缓存（MHA）= 2 × H × d_head × dtype字节数
```

对于 H=64 个头、d_head=128、BF16（2 字节）的模型：2 × 64 × 128 × 2 = 32,768 字节/token/层。Llama 3 70B（80 层）中，单个 token 占用约 2.6 MB KV 缓存。批量 32 条序列、每条 4,096 token 时：32 × 4096 × 2.6 MB ≈ 340 GB——超过四张 A100-80GB 的显存。KV 缓存大小直接限制吞吐量：缓存越小，批量越小，GPU 利用率越低。

### MQA：激进的基线方案

Shazeer（2019）提出多查询注意力（MQA）：所有 H 个查询头共享**单个** K/V 头。KV 缓存降至 MHA 的 1/H。长序列吞吐量提升 2–3×。但 MQA 会降低质量，尤其在长上下文任务和推理基准上——单个共享 K/V 头必须表示原来 H 个独立头才能覆盖的所有信息。PaLM 和 Falcon 使用了 MQA，但它被认为是质量与计算之间的折中，而非无代价的优化。

### 中间路线的需求

各研究机构需要一种方案，同时满足：
1. 大幅缩减 KV 缓存（支持更大批量和更长上下文）。
2. 保持 MHA 级别的质量（无任务特定的性能回退）。
3. 能从现有 MHA 检查点转换，无需从头完整训练。

GQA 同时满足这三个条件。

## 核心方法

### 三种注意力变体

**多头注意力（MHA）**
- H 个查询头，H 个 K/V 头（每个查询头对应一个独立 K/V 头）。
- 每 token 每层 KV 缓存：`2 × H × d_head × dtype字节数`

**多查询注意力（MQA）**
- H 个查询头，1 个 K/V 头（所有查询共享同一 K/V）。
- 每 token 每层 KV 缓存：`2 × 1 × d_head × dtype字节数`
- 相对 MHA 节省：H 倍。

**分组查询注意力（GQA）**
- H 个查询头，G 个 K/V 组，G ∈ [1, H]。
- 每组 H/G 个查询头共享一对 K/V。
- 每 token 每层 KV 缓存：`2 × G × d_head × dtype字节数`
- 相对 MHA 节省：H/G 倍。
- G=1 时退化为 MQA；G=H 时退化为 MHA。

### 内存计算公式

对批量大小 B、序列长度 S、L 层、G 个 K/V 组、头维度 d_head、dtype 字节数 b：

```
KV 缓存总量 = B × S × L × 2 × G × d_head × b
```

**Llama 3 70B 具体示例**：H=64 查询头，G=8 K/V 组，d_head=128，BF16（b=2），L=80 层。

- MHA 每 token KV 缓存：2 × 64 × 128 × 2 = 32,768 字节
- GQA 每 token KV 缓存：2 × 8 × 128 × 2 = 4,096 字节
- 缩减比例：**相比 MHA 节省 8 倍**

批量 32 条序列 × 8,192 token 时：GQA 仅需约 42 GB，MHA 需要约 340 GB——前者可以放进两张 A100，后者需要四张。

### 从 MHA 检查点再训练

GQA 获得广泛采用的关键在于不需要从头训练：

1. 取一个预训练 MHA 检查点，共有 H 个 K/V 头。
2. 将 H 个 K/V 头均匀分为 G 组，每组 H/G 个头。
3. 在每组内对 H/G 个 K/V 权重矩阵做**均值池化**，合并为一个 K/V 头。
4. 使用标准语言模型损失对结果模型再训练少量步骤（约为原始训练计算量的 5%）。

再训练后质量接近 MHA。均值池化初始化比随机初始化效果好很多。论文表明：用相同的有限计算预算，从 MHA 做再训练明显优于从头训练 GQA——预训练 MHA 权重为查询投影和 FFN 层提供了良好初始化。

### 注意力计算

前向传播时，每个 K/V 组的单个 K 和 V 被广播到该组内的所有 H/G 个查询头。组内注意力计算：

```
对于组 g 内的每个查询头 q：
  Attention(Q_q, K_g, V_g):
    score = softmax(Q_q · K_g^T / sqrt(d_head))
    output = score · V_g
```

K_g 和 V_g 对组 g 内所有 H/G 个查询头是同一张量。这种广播操作在大多数注意力内核中是免费的——FlashAttention 原生支持 GQA。

## 工程权衡

| 决策 | 收益 | 代价 |
|---|---|---|
| MQA（G=1）vs MHA | KV 缓存缩减最大（H 倍）；长序列吞吐量最高 | 质量明显下降，尤其在长上下文推理和复杂任务上 |
| GQA G=8（Llama 3 风格） | KV 缓存缩减约 8 倍；吞吐量接近 MQA；质量与 MHA 持平 | K/V 计算量略多于 MQA（8 个 K/V 头 vs 1 个） |
| GQA G=H/2（适度分组） | 适度缓存缩减（2 倍），质量影响极小 | 吞吐提升不如 G=8 显著 |
| 从 MHA 检查点再训练 | 避免完整重训；复用所有预训练权重；约 5% 计算量即可恢复质量 | 初始化不如从头训练 GQA 的最优状态 |
| 更小的 KV 缓存（GQA） | 更高吞吐、更大批量、相同显存下更长上下文 | 基本无代价——在任意内存预算下 GQA 都优于 MHA |
| GQA vs MLA（DeepSeek-V2/V3） | 实现更简单；无低秩投影开销；与所有注意力内核兼容 | MLA 通过低秩潜在投影实现更大 KV 压缩；GQA 是 MLA 超越的基线 |

### 实践中如何选择 G

选择 G 需要结合硬件做权衡：

- **G=H（MHA）**：KV 缓存不是瓶颈时（短序列、小批量）。
- **G=1（MQA）**：KV 缓存是主要约束且可以接受质量下降时。
- **G=8（Llama 3 70B 风格）**：大模型的最佳平衡点——8 倍缓存缩减，质量接近 MHA。G 需整除 H。
- **与张量并行对齐**：张量并行度为 T 时，设 G ≥ T 以确保每个设备至少有一个 K/V 头。Llama 3 70B 使用 G=8、TP=8，每台设备恰好一个 K/V 头。

## 实验结果

原论文将 T5-XXL（11B）和 PaLM（540B）检查点再训练为 GQA，在 SuperGLUE 和多个生成任务上进行评估。

**质量**：G ≥ 2 的 GQA 在所有任务上与 MHA 差异在噪声范围内。MQA（G=1）在 SuperGLUE 任务上下降 0.5–2 个百分点，差距最大的是需要精确 token 级注意力的任务（如 WiC、MultiRC）。G=8 的 GQA 弥合了这一差距。

**吞吐量**：在长序列解码（序列长度 ≥ 2,048）上，相比 MHA 吞吐量提升 2–3 倍。提升来源于内存带宽——GQA 减少了每次解码步骤中 GPU 需要读取的数据量。

**再训练效率**：用约 5% 的原始训练计算量做再训练，质量可恢复到 MHA 的 1% 以内。用相同 5% 预算从头训练 GQA，效果明显差于再训练——MHA 权重提供了强有力的初始化。

## 可复现性说明

GQA 在 HuggingFace Transformers 中通过 `num_key_value_heads` 配置参数完全支持。设置 `num_key_value_heads < num_attention_heads` 即可在所有支持的模型（Llama、Mistral、Gemma、Falcon 等）中自动启用 GQA。

```python
from transformers import LlamaConfig
config = LlamaConfig(
    num_attention_heads=64,      # H 个查询头
    num_key_value_heads=8,       # G 个 K/V 组
    hidden_size=8192,
    num_hidden_layers=80,
)
```

**vLLM**：分页 KV 缓存分配器使用 `num_key_value_heads` 进行块大小计算。GQA 模型使用更少的 KV 缓存块，自动支持更大的有效批量大小。

**FlashAttention 2+**：原生支持 GQA。内核在组内广播 K/V 张量，无需展开——内存高效。

**实现复杂度**：在自定义注意力内核中添加 GQA 约需 10 行代码——将 K/V 从 `[batch, seq, G, d_head]` 形状重复展开为 `[batch, seq, H, d_head]`，或直接在 einsum 中处理分组逻辑。

## 综合评述

GQA 已是新的 MHA——是默认选择，而非优化手段。2023 年中期以后发布的所有主要开源模型都在使用它。再训练方案是推动大规模采用的决定性因素：各研究机构可以转换现有 MHA 检查点，而无需承担完整重训的成本。质量结论清晰：G=8 的 GQA 在各类下游基准上与 MHA 无可区分的差异，同时 KV 缓存仅需 1/8。

GQA 的下一步是 DeepSeek-V2 引入的多头潜在注意力（MLA）。MLA 通过将键值投影到低秩潜在空间并单独分解 RoPE，进一步压缩 KV 缓存，压缩比可达 10–20 倍，且无质量损失——远优于 G=8 的 GQA。GQA 是现有方案的下限，MLA 是当前的上限。GQA 仍是实践默认选择，因为 MLA 的低秩投影增加了实现复杂度，并破坏了标准 FlashAttention 内核的兼容性。

对于 2026 年的工程实践者：训练新模型时，使用 GQA，G 的选择满足 G/H ≈ 1/8（如 64 头模型选 G=8）。若已有 MHA 检查点需要支持更长上下文或更大批量，再训练方案是最快捷的 KV 缓存缩减路径。

## 参考文献

- [1] Ainslie et al. 2023，"GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints"，arXiv:2305.13245。
- [2] Shazeer 2019，"Fast Transformer Decoding: One Write-Head is All You Need"，arXiv:1911.02150。
- [3] Touvron et al. 2023，"Llama 2: Open Foundation and Fine-Tuned Chat Models"，arXiv:2307.09288。
- [4] Dubey et al. 2024，"The Llama 3 Herd of Models"，arXiv:2407.21783。
- [5] Liu et al. 2024，"DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts Language Model"，arXiv:2405.04434。
