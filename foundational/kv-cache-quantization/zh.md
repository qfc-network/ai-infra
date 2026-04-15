# KV Cache 量化 —— KIVI 与 KVQuant

- **论文**：
  - **KIVI** —— Zirui Liu 等，Rice University。2024-02。
  - **KVQuant** —— Coleman Hooper 等，UC Berkeley。2024-01。
- **链接**：[KIVI (arXiv:2402.02750)](https://arxiv.org/abs/2402.02750) · [KVQuant (arXiv:2401.18079)](https://arxiv.org/abs/2401.18079) · [KIVI 代码](https://github.com/jy-joy/KIVI) · [KVQuant 代码](https://github.com/SqueezeAILab/KVQuant)

## 一句话总结

KV cache 是长上下文与大批量下的主导内存消耗者。KIVI 无需重训，对 Key 做 INT2、对 Value 做 INT4 的逐通道量化，将 KV cache 内存压缩 2.6×，同时保持 FP16 质量。KVQuant 更进一步——逐通道非均匀量化加 outlier 处理——在相近质量损失下实现 INT4 Key 与 INT2 Value。两者合力完成了量化闭环：权重（GPTQ/AWQ）→ 激活（SmoothQuant）→ KV cache（KIVI/KVQuant）。

实际意义：在 batch size 64、序列长度 32k 下，LLaMA-2-13B 的 KV cache 单项就超过 330 GB（FP16）。INT4 KV 将其压到约 85 GB；KIVI 的非对称 2/4 位方案进一步降至 50 GB 以下。这个量级的差距，决定了长上下文 serving 是否在经济上可行。

## 背景与动机

权重量化（GPTQ/AWQ）压缩的是 LLM 内存的静态部分——模型参数。而在推理服务阶段，主导内存的是另一个动态内存池：**KV cache**，它存储每个激活请求中每一个 token、每一个注意力层的 Key 与 Value 张量。

具体来说，以 LLaMA-2-13B（40 层，40 头，d_head = 128，FP16）为例：

```
KV cache = 2 (K+V) × 40 层 × 40 头 × 128 d_head × 2 字节 × B × S
         ≈ 每（批次条目 × token）约 819,200 字节
```

在 B=64、S=4096 时：约 42 GB——超过 26 GB 的模型权重本身。S=32k 时则扩展到约 330 GB。GQA（分组查询注意力）通过减少 KV 头数量来缓解，但即使 G=8，压力依然显著。与权重不同，KV cache 无法离线量化——它必须在前向传播的热路径中，针对每个请求实时在线量化。

统计挑战是真实存在的：KV 张量与权重矩阵有本质不同。Key 向量在特定通道上有大幅 outlier，主导注意力点积计算。Value 向量分布更均匀，但由于要与 softmax 系数加权求和，需要更高精度。朴素的逐张量量化忽视了这些统计差异，即便在 INT8 下也会造成明显精度损失。

KIVI 与 KVQuant 都继承了 SmoothQuant 的核心洞察——outlier 是逐通道的，必须逐通道处理——并在 INT2–INT4 范围内激进地应用到 KV 张量。

## 核心方法

### KIVI：带残差缓存的非对称逐通道量化

KIVI 的核心观察是 Key 与 Value 对量化的敏感性不同，因此应分配不同的比特宽度：

- **Key → INT2 逐通道**：每个 Key 向量的每个维度（通道）都有独立的 scale 和 zero-point，在序列维度上计算。Key 通过与 Query 向量的点积进入注意力分数计算，结果再经 softmax，因此只有*相对*大小重要，绝对精度要求较低——这使得 INT2 对 Key 可行。

- **Value → INT4 逐通道**：Value 直接与后 softmax 注意力权重相乘并累加为上下文向量。累积误差通过输出线性传播，因此 Value 需要比 Key 更高的精度。

两者的量化公式均为标准逐通道均匀量化：

```
q(x) = clamp(round((x − z) / s), 0, 2^b − 1)
s = (max(x) − min(x)) / (2^b − 1),   z = min(x)
```

其中 `s` 和 `z` 对每个通道独立计算，`x` 在该通道的序列维度上取值。反量化时：`x̂ = q(x) · s + z`。

**FP16 残差缓存**：最近生成的 R 个 token（例如 R=128）保留为完整的 FP16。近期 token 被大量关注（注意力 sink + 近因效应），其量化误差对生成质量的影响不成比例。在大 S 场景下，残差缓存的额外开销微乎其微——S=4096 时 R=128 仅占 3% 的额外开销——但能显著恢复 INT2 下的质量。

推理时，某个位置的完整 KV 由两部分拼接而成：对压缩的历史部分实时反量化，再与残差部分拼接。这需要一个自定义融合注意力 kernel，实现难度不低。

**内存降低**：KIVI 的 K（INT2）与 V（INT4）平均比特宽度为 3 位，相比 FP16 的 16 位，理论压缩比为 16/3 ≈ 5.3×。考虑逐通道 scale 存储（每通道每层一个 FP32）和残差缓存，实际有效压缩约 2.6×。

### KVQuant：非均匀量化、RoPE 前量化与 Outlier 隔离

KVQuant 在 KIVI 思路基础上做了四个工程扩展：

**1. 逐通道非均匀量化（NUQ）**：不再使用均匀 INT 分档，KVQuant 通过类似 NF4（Normal Float 4）的方式，将量化级别拟合到每个通道的实际分布：分档落在校准分布的分位数处，使更多分档覆盖高密度区域。对于 INT2 Key，这在不改变比特宽度的情况下有效提升了常见值的表达精度。

**2. RoPE 前 Key 量化**：标准 Transformer 在注意力点积前对 Query 和 Key 施加 RoPE（旋转位置编码）。KVQuant 在施加 RoPE *之前*量化 Key。直觉是：RoPE 对头维度的 2D 子空间做旋转，原始域的小量化误差在 RoPE 后会变成旋转后的误差；在 RoPE 前量化能确保旋转作用于精确的（整数）值，使最终误差更可预测、更小。

**3. Outlier 通道隔离**：Key 矩阵中某些通道存在极端 outlier，即便有逐通道 scale，均匀 2 位范围也无法良好表达。KVQuant 通过一次性校准识别每层中的 top-k outlier 通道（通常约占通道数的 1%），以 FP4 或 FP8 存储，其余通道量化到 INT2。这与 SmoothQuant/LLM.int8() 对激活 outlier 的处理思路完全一致，只是应用到了 KV 向量。

**4. 敏感度加权比特分配**：并非所有注意力层和头对量化误差同等敏感。KVQuant 运行轻量级敏感度分析（基于梯度或激活统计），对敏感的头/层分配更多比特，对其余位置更激进地压缩。

## 工程 Tradeoff

| 决策 | 得到什么 | 放弃什么 |
|---|---|---|
| 逐通道 vs 逐张量量化 | 处理通道级 outlier；INT2/INT4 下质量可用 | 逐通道 scale 存储（每通道每层一个 FP32，小但非零） |
| INT2 Key vs INT4 Key | Key 内存进一步减半 | 量化误差更高；需要残差缓存才能与 FP16 持平 |
| 近期 token FP16 残差缓存 | 保护高关注度的近期 token 质量；大 S 时开销极小 | 每请求 R × 全精度字节额外开销；注意力 kernel 逻辑更复杂 |
| RoPE 前 Key 量化（KVQuant） | RoPE 作用于精确值；INT2 Key 质量更好 | 需要修改前向传播中量化的位置，有架构侵入性 |
| KV 量化 vs GQA | 两者正交可叠加——INT4 KV 量化与 GQA 可同时使用，收益相乘 | GQA 减少 KV 头数；KV 量化减少每值比特数；两个独立旋钮，无交互惩罚 |
| 自定义反量化 kernel | 实时反量化隐藏在注意力计算中，静态存储不膨胀 | 较大 kernel 工程量；INT2 融合注意力在标准 vLLM 中（截至 2024 年）尚未支持 |

## 实验与结果

**KIVI** 在 LLaMA-2-7B 和 LLaMA-2-13B 上评估：

- INT2 Key + INT4 Value：7B 模型 Wikitext-2 困惑度与 FP16 基线相差 0.1；13B 相差 0.15。下游任务（MMLU、GSM8K、HumanEval）稳定在 FP16 1% 以内。无需重训。
- 吞吐量：在相同 GPU 内存预算下，KIVI 最多可支持 2.35× 更大的 batch size。优势随序列长度增长而放大——序列越长，KV cache 越是瓶颈，KIVI 的优势越显著。
- 残差缓存的重要性：若无残差缓存（纯 INT2），7B 模型困惑度上升约 0.5–1.0。

**KVQuant** 在最大 70B 参数的模型上评估：

- INT4 平均比特宽度：LLaMA-2 全系列（7B/13B/70B）困惑度与 FP16 差距 < 0.1。因非均匀分档，比 KIVI 更贴近 FP16。
- INT2 Key 加 outlier 隔离：LLaMA-2-7B 困惑度差距 < 0.5。仅 RoPE 前量化一项相比 RoPE 后 INT2 即可恢复约 0.2 困惑度点。
- 敏感度分析显示，第 0 层（第一个注意力层，处理原始 token 嵌入）和最后几层对量化始终最敏感，需要更多比特。
- 非均匀量化对长尾分布的 Key 通道特别有效——这在第一层和最后几层中是普遍规律。

两篇论文均报告在推荐配置下，长上下文任务（LongBench、SCROLLS）上无质量退化——这是最核心的实用声明：内存压缩不损失长上下文推理能力。

## 复现要点

**KIVI**：完全开源，位于 [github.com/jy-joy/KIVI](https://github.com/jy-joy/KIVI)。通过 monkey-patch 注意力实现接入 HuggingFace `generate`，支持 LLaMA 和 Mistral 系列。需要一个自定义 CUDA kernel 在注意力循环中进行 INT2 实时反量化，仓库已提供。无需校准——KIVI 是真正的无训练量化。典型开销：每 token 比 FP16 增加约 5–10% 延迟（反量化所致），与大 batch 下内存节省带来的吞吐提升相比微不足道。

**KVQuant**：代码位于 [github.com/SqueezeAILab/KVQuant](https://github.com/SqueezeAILab/KVQuant)，来自 Berkeley 同一团队（SmoothQuant、SpAtten 也出自此处）。需要一次性校准（128–512 样本）来计算逐通道量化级别并识别 outlier 通道。RoPE 前量化需要修改模型的注意力前向函数。非均匀量化级别以查找表形式存储（256 个条目，INT8 索引到 FP16 级别）；存储开销约为 KV cache 的 0.1%。

**集成状态**：截至 2024 年初，两者均未进入 vLLM 或 TGI 主线。两者都需要运行自定义注意力实现，限制了在标准 serving 栈上的便捷部署。vLLM 社区正在积极推进集成。对于生产环境，INT8 KV cache（vLLM 原生支持）是务实的第一步；KIVI/KVQuant 则为愿意维护自定义 kernel 的团队提供下一层级的压缩。

## 个人评注

KV cache 量化为 LLM serving 的量化体系画上了完整的句号：

- **GPTQ/AWQ** 处理权重——离线，serving 前，每个模型 checkpoint 处理一次。
- **SmoothQuant** 处理激活——在推理时，将权重和激活都保持在 INT8。
- **KIVI/KVQuant** 处理 KV cache——在推理时，处理与上下文和 batch size 成正比增长的那部分内存。

对部署工程师而言，实际操作顺序是：在训练时应用 GQA（免费的质量保留型 KV 头数压缩），然后在 serving 时应用 INT4 KV 量化（再压 2–4×，质量损失极小）。这两个旋钮完全正交——有 GQA 的模型再加 INT4 KV 量化，收益是相乘的。

KIVI 的非对称设计选择——Key 用 INT2、Value 用 INT4——值得深入理解。它来自于 Key 与 Value 使用方式的不同：Key 进入点积后结果被 softmax 压缩（容错性高），而 Value 被直接加权累加到输出中（误差敏感）。同样的差异化精度逻辑也出现在混合精度训练（前向和反向精度要求不同）以及投机解码（草稿 token 廉价，验证精确）中。

残差缓存的思想同样具有广泛适用性：每当你有一个频繁在近端访问的量化缓冲区时，将近期部分保留为全精度，可以以极低内存代价恢复大部分质量。这一模式出现在 KV cache 管理（KIVI）、环形注意力实现（最后一个 chunk 全精度）以及注意力 sink 研究（保留首尾 token 全精度）中。

展望未来：Blackwell 一代硬件已原生支持 FP8 KV cache，使 INT8 KV 成为轻而易举的选项，INT4 也将可在硬件层实现。KIVI 与 KVQuant 是这一趋势的研究基础——随着硬件支持的成熟，INT4 乃至 INT2 KV cache 很可能成为生产 serving 栈的标配。

## 参考

- [1] Liu et al. _KIVI: A Tuning-Free Asymmetric 2bit Quantization for KV Cache._ arXiv:2402.02750，2024。
- [2] Hooper et al. _KVQuant: Towards 10 Million Context Length LLM Inference with KV Cache Quantization._ arXiv:2401.18079，2024。
- [3] Ainslie et al. _GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints._ arXiv:2305.13245，2023。
- [4] Xiao et al. _SmoothQuant: Accurate and Efficient Post-Training Quantization for Large Language Models._ ICML '23 / arXiv:2211.10438。
- [5] Dettmers et al. _LLM.int8(): 8-bit Matrix Multiplication for Transformers at Scale._ NeurIPS '22 / arXiv:2208.07339。
- [6] Kwon et al. _Efficient Memory Management for Large Language Model Serving with PagedAttention._ SOSP '23 / arXiv:2309.06180。
