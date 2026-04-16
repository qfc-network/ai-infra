# Llama 4 — Meta 首个原生 MoE 模型

- **作者 / 机构**: Meta AI
- **发表时间**: 2025-04
- **链接**: [Llama 4 Scout (HuggingFace)](https://huggingface.co/meta-llama/Llama-4-Scout-17B-16E) · [Llama 4 Maverick (HuggingFace)](https://huggingface.co/meta-llama/Llama-4-Maverick-17B-128E) · [Meta 博客](https://ai.meta.com/blog/llama-4-multimodal-intelligence/)

## TL;DR

Llama 4 是 Meta 首个基于 MoE 的模型系列，于 2025 年 4 月发布。两个开放权重变体：**Scout**（每个专家 17B 参数 × 16 个专家，激活 17B）和 **Maverick**（17B × 128 专家，激活 17B，总参数 400B+）。两者均**原生多模态**：视觉编码器从零开始与语言模型联合训练，而非事后附加。注意力设计引入了 **iRoPE**——局部与全局注意力层交替排列，RoPE 仅作用于局部层，使极长上下文（Scout 支持 1000 万 token）在避免全局注意力二次代价的同时成为可能。本文聚焦基础设施含义：MoE 路由、多模态 token 预算、iRoPE serving，以及 Llama 4 与 [Llama 3 稠密架构](../llama3/zh.md)和 [DeepSeek V3 MoE](../../deepseek/v3-tech-report/zh.md) 的对比。

## 背景与动机

Llama 3 405B 是稠密模型，证明了 16k H100 规模的稠密训练可达前沿质量，但 serving 经济性不佳：405B 参数 × 2 字节（BF16）= 仅权重就需 810 GB。对每个 token 激活整个模型意味着每步解码都要从 HBM 读取 810 GB 数据。

MoE 直接解决了这个问题：保留大总参数量以获得容量，但每个 token 只激活一小部分。Llama 4 的设计——无论总规模如何，激活参数均为 17B——给出了明确的 serving 目标：即使 Maverick 有 128 个专家（总参数 400B+），解码显存占用与 17B 稠密模型相当。

多模态决策反映了 Meta 策略的转变：不再分别发布独立的视觉模型，而是将视觉作为从一开始就共同训练的一等模态。这影响了数据管线、tokenization 以及 serving 架构，远超事后附加的视觉适配器方案。

## 架构

### MoE 设计

Scout 和 Maverick 均采用标准 MoE Transformer 块结构：

- 注意力层：标准多头注意力 + GQA（与 Llama 3 相同）。
- FFN 层：替换为 MoE FFN 块。
  - `E` 个标准大小的专家 FFN。
  - top-K 路由器每个 token 选择 `K = 1` 个专家（Scout/Maverick 使用 top-1 路由，不同于 Mixtral 或 DeepSeekMoE 的 top-2）。
  - 一个共享专家（始终激活）与路由专家并列——类似 [DeepSeekMoE](../../deepseek/moe/zh.md) 的共享专家概念。

总参数 = `(注意力 + 共享 FFN + E × 专家 FFN) × 层数`。每个 token 激活的参数 = `(注意力 + 共享 FFN + 1 × 专家 FFN) × 层数` ≈ 两种变体均为 17B。

top-1 路由相比 top-2 简化了负载均衡（一个 token 对应一个专家，无需同步部分权重），但增加了路由方差——单次错误路由会完全丢弃该 token，而不能靠第二个专家部分挽回。

### iRoPE — 交替注意力

在 1000 万 token 上使用标准 RoPE + 全局注意力的二次代价不可承受。Llama 4 的解决方案：**交替**局部注意力与全局注意力层。

- **局部注意力层**（占多数）：标准滑动窗口注意力，固定窗口大小（如 8192 token）。RoPE 位置编码仅在此类层应用。
- **全局注意力层**（占少数，如每 4 层一个）：对整个序列做全量注意力。**全局层不使用 RoPE**——位置信息已由前面的局部层编码。

直觉：带 RoPE 的局部层处理窗口内的相对位置信息；全局层在不需要显式位置编码的情况下跨全序列聚合，因为其操作的表示已经嵌入了局部位置结构。

这与 [Gemma 2 的](../../google/gemma2/zh.md)交替局部/全局注意力不同——后者两类层各自有独立的位置编码。iRoPE 的不对称是刻意设计：局部层有 RoPE，全局层裸注意力。

**Serving 含义**：KV cache 需求因层类型而异。局部层 KV cache 有上界 = `窗口大小 × 局部层数 × 2 × d_head × n_kv_heads`；全局层 KV cache 无上界 = `seq_len × 全局层数 × ...`。Scout 1000 万 token 上下文中，全局层 KV cache 占主导；高效 serving 需要 CPU 卸载或精细的内存管理。关于 radix-trie 缓存与混合注意力类型的交互，见[前缀缓存](../../foundational/prefix-caching/zh.md)。

### 视觉编码器

Llama 4 包含一个原生视觉编码器（基于 ViT 衍生架构），与语言模型联合训练。图像被 tokenize 为固定数量的视觉 token（patch），投影到语言模型的嵌入空间后与文本 token 交替排列。

关键基础设施要点：
- **变分辨率编码**：图像缩放至 token 预算内（如 4096 个视觉 token）；高分辨率图像被分块处理，各块独立编码后拼接。
- **无独立交叉注意力**：视觉 token 直接占据序列中的位置，通过普通 transformer 注意力与文本 token 互相 attend。架构简单，但序列长度增加。
- **Prefill 代价**：单张高分辨率图像可产生数千个 token，显著影响 TTFT。分块 prefill（见 [Sarathi-Serve](../../foundational/chunked-prefill/zh.md)）的重要性因此提升。

## 训练基础设施

Meta 的博客和模型卡片描述了数万张 H100 的训练集群。继承自 Llama 3 的关键选择（完整细节见 [Llama 3 条目](../llama3/zh.md)）：

- 4D 并行：TP + PP + CP + DP。
- 全程 BF16 混合精度。
- 视觉编码器使用梯度检查点。

MoE 带来的新考量：
- **专家并行（EP）**：`E` 个专家 FFN 分片到 EP 组。每张 GPU 持有部分专家；all-to-all 通信将 token 路由到正确的专家设备。128 个专家时，EP 不可或缺——将所有专家存储在单节点上不可行。
- **负载均衡**：辅助损失项惩罚不均匀的专家利用率。top-1 路由增加负载不均风险；共享专家吸收路由器不确定的 token。
- **通信模式**：EP all-to-all 在每个 MoE 层增加一次网络密集步骤，类似 [DeepEP](../../deepseek/open-source-week/deep-ep/zh.md) 的方案。NVLink 处理节点内通信；RDMA/InfiniBand 处理节点间通信。

## 模型规模与关键数字

| 变体 | 专家数 | 激活参数 | 总参数 | 上下文 | 视觉 |
|---|---|---|---|---|---|
| Scout | 16 | ~17B | ~109B | 1000 万 token | 是 |
| Maverick | 128 | ~17B | ~400B+ | 100 万 token | 是 |

两种变体使用相同的 17B 专家大小，区别在于专家数量（以及总容量）。Scout 针对长上下文效率优化；Maverick 针对标准上下文长度的质量优化。

## 工程权衡

| 决策 | 收益 | 代价 |
|---|---|---|
| top-1 路由（vs top-2） | 负载均衡更简单；每层一次 all-to-all | 路由方差更高；错误路由无法用第二专家补救 |
| 共享专家与路由专家并列 | 稳定基础表示；吸收不确定 token | 每个 token 额外计算（共享专家始终激活） |
| iRoPE（RoPE 仅在局部层） | 无二次代价的 1000 万上下文 | 混合 KV cache 大小使内存管理复杂；非标准注意力 |
| 原生多模态（联合训练） | 更紧密的视觉-语言对齐 | 更大的训练数据集和管线复杂度；视觉 token 增加 prefill 代价 |
| 变分辨率分块 | token 预算内处理高分辨率图像 | 高分辨率图像 → 大 prefill 批次 → TTFT 压力 |
| 无论总规模均激活 17B | Serving 代价与激活量而非总量成正比 | Maverick 400B+ 总权重仍需存储（冷存储或 EP 分片） |

## 对比：Llama 4 vs DeepSeek V3

两者都是在数月内发布的大型 MoE 模型，针对相似的质量水平。关键差异：

| 方面 | Llama 4 Maverick | DeepSeek V3 |
|---|---|---|
| 总参数 | ~400B+ | 671B |
| 激活参数 | ~17B | ~37B |
| 路由 | top-1 + 共享专家 | top-2 + 共享专家（DeepSeekMoE 细粒度） |
| 注意力 | iRoPE（局部+全局交替） | MLA（潜在 KV 压缩） |
| 视觉 | 原生联合训练 | 纯文本 |
| 训练精度 | BF16 | FP8（首个主要模型） |
| 上下文 | 100 万（Maverick）/ 1000 万（Scout） | 128k |

DeepSeek V3 每个 token 激活更多参数（37B vs 17B），通常意味着更高的每 token 质量，但 serving 计算代价更高。Llama 4 的 iRoPE 提供了主要的上下文长度优势；V3 的 MLA 提供了主要的 KV cache 压缩优势。

## 深度解析

Llama 4 代表 Meta 追赶 DeepSeek 和 Mistral 所建立的 MoE 趋势。技术贡献与其说是全新机制（iRoPE 是已有局部/全局注意力思想的合理延伸），不如说是**集成**：原生多模态 + MoE + 超长上下文集成在单一开放权重模型中。

基础设施挑战是三个各自棘手的组件叠加的产物。MoE serving 需要专家并行和负载均衡；超长上下文 serving（1000 万 token）需要分层 KV cache 管理；多模态 serving 需要处理可主导 prefill 代价的可变长度视觉 token 序列。三者叠加使 Llama 4 成为比 Llama 3 复杂得多的 serving 目标——即使激活参数量更低。

对实践者而言：Scout 的 1000 万上下文 + 17B 激活参数是最有趣的效率点——有竞争力的质量、超长上下文，且 serving 占用与标准 17B 稠密模型相当。代价是全局注意力层无上界的 KV cache。

## 参考文献

- [1] Meta AI. _Llama 4: Multimodal Intelligence at Scale._ Blog post, April 2025.
- [2] Meta AI. 模型卡片：`meta-llama/Llama-4-Scout-17B-16E`，`meta-llama/Llama-4-Maverick-17B-128E`。HuggingFace，2025.
- [3] MoE 路由背景：[DeepSeekMoE](../../deepseek/moe/zh.md)、[DeepSeek V3](../../deepseek/v3-tech-report/zh.md)、[Mixtral](../../mistral/mixtral/zh.md)。
- [4] 注意力背景：[GQA](../../foundational/gqa/zh.md)、[RoPE](../../foundational/rope/zh.md)、[Gemma 2 交替注意力](../../google/gemma2/zh.md)。
- [5] Serving 背景：[前缀缓存](../../foundational/prefix-caching/zh.md)、[分块 Prefill](../../foundational/chunked-prefill/zh.md)。
