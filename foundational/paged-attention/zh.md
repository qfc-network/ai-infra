# PagedAttention / vLLM

- **作者 / 机构**：Kwon 等，UC Berkeley
- **发表时间**：2023-09（SOSP '23）
- **链接**：[论文 (arXiv:2309.06180)](https://arxiv.org/abs/2309.06180) · [vLLM](https://github.com/vllm-project/vllm)

## 一句话总结

LLM 推理的瓶颈不是算力，而是 **KV cache 显存碎片化**。传统系统按最大长度给每个请求预留一整块连续 KV cache，60–80% 显存浪费在 padding 和预留上。PagedAttention 从操作系统借来**虚拟内存分页**：每个请求的 KV cache 切成固定大小的块，维护每请求的 block table，让 attention kernel 从非连续块中 gather K、V。由此**碎片几乎归零**、**连续批处理（continuous batching）**可行、共享前缀 / beam search 可以 copy-on-write，吞吐比此前 serving 系统高 2–4×。参考实现 vLLM 已成为开源推理的事实标准。

## 背景与动机

2023 年 FasterTransformer、HF TGI 等 serving 系统把每个请求的 KV cache 分配为一个连续张量，尺寸按最大可能长度预留。两个复合问题：

1. **内部碎片**——多数请求提前结束，预留空间浪费。
2. **外部碎片**——请求结束后释放的连续块，形状匹配不上新请求的需求。

结果：GPU 显存成为 serving 瓶颈，吞吐低，跨请求共享 KV（前缀缓存、beam search、并行采样）在工程上不现实——cache 被物理绑到单一请求上。

## 核心方法

### 块与 block table

KV cache 以**固定大小的块**存储（典型每块每层 16 个 token 的 K 和 V）。每个请求有一张 **block table**，把逻辑 token 位置映射到物理 block ID，完全对应 OS 里的页表。

- 新请求 → 按需分配块。
- 请求结束 → 块回收到空闲池；所有块同形同大，无碎片。
- 碎片率从 60–80% 降到 4% 以下。

### PagedAttention kernel

Attention kernel 改写为**通过 block table 间接 gather 非连续物理块的 K、V**。块大小与 warp / 内存事务对齐，gather 开销很小。

### 连续批处理

分页存储让请求可以**在任意步被加入或移出 in-flight batch**，而不只是在 batch 边界。配合迭代级调度，变长 workload 下 GPU 仍能打满。

### 共享块：CoW 与前缀缓存

多个逻辑序列在 KV 内容相同时可以**共享同一个物理块**，适用于：

- **Beam search / 并行采样**：兄弟序列共享 prompt 前缀。
- **前缀缓存**：重复的 system prompt / few-shot 样例只编码一次，跨请求复用。
- **Copy-on-write**：共享块需要分叉时，在那一刻才复制。

这些对模型透明，全部逻辑在 block table 与 kernel 内。

## 工程 Tradeoff

| 决策 | 得到什么 | 放弃什么 |
|---|---|---|
| 固定块大小 | 无外部碎片，分配器简单 | 每序列最后一块有少量内部碎片 |
| Block table 间接寻址 | 灵活共享，无需物理连续 | Attention kernel 要 gather，per-op 略有开销 |
| 连续批处理 | 变长 workload 吞吐高 | 调度复杂度上升；延迟尾部更难推理 |
| CoW 共享前缀 | 多采样 / 共享 system prompt 场景大幅受益 | 引用计数 + CoW 簿记 |
| Python + 自定义 CUDA 混合栈 | 迭代快、模型覆盖广 | 小批量低延迟场景 Python 开销可见；后续以 C++ 引擎缓解 |

## 实验与结果

- OPT 与 LLaMA 13B–175B 规模，较 FasterTransformer、HuggingFace TGI **吞吐提升 2–4×**。
- GPU 显存接近满载，碎片 < 4%。
- vLLM 采用度：几乎所有模型家族（Llama、Mistral、Qwen、DeepSeek 等）的默认开源推理栈；支持张量并行、量化（AWQ / GPTQ / FP8）、投机解码、prefill/decode 解耦部署。

## 复现要点

- 完全开源，`pip install vllm` 即可本地跑。
- 块大小（默认 16）可调；更大块降低开销、增加内部碎片。
- 主流后端均有对应 kernel（CUDA、ROCm、TPU via JAX port、Inferentia）；部分功能（CoW、某些量化方案）在非 CUDA 路径上滞后。

## 个人评注

PagedAttention 是 LLM infra 里 **"OS 思想在 ML 上兑现红利"** 最清晰的例子。分页类比不是装饰——论文把 `malloc/free` 的低效与碎片直接映射到 KV cache，再套用已知的解法。更深的启示是：**LLM serving 是内存管理问题，不是算力问题**，这个框架还在持续产出新工作——基于块的 radix 树（SGLang）、prefill/decode 解耦部署（DistServe）、多级 KV cache（host DRAM / SSD）——都是直系后代。做规模化推理的，vLLM 是默认起点；理解它的内部机制不再是可选项。

## 参考

- [1] Kwon et al. _Efficient Memory Management for Large Language Model Serving with PagedAttention._ SOSP '23 / arXiv:2309.06180.
- [2] vLLM project. https://github.com/vllm-project/vllm
- [3] Zheng et al. _SGLang._ arXiv:2312.07104, 2023.（基于 radix 树的扩展）
- [4] Zhong et al. _DistServe._ OSDI '24.（prefill/decode 解耦）
