# SGLang —— RadixAttention 与结构化 LLM 程序前端

- **作者 / 机构**：Zheng 等，LMSYS Org + UC Berkeley + Stanford + CMU
- **发表时间**：2023-12（预印本），NeurIPS '24
- **链接**：[论文 (arXiv:2312.07104)](https://arxiv.org/abs/2312.07104) · [代码](https://github.com/sgl-project/sglang)

## 一句话总结

SGLang 把一个**结构化的多 call LLM 程序前端语言**（分支、循环、并行采样、工具调用）与一个 **RadixAttention 运行时**配对——后者通过**基数树**自动跨 call 复用 KV cache。核心洞察：LLM 应用已不再是单次 forward，而是一段包含很多相关 call 的程序，推理成本的大头是为共享前缀反复重算 KV。基于 token 的基数树以 **O(前缀长度)** 成本做前缀级 KV 复用，无需用户手动标注。论文报告 agentic workload 上相对 vLLM 时代基线 **最高 5× 吞吐**，输出完全等价。今天 SGLang 是与 vLLM、TensorRT-LLM 并列的两三个主流生产 serving 栈之一。

## 背景与动机

到 2023 年底，LLM workload 已从"一 prompt → 一 completion"转向：

- 大量共享历史的**多轮对话**。
- **Few-shot** 下成千上万请求使用相同指令前缀。
- **Agent / 工具调用**有公共 system prompt 与分支探索。
- **Tree-of-thought / self-consistency** 从一个前缀出发生出很多兄弟 completion。
- **RAG** 有大段重复的 context 片段。

那时代的 serving（vLLM、TGI）把每个请求当独立。KV cache 在请求**内部**存在，请求结束就扔；不同请求共享前缀也各自重算。对某些生产流量，这种冗余 prefill **占了算力的多数**。

vLLM 有原始的前缀缓存，但匹配逻辑粗（精确前缀、按请求显式开启）。SGLang 的目标是让前缀复用**自动、细粒度、程序感知**。

## 核心方法

### RadixAttention

一棵以 token ID 为索引的**基数树**（压缩 trie）。每个节点存一块 KV cache。新请求到来时：

1. 用请求的 prompt token 走树，找到**最长已缓存前缀**。
2. 该前缀的 KV 直接复用——不重算。
3. 对请求未命中的尾部做 prefill，生成的 KV 延展树。

关键是：这棵树在 serving 进程里**跨所有请求共享**。system prompt 重的 workload 下所有请求都汇聚到同一条祖先路径，它只加载一次。分支 workload（同前缀的多个 completion、beam search、tree-of-thought）共享内部节点。

淘汰：按树叶做 LRU，正在服务的请求用引用计数保护。

与 vLLM 逐 block 的前缀匹配不同，RadixAttention 在**任意前缀边界**工作——共享的 system prompt 可以在 block 中间结束。显存按页管理（与 PagedAttention 同思路），但索引是树，不是 hash。

### 前端语言

嵌入 Python 的 DSL 用于描述多 call LLM 程序：

```python
@sgl.function
def multi_turn(s, question):
    s += sgl.system("You are a helpful assistant.")
    s += sgl.user(question)
    s += sgl.assistant(sgl.gen("answer", max_tokens=512))
    s += sgl.user("Explain it simpler.")
    s += sgl.assistant(sgl.gen("simpler", max_tokens=512))
```

前端：

- 让前缀共享**显式**——每个 `s += ...` 扩展程序状态，跨 call 的共享前缀结构一目了然。
- 支持 **fork / merge**：从一个状态并行探索多条 continuation，再合并输出。
- 支持**约束解码**（regex、语法）——编译高效。
- 与工具调用、JSON mode、流式结合良好。

运行时利用前端结构提前准备 RadixAttention 查找、并对相关请求做批量化。

### 约束解码的压缩 FSM

对 regex / 语法约束生成（JSON 输出、结构化响应），SGLang 编译自动机并**把确定性转移折叠成单步**——当前 FSM 状态下一 token 唯一确定时跳过前向。真实带 JSON schema 的程序能获得显著加速。

### 后续性能工作

后续 SGLang 版本加入：

- **Flashinfer 集成**用高效 CUDA attention kernel。
- **张量并行**与混合 PD 解耦（与 DistServe / Mooncake 收敛）。
- **投机解码**——MTP、EAGLE 变体。
- **层次 KV cache**——把基数树扩到 CPU DRAM 与 SSD，在同一 RadixAttention 抽象下实现 Mooncake 式的缓存池。
- **调度改进**——chunked prefill、优先级队列、公平共享。

前端 + RadixAttention 的配对仍是它的独特核。

## 工程 Tradeoff

| 决策 | 得到什么 | 放弃什么 |
|---|---|---|
| 基于 token 的基数树 | 自动、细粒度前缀复用 | 树查找成本；淘汰逻辑不平凡 |
| 全请求共享一棵树 | 复用最大化 | 根节点附近争用；需要无锁或良好设计的并发 |
| 前端语言 | 把程序结构暴露给运行时 | 应用需采用 DSL 才能最大化收益（但 REST 风格 API 仍可用） |
| 压缩 FSM 约束解码 | JSON / 语法输出快 | 仅在 FSM 有确定性段落时有效 |
| 层次 KV cache | 冷前缀下沉到 CPU/SSD | 复杂度上升；miss 时延迟高 |

## 实验与结果

- 原论文：多轮 / RAG / agentic benchmark 上相对基线 serving **最高 5× 吞吐**；质量与朴素基线完全相同。
- **无前缀复用**的 workload 上，SGLang 与 vLLM 表现相当——采用不带惩罚。
- 更晚的 SGLang（2024–2025）整合了 PD 解耦和层次缓存，生产数字已接近 Mooncake 级架构。
- 采用：许多开源部署（Grok、Kimi、多种 Qwen 部署），学术推理研究常见选择。

## 复现要点

- 完全开源（`pip install sglang`）。
- 原生支持 OpenAI 兼容 API，同时提供原生 DSL。
- 与 PyTorch / Triton / FlashInfer 生态集成。
- 参考实现在 LMSYS 与合作组织中生产使用。

## 个人评注

SGLang 的持久力不寻常。很多推理论文被引用后，想法被 vLLM / TensorRT-LLM 吸收，原代码淡出。SGLang 反而**成了一个竞争性生产栈**，因为它的两个主张都强且组合得好：

1. **RadixAttention** 是前缀缓存的正确数据结构。hash 索引的 block 缓存（vLLM 早期）只能找精确匹配；基数树能找最长前缀且支持分支。两年后，大家都用某种基数树形的东西。
2. **前端语言**是**多 call LLM 程序**的正确抽象。agent 的兴起——每个 call 隐含是程序里的节点——验证了这个赌注。

2026 年 SGLang 与同辈的站位：

- **vLLM**：采用最广，已吸收基数树风格缓存与 PD 解耦。生态最大。
- **SGLang**：前缀缓存与结构化程序最激进。agentic 与 RAG workload 上强。
- **TensorRT-LLM**：NVIDIA 特化峰值最优。kernel 集成最紧。
- **NVIDIA Dynamo**：最年轻，押解耦与多模型 serving。

按 workload 选。重前缀复用或结构化多 call 程序——SGLang 是默认。纯 NVIDIA 部署上简单 workload 追求裸峰值——TensorRT-LLM。硬件与社区覆盖最广——vLLM。

## 参考

- [1] Zheng et al. _SGLang: Efficient Execution of Structured Language Model Programs._ NeurIPS '24 / arXiv:2312.07104.
- [2] SGLang 仓库：https://github.com/sgl-project/sglang
- [3] Kwon et al. _PagedAttention / vLLM._ SOSP '23.（基线与背景）
- [4] Zhong et al. _DistServe._ OSDI '24.（解耦，后被 SGLang 采纳）
- [5] Qin et al. _Mooncake._ FAST '25.（层次缓存，后被 SGLang 采纳）
