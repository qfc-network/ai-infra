# Orca —— 大模型服务的持续批处理

- **作者 / 机构**：Gyeong-In Yu 等，首尔国立大学 / NAVER
- **发表时间**：2022-11（OSDI '22）
- **链接**：[论文](https://www.usenix.org/conference/osdi22/presentation/yu) · [代码（未开源；vLLM 是标准开源实现）](https://github.com/vllm-project/vllm) · [博客（Anyscale 解读）](https://www.anyscale.com/blog/continuous-batching-llm-inference)

## 一句话总结

Orca 之前，所有大模型推理系统都在**请求级别**做批处理：把一批请求收集齐、全部 pad 到最长序列、等所有请求都生成完毕、才放入下一批。短请求被长请求拖累，GPU 算力大量消耗在填充上。Orca 的答案是**迭代级别调度**（也叫持续批处理或 in-flight batching）：调度器在每一个解码步之后重新决策，完成的序列立即退出，新序列无需等待整批结束就能进入。在 OPT-30B 上，Orca 相比当时最优的 FasterTransformer 获得了 **36.9 倍吞吐量提升**。Orca 没有解决 KV 缓存内存碎片问题（那要等 PagedAttention / vLLM），但它确立了现代推理系统普遍采用的调度原语。

## 背景与动机

### 请求级调度的瓶颈

2022 年，FasterTransformer、早期 HuggingFace TGI、DeepSpeed Inference 等主流推理系统共用同一个服务循环：

1. 收集请求，直到凑满一批或超时。
2. 将批内所有序列 pad 到最长序列的长度。
3. 把整批跑完，直到**所有**序列都输出了 EOS。
4. 释放这批，接入下一批。

这个设计有两个叠加问题：

**队头阻塞。** 如果一批中某个请求需要生成 2,000 个 token，其余的只需 50 个，短请求就要白白等待 1,950 个额外的解码步。短请求的尾延迟完全被批内最长请求控制。

**填充浪费。** 序列长度分布高度偏斜，少数长序列迫使绝大多数成员做大量 pad。实际 token 计算的 GPU 利用率可能远低于 50%。

这两个问题造成了一个根本性的困境：要提吞吐就要大批次（加剧队头阻塞）；要降延迟就要小批次（降低 GPU 利用率）。请求级调度框架内没有好答案。

### 大模型推理的两阶段结构

Orca 的关键观察是大模型推理有两个结构迥异的阶段：

- **预填充（Prefill）**：整个 prompt 在一次前向传播中处理完，注意力跨所有 prompt token 并行计算。这个阶段**受计算瓶颈约束**——即使在小批次下也能让 GPU 保持高利用率。
- **解码（Decode）**：每次自回归地生成一个 token，每步模型都要对整个 KV 缓存（所有历史 token）做注意力。这个阶段**受内存带宽约束**——每步都要从 HBM 加载 KV 缓存，单个新 token 给 GPU 提供的计算量不足以掩盖内存延迟。

请求级调度器把这两个阶段混在一起对待。Orca 的调度器将它们分开处理。

## 核心方法

### 迭代级调度

核心思想：不再以请求（完整序列）为调度粒度，改以**迭代**（单个解码步）为粒度。每次解码步之后，调度器可以：

- 移除已输出 EOS 的序列。
- 从等待队列中加入新序列。
- 继续运行生成中的序列。

批的组成每步都在变化。不再有等待批内最慢序列的问题。新序列在预填充完成后立即进入，无需等待。

Orca 引入的关键吞吐量指标是**有效吞吐量（goodput）**：单位时间内成功完成的请求数，而非原始的 GPU FLOP 利用率。Goodput 把优化目标对齐到用户真正关心的事——请求完成——而不是不一定转化为业务价值的硬件效率指标。

### 选择性批处理

Transformer 中并非每个操作都能在不同长度的序列上直接批处理。最大的挑战是**注意力层**：标准批注意力要求序列等长，否则就要 pad 到最大长度。

Orca 的选择性批处理识别出哪些层可以直接跨变长序列批处理、哪些需要特殊处理：

- **前馈层、LayerNorm、投影层**：这些操作对每个 token 独立执行，来自不同序列的 token 可以在批维度上拼接、一起计算，完全不需要 pad。Orca 正是这样做的——这些层的"批"是所有活跃 token 拼成的一维张量。
- **注意力层**：KV 缓存是按序列、按层分开的；注意力运算必须尊重序列边界。Orca 用一个自定义注意力核来分别处理每条序列的注意力，然后拼接输出。这比完全融合的批注意力效率略低，但避免了 pad。

正是这一区分让迭代级调度在工程上可行：对廉价层拼接不同序列的 token，Orca 即使在不同位置的混合批次下也能维持高 GPU 利用率。

### 调度循环

```
while serving:
    # 预填充新接入的请求
    for each new_request in admitted:
        KV[new_request] = prefill(new_request.prompt)

    # 解码步：每条活跃序列生成一个 token
    new_tokens = selective_batch_decode(all_live_sequences)

    for seq, token in zip(all_live_sequences, new_tokens):
        seq.append(token)
        if token == EOS or len(seq) >= seq.max_length:
            emit(seq)
            free(KV[seq])
            admit_next_from_queue()
```

调度器持续运行这个循环。序列在中途进出批次，批始终保持满负荷（上限为内存预算），消除了请求级系统里的空闲时间。

### 内存：静态 KV 分配

Orca 早于 PagedAttention。KV 缓存在请求接入时**按最大可能输出长度静态分配**。这是 Orca 设计的主要局限：对提前结束的序列，预留但未用的空间白白占用 GPU 内存；若实际输出长度超过预分配上限，请求无法服务。

迭代级调度（Orca）+ 虚拟分页 KV 缓存（vLLM/PagedAttention）的组合，才是现代大模型服务高效运转的原因。Orca 贡献了调度策略，PagedAttention 贡献了内存管理。

## 工程 Tradeoff

| 决策 | 得到什么 | 放弃什么 |
|---|---|---|
| 迭代级 vs 请求级调度 | 几乎零空闲时间；短请求不再被长请求阻塞；36.9× 吞吐提升 | 调度器每解码步运行一次，控制面开销更大；批组成难以静态分析 |
| 预填充与解码作为独立阶段 | 可分别优化调度；对计算 vs 内存瓶颈有更清晰的成本核算 | 预填充会抢占正在进行的解码迭代，给在途序列引入延迟毛刺（TTFT vs TPOT 张力） |
| Goodput 作为优化目标 | 对齐到用户可见延迟；暴露调度策略的真实权衡 | Goodput 可以在牺牲尾延迟的情况下被最大化——持续被降优先级的请求永远完不成 |
| 静态 KV 缓存分配 | 分配器简单；单请求生命周期内无碎片 | 最多 60–80% GPU 内存浪费在 pad 和提前终止的剩余空间；批大小上限严苛 |
| 中心化调度器 | 便于推理全局内存预算和调度策略 | 单点瓶颈；不能自然扩展到预填充/解码拆分部署（DistServe、Mooncake 架构） |

## 实验与结果

Orca 论文在 OPT-30B 和 OPT-66B 上与 FasterTransformer（2022 年最优推理库）对比，在不同请求率和输出长度下衡量吞吐（请求/秒）。

核心数据：

- 在匹配延迟 SLO 的条件下，OPT-30B 上相比 FasterTransformer **吞吐提升 36.9×**。这是标题数字，基于真实 trace 数据的混合 prompt 与输出长度分布。
- OPT-66B 上提升约 22×，略低是因为更大的模型受内存带宽约束更强，对调度效率的敏感度稍低。
- 平均延迟等比提升；P99 尾延迟改善更显著——队头阻塞被消除。
- GPU 利用率从 FasterTransformer 满载下的约 30–40% 提升至 Orca 的 70–90%。

评估使用的是 NAVER 生产服务系统的真实 trace，让结果的可信度超越合成 benchmark。输出长度分布是长尾的，正是请求级调度表现最差、迭代级调度收益最大的场景。

## 复现要点

原始 Orca 代码库未开源，但迭代级调度现已是主流开源服务框架的默认行为：

- **vLLM**：将 Orca 式调度与 PagedAttention 结合的完整开源实现。`pip install vllm` 即可两者兼得。
- **TGI（HuggingFace Text Generation Inference）**：v0.9（2023）起添加持续批处理，明确引用 Orca。
- **LightLLM**、**TensorRT-LLM**：均以迭代级调度为默认。

复现核心行为：部署 vLLM，启用异步引擎模式，用混合长度 workload（如 ShareGPT trace）压测，与禁用持续批处理的朴素请求级基线对比。吞吐差距显著且可重现。

关键参数：
- `max_num_seqs`：in-flight 批的最大序列数。增大提升吞吐，代价是内存和尾延迟。
- `max_num_batched_tokens`：每次迭代处理的 token 上限。控制每步的预填充预算。
- 预填充/解码协同调度策略：vLLM 默认优先预填充；调整这个参数在 TTFT 和 TPOT 之间权衡。

## 个人评注

Orca 的重要性不在于思想复杂——其实并不复杂。迭代级调度的想法一旦说出来就很直白。使它产生影响的是：有人认认真真地把它实现好、仔细测量、然后发在顶级系统会议上。36.9× 这个数字给整个社区提供了一个共同的参照点。

更深的教训是关于**抽象层边界**。请求级调度是从经典批量推理（图像分类、embedding 生成）继承的——那些场景下所有请求等长，也没有自回归解码阶段。这个抽象对大模型是错的，修正这个抽象——不改模型本身——就恢复了一个数量级的性能。

Orca 设计的遗留局限直接对应它选择不处理的问题：

- **静态 KV 分配**：由 PagedAttention（vLLM，2023）解决。
- **预填充/解码相互干扰**：长 prompt 的预填充会阻塞所有在途序列的解码进度。Sarathi（Agrawal 等，2024）用分块预填充解决——把大预填充切成小块、与解码步交错执行，同时平滑 TTFT 和 TPOT。
- **规模下的中心化调度器**：DistServe 和 Mooncake 彻底把预填充和解码池拆开，跑在独立硬件上、用独立调度器。Orca 的中心化设计是它们改进的基线。

学推理 infra 的人：先把 Orca 搞懂。后续每一篇大模型服务论文——vLLM、SGLang、Sarathi、DistServe、Mooncake——要么建立在迭代级调度之上，要么在明确解决 Orca 留下的某个问题。

## 参考

- [1] Yu et al. _Orca: A Distributed Serving System for Transformer-Based Generative Models._ OSDI '22. https://www.usenix.org/conference/osdi22/presentation/yu
- [2] Kwon et al. _Efficient Memory Management for Large Language Model Serving with PagedAttention._ SOSP '23 / arXiv:2309.06180.
- [3] Agrawal et al. _Sarathi: Efficient LLM Inference by Piggybacking Decodes with Chunked Prefills._ arXiv:2308.16369, 2023.
- [4] Zhong et al. _DistServe: Disaggregating Prefill and Decoding for Goodput-Optimized Large Language Model Serving._ OSDI '24.
- [5] Holmes et al. _Deepspeed-FastGen._ arXiv:2401.08671, 2024.
- [6] vLLM 项目. https://github.com/vllm-project/vllm
