# NVIDIA Dynamo — 集群级分离式推理编排框架

- **作者 / 机构**：NVIDIA
- **发布**：2025 年 3 月开源
- **链接**：[GitHub](https://github.com/ai-dynamo/dynamo) | [博客："NVIDIA Dynamo: A Scalable Inference Framework"](https://developer.nvidia.com/blog/nvidia-dynamo-a-scalable-inference-framework/)

## 一句话总结

NVIDIA Dynamo 是用于 LLM 推理的集群级编排框架，而非内核库。它将 prefill 和 decode 分离为独立的工作池，将请求路由到已缓存相关 KV 数据的 prefill 工作节点，并通过 RDMA/NVLink 在节点间传输 KV 块——将缓存局部性转化为减少的重复计算。Dynamo 以 TensorRT-LLM 作为单 GPU 计算引擎，将学术界的分离式推理研究（DistServe、Mooncake）推进到 H100/B200 集群上的生产级硬件加速部署。

## 背景与动机

生产 LLM 推理集群存在两种根本不同的工作负载模式，而标准的一体化服务系统强制将它们混在同一张 GPU 上：

- **Prefill（预填充）**：受计算瓶颈。在单 GPU 上处理一个 1000-token 的提示是大规模稠密矩阵乘法——算术强度高，GPU 算力是瓶颈。
- **Decode（解码）**：受内存带宽瓶颈。每次只生成一个 token 需要为每一步加载完整的 KV 缓存和所有模型权重——内存带宽是瓶颈，而非算力。

将两者混在同一张 GPU 上导致不良妥协：GPU 针对一种模式调优，在另一种模式下利用率低下。大规模下，这会叠加成 30–50% 的 GPU 浪费（NVIDIA 内部基准数据）。

学术界的框架来自 DistServe（见 [../../foundational/distserve/zh.md](../../foundational/distserve/zh.md)）和 Mooncake（见 [../../moonshot/mooncake/zh.md](../../moonshot/mooncake/zh.md)）：将 prefill 和 decode 阶段分离到不同的工作池。Dynamo 是 NVIDIA 对这一理念的生产实现，具备硬件级集成（节点内 KV 传输用 NVLink，跨节点用 RDMA/InfiniBand）、KV 感知路由器和在线动态调整工作池大小的规划器。

第二个动机是**跨请求的 KV 缓存复用**。在朴素的服务设置中，即使每个请求与其他千个请求共享同一个长系统提示，prefill 也会从头重新计算。Dynamo 的 KV 路由器跟踪哪些 prefill 工作节点持有哪些前缀 KV 块，并将新请求路由到已缓存匹配块的节点——消除重复计算。

## 核心抽象

### Processor（处理器）

Processor 是处理一组请求的 prefill 或 decode 的工作进程。每个 Processor 在一张或多张 GPU 上运行 TensorRT-LLM 引擎。从 Dynamo 的视角看，Processor 是无状态的——接收来自 Dynamo 层的批次描述符，执行前向传播，返回 KV 块或生成的 token。

Prefill Processor 和 Decode Processor 是具有不同资源特征的不同类型：
- **Prefill Processor**：需要高算力吞吐（H100 SXM 为佳），受益于大 L2 缓存进行权重复用（长提示），但工作 KV 缓存相对较小（prefill 只处理每个提示一次然后将 KV 传出）。
- **Decode Processor**：需要高 HBM 带宽，受益于与大量并发序列批处理，需要更大的 KV 缓存保存活跃解码状态。NVLink 连接的多 GPU 配置更优，以最大化跨 GPU KV 带宽。

### KV 路由器

KV 路由器是 Dynamo 的核心调度器。对于每个入站请求，它：

1. 对输入前缀进行分词。
2. 查询分布式 KV 索引，找到哪个 Prefill Processor 已缓存了最长的前缀匹配。
3. 将请求路由到该 Processor——若无好的匹配，则路由到负载最轻的 Prefill Processor。
4. Prefill 完成后，KV 路由器选择一个 Decode Processor 并启动 KV 传输。

这是缓存感知路由：路由决策是缓存状态的函数，而非仅仅是负载。其直觉与 [SGLang 的 RadixAttention](../../foundational/sglang/zh.md)（在缓存树中匹配最长公共前缀）类似，但在集群级别跨多个工作节点运行，而非在单个工作节点的本地缓存内。

分布式 KV 索引由每个 Prefill Processor 通过类 gossip 协议向路由器报告其缓存前缀集来维护。索引是最终一致的；在索引更新和请求到达之间，路由器偶尔可能路由到已驱逐相关块的 Processor。

### KV 传输

Prefill 完成后，计算好的 KV 块必须从 Prefill Processor 移动到 Decode Processor。Dynamo 使用：

- **NVLink** 用于节点内传输（在 NVLink 连接的多 GPU 服务器内）：约 600 GB/s 双向带宽。以此带宽，传输 Llama-70B 模型中 4096-token 提示的 KV 缓存（FP8 约 1.5 GB）需约 2.5ms——与一个 decode 步骤相当。
- **RDMA/InfiniBand** 用于跨节点传输：RDMA 约 400 Gb/s，相同 1.5 GB 的传输约需 50ms。这不可忽视，意味着跨节点 KV 传输会显著增加首 token 时间（TTFT）。

Dynamo 的传输实现对节点内使用 CUDA IPC，对跨节点使用 NCCL/UCX，通过 GPU 直接点对点访问绕过 CPU 拷贝路径。

这带来一个关键约束：**尽可能将 prefill 和 decode 工作节点同置于同一节点**，以利用 NVLink 带宽。仅当工作池扩展需要时才采用跨节点分离，并接受 InfiniBand 延迟惩罚。

### Planner（规划器）

Planner 是 Dynamo 的在线性能分析器和资源管理器。它持续观察：

- **前缀命中率**：在 KV 路由器中找到显著前缀匹配的请求比例。若命中率下降，Planner 可能扩展 Prefill Processor 池或调整驱逐策略。
- **Decode 长度分布**：输出长度的 p50/p90/p99。较长的解码序列需要更多 Decode Processor 容量。
- **队列深度**：每个池的等待时间。若 prefill 队列积压，增加 Prefill Processor；若 decode 队列积压，增加 Decode Processor。

Planner 在池级别做出扩缩容决策——不抢占单个请求。它将扩缩容决策传达给集群编排层（Kubernetes 或 NVIDIA Base Command）。扩展 Processor 池需要启动新的 TRT-LLM 引擎进程，大型模型需要 30–120 秒（权重加载时间），因此 Planner 的决策具有显著的惯性。

## 与 TensorRT-LLM 的集成

Dynamo 处理编排；TensorRT-LLM（见 [../tensorrt-llm/zh.md](../tensorrt-llm/zh.md)）处理单 GPU 推理。两者之间的接口：

- Dynamo 通过 gRPC 将批次调度到 TRT-LLM 引擎。
- TRT-LLM 引擎为 prefill 批次返回 KV 块，为 decode 批次返回生成的 token。
- Dynamo 的 KV 传输层随后将 KV 块移动到目标 Decode Processor。

TRT-LLM 提供 Dynamo 所依赖的底层内核效率（融合注意力、FP8 量化、飞行中批处理）。Dynamo 提供 TRT-LLM 所不具备的集群级路由、KV 跟踪和池管理。

这种分离意味着 Dynamo 在原则上与后端无关。Processor 的 gRPC 接口也可以由 vLLM 或 TGI 提供，尽管在发布时 TRT-LLM 是主要集成目标。

## 内存架构

在分离式设置中，KV 缓存内存在每个池中独立管理：

**Prefill Processor**：仅在活跃 prefill 计算期间持有 KV 缓存，完成后立即传输并驱逐。每个请求的 KV 驻留时间：O(提示长度 / prefill 吞吐量)——通常 50–200ms。工作 KV 占用小；瓶颈是 prefill 计算期间的 HBM 带宽。

**Decode Processor**：在整个解码期间持有 KV 缓存——从 TTFT 到最后一个输出 token。在 Llama-70B 模型中，100 个并发序列每个 2048 token：FP8 下约 30 GB 活跃 KV。HBM 容量是 decode 池的主要规模约束。

集群总 KV 内存大致为：
```
KV_total ≈ (prefill 批大小 × 提示长度 × 每 token KV_prefill)
          + (decode 并发数 × decode 长度 × 每 token KV_decode)
```
其中 `每 token KV` 取决于模型架构和量化精度。

## 工程权衡

| 决策 | 收益 | 代价 |
|---|---|---|
| 分离 prefill/decode 池 | 每个池针对其瓶颈（算力 vs. 带宽）单独调优；无混合模式妥协 | KV 传输开销（每请求 2–50ms，取决于同置情况）；路由复杂性 |
| KV 感知路由（缓存感知调度） | 消除共享前缀的重复 prefill；高前缀复用下平均 TTFT 降低 | 路由器维护最终一致的分布式 KV 索引；快速驱逐下存在陈旧缓存未命中 |
| Planner 驱动的动态池扩缩容 | 无需人工干预即可适应工作负载变化 | 扩缩容延迟（新 Processor 上线需 30–120s）；突发负载下 Planner 可能震荡 |
| TRT-LLM 作为计算引擎 | H100 顶级内核效率；FP8 支持；融合操作 | 绑定 NVIDIA 硬件；TRT-LLM 引擎构建时间（每模型需数分钟）；灵活性不如 PyTorch 后端 |
| 节点内优先使用 NVLink KV 传输 | 同置工作节点近零开销 KV 传输（2–5ms） | 强制同节点 prefill/decode 紧耦合；跨节点扩缩容承受 InfiniBand 延迟惩罚 |
| Dynamo 与 Processor 间使用 gRPC | 清晰分离；与后端无关的接口 | 每批次额外序列化开销；延迟下限高于进程内批处理 |

## 生产部署注记

Dynamo 面向 10 到 100 张 GPU 规模的 H100/B200 集群部署。关键操作特性：

- **最小有效规模**：仅当集群有足够 GPU 有意义地分离 prefill 和 decode 池时，分离才有回报（每个池至少约 8 张 GPU）。对于较小部署，直接使用 TRT-LLM 的单体服务更简单。
- **前缀命中率敏感性**：Dynamo 的效率优势强烈依赖于前缀命中率。对于提示多样、无重叠的工作负载（通用聊天机器人流量），命中率可能为 10–20%；对于具有共享文档上下文的 RAG 管道或具有共享系统提示的编程助手，命中率可达 60–80%。
- **模型更新开销**：当模型权重更新时（微调切换、版本更新），所有 Processor 上的全部 KV 缓存失效。Dynamo 需要协调刷新，暂时降低缓存命中率。
- **可观测性**：Planner 的指标（命中率、队列深度、decode 长度分布）是主要运维信号。提供 Prometheus/Grafana 集成。

## 评述

Dynamo 最好被理解为 DistServe/Mooncake 研究洞察的产品化：分离 prefill 和 decode 是集群规模下正确的架构选择，但要正确实现需要同时解决五个复杂系统问题——KV 路由、KV 传输、池规模调整、缓存失效和容错。学术论文干净地解决了其中一两个；Dynamo 试图用硬件加速基础设施一次解决全部五个。

技术上最有趣的组件是 KV 路由器。集群级缓存感知路由是一个分布式系统问题，学术文献低估了其难度：在持续驱逐压力下，跨数百个工作节点维护哪个节点持有哪些 KV 前缀块的一致视图，需要仔细的一致性权衡。Dynamo 的最终一致性模型（类 gossip 索引更新）是正确选择——强一致性会为每个路由决策增加不可接受的延迟——但这意味着系统必须优雅处理由陈旧索引状态导致的缓存未命中。

Dynamo 的长期问题是与 TRT-LLM 和 NVIDIA 硬件的紧密耦合究竟是特性还是局限。对于在 H100 或 B200 的 NVLink 集群上部署的企业，答案显然是"特性"——FP8 融合内核和 NVLink KV 传输的性能提升会相互叠加。对于多云或异构部署，gRPC 抽象边界是支持其他后端工作发生的地方。随着分离式推理模式成为主流服务架构，可以预期 Dynamo 的 KV 路由器和 Planner 将逐步演进为支持非 NVIDIA 加速器。

## 参考文献

- [1] NVIDIA. _Dynamo: A Scalable Inference Framework._ https://github.com/ai-dynamo/dynamo
- [2] Zhong et al. _DistServe: Disaggregating Prefill and Decoding for Goodput-Optimized LLM Serving._ arXiv:2401.09670。见 [../../foundational/distserve/zh.md](../../foundational/distserve/zh.md)
- [3] Qin et al. _Mooncake: A KVCache-centric Disaggregated Architecture for LLM Serving._ 见 [../../moonshot/mooncake/zh.md](../../moonshot/mooncake/zh.md)
- [4] NVIDIA TensorRT-LLM。见 [../tensorrt-llm/zh.md](../tensorrt-llm/zh.md)
- [5] Zheng et al. _SGLang: Efficient Execution of Structured Language Model Programs._ 见 [../../foundational/sglang/zh.md](../../foundational/sglang/zh.md)
- [6] 前缀缓存。见 [../../foundational/prefix-caching/zh.md](../../foundational/prefix-caching/zh.md)
