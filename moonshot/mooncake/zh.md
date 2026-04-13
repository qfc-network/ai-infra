# Mooncake —— 以 KVCache 为中心的解耦推理架构

- **作者 / 机构**：Qin 等，Moonshot AI + 清华
- **发表时间**：2024-06（预印本），FAST '25（USENIX）
- **链接**：[论文 (arXiv:2407.00079)](https://arxiv.org/abs/2407.00079) · [代码](https://github.com/kvcache-ai/Mooncake)

## 一句话总结

Mooncake 是 Moonshot AI 旗下 **Kimi** 生产 LLM 服务的推理架构。核心动作：把 **prefill 与 decode 物理拆到不同 GPU 池**、把 **KV cache 当成一等分布式资源**（横跨集群的 CPU DRAM + SSD 池）、由全局调度器按缓存命中最大化分发请求。论文报告：在生产 trace 上吞吐相对 vLLM 基线提升 **75%–525%**，同时仍满足 SLO。Mooncake 提出了 "**KVCache-centric**" 设计哲学和 "**conditioned PD-disaggregation**" 的工程化答案——现已成为严肃推理系统的参考模型（vLLM、SGLang、NVIDIA Dynamo 都向类似设计收敛）。

## 背景与动机

标准 LLM serving（2023 年前后的 vLLM、TGI 等）把 prefill 和 decode 同卡共置。两个结构性问题：

1. **Prefill 是算力瓶颈**（输入序列长、算术强度高）；**decode 是显存带宽瓶颈**（一次一个 token，每步重读权重）。共置意味着总有一边在浪费另一边需要的资源。
2. **请求结束后 KV cache 直接丢弃** —— 即便下一个请求可能有大段共享前缀（system prompt、多轮对话历史、RAG 上下文）。

到了 Kimi 这种规模（超长上下文、多轮、重前缀共享），上述低效成为支配项。Mooncake 的贡献是说明：**要解决这两个问题就得围绕 KV cache 重做架构**，而不是只优化 kernel。

## 核心方法

### Prefill / decode 解耦

两个 GPU 池：

- **Prefill 池** —— 跑 prompt forward，产出初始 KV cache，再通过 RDMA 传给 decode 池。
- **Decode 池** —— 接收 KV cache，跑自回归生成。

两池独立 sizing 与调优。Prefill GPU 可重批（算力瓶颈喜欢批量）；decode GPU 优先低延迟。两池的 batch size、并行选择（TP/PP）、甚至 GPU 型号都可不同。

池间 KV cache 传输是新的热路径。Mooncake 用 RDMA + 精细调度把传输时间压进 SLO 预算。

### KVCache 池 —— 把缓存做成一等资源

集群里所有节点的 CPU DRAM 与 SSD 被池化成**分布式 KV cache 存储**，按 (model, prefix-hash, position) 索引。新请求来时：

1. 调度器 hash 它的 prompt 前缀，查 KV cache 池。
2. 命中则把 cache block 拉过来（从其他节点 DRAM 走 RDMA，或从 SSD），直接进 prefill GPU 的 HBM。
3. Prefill 只算 prompt 中**未命中**的那部分。
4. 生成的 KV 写回池，供后续复用。

对前缀共享重的 workload（system prompt、多轮对话、RAG），效果显著：5 万 token 的 system prompt 算一次，能服务后续上千个请求几乎零 prefill 成本。

### Conditioned PD-disaggregation 与调度器

全局调度器（Conductor）按请求决定：

- 分到哪个 prefill 实例 —— 看缓存命中率、当前负载、与命中 cache 块的网络近邻关系。
- 分还是不分 —— 短 prompt 可能传输开销不划算，干脆 colocate。
- KV 传输如何与计算重叠 —— prefill 算后几层时，前几层的 KV 已开始流向 decode 池。

调度是**SLO 感知**的：高负载时，宁可丢/延 那些反正满足不了 TTFT（time-to-first-token）的请求，也不要全收下让大家一起退化。论文显示这避免了朴素准入下"所有人一起变慢"的失败模式。

### Chunked prefill、按层传输、KV 流式

几个让架构能跑起来的工程细节：

- **Chunked prefill** 把 prompt 按固定大小切块，让调度器在突发负载时能与 decode 请求穿插共用资源。
- **按层传 KV** —— prefill 还在算第 1 层时，第 0 层的 cache 已开始送往 decode 池——重叠而非分阶段。
- **缓存淘汰**按 (recency、prefix 长度、命中率) 加权，长共享前缀粘性高。

## 工程 Tradeoff

| 决策 | 得到什么 | 放弃什么 |
|---|---|---|
| Prefill / decode 解耦 | 各池按瓶颈调优；扩缩容清晰 | 池间 KV 传输（RDMA 带宽、延迟、复杂度） |
| 分布式 KV cache 池 | 共享前缀场景节省巨大 | 缓存一致性 + 淘汰逻辑；miss 时代价高 |
| 缓存感知的全局调度器 | 命中率高、SLO 达成更稳 | 调度成单点热组件；策略面复杂 |
| SLO 感知准入 | 高负载下尾延迟可控 | 简单系统能勉强跑过去的请求会被丢掉 |
| Chunked prefill + 按层传 | PD 传输被算力遮蔽 | 每请求的状态更多；难以推理 |
| 重度依赖 RDMA | 微秒级传输 | fabric 要求高；普通以太网迁过去掉链 |

## 实验与结果

- 合成 workload：相对 vLLM 基线吞吐提升 **75%–525%**，随前缀共享强度变化。
- Kimi 生产 trace（2023 年 9 月样本）：相同 SLO 下处理更多请求；缓存命中时 TTFT 显著下降。
- KV cache 命中率：某些 trace 下高到让 **prefill 反成少数计算量**——cache 在干活。

注意：525% 是 workload 相关。无前缀共享的（一次性独立 prompt）收益小很多。架构价值与你的流量长得像不像 Kimi 成正比（长上下文、多轮、RAG）。

## 复现要点

- 参考实现 [Mooncake](https://github.com/kvcache-ai/Mooncake) 已开源，作为 transfer engine + KV cache 后端集成进 vLLM 与 SGLang。
- 硬要求：RDMA fabric（IB / RoCE）、足够多的 host DRAM 才能让池有意义、能从中受益的 workload。
- 多个后续系统功能上兼容（NVIDIA Dynamo、SGLang 分层 KV cache）。
- 完整的 Conductor 调度器 + SLO 感知是相当大的运维投入。

## 个人评注

Mooncake 留下的真正贡献是**框架命名**，而非某个具体技术。把架构叫做 "KVCache-centric"，迫使全行业承认：**KV cache 是 LLM 推理的工作集，推理系统应围绕 KV cache 管理设计，而不是围绕算力管理**。Prefill/decode 解耦此前已经在空气里（DistServe、Splitwise），但 Mooncake 用真实盈利服务的生产数据证明它在规模上跑得通。

回头看，业界收敛速度很快。到 2025 年，vLLM、SGLang、TensorRT-LLM、NVIDIA Dynamo、几乎所有生产栈都已实现或规划了 PD 解耦 + 分布式 KV cache 池。DeepSeek 的推理系统综述描述的 V3/R1 服务架构也类似。**Mooncake 没发明这些想法，但它带着生产数据公开了，这就把共识锚定下来了。**

开放前沿是**前缀共享低的时候怎么办**——多数收益依赖前缀。Agentic workload（工具调用分支多、复用少）会让 cache-pool 收益缩水，得只靠解耦扛。下一代问题是**前缀共享缺失时如何让 cache 池仍然有用**——投机式缓存、部分缓存复用、学习式预取。

## 参考

- [1] Qin et al. _Mooncake: A KVCache-centric Disaggregated Architecture for LLM Serving._ FAST '25 / arXiv:2407.00079.
- [2] Patel et al. _Splitwise._ ISCA '24.
- [3] Zhong et al. _DistServe._ OSDI '24.
- [4] Mooncake 参考实现：https://github.com/kvcache-ai/Mooncake
- [5] DeepSeek 推理系统综述：https://github.com/deepseek-ai/open-infra-index/blob/main/202502OpenSourceWeek/day_6_one_more_thing_deepseekV3R1_inference_system_overview.md
