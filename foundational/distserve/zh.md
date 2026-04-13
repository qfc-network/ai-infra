# DistServe —— 为 Goodput 解耦 Prefill 与 Decode

- **作者 / 机构**：Zhong 等，北京大学 + UC San Diego + Microsoft Research
- **发表时间**：OSDI '24
- **链接**：[论文 (arXiv:2401.09670)](https://arxiv.org/abs/2401.09670) · [代码](https://github.com/LLMServe/DistServe)

## 一句话总结

Mooncake 的学术兄弟：系统性地论证并实现**把 prefill 与 decode 拆到不同 GPU 组**，也是第一篇把问题用 **goodput** 框起来的论文——单位时间内同时满足 TTFT（首 token 延迟）与 TPOT（每 token 延迟）两个 SLO 的请求数，而不是原始吞吐。论文展示两个阶段有根本冲突的资源偏好，同卡共置迫使两边妥协，把 goodput 压掉 ~2–4×；好好设计的解耦系统能同时提 goodput 并收紧延迟尾部。这篇论文的框架（goodput、阶段解耦、placement 与并行协同设计）现在是推理 serving 问题的标准视角。

## 背景与动机

2023 年的 serving 由连续批处理（Orca、vLLM）主导，prefill 与 decode 在同卡穿插。在"只是 forward"这种视角下没问题。但一旦加上真实 SLO——API 要同时承诺"首 token X 毫秒内到"和"后续每 token Y 毫秒一个"——系统就开始病态：

- 高 prefill 负载下，decode 被大块算力绑定的 prefill 挤死（队头阻塞），TPOT 爆。
- 高 decode 负载下，新 prefill 等资源，TTFT 爆。
- 同一个 batch 里不同请求想要不同并行（TP/PP/batch size），单配置只能折中。

论文的核心观察：**prefill 是算力瓶颈，decode 是显存带宽瓶颈，两者不该共享资源。**

## 核心方法

### Goodput 作为正确的度量

`throughput = requests/s` 忽略了请求是否满足 SLO。一个跑 100 req/s 但延迟是目标 10× 的系统不比一个 40 req/s 满足目标的更好。DistServe 形式化：

```
goodput = 同时满足 SLO_TTFT 与 SLO_TPOT 的请求数/秒
```

论文其余部分都在优化这个，不是原始吞吐。这一单一重构很可能是论文被复用最多的想法——Mooncake、Splitwise、NVIDIA Dynamo 都继承了它。

### 阶段解耦

两个 GPU 池：

- **Prefill 实例** —— 接 prompt，前向，产出 KV cache，交接。
- **Decode 实例** —— 接收 KV，自回归生成 token。

阶段间 KV 传输不简单——长 prompt 请求动辄数百 MB。DistServe 在节点内用 NVLink，跨节点用 RDMA/IB，并精细让传输与目标实例的下一次计算 overlap。

### 各池独立 placement 与并行

每个池按自己 workload 特征挑 **TP / PP / 副本数**：

- Prefill：TP 更划算（每层一次大 matmul，all-reduce 被长序列摊薄）。PP 收益小——短 prompt 里流水线填充占大头。
- Decode：PP 更划算（每 token 延迟由权重加载主导，PP 把这个跨 stage 流水线化），小 batch 下 TP 收益低。

统一 serving 系统只能选一套配置；解耦让每池独立取最优。

### Placement 优化器

论文给了一个 **placement 算法**：给定集群和 workload 分布，输出

- prefill 与 decode 实例各多少。
- 各自 (TP, PP) 怎么选。
- 物理节点落在哪（与 KV 传输路径的亲和）。

基于解析性能模型做优化搜索——不是在线调度，是一次性的集群级计划。

### 延迟感知的请求路由

入站请求按 prompt 长度匹配到当前队列最合适的 prefill 实例（短 prompt 去负载轻的实例以命中 TTFT；长 prompt 批起来提高 FLOP 利用）。生成的 KV 再送到负载最轻的 decode 实例。

## 工程 Tradeoff

| 决策 | 得到什么 | 放弃什么 |
|---|---|---|
| 阶段解耦 | 每阶段最优；SLO 归属清晰 | KV 传输成本；多一跳 |
| Goodput 度量 | 对齐真实 API 合约 | 比裸吞吐难测——要定义 SLO |
| 各池独立并行 | 匹配每阶段的 workload 形状 | 部署复杂度；配置面扩大 |
| 离线 placement 优化 | 集群布局接近最优 | 需 workload 画像；适应慢 |
| 解析性能模型 | 布局搜索快 | 模型精度限制优化器 |
| KV 走高带宽互联 | 传输进 SLO 预算 | 依赖 fabric（NVLink / IB / RoCE） |

## 实验与结果

- 在生产式 trace（ShareGPT、LongBench）上，DistServe 相对 vLLM、DeepSpeed-MII，在同 SLO 下 goodput 提升 **2.0–4.48×**。
- 更惊人的是：在 vLLM 几乎完全无法满足的紧 SLO 下，DistServe 仍能到 ~70% 峰值——因为 vLLM 的共置设计根本无法同时满足紧 TTFT + TPOT。
- KV 传输开销：NVLink 连接的实例个位数毫秒；典型 shape 下 RDMA 上 ~10–20 毫秒。相对 TTFT 预算在任何非短 prompt 下都可忽略。

## 复现要点

- 参考实现已开源。研究集群上 paper 质量，不是生产级（无准入控制、无磁盘 KV 池、无多模型）。
- 实验用 workload trace 均公开（ShareGPT、合成分布）。
- **想法**此后被每个主要生产栈重新实现。vLLM 有 PD 解耦模式，SGLang 原生支持，TensorRT-LLM 加进去了，NVIDIA Dynamo 整个架构围绕它。

## 个人评注

DistServe 与 Mooncake 在 2024 年初几周内相继出现，核心想法近似——业界把它们视为互补而非竞争。分工：

- **DistServe** 是*学术*框架：goodput、形式化 placement 优化、干净 ablation。读这篇是为论证结构。
- **Mooncake** 是*生产*框架：缓存池、SLO 感知准入、来自盈利服务的真实 trace。读这篇是为看"压力下解耦真实长什么样"。

两篇一起奠定了：**PD 解耦不是小优化，而是 LLM serving 规模化的默认架构**。12 个月内每个主要生产栈都采纳了某种形式。

开放前沿恰是 DistServe 没处理的：**workload 漂移下如何动态再解耦**。论文的 placement 是离线的；如果一天里流量从短 prompt 聊天漂到长 prompt RAG，你得重新规划。下一代系统（尤其 NVIDIA Dynamo）在做在线重规划。还有：**解耦怎么与投机解码、前缀缓存、MoE 专家并行组合**——组合爆炸非平凡，统一框架尚未定型。

2026 年设计 LLM serving 系统：**从解耦架构起步、偏离要有理由**，不是反过来。DistServe 是让这成为默认的那篇。

## 参考

- [1] Zhong et al. _DistServe: Disaggregating Prefill and Decoding for Goodput-Optimized LLM Serving._ OSDI '24 / arXiv:2401.09670.
- [2] Qin et al. _Mooncake._ FAST '25 / arXiv:2407.00079.（配对读）
- [3] Patel et al. _Splitwise._ ISCA '24.（同期，互补）
- [4] Agrawal et al. _Sarathi-Serve._ OSDI '24.（统一池内的 chunked prefill——对照）
- [5] DistServe 参考实现：https://github.com/LLMServe/DistServe
