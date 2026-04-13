# DeepSeek-V3 技术报告

- **作者 / 机构**：DeepSeek-AI
- **发表时间**：2024-12
- **链接**：[论文 (arXiv:2412.19437)](https://arxiv.org/abs/2412.19437) · [代码](https://github.com/deepseek-ai/DeepSeek-V3) · [模型](https://huggingface.co/deepseek-ai/DeepSeek-V3)

## 一句话总结

671B MoE 模型（每 token 激活 37B），在 14.8T tokens 上仅用 **278.8 万 H800 GPU 小时**（按 $2/小时约 558 万美元）完成训练，效果对标 Llama 3.1 405B、在多项 benchmark 上逼近 GPT-4o / Claude 3.5 Sonnet。真正的贡献不在模型本身，而在让训练如此便宜的 **infra 栈**：大规模 FP8 混合精度、DualPipe 流水线、手写 all-to-all kernel，以及 auxiliary-loss-free 的 MoE 负载均衡方案。

## 背景与动机

在 V3 之前，这种规模的前沿训练通常被认为需要 ≥10k 张 H100 级 GPU、数千万美元。两个结构性瓶颈主导：

1. **通信**：MoE 跨节点专家并行的 all-to-all 会打满节点间带宽，GPU 被卡住空转。
2. **精度**：BF16 激活和梯度的显存与 HBM 带宽占用都大；FP8 此前只在小规模上验证过。

H800（面向中国市场的受限版本）NVLink 带宽大约只有 H100 的一半，让通信问题雪上加霜。V3 的 infra 选择都是对这个约束的直接回应。

## 核心方法

### 架构
- **MoE**：每个 MoE 层 256 个路由专家 + 1 个共享专家，top-8 路由；共 61 层。
- **MLA（Multi-head Latent Attention）**：将 KV cache 压缩到低秩 latent；推理时每 token 的 KV cache 约为标准 MHA 的 1/10。细节见 V2 论文。
- **Multi-token Prediction (MTP)**：辅助目标，预测接下来 2 个 token；推理阶段也可用于投机解码。

### 训练 infra
- **FP8 混合精度**：大多数 GEMM 用 FP8 E4M3；激活按 1×128 tile、权重按 128×128 block 做细粒度 scaling 控制 outlier；master weights 与 optimizer state 保留 BF16/FP32。这是 FP8 端到端训练在大规模上的首次公开验证。
- **DualPipe**：双向流水线调度，相邻 micro-batch 的前向与反向阶段互相 overlap，pipeline bubble 小于 1F1B / ZeroBubble。配合 PTX 级手写 all-to-all kernel，算力与跨节点通信几乎完全重叠。
- **2048 张 H800 上的专家并行**：all-to-all 按 H800 的 NVLink + IB 拓扑调优（每个 token 最多分发到 4 个节点）。
- **无辅助损失的负载均衡**：在路由 logits 上加 per-expert 偏置项，根据实时负载在线更新。规避了标准 aux loss 带来的精度损失。

## 工程 Tradeoff

| 决策 | 得到什么 | 放弃什么 |
|---|---|---|
| FP8 GEMM + 细粒度 scaling | 吞吐 ~2×，显存减半 | scaling/累加复杂度上升；需高精度累加、fwd/bwd 激活都用 E4M3 等补偿手段 |
| MLA | KV cache 缩小 10×，长上下文与推理成本双降 | 额外投影计算；架构复杂度高于朴素 MHA |
| Aux-loss-free 均衡 | 避免均衡损失带来的精度代价 | 需要额外 bookkeeping；对偏置更新速率敏感 |
| DualPipe | 流水线 bubble 接近零 | 激活显存是 1F1B 的 2 倍（但 FP8 + MLA 已腾出空间，可接受） |
| token 分发限 ≤4 节点 | 通信开销可控 | 路由受限，专家利用率的灵活度下降 |

## 实验与结果

- **成本**：预训练 266.4 万 + 长上下文与后训练 12.4 万 = 合计 278.8 万 H800 小时，按 $2 估算约 558 万美元。
- **吞吐**：尽管 H800 互联较慢，持续 MFU 与同规模 H100 BF16 训练相当。
- **质量**：发布时开源模型中最强，MMLU-Pro、MATH、代码 benchmark 逼近闭源前沿；长上下文 128k 表现稳健。

注意：558 万只算最终一次训练。研发、失败尝试、数据整理、人力都没计入。即便如此，相对同期的报告提升仍显著。

## 复现要点

- 开源了权重与推理代码，**训练代码未开源**。
- PTX 级 all-to-all kernel 后来作为 **DeepEP** 开源（2025 Open Source Week），FP8 GEMM 为 **DeepGEMM**，MLA 推理 kernel 为 **FlashMLA**。
- 复现训练需要：~2k H800/H100 集群、FP8 可用的 kernel 栈、多周稳定运行。对大多数实验室不现实，但各模块的 recipe 已可复用。

## 个人评注

真正新颖的贡献是：(a) FP8 在这个规模下真的跑通了，且给出了可复制的 recipe；(b) aux-loss-free 均衡作为困扰 MoE 多年的 hack 的干净替代；(c) DualPipe + 自定义 kernel 的协同设计证明了 H800 级集群只要愿意做系统工作，同样能训前沿模型。模型本身强但不算架构革命——MLA 与细粒度 MoE 在 V2 就已定型。对业界真正的启示：**前沿规模训练是系统问题**，compute-optimal 边界之下还有大量 infra 红利可挖。

## 参考

- [1] DeepSeek-AI. _DeepSeek-V3 Technical Report._ arXiv:2412.19437, 2024.
- [2] DeepSeek-AI. _DeepSeek-V2._ arXiv:2405.04434, 2024.（MLA、DeepSeekMoE）
- [3] DeepSeek Open Source Week 开源仓库：FlashMLA、DeepEP、DeepGEMM、3FS、DualPipe（2025）。
