# Mixtral of Experts

- **作者 / 机构**：Mistral AI
- **发表时间**：2024-01
- **链接**：[论文 (arXiv:2401.04088)](https://arxiv.org/abs/2401.04088) · [权重](https://huggingface.co/mistralai/Mixtral-8x7B-v0.1)

## 一句话总结

Mixtral 8×7B 是基于 Mistral 7B 的**稀疏 MoE**：每层 8 个 FFN 专家，top-2 路由，**总参数 46.7B、每 token 激活 12.9B**。多数 benchmark 上匹敌甚至超过 Llama 2 70B，而推理成本接近 13B dense。架构保守——**少、大、同质专家，没有共享专家，没有细粒度切分**——但它是开源界第一个有意义规模的 MoE。放在 DeepSeekMoE 旁边读，刚好是"粗粒度 MoE"那一端。

## 背景与动机

2023 年底 / 2024 年初 Mixtral 发布时，开源权重领域清一色是 dense。MoE 在 GPT-4 传闻已久，但没有公开验证。Mistral 的目标很务实：证明 MoE 在 ~50B 参数段可用、推理便宜、量化后能塞进两张消费级 GPU。论文短（~15 页），刻意少谈创新——这是**发布论文，不是架构论文**。

## 核心方法

### 架构
- 基座：Mistral 7B（GQA、RoPE、SwiGLU、滑窗注意力）。
- **把每个 FFN 替换为稀疏 MoE 层**：8 个专家，top-2 路由。
- Router：对 token 隐状态做线性投影，softmax，取 top-2。
- 输出：两个被选专家的输出加权求和。
- 32 个 MoE 层，32k 词表，32k 上下文。

### 路由与均衡
- 标准 top-2 + softmax 归一化。
- 均衡用**辅助损失**（Switch Transformer 式）加 **router z-loss** 稳定训练。
- 没有共享专家、没有细粒度切分、没有值得一提的 capacity factor 花样——重点就是简单。

### 训练
- 论文几乎不披露训练细节——无总 token 数、无集群规模、无吞吐数据。与 V3、Llama 3 形成明显对照。
- 同时发布微调版 Mixtral 8×7B Instruct。

## 工程 Tradeoff

| 决策 | 得到什么 | 放弃什么 |
|---|---|---|
| 粗粒度 8 专家、top-2 | 路由简单、分发开销低 | 专业化粒度不如 DeepSeekMoE 的 256 top-8 |
| 无共享专家 | 架构更简单 | 路由专家必须重复学通用特征 |
| 替换 FFN 即可 | 工具链天然支持 | 无法在 attention 与 MoE 间重新分配算力 |
| 标准 aux 均衡 loss | 稳定、认知成本低 | 与 V3 的 aux-loss-free 相比略有精度税 |
| 46.7B 总 / 12.9B 激活 | 4bit 量化后可在 2×24GB 上跑 | 显存占用尴尬——介于 13B 与 70B 部署类别之间 |

## 实验与结果

- 多数 benchmark 对标 Llama 2 70B，数学/代码上胜出。
- 推理吞吐与延迟接近 13B dense。
- Instruct 版 MT-Bench 超过 GPT-3.5。
- **未披露任何训练成本。** 对 infra 分析是实质性限制。

## 复现要点

- 权重开源（Apache 2.0）；推理在 `transformers`、vLLM、llama.cpp 等主流栈里都支持。
- 训练代码、数据、infra 细节均未公开。
- 对 MoE 研究而言是**干净、被充分理解的 baseline**——大多数开源 MoE 工具都围绕 Mixtral 的 shape 打磨过。

## 个人评注

Mixtral 的贡献是**生态，不是架构**。它用宽松许可证把一个可用质量的 MoE 放进开源世界，迫使所有推理框架、量化库、serving 栈都支持稀疏路由。DeepSeekMoE 在架构上更激进；Mixtral 则是把 MoE 拉入开源主流的那个。两篇论文对照读：Mixtral 给出 **最小可行 MoE**（简单、能用、能发），DeepSeekMoE 展示 **把架构推得更狠**能多拿多少（更多专家、更细粒度、共享隔离）。业界正在向 DeepSeek 这一端收敛——下一代 Mixtral 很可能长得像 V3 的配方。

## 参考

- [1] Mistral AI. _Mixtral of Experts._ arXiv:2401.04088, 2024.
- [2] Jiang et al. _Mistral 7B._ arXiv:2310.06825, 2023.
- [3] Fedus et al. _Switch Transformers._ JMLR, 2022.（aux loss 基线）
- [4] Dai et al. _DeepSeekMoE._ arXiv:2401.06066, 2024.（对照点）
