# The Llama 3 Herd of Models

- **作者 / 机构**：Meta AI
- **发表时间**：2024-07
- **链接**：[论文 (arXiv:2407.21783)](https://arxiv.org/abs/2407.21783) · [模型](https://huggingface.co/meta-llama)

## 一句话总结

一份 90 页的工程报告，记录如何用最多 **16000 张 H100** 在 **15.6T tokens** 上训练 **405B dense transformer**。架构保守（dense、GQA、RoPE、标准 transformer），价值全在系统工程：4D 并行（TP + PP + CP + DP）、全程 BF16、详尽的大规模硬件故障统计、以及分阶段的长上下文扩展（8k → 128k）。这是 2024 年公开资料里最清晰的"前沿规模 dense 预训练"全景。

## 背景与动机

2024 年中，前沿训练分两条路：稀疏 MoE（DeepSeek、Mistral、坊间的 GPT-4）和 dense（Anthropic、Meta）。Meta 的判断是：dense 更容易训练、部署、微调，到 405B 这个规模，相比同激活参数的 MoE 的能力差距可接受。论文的任务，就是在没人公开做过的规模上把这个赌注证明出来。

## 核心方法

### 架构（刻意无聊）
- Dense transformer，405B 参数，126 层，hidden 16384，128 heads。
- GQA，8 个 KV head。
- RoPE，base frequency 为长上下文做了缩放。
- Tokenizer：128k 词表。

没有 MoE，没有 MLA，没有奇巧注意力。论文的立场是：**数据和系统做对了，前沿规模下架构没那么重要**。

### 数据
- 过滤后 15.6T tokens。多级分类器过滤、大规模去重、质量启发。
- 混合比例通过**小模型上几千次 scaling-law 实验**调出——数据比例被当作可学超参。
- 最后阶段 annealed LR + 数据配比切换，数学和代码显著提升。

### 训练 infra
- **16k H100** 单集群；最大一次训练用 RoCE 以太网，也有 InfiniBand 变体。
- **4D 并行**：tensor (TP=8, 节点内) + pipeline (PP=16) + context (CP=2, 长上下文阶段) + data (DP=128)。CP 维度是对长上下文激活显存的正面回应。
- **全程 BF16**，不用 FP8。Meta 的立场：稳定性与成熟工具链在这个规模下比算力节省更重要。
- **可靠性工程**：54 天内 419 次非预期中断，78% 来自硬件（GPU/HBM/NVLink 故障）。详细 MTBF 分析、自动 checkpoint 重启、burn-in 流程。

### 长上下文扩展
分阶段：8k 预训练 → 用长文档继续训练、调整 RoPE base frequency 扩到 128k。CP 并行就是在这一阶段引入的。

### 后训练
- SFT + DPO，不是完整 RLHF（号称更简单、稳定、效果接近）。
- 多轮迭代 + reward model 对合成数据做过滤。

## 工程 Tradeoff

| 决策 | 得到什么 | 放弃什么 |
|---|---|---|
| dense 而非 MoE | 训练、部署、微调都更简单；scaling 可预测 | 每 token 激活 FLOPs 更高；每激活参数的质量劣势 |
| BF16 而非 FP8 | 稳定、工具链成熟 | 相对 FP8 吞吐少了约 2× |
| 16k 规模的 RoCE 以太网 | 绕开大规模 IB 供给约束 | 更多网络层工程；自研拥塞控制 |
| DPO 而非 PPO-RLHF | 稳定、infra 简单 | 对齐质量上限略降 |
| 带 CP 的 4D 并行 | 长上下文在此规模可训 | 调度复杂度；CP 的通信不是免费的 |
| scaling-law 驱动的数据配比 | 有原则的数据决策 | 正式训练前要跑几千次小实验，准备成本高 |

## 实验与结果

- **405B** 在发布时多项 benchmark 上对标 GPT-4 级模型。
- **8B / 70B** 衍生版是业界开源微调的主力。
- 论文详细披露 MFU、故障率、scaling 曲线——是目前规划大规模训练最好的公开数据集。

## 复现要点

- 开源权重（8B / 70B / 405B），训练代码未开源，但架构与数据流水线细节足以复现。
- 集群级工程（网络拓扑、调度、故障处理）是 Meta 内部的，非 hyperscaler 无法复刻。
- 与 DeepSeek-V3 对照阅读极佳：同期、相反的架构赌注、同等的 infra 严谨度。

## 个人评注

读这篇论文就是为了理解——当你**拒绝架构捷径**时，前沿规模训练的成本与复杂度到底是什么样子。与 V3 的对照很有启发：V3 把钱花在 infra（FP8、DualPipe、手写 kernel）以低算力买能力；Llama 3 把钱花在算力（BF16、dense、16k GPU）以买简洁。两条路都走得通，没有一条能推广到所有实验室。论文最被低估的部分是**详尽的故障统计**——如果你在规划 ≥1k GPU 的训练，**把可靠性工程当成一等公民设计维度**，而不是事后补救。下一代 Llama 大概率会转向 MoE（经济压力摆在那里），但 Meta 的稳定性优先文化应该会保留：BF16、成熟并行、偏执的 checkpoint。

## 参考

- [1] Meta AI. _The Llama 3 Herd of Models._ arXiv:2407.21783, 2024.
- [2] Meta 团队关于 Llama 3 训练 infra 的配套博客。
