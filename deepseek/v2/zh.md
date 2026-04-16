# DeepSeek-V2 — 经济高效的 236B MoE

- **作者 / 机构**：DeepSeek-AI
- **发表时间**：2024-05
- **链接**：[论文 (arXiv:2405.04434)](https://arxiv.org/abs/2405.04434) · [模型 (HuggingFace)](https://huggingface.co/deepseek-ai/DeepSeek-V2) · [代码](https://github.com/deepseek-ai/DeepSeek-V2)

## 一句话总结

DeepSeek-V2 是首个公开证明 MLA（多头潜注意力）与 DeepSeekMoE 两项架构创新可以在前沿规模上协同部署的模型——它同时做到了能力更强、训练更便宜、推理更高效。总参数 236B，每个 token 仅激活 21B，KV cache 相比同等 MHA 模型缩减约 93%，生成吞吐量较 DeepSeek 67B 提升 5.76×，同时在大多数 benchmark 上表现持平或更好。V2 是架构层面的概念验证；V3 随后将同一套方案扩展到 671B。两篇论文构成一条连贯的工程叙事。

## 背景与动机

### 2024 年的推理成本问题

到 2024 年中，前沿 LLM 推理经济学已引发广泛关注。GPT-4 级稠密模型面临：

1. **KV cache 随序列长度与头数线性增长**：长上下文场景下，单条请求的 KV cache 内存可超过模型权重本身。
2. **每个 token 激活全部参数**：70B 稠密模型每生成一个 token 都要读取并计算完整的 70B 参数——没有稀疏性，没有复用。
3. **推理成本随模型规模增长**：服务 70B 模型的单 token 成本约是 7B 模型的 10 倍。

MoE 方案解决了问题 (2)，每 token 只激活一部分参数。KV cache 问题 (1) 与之正交，标准 MoE 无法解决。DeepSeek-V2 是首个在规模上同时攻克两者的模型。

### V2 的假设

既然 MLA 能在不损失质量的前提下将 KV cache 缩减约 10 倍（详见 [MLA 条目](../mla/)），DeepSeekMoE 能将每 token 激活参数量降至约 9%（详见 [MoE 条目](../moe/)），问题是：这两项创新在规模上能否良好组合？V2 给出了肯定的答案。

目标：一个可与 GPT-4 级系统媲美、但服务成本更接近 21B 稠密模型而非 236B 稠密模型的模型。

## 核心方法

### 架构概览

| 维度 | 数值 |
|---|---|
| 总参数量 | 236B |
| 每 token 激活参数 | 21B（约 9%） |
| Transformer 层数 | 60 |
| 上下文窗口 | 128k tokens |
| 注意力机制 | 全部采用 MLA |
| FFN 层 | DeepSeekMoE（细粒度专家 + 共享专家） |

第一层使用标准稠密 FFN，后续所有 FFN 层均采用 MoE。这一"首层稠密 + 其余稀疏"的模式通过确保初始表示不立即经过路由筛选来稳定训练。

### MLA：消除 93% 的 KV cache

完整原理详见 [MLA 条目](../mla/)。系统层面的关键推论：在 128k 上下文下，每条序列的 KV cache 从等效 MHA（约 12 GB）压缩至约 800 MB。这从根本上改变了批量推理的内存算术——在 HBM 耗尽之前，可以容纳远更多的并发序列，从而在相同硬件上实现更高吞吐，而不仅仅是降低内存。

解耦 RoPE 设计（接收位置编码的小维度分量单独缓存）意味着 MLA 与任意位置编码扩展技术兼容。V2 的 128k 窗口通过 YaRN 风格的 RoPE 频率基数缩放实现，无需架构改动。

### DeepSeekMoE：细粒度专家 + 共享专家隔离

完整原理详见 [MoE 条目](../moe/)。在 V2 规模下，细粒度配置意味着大量较小的路由专家配合 top-K 路由，加上少量每 token 必经的共享专家。共享专家吸收高频跨域特征；路由专家专注细分领域。

系统层面含义：专家分发的 all-to-all 通信模式涉及的专家数量多于标准 8-expert MoE，但每个专家更小，单专家激活时间更短，为网络栈提供了更多计算与通信重叠的机会——V3 通过 DualPipe 明确利用了这一特性。

### V2 规模下的专家并行

训练 V2 需要跨多节点切分专家。每个节点持有专家池的一个分片；all-to-all 分发将 token 表示发送到持有所选专家的节点，结果回传。在 V2 的专家数量下，每个 token 的每个 MoE 层分发会跨越多个节点，且每个 micro-batch 的每个 token 都会发生这种情况。

DeepSeek 通过以下方式缓解这一问题：
1. **专家负载均衡**：对路由 logits 施加辅助损失，防止路由坍塌到少数专家（V3 后来用更简洁的偏置方案取代了这一做法）。
2. **拓扑感知路由约束**：限制一个 token 的专家可跨越的节点数，以限定 all-to-all 体积，用路由灵活性换取可预测的通信开销。

### 训练数据与课程

V2 在 8.1T tokens 上训练，以英文和中文文本为主，包含代码与数学数据。训练分两阶段：

1. **8.1T tokens / 4k 上下文预训练**：标准 next-token prediction 损失；余弦学习率衰减。
2. **扩展至 128k 上下文**：在长文档（书籍、代码库、多轮对话）上继续预训练，采用 YaRN 调整 RoPE 频率，额外约 100B tokens。

MLA 架构意味着第二阶段的上下文扩展不会显著增加训练内存——压缩后的 latent 在 BPTT 跨长序列的激活缓存中同样受益于 KV cache 压缩。

### 后训练：SFT → DPO

V2 的对齐流程早于 GRPO（2025 年随 R1 出现）。方案为：

1. **有监督微调（SFT）**：120 万条精选指令样本，覆盖通用对话、代码、数学与工具调用。对补全内容施加标准交叉熵损失。
2. **直接偏好优化（DPO）**：通过对 SFT 模型的拒绝采样收集偏好对，由奖励模型打分。DPO 直接在偏好对上优化，无需单独的 RL 循环，以参考 SFT 模型作为 KL 锚点。详见 [DPO 条目](../../foundational/dpo/)。

V2 的流程中没有在线 RL 或基于回滚的训练——那一迭代随 R1 而来。

## 工程权衡

| 决策 | 得到什么 | 放弃什么 |
|---|---|---|
| MLA + MoE 协同 | KV cache 与激活参数成本双降；互补收益 | 两项新组件相互影响，训练不稳定时排查难度更高 |
| 128k 上下文（YaRN 扩展） | 长上下文能力，额外成本低 | YaRN 扩展在极端长度下质量下降；从头训练长上下文效果更好 |
| 首层稠密 | 训练稳定；避免早期层路由混乱 | 放弃了 1/60 的参数稀疏收益 |
| 辅助负载均衡损失 | 防止专家坍塌 | 损失项带来的精度代价（V3 通过偏置均衡消除了这一问题） |
| DPO 代替 PPO | 流程简单；无需回滚基础设施 | 对困难推理任务的后训练能力受限；受限于离线偏好对的质量 |
| 8.1T token 训练预算 | 强大的基础模型，成本合理 | 相同激活参数量（21B）的稠密模型可用相同成本训练更多 token；V2 的优势在推理端，非训练效率本身 |

## 实验与结果

**与 DeepSeek 67B（稠密前代模型）相比：**
- 性能：V2 在所有主要 benchmark（MMLU、HumanEval、MATH、C-Eval）上持平或提升。
- 训练成本：低约 42.5%（稀疏激活降低了每 token FLOPs）。
- KV cache：缩小 93.3%。
- 峰值生成吞吐：提升 5.76×（相同硬件，显存可容纳更多并发序列）。

**与同期模型（2024 年中）相比：**
- 在中文 benchmark（C-Eval、CMMLU）上与 GPT-4 级模型竞争。
- 在英文推理与代码上与 Llama-3 70B 相当或更好。
- 发布时激活参数量级内最强开源模型。

**后训练增益：**
- DeepSeek-V2-Chat 在指令跟随 benchmark（MT-Bench、AlpacaEval 2）上比 V2 基础模型高约 10 分。
- DPO 在偏好敏感任务上明显优于纯 SFT；在客观任务（数学、代码）上差距收窄，因为 SFT 本身已表现良好。

## 可复现性说明

- V2 模型权重在 HuggingFace 公开发布，SFT checkpoint 也单独提供。
- 训练代码未公开（与 V3 情况相同）。
- DPO 使用的偏好数据未公开；采用该 recipe 的从业者需自行构建偏好对。
- MLA 推理请使用 **FlashMLA**（DeepSeek OSW 2025）获取优化的 decode kernel。吸收技巧（W_UK 折叠进 W_Q 等）必须在权重加载步骤完成；朴素实现在每步重新投影 K、V 会损失大部分性能增益。
- 专家并行服务：vLLM 和 SGLang 均支持 V2 服务，MoE 跨 GPU 专家切分自动处理。

## 评注

V2 的意义在于架构与经济层面。"服务成本接近 21B 的 236B 模型"并非营销——KV cache 缩减与稀疏激活均可量化，吞吐数字验证了这一点。这是业界意识到 MoE + 注意力压缩可以打破推理成本曲线的时刻。

V2 确立的三件事，V3 随后全部继承：

1. **MLA 在规模上有效**。KV cache 的理论缩减直接转化为服务吞吐提升，质量无回退。FlashMLA 后来证明这在完全优化的 kernel 下依然成立。

2. **带共享专家的细粒度 MoE 在 200B+ 总参数规模下稳定**。路由未坍塌；辅助损失下专家利用率保持均衡。这为进一步扩展至 671B 消除了风险。

3. **DPO 足以产出强大的对话模型**。V2-Chat 的对齐效果无需在线 RL 即可与采用更复杂 RLHF 流程的模型竞争。这为衡量 V3/R1 的 GRPO 跃升设定了参照——它是在强大的 DPO 基线上的提升，而非弱 SFT 基线。

**V2 未能解决、由 V3/R1 补足的问题：**

- **推理深度**：DPO 对齐的 V2 是强通用助手，但不是强多步推理器。GRPO + 思考 token（R1）解锁了这一能力。
- **训练精度效率**：V2 用 BF16 训练；V3 切换到 FP8，再次将训练成本减半。
- **流水线 bubble**：V2 使用标准流水线调度；V3 的 DualPipe 几乎消除了 bubble。
- **无精度代价的负载均衡**：V2 的辅助损失已知会损失精度；V3 的偏置方案干净地消除了这一代价。

V2 是"验证"跑；V3 是同一套方案的优化生产跑。

## 参考文献

- [1] DeepSeek-AI. _DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts Language Model._ arXiv:2405.04434, 2024.
- [2] DeepSeek-AI. _DeepSeek-V3 Technical Report._ arXiv:2412.19437, 2024.
- [3] DeepSeek-AI. _DeepSeekMoE: Towards Ultimate Expert Specialization in Mixture-of-Experts Language Models._ arXiv:2401.06066, 2024.
- [4] Peng et al. _YaRN: Efficient Context Window Extension of Large Language Models._ arXiv:2309.00071, 2023.
- [5] Rafailov et al. _Direct Preference Optimization._ NeurIPS '23 / arXiv:2305.18290.
- [6] DeepSeek-AI. _DeepSeek-R1._ arXiv:2501.12948, 2025.
- [7] DeepSeek. _FlashMLA._ https://github.com/deepseek-ai/FlashMLA, 2025.
