# DeepSeek-R1

- **作者 / 机构**：DeepSeek-AI
- **发表时间**：2025-01
- **链接**：[论文 (arXiv:2501.12948)](https://arxiv.org/abs/2501.12948) · [模型](https://huggingface.co/deepseek-ai/DeepSeek-R1)

## 一句话总结

一套把强基座（DeepSeek-V3-Base）训成前沿**推理模型**的开源配方，在数学 / 代码 / 逻辑等 benchmark 上对标 OpenAI o1，且**不依赖大规模人工 SFT 数据**。两个结果并列：**R1-Zero**——从基座出发纯 RL，规则式奖励，没有 SFT，从零涌现长链思维、自我验证、反思；**R1**——加了小规模 "cold-start" SFT 提升可读性，随后反复 RL + SFT 的工程可用版本。论文同时推出 **GRPO**（Group Relative Policy Optimization），一种去掉 value 网络的 PPO 变体。权重与蒸馏版学生模型开源。

## 背景与动机

R1 之前，推理模型配方被 o1 锁着（闭源），而开源界的 SFT 重度路线卡在前沿之下。开放文献里有两个未知：

1. **长链推理能否仅靠 RL 涌现**，而不需要模仿数据？
2. **围绕 RL 最小的脚手架是什么**，才能在规模下训得动（奖励 hacking、可读性、中英混杂）？

R1 同时回答两个问题。R1-Zero 是科学结果（能涌现），R1 是工程结果（这套脚手架能出厂）。

## 核心方法

### R1-Zero —— 纯 RL 从基座训

起点 V3-Base，无 SFT。RL 配置：

- **Prompt**：可验证答案的推理任务（有已知解的数学题、带单测的代码题）。
- **奖励**：纯规则。
  - **准确性奖励**：最终答案是否匹配 ground truth？
  - **格式奖励**：思维链是否包在 `<think>...</think>` 标签内？
- **无学习式奖励模型、无偏好数据。** 结构上消除了奖励 hacking——只能 hack 规则可验证的东西。

训练中，**响应长度自发增长**——模型发现长推理能换更高准确性奖励。自我反思（"让我验证一下"、"等等，错了"）在无监督下自发出现。Benchmark 单调上行。

缺点：输出不可读——中英混杂、格式差。benchmark 能看，产品不行。

### R1 —— 冷启 SFT + 多阶段 RL

四阶段：

1. **冷启 SFT**：数千条精挑的长 CoT 样本，给模型一个可读的"推理腔"，再进 RL。
2. **推理 RL**：与 R1-Zero 同规则奖励，额外加一条语言一致性奖励抑制混语言。训到收敛。
3. **Reject-sample SFT**：用 RL 后的模型生成 CoT，剔除低质，加上非推理 SFT 数据（写作、自我认知），合成约 80 万样本的更广能力 SFT 集。从 V3-Base 再训。
4. **对齐 RL**：再一轮 RL，覆盖 helpfulness 和 harmlessness，这些不可验证的维度用学习式奖励模型。推理维度保留规则奖励。

最终模型达到 o1 级别 benchmark，同时可生产化。

### GRPO —— 组相对策略优化

对每条 prompt 从当前策略采样一组 `G` 个回答（`G = 8–16`），以组内均值奖励作为 baseline，替代学习式 value 网络：

```
A_i = (r_i − mean({r_j})) / std({r_j})
```

标准 PPO clipping，用这个 advantage。好处：

- **无 value 网络**——相对 PPO 省一半显存和算力；前沿规模下至关重要。
- **组内相对归一化**在不同难度 prompt 间更稳。
- 与规则奖励搭配更好——规则奖励下 value function 本来就不是必需的。

GRPO 首次出现在 DeepSeekMath，R1 在更大规模上验证了它。

### 蒸馏

80 万 SFT 集用于把推理行为蒸馏到更小的 dense 模型（Qwen 与 Llama，1.5B–70B）。在这些小模型上，**蒸馏胜过直接 RL**——小模型规模下 RL 不稳，模仿强教师更可靠。

## 工程 Tradeoff

| 决策 | 得到什么 | 放弃什么 |
|---|---|---|
| 从基座纯 RL（R1-Zero） | 证明不靠模仿也能涌现推理 | 输出不可读，不能直接做产品 |
| RL 前加冷启 SFT | 产品可用；收敛更快 | 有少量人工数据依赖，科学主张被稀释 |
| 推理任务仅用规则奖励 | 无奖励 hacking；无 RM 算力 | 只在验证廉价的任务（数学、代码）可用 |
| GRPO 替代 PPO | RL 便宜约 2×；跨难度稳定 | 需组采样（每 prompt `G=8–16` rollout） |
| 小模型用蒸馏而非 RL | 小模型质量稳定 | 继承教师偏差；小规模无新涌现 |
| 多阶段流水线 | 能力与对齐兼顾 | 复杂度高；正确复现需要四阶段都到位 |

## 实验与结果

- **R1-Zero**：AIME 2024 pass@1 从 ~15.6%（V3-Base）升到 ~71%。语言混杂与可读性问题被证实。
- **R1**：MATH-500 97.3%，AIME 79.8%，Codeforces 96.3 百分位，MMLU 90.8%。多项硬推理 benchmark 持平或胜过 o1；通用任务有竞争力。
- **蒸馏**：R1-distilled-Qwen-32B 在推理 benchmark 上超 o1-mini；distilled-7B 是发布时最强的小推理模型。

## 复现要点

- R1、R1-Zero 及所有蒸馏版均开源权重。
- 论文把流水线描述得很清楚，训练代码未开源，但已有多个复现（HuggingFace **Open-R1**、**simpleRL**、清华团队等）。
- 规则奖励 RL 是可复现性最高的部分——只需基座、可验证数据、PPO/GRPO loop。
- 在 V3 规模（671B MoE）跑 RL 对多数实验室仍不现实；Qwen-7B 级别的小规模复现用几台机器即可。

## 个人评注

真正被记住的贡献是 **R1-Zero**，而不是 R1。证明**基座 + 规则奖励 RL** 就能涌现思维链、自我验证、反思，是一个有分量的科学结论——它把"推理能力"与"推理数据"解耦，暗示后者更多是捷径。R1 本身是能出厂的工程版本。GRPO 值得单独重视：在 RLHF 规模下砍掉 value 网络是一次重大效率胜利，已经在其他工作中被采用。开放问题是纯 RL 能走多远——R1-Zero 能成是因为数学和代码有廉价 verifier；对没有 ground truth 的任务（写作、开放式问题），同样路径能否产出前沿推理仍未知。下一波工作已经在攻这个方向：**process reward model、树搜索增强 RL、self-consistency 作为伪奖励**。预期到 2026 年，开源推理模型的模板仍是 R1。

## 参考

- [1] DeepSeek-AI. _DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning._ arXiv:2501.12948, 2025.
- [2] Shao et al. _DeepSeekMath._ arXiv:2402.03300, 2024.（GRPO 首次出现）
- [3] OpenAI. _Learning to Reason with LLMs (o1 system card)_, 2024.
- [4] HuggingFace. _Open-R1._ 2025.
