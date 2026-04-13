# RLHF / InstructGPT —— 用人类偏好对齐 LLM

- **作者 / 机构**：Ouyang 等，OpenAI
- **发表时间**：2022-03（InstructGPT）；基于 Christiano 等（2017）、Stiennon 等（2020）
- **链接**：[InstructGPT 论文 (arXiv:2203.02155)](https://arxiv.org/abs/2203.02155) · [RLHF 摘要 (arXiv:2009.01325)](https://arxiv.org/abs/2009.01325) · [Deep RL from HF (arXiv:1706.03741)](https://arxiv.org/abs/1706.03741)

## 一句话总结

三阶段配方——**SFT → 奖励模型 (RM) → 对 RM 做 PPO 的 RL**——把裸的 next-token 预测 LLM 变成人们真正愿意用的模型。InstructGPT 证明经 RLHF 对齐的 1.3B 模型在指令遵循任务上人类评估胜过 175B 基座 GPT-3。这篇奠定了此后每个主流 LLM 产品的模板：GPT-3.5/4、Claude、Llama 2 Chat、DeepSeek，几乎一切都是它的变种。infra 层面的影响与算法层面同等重要：RLHF 把**重 rollout 的 RL loop** 引入 LLM 训练，迫使业界建立训练时给策略模型做 serving（actor）、内联跑 reward model、跨数千 GPU 协调 PPO 更新的系统。所有主流后训练栈（TRL、DeepSpeed-Chat、OpenRLHF、verl）之所以存在，都是因为 RLHF 在运维上比预训练更难。

## 背景与动机

GPT-3 是基座模型：足够 prompt 工程后能遵循指令，但开箱即用会续写 prompt 而非回应它。核心问题：

- **预训练损失（next-token）与"有用且无害"目标不对齐。**
- 为 SFT 收集足够多的"有用响应"精标数据可行但有限。
- 成对偏好比黄金示范更便宜，也更有信息量。

Christiano 等 2017 已经证明人类偏好 RL 在简单 RL 任务上可行。Stiennon 等 2020 把它用到 GPT-2 的摘要上。InstructGPT 把配方扩到 175B，之后几年成为默认对齐方法。

## 核心方法

### 阶段 1 —— 监督微调（SFT）

- 从合同标注员处收集约 1.3 万条 prompt → 高质量响应的示范。
- 用标准语言建模损失在这些样本上微调基座。
- 产出"SFT 模型"——不错但常啰嗦、过度自信的指令遵循者。

这一阶段锚定分布——RL 阶段需要保持在某个合理点附近。

### 阶段 2 —— 奖励模型训练

- 对每个 prompt 从 SFT 模型采样 K 个（常 4–9）响应。
- 标注员从最好到最差排序。
- 在偏好数据上用 **Bradley-Terry 成对比较损失**训练奖励模型（架构同 LM，头不同）：

```
L_RM = -E[(y_w, y_l) ~ D] log σ(r(x, y_w) − r(x, y_l))
```

`r(x, y_w) − r(x, y_l)` 越高，匹配人类偏好的概率越大。RM 学到预测一个标量"这个响应多好"，与人类排序一致。

产出：打分 (prompt, 响应) → 标量 的奖励模型。

### 阶段 3 —— 对 RM 做 PPO

从 SFT 出发用 **近端策略优化（PPO）**训策略，以 RM 为奖励：

```
objective = E_prompt [ r(x, y) − β · KL(π_θ(y|x) || π_SFT(y|x)) ]
```

两点值得注意：

- **对 SFT 的 KL 惩罚**防止策略漂到奖励 hack 的输出——RM 给高分但离正常语言很远。
- **PPO** 让每次更新靠近前一版策略，稳定训练——RM 奖励噪声大，一次坏更新可能崩坏生成能力。

完整 PPO loop 同时需要四个模型在内存（或分片跨设备）：

1. 策略（正在训练）。
2. 参考策略（冻结的 SFT，用于 KL）。
3. 价值网络 / critic（给 advantage 做基线）。
4. 奖励模型（冻结，给 rollout 打分）。

**这是昂贵的部分。** 175B 策略在 GPU 集群上做 PPO，是前沿规模下"训练"与"推理"第一次被迫在同一批硬件上共存。

## 工程 Tradeoff

| 决策 | 得到什么 | 放弃什么 |
|---|---|---|
| 三阶段（SFT → RM → PPO） | 质量 + 标注效率 | 流水线复杂度；阶段失败会累加 |
| 成对偏好替代直接打分 | 标注更便宜更可靠 | 损失基数信息；只有相对 |
| 对 SFT 的 KL 惩罚 | 防奖励 hack / mode collapse | 限制策略能走多远 |
| PPO（vs 更简单 RL） | 更新稳定；研究成熟 | 价值网络翻倍显存；rollout/训练协调复杂 |
| 奖励模型为标量 | 单目标，好优化 | RM 的缺陷成为策略的缺陷（Goodhart） |
| 人工标注员 | 基于真实偏好 | 慢、贵、不一致 |

## 实验与结果

- **1.3B InstructGPT > 175B GPT-3** 在指令遵循的人类偏好评估上——差距很大（~85% 胜率）。主头条：对齐在这项任务上胜过规模。
- 在 truthfulness（TruthfulQA）、毒性减低、指令遵循上均改进。
- 代价：原始 NLP benchmark 轻微下降（"对齐税"），大部分可通过在 RL 里混入预训练数据恢复。
- 此后广泛复现：ChatGPT（2022-11）本质是产品规模的 InstructGPT；到 2023 年前后几乎每个主流后训练 LLM 都用同一配方。

## 复现要点

- 论文清楚描述方法；原代码和偏好数据未公开。
- 开源实现：**TRL**（HuggingFace）、**DeepSpeed-Chat**、**OpenRLHF**、**verl**（字节，2024+）。细节各有差异，但都走三阶段。
- **贵的不是代码，是偏好数据** —— 大规模高质量成对比较需要成熟标注运营。开放数据集（Anthropic HH-RLHF、OpenAssistant、UltraFeedback）让学术复现成为可能。
- 大规模 PPO 仍然脆弱。种子、超参、RM 质量都显著影响结果。

## 个人评注

InstructGPT 是 **LLM 史上单篇最重要的后训练论文**。不仅方法管用，而且**重定义了什么是语言模型产品**。之前：LLM 是文本补全器，靠 prompt 工程；之后：LLM 是显式对齐到人类偏好的系统，偏好本身就是产物的一部分。

infra 后果向外扩散。RLHF 迫使每个严肃 LLM 机构建或采用：

- **分布式 rollout 基础设施** —— 训练时大规模生成 completion。这也是 vLLM / SGLang 对训练重要（不只 serving）的原因。
- **奖励模型 serving** —— 第二条内联推理路径。
- **KL / ref-policy 机制** —— 第三条推理路径（冻结参考）。
- **显存高效 PPO** —— 四个模型同驻内存没法搞，必须分片、offload，或用后面提到的继任方法。

这份运维负担是过去两年里业界拼命找替代方案的原因。**DPO**（下一篇）干掉了显式奖励模型。**GRPO**（见 DeepSeek R1）干掉了价值网络。后 RLHF 的整个算法景观，都在优化 InstructGPT 起头的 infra 成本。

还值得单独命名：**奖励模型是承重构件，也是最薄弱的一环**。几十万对上训的 RM 不可能捕捉完整人类偏好空间；策略对它训练就会学会利用它的盲点（Goodhart）。2022 年之后的对齐研究很多都在让这件事不那么糟：process reward model、constitutional AI、AI feedback (RLAIF)、规则奖励（R1）。没有一个彻底解决，但方向是远离一次性标量 RM。

2026 年做后训练：RLHF-PPO 如今多是 baseline，不是默认。小规模/简单偏好微调用 DPO，带可验证奖励的推理任务用 GRPO，完整 PPO 只留给大规模自研产品（那份运维投入才划算）。读 InstructGPT 拿概念基础；实务里用它的继任者。

## 参考

- [1] Ouyang et al. _Training Language Models to Follow Instructions with Human Feedback._ arXiv:2203.02155, 2022.
- [2] Stiennon et al. _Learning to Summarize with Human Feedback._ NeurIPS '20 / arXiv:2009.01325.
- [3] Christiano et al. _Deep Reinforcement Learning from Human Preferences._ NeurIPS '17 / arXiv:1706.03741.
- [4] Schulman et al. _Proximal Policy Optimization Algorithms._ arXiv:1707.06347, 2017.
- [5] Bai et al. _Training a Helpful and Harmless Assistant with RLHF._ arXiv:2204.05862, 2022.（Anthropic 版本）
