# DPO —— 直接偏好优化

- **作者 / 机构**：Rafailov 等，Stanford
- **发表时间**：2023-05，NeurIPS '23（Outstanding Paper Award）
- **链接**：[论文 (arXiv:2305.18290)](https://arxiv.org/abs/2305.18290) · [代码](https://github.com/eric-mitchell/direct-preference-optimization) · [TRL 实现](https://github.com/huggingface/trl)

## 一句话总结

DPO 把 RLHF 三阶段流水线（SFT → RM → PPO）压进**一个监督训练步**。关键洞察：在 RLHF 目标下，最优策略与奖励函数有封闭形式关系，所以**可以直接在偏好对上优化，永远不训奖励模型、也不做 RL**。损失是在 `(prompt, chosen, rejected)` 三元组上的简单 log-likelihood ratio。结果：对齐质量与 PPO-RLHF 相当甚至更好，同时抹掉一半 infra（无 RM、无 rollout、无价值网络、无 KL 簿记——只是个改装的交叉熵）。DPO 一夜成为开源后训练的默认方法，Llama 3、Mistral 模型、Qwen、Mixtral-Instruct、无数社区微调都在用。这是 RLHF 引入以来后训练 infra 最大的单次简化。

## 背景与动机

PPO-RLHF 的运维成本在吞噬对齐研究。一次 RLHF 运行需要：

- 四个模型同驻（策略、参考、价值、RM）。
- 训练时做 rollout 生成。
- PPO 超参调优（出了名的脆弱）。
- 即便用开源实现，也要投入一人月级别 infra。

与此同时，理论界注意到：RLHF 目标

```
max_π E_π[r(x,y)] − β · KL(π || π_ref)
```

有**封闭形式最优策略**：

```
π*(y|x) ∝ π_ref(y|x) · exp(r(x,y) / β)
```

重新整理：给定最优策略和参考，可以**反推隐含奖励**：

```
r(x,y) = β · log(π*(y|x) / π_ref(y|x)) + 常数
```

DPO 的动作：把这个代入原本用于训 RM 的 Bradley-Terry 偏好损失。损失就落在*策略本身*上，不是一个独立 RM。直接优化它就得到原 RLHF 问题的最优策略——无中间步骤。

## 核心方法

### DPO 损失

给定偏好对 `(x, y_w, y_l)`（prompt、chosen、rejected）：

```
L_DPO = -E[(x,y_w,y_l)]  log σ( β · log(π_θ(y_w|x) / π_ref(y_w|x))
                                − β · log(π_θ(y_l|x) / π_ref(y_l|x)) )
```

等价：让 `log π_θ(y_w|x) − log π_ref(y_w|x)` 相对 `y_l` 的同量最大化，系数 `β` 控制。全程监督，训练中不从 `π_θ` 采样。

直观：模型被鼓励**相对参考策略**提高 chosen 响应的对数似然、降低 rejected 的。参考锚取代了 PPO 的 KL 惩罚。

### 训练时你需要的

- 要训的策略（`π_θ`，从 SFT 初始化）。
- 参考策略（`π_ref`，通常就是冻结的 SFT）。
- 偏好数据（`x, y_w, y_l` 三元组）。

就这些。无奖励模型、无 rollout、无 critic。chosen/rejected 响应在两个模型下的 log probability 各跑一次前向即可；损失水到渠成。

### β 的行为

`β` 控制最优策略偏离参考的幅度：

- `β` 大 ≈ 非常靠近参考（偏好拟合弱）。
- `β` 小 ≈ 强拟合偏好（可远离；mode collapse 风险）。

典型 0.1–0.5；正确值取决于偏好数据质量与覆盖。

## 扩展与家族

DPO 事实上是一个模板，不是一个点解。DPO 后的家族：

- **IPO**（Identity PO，Azar 等 2023）—— 用 MSE 损失替代 DPO 的 log-sigmoid，对偏好噪声更稳。
- **KTO**（Kahneman-Tversky Optimization，Ethayarajh 等 2024）—— 用一元信号（此响应好/坏）而不是成对偏好。能标个体不能标对时有用。
- **ORPO**（Hong 等 2024）—— SFT 与偏好优化合一阶段，不需要参考模型。
- **SimPO**（Meng 等 2024）—— 用长度归一化的对数概率作为隐含奖励，彻底扔掉参考模型；更简单，有时更好。
- **NCA、RRHF、RSO** —— 探索不同损失或采样策略的变体。

都共享 DPO 的中心思想：**把偏好优化表达成对数概率上的监督损失，跳过奖励模型**。

## 工程 Tradeoff

| 决策 | 得到什么 | 放弃什么 |
|---|---|---|
| 无独立奖励模型 | 模型减半，infra 减半 | 偏好数据与策略训练不能解耦 |
| 无 rollout | 不需要 RL 基础设施 | 无法用需要生成样本才能评的奖励（如规则 / 工具式） |
| 监督损失 | 稳定、快、与标准训练器兼容 | 对探索的显式控制弱 |
| 参考策略作为锚 | 理论干净 | 训练时仍需参考模型在内存里 |
| 任意偏好对都能用 | 兼容现有数据 | 难与 RL 式可验证奖励在训练中途混用 |

## 实验与结果

- 标准对齐 benchmark（摘要、helpfulness）上，DPO 在胜率评估中匹平或超过 PPO-RLHF。
- 同等数据下训练 wall-clock 快 2–5×——省掉 RM 训练阶段与 rollout 成本。
- Llama 3 后训练采纳（结合迭代 DPO 多轮，用 RM 过滤的合成数据），Mistral Instruct、Qwen 1.5+、2024 年起几乎所有开源 instruct 微调都在用。
- 实践中观察到的缺点：**DPO 对偏好数据分布敏感**。chosen/rejected 对全来自窄分布时，策略会过拟合该区域、在 held-out prompt 上退化。迭代 DPO（多轮带新样本）能缓解但不根治。

## 复现要点

- 实现极简；**TRL 的 DPOTrainer ~500 行**。任何有 SFT checkpoint 和偏好数据的组，在单个 8×A100 节点上几小时就能跑。
- 超参（β、LR、epoch）要调但搜索空间比 PPO 窄得多。
- 这阶段偏好数据的重要性高于算法选择；多数 DPO 失败根源是数据质量不是损失函数。

## 个人评注

DPO 是最近 ML infra 里"**正确理论省 90% 代码**"的最佳例子。数学一直在 RLHF 目标里躺着——没人愿意做代入。Rafailov 等做了，一年的 PPO 工具投入就被抹掉。

更广的模式值得命名：**后训练算法现在部分被 infra 成本度量**，不只是质量。DPO 胜在简单；GRPO（DeepSeek R1）在推理任务胜过 PPO 是因为更简单；Constitutional AI 胜在无害性是因为比策划人工安全数据更简单；等等。"折叠一条流水线的理论洞察"的影响远超"边际质量提升"。

DPO 不适用的地方：

- **可验证奖励**（数学、代码）：规则奖励不能干净表达为偏好对，DPO 直接用不上。这正是 R1 的场景，GRPO + 规则奖励占优。
- **在线迭代**：DPO 按构造是离线的。对策略需要探索和适应的任务（agentic workload、长周期 RL）仍需在线方法。
- **安全关键对齐**：没有显式 RM 更难推理策略在优化什么，这对安全审计重要。

2026 年实务后训练格局大致：

- **DPO**（以及 SimPO、KTO 等后代）—— 开源 instruct tuning 默认。
- **PPO-RLHF** —— 有运维投入能负担的大型自研部署（ChatGPT、Claude）。
- **GRPO + 规则奖励** —— 推理 / 可验证任务微调。
- **Constitutional AI / RLAIF** —— 安全 / 无害性调优。

读 InstructGPT 看"无限工程资源下对齐长什么样"；读 DPO 看"没有那种资源时对齐长什么样"。多数人大多时候想要 DPO。

## 参考

- [1] Rafailov et al. _Direct Preference Optimization._ NeurIPS '23 / arXiv:2305.18290.
- [2] Azar et al. _A General Theoretical Paradigm to Understand Learning from Human Preferences (IPO)._ arXiv:2310.12036, 2023.
- [3] Ethayarajh et al. _KTO._ arXiv:2402.01306, 2024.
- [4] Hong et al. _ORPO._ arXiv:2403.07691, 2024.
- [5] Meng et al. _SimPO._ NeurIPS '24 / arXiv:2405.14734.
- [6] TRL DPOTrainer：https://github.com/huggingface/trl
