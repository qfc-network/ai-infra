# Building Effective Agents

- **作者 / 机构**：Anthropic Engineering（Schluntz & Zhang）
- **发表时间**：2024-12
- **链接**：[博客原文](https://www.anthropic.com/engineering/building-effective-agents) · [cookbook 例子](https://github.com/anthropics/anthropic-cookbook/tree/main/patterns/agents)

## 一句话总结

不是论文，是 Anthropic 的一篇**工程博客**，为 LLM 应用建立一套词汇。把 **workflow**（通过预定义代码路径组合 LLM 调用）与 **agent**（LLM 自主选择工具与轨迹）分开，主张**多数情况下 workflow 才是对的答案**，并给出五种具体 workflow 模式和"什么时候真的需要 agent"的判断。持久影响是**定义层面**：这篇之后业界在 workflow 上统一到 **prompt chaining / routing / parallelization / orchestrator-workers / evaluator-optimizer** 的词汇，agent 则定义为"loop + 工具使用 + 环境反馈"。读这篇代替 agent 框架营销材料，才能看清大多数生产"agentic"系统背后真正在发生什么。

## 背景与动机

到 2024 年底，"agent"已是 LLM 工程里被滥用得最厉害的词。Anthropic 从客户工作里观察到：绝大多数成功的生产部署都是**结构化的 LLM workflow**，而非自治 agent——但营销压力把什么东西都叫 agent，遮蔽了真正有效的东西。同时，同期多数复杂 agent 框架（当时的 LangChain、LangGraph、AutoGen）引入的抽象层常常帮倒忙。

博客的立场：

> 最成功的实现并不用复杂框架或专门库，它们用简单、可组合的模式搭起来的。

目的：**切开 hype，把真正有效的模式命名出来，并说清各自适用场景**。

## 核心区分

### Agentic 系统 = workflow + agent

- **Workflow**：由**预定义代码路径**编排的 LLM + 工具。控制流你写，LLM 填决策和内容。
- **Agent**：LLM **自主指挥自己的过程与工具使用**，主导如何完成任务。loop + 自主行动。

Workflow 可预测、便宜、好调试。Agent 灵活、昂贵、难调试。博客的强默认建议：**从 workflow 起步；只有任务范围真需要时才升级到 agent**。

### 起点：augmented LLM

workflow 和 agent 之前的基础积木是一个**带检索、工具、记忆增强的 LLM**。多数应用不需要更多。单次 augmented LLM 调用能解决问题，就到此为止。

## 五种 workflow 模式

### 1. Prompt chaining

把任务拆成固定顺序的几个 LLM 调用，每步消费上一步输出。步骤之间可用**程序化 gate** 校验或短路。适用于任务能被干净线性分解的情况（起草 → 批评 → 重写；抽取 → 翻译 → 格式化）。

```
input → LLM_1 → [gate 校验] → LLM_2 → [gate] → LLM_3 → output
```

得：每步更简单、每次调用更便宜、调试更局部。
失：每步多一次调用；如果任务需要跳步则不够灵活。

### 2. Routing

对输入分类，路由到专用 prompt / 模型 / workflow。小快模型（或规则）决策；大模型只处理匹配其专长的输入。

```
input → router → {workflow_A | workflow_B | workflow_C}
```

得：成本（多数输入走小模型）、质量（专项处理）。
失：分类必须可靠；误路由代价高。

### 3. Parallelization（sectioning 与 voting）

子任务并发跑、再聚合。两种变体：

- **Sectioning**：任务切成独立片段，LLM 处理各片段，结果合并。
- **Voting**：同一任务用不同 prompt 或 temperature 多跑几次，聚合（多数、过滤、best-of-N）提升可靠性。

适用于子任务独立，或单次调用可靠性不够（安全关键、高风险）场景。

### 4. Orchestrator-workers

一个中央 LLM **动态分解**任务为子任务，委派给 worker LLM，再综合结果。子任务集合**不预定义**——orchestrator 按输入自己决定。

```
input → orchestrator（规划） → worker_1 ∥ worker_2 ∥ ... → orchestrator（综合） → output
```

类似 parallelization 但分解是动态的。更强大；更贵；失效模式（分解错、上下文丢失）是新的。

### 5. Evaluator-optimizer

一个 LLM 生成，另一个 LLM 评审并给反馈；循环直到评审通过（或达上限）。适用于：

- 能定义清晰评估标准。
- 一稿输出通常需要再打磨。
- 迭代真的能改善结果（并非所有任务都受益——有些"改进"反而退化）。

```
generator → output → evaluator → [反馈] → generator → ... → final
```

## Agent：workflow 不够用时

Agent 是一个 LLM 在 loop 里自主选择工具使用与继续方式。好的 agent 有：

- 清晰的**任务**。
- 文档良好的**工具集**。
- **环境反馈**以支撑决策（工具输出、代码执行结果、搜索命中）。
- 某种**停止条件**（任务完成、预算耗尽、人工接管）。

何时用 agent（博客观点）：

- **开放性任务**，路径无法预定义。
- **交互式、长周期**问题（跨 codebase 写代码、多步研究、操作电脑）。
- 灵活性的价值超过不可预测性的代价时。

何时不用：

- workflow 能胜任的任何任务——agent 更贵、失败方式更多。
- 对错误行为容忍度低的任务（agent 时不时会干错事）。

## 工程 Tradeoff

| 模式 | 何时合适 | 代价 |
|---|---|---|
| Augmented LLM（单次调用） | 简单一次性任务 | 质量天花板有限 |
| Prompt chaining | 能线性分解且有自然 gate | 每个输出多延迟 + 多成本 |
| Routing | 输入类型混合且需不同处理 | 需分类准确 |
| Parallelization | 子任务独立或靠投票提可靠性 | 算力倍数 |
| Orchestrator-workers | 需要动态分解 | 失效模式复杂 |
| Evaluator-optimizer | 迭代打磨确实能改善 | loop 边界、迭代成本 |
| Agent | 开放、长周期、需灵活 | 贵、不可预测、难调试 |

## 个人评注

这篇博客是"**agent 框架本该长什么样**"最清晰的公开表达。之前那些复杂框架（一层套一层的 chain 抽象、自动生成图、冗长编排 DSL），大多数都在解决一个——对多数任务来说——不需要解决的问题。多数生产"agent"工作其实是五种 workflow 之一，而那些**就是普通代码加 LLM 调用**。Anthropic 工程团队把这点挑明，业界悄悄对齐了。

定义也澄清了*真正的* agent 有什么不同：**是模型、而不是框架，在选轨迹**。这意味着**你的投入应从编排代码转向工具质量、环境反馈和可观测性**——那才是模型真正在使用的东西。Anthropic 自己的产品（Claude Code）就是这个意义上的 agent；每个设计决策都围绕给模型好用的工具、清晰的反馈、从错误中恢复的能力，而不是给它套控制流脚手架。

对 infra 人员，相关启示是**可观测性比框架选择更重要**。没有好 tracing、可重放 run、清晰 SLO 的 agent，生产里就是不可控的，跟你选哪个 agent 库没关系。cookbook 例子刻意精简——模式在几百行 Python 内能跑，你多 import 的都在带自己的 bug。

更大趋势：到 2026，**按这个定义多数生产"agentic"系统其实是 workflow 而非 agent**。真正的 agent——长周期、灵活、自主——能跑但仍贵且挑剔。Anthropic 这篇框架经住了时间：workflow 覆盖 80% 价值；agent 覆盖新前沿，但尚未替掉已有用例里的 workflow。

## 参考

- [1] Anthropic Engineering. _Building Effective Agents._ 2024-12. https://www.anthropic.com/engineering/building-effective-agents
- [2] Anthropic Cookbook agent 模式：https://github.com/anthropics/anthropic-cookbook/tree/main/patterns/agents
- [3] Schluntz & Zhang 在 Latent Space 播客（2024）——对相同想法的更深讨论。
- [4] 相关：[我们是如何搭 Claude Code 的](https://www.anthropic.com/engineering) —— 应用实例。
