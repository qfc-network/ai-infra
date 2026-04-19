# Agent 框架全景

_系统设计条目，涵盖四个主要 Agent 框架——LangGraph、AutoGen、OpenAI Swarm 和 DSPy——重点关注状态管理、上下文窗口压力、延迟特性和可观测性。底层工具调用原语详见 [../tool-use-infra/](../tool-use-infra/)；MCP 作为工具/资源注册层详见 [../../anthropic/mcp/](../../anthropic/mcp/)；这些框架所实现的工作流分类详见 [../../anthropic/building-effective-agents/](../../anthropic/building-effective-agents/)；SGLang 作为多次调用 LLM 程序的低层次替代方案详见 [../../foundational/sglang/](../../foundational/sglang/)。_

## 一句话总结

原始 LLM API 只提供单次补全。真实的 Agent 任务需要循环控制：工具失败时重试、基于中间结果分支、跨多次模型调用维护状态、在专业子 Agent 之间协调。Agent 框架提供这些脚手架。框架的选择本质上不是功能列表的比较，而是关于你愿意承担哪种失败模式。LangGraph 将状态显式化、图结构可调试，代价是冗长的配置。AutoGen 将 Agent 建模为对话参与者，可扩展到复杂协调，但产生难以追踪的非确定性流程。OpenAI Swarm 刻意极简，适合理解切换模式，而非生产级多 Agent 系统。DSPy 在类别上完全不同：它是提示编译器而非运行时，属于评估流水线而非服务循环。状态持久化大小、上下文窗口倍增、p99 延迟放大和可观测性开销，才是在生产规模上真正区分这些框架的实际指标。

## 背景与动机

### 没有框架时原始循环的样子

没有框架时，开发者手动编写 Agent 循环：

```python
messages = [{"role": "system", "content": system_prompt}]
messages.append({"role": "user", "content": user_input})

while True:
    response = llm.complete(messages, tools=tools)
    if response.stop_reason == "tool_use":
        for tool_call in response.tool_calls:
            result = dispatch(tool_call)
            messages.append(tool_result_message(tool_call.id, result))
    else:
        return response.text
```

对于简单的线性链，这是可行的。但当需要以下能力时就会崩溃：并行子图执行、基于中间结果的条件分支、跨会话持久化状态（Agent 被中断后需要恢复）、人工审批门控、多专业 Agent 协调、带退避的错误重试，以及随消息列表增长的上下文窗口管理。框架处理这些关切，代价是引入学习曲线和运维复杂性。

## LangGraph

### 设计模型

LangGraph（LangChain 生态的一部分，2024 年发布）将 Agent 工作流建模为**有向图**，其中节点是 Python 函数，边是状态转移。每个节点从**共享状态字典**（Python 中为类型化的 TypedDict）读取并写入数据。图有一个定义好的起始节点和一个或多个终止节点；执行沿边推进，直到到达终止节点或图抛出异常。

条件边接受一个路由函数，该函数检查当前状态并返回下一个节点的名称。这是分支逻辑所在的位置——路由器决定是继续工具调用循环、切换到专业子图，还是终止。并行节点执行通过 `Send` 原语支持：一个节点可以将工作扇出到多个并发节点，然后在减少节点收集结果。

### 状态持久化与检查点

LangGraph 的核心生产特性是其**检查点器**抽象。在每次节点转移时，完整的状态字典被序列化并写入持久化后端。支持的后端包括：内存（默认，用于测试）、SQLite（单进程）、Postgres（分布式）和 Redis。

检查点有两个用途：可恢复性（被中断的 Agent 可以从最后一个检查点恢复，而不是从头重启）和人工审批（图可以配置为在指定节点前暂停，等待外部输入，然后恢复——即中断/恢复模式）。

检查点的内存成本与状态字典大小成正比。在 Agent 工作流中，工具结果通常存储在状态中（在节点间传递结果的常见模式），状态随每次迭代增长。一个累积了 10 个各 2,000 token 工具结果的工作流，其状态约为 20,000 tokens（约 80 KB），必须在每次节点转移时序列化并写入检查点器。在 100 个并发 Agent 会话、每个会话 10 个节点转移的情况下，Postgres 后端必须处理每次会话执行 1,000 次写入，每次可达数 MB。低规模下这不是问题；高并发时会成为吞吐瓶颈。

**缓解策略**：将工具结果存储为外部引用（将实际数据存入对象存储或数据库，状态中只保留 ID）；定义显式的状态压缩节点，在检查点前对先前轮次进行摘要。

### LangGraph 中的多 Agent 模式

LangGraph 支持子图组合：一个节点本身可以是一个已编译的 LangGraph，允许构建分层 Agent 架构。一个监督者图协调专业子图（研究 Agent、写作 Agent、代码审查 Agent），根据当前状态路由任务。每个子图维护自己的状态字典，结果向上层图暴露。

延迟影响显著：每个子图是独立的 LLM 调用序列。监督者顺序调用三个专业 Agent，总延迟增加 3 倍单 Agent 延迟。LangGraph 通过 `Send` 支持并行节点执行以减少这一影响。

### 可观测性

LangGraph 与 **LangSmith** 原生集成。每次图执行产生一个追踪树：每个节点调用是一个带有输入状态、输出状态和计时信息的 span。这个追踪使得精确查看哪个状态触发了哪次 LLM 调用、调用了哪些工具、延迟花在哪里成为可能——这是相对于手动循环的主要调试优势。

## AutoGen

### 设计模型

AutoGen（微软研究院，2023 年）采用不同的方法：Agent 是通过发送和接收消息进行通信的**对话参与者**。基类 `ConversableAgent` 封装一个 LLM 并定义自动回复机制：收到消息时，使用对话历史调用 LLM，可选执行工具，然后返回结果。

`GroupChat` 协调多个 Agent。`speaker_selection_method` 控制下一个发言者：`"round_robin"` 按顺序轮流；`"auto"` 使用基于 LLM 的选择器为每轮选择最相关的 Agent；`"manual"` 需要人工选择。

### 对话模型与非确定性

AutoGen 的对话模型产生静态难以推理的涌现行为。基于 LLM 的发言者选择器引入了非确定性：相同的任务在不同运行中可能产生不同的 Agent 对话模式。这既是特性（比固定图更灵活），也是缺陷（非确定性流程难以调试、难以测试、难以对延迟进行上界估计）。

上下文窗口管理挑战在 AutoGen 中尤为突出：每个 Agent 维护自己的对话历史，但在 GroupChat 中，所有 Agent 都看到完整对话。在第 N 轮，GroupChat 管理器将所有 N 条先前消息提供给发言者选择器 LLM。一个涉及 4 个 Agent 进行 20 轮对话的复杂任务，在第 20 轮时选择器 LLM 面对包含 20 条消息的上下文（每条消息可能包含代码或工具输出）。这种上下文膨胀随 Agent 数量和轮次数加速。

AutoGen 的缓解方案：`max_consecutive_auto_reply` 参数限制 Agent 在要求人工输入前自动回复的轮数，约束失控的对话；可通过 `compress_history`（摘要化）或 `clear_history`（完全重置）在可配置的间隔截断每个 Agent 的对话历史。

### 代码执行循环

AutoGen 的设计优势在于代码编写 + 执行循环：`coder` Agent 编写 Python，`executor` Agent 在沙箱中运行代码并返回标准输出/错误，`coder` 看到输出后迭代。沙箱默认是 Docker 容器（`docker_code_execution_config`），提供文件系统和进程隔离，无需手动管理容器。

此循环的延迟主要由容器启动时间（Docker 冷启动每次执行 0.5–2 s）、Python 执行时间和每次迭代的 LLM 调用决定。对于需要 5 次代码迭代的任务，预计 5 × (LLM 调用 + 容器执行) ≈ 5 × (2 s + 1 s) = 15 s 才能得出正确结果。跨迭代复用暖容器可将此降至 5 × (2 s + 0.1 s) = 10.5 s，节省约 30%。

### 可观测性

AutoGen 提供 **AutoGen Studio**（用于构建和运行 Agent 工作流的 Web UI）和消息交换结构化日志。挑战在于对话日志是消息的平坦序列，要重建哪个 Agent 决策导致了下游失败需要仔细的日志分析。AutoGen 0.4+ 包含 OpenTelemetry 追踪导出，将 Agent 对话映射到标准 span，实现与 Jaeger、Honeycomb 或任何 OTel 兼容后端的集成。

## OpenAI Swarm

### 设计模型

Swarm（OpenAI，实验性，2024 年）刻意极简。Agent 由系统提示和可用函数列表定义。一个 Agent 可以通过从函数返回一个 `Agent` 对象来切换到另一个 Agent——运行器将上下文切换到该 Agent 并继续执行。没有持久化状态，没有图，没有对话管理器。

切换模式的工作原理如下：当前 Agent 的 LLM 调用一个工具，该工具的返回值是另一个 `Agent` 对象；Swarm 运行器检测到这个返回类型，将当前 Agent 的系统提示替换为新 Agent 的系统提示，并用新 Agent 的工具列表继续后续的对话轮次。这在 LLM 层面完全是标准工具调用；切换逻辑只存在于运行器解释返回值的方式中。

### Swarm 的定位

Swarm 最好理解为切换模式的参考实现，而非生产框架。它表明 Agent 切换不需要任何特殊机制，超出标准工具调用协议的范围之外：Agent"切换"只是将新 Agent 对象作为工具结果返回。其局限性是刻意为之：无持久化、无分支、无并行执行、无检查点。整个 Swarm 源代码约 300 行，是最小循环的可读示范。

对任何必须在进程重启后存活、支持人工审批或并行运行子 Agent 的 Agent 系统，Swarm 都是错误的工具。它的价值在于教学：在引入框架复杂性之前学习 Agent 模式的入口。

## DSPy

### 设计模型

DSPy（斯坦福 NLP，2023–2024）在类别上与前三者完全不同。它是**提示编译器和优化器**，而非 Agent 运行时。DSPy 程序定义模块——`Predict`、`ChainOfThought`、`ReAct`、`MultiChainComparison`——指定 LLM 调用的*结构*，而不编写提示文本。优化器（`BootstrapFewShot`、`MIPRO`、`BayesianSignatureOptimizer`）然后在开发集上以最大化某个指标为目标，搜索提示空间（示例、指令）。

```python
import dspy

class RAGPipeline(dspy.Module):
    def __init__(self):
        self.retrieve = dspy.Retrieve(k=5)
        self.generate = dspy.ChainOfThought("context, question -> answer")

    def forward(self, question):
        context = self.retrieve(question).passages
        return self.generate(context=context, question=question)

# 针对指标进行优化
optimizer = dspy.BootstrapFewShot(metric=answer_exact_match)
optimized_rag = optimizer.compile(RAGPipeline(), trainset=train_examples)
```

优化器的 `compile` 步骤在训练样本上运行流水线，收集成功的示例，并将最佳示例作为少样本示例注入最终提示。`MIPRO` 更进一步：使用元 LLM 提出指令变体，并对指标进行评估，实际上是在执行无梯度的提示梯度下降。

$$\text{最优提示} = \arg\max_{p \in \mathcal{P}} \mathbb{E}_{x \sim D_{\text{dev}}} [\text{metric}(f_p(x), y)]$$

其中 $\mathcal{P}$ 是由指令变体和示例排列组成的提示空间，$f_p$ 是使用提示 $p$ 的程序，$D_{\text{dev}}$ 是开发集。

### 关注点分离

DSPy 的核心洞察是：提示文本是超参数，而非手工打磨的工件。将其视为超参数可以实现系统性优化，而非手动调整。程序逻辑（`RAGPipeline.forward`）与提示内容（由编译器优化）之间的分离，类似于传统软件中算法与实现的分离。

基础设施结果：DSPy 程序不是直接部署的。编译器离线运行（`compile` 步骤可能在训练集上进行数百或数千次 LLM 调用），输出是一个优化后的程序对象，其提示在编译时已固定。服务基础设施看到的是标准 LLM 调用（带有固定提示）；DSPy 在推理时不可见。

### DSPy 在技术栈中的位置

DSPy 不是 LangGraph、AutoGen 或 Swarm 的替代品。它在评估和编译流水线中运行：

```
DSPy（离线优化）→ 优化后的提示工件
    ↓
LangGraph / AutoGen / Swarm（在线服务）→ 使用优化后的提示
```

构建 Agent 系统的团队可以使用 DSPy 优化每个 Agent 节点内部的提示（系统提示、少样本示例），同时使用 LangGraph 或 AutoGen 在运行时管理 Agent 循环。

### 编译的计算成本

`BootstrapFewShot` 优化器在编译期间大约需要 `num_bootstraps × trainset_size` 次 LLM 调用。10 次引导迭代和 100 个训练样本意味着 1,000 次 LLM 调用。以 GPT-4o 定价（约 $5/百万 token），每次调用 500 token 的提示，一次编译运行成本约 $2.50。`MIPRO` 的指令搜索可能需要多 10–50 倍的调用量。对于周期性重新优化（每周与新评估数据绑定的提示刷新）这是可以接受的，但对于服务时的每次请求优化则成本过高。

## 跨框架的基础设施影响

### 状态持久化规模

LangGraph 检查点器在每次节点转移时序列化状态。一个持续 3 小时、图转移 200 次、每次状态约 5,000 tokens（约 20 KB）的 Agent 会话，每个会话写入 4 MB 的检查点数据。在 1,000 个并发会话下，每小时产生 4 GB 的检查点写入。Postgres 可以处理这个量级；瓶颈通常是高并发下的连接池耗尽（使用 pgBouncer 或等效代理）。

AutoGen 没有内置持久化——对话历史存储在内存中。会话恢复需要从头重新运行，或在 `messages` 列表上实现自定义序列化层。

### 多 Agent 负载下的上下文窗口管理

在多 Agent 系统中，上下文压力成倍放大。一个监督者顺序调用 3 个专业 Agent，每次轮次每个 Agent 消耗 5,000 tokens 的上下文，监督者将所有结果回注入自己的上下文，不到 10 次监督者轮次就能耗尽 128k token 的上下文窗口。缓解方案是**将 Agent 边界作为上下文边界**：每个 Agent 维护自己的上下文；跨越边界的信息经过摘要而非逐字传递。LangGraph 通过状态压缩节点自然地执行这一原则；AutoGen 需要显式的 `compress_history` 调用。

### 深链中的延迟放大

Agent 链中每次 LLM 调用增加 1–5 s 的首 token 延迟。一个没有并行性的 5 节点 LangGraph 流水线，p50 延迟为 5 × 2 s = 10 s，p99 延迟随 LLM 调用尾延迟扩展——轻松达到 25–40 s。并行性（LangGraph 的 `Send`、AutoGen GroupChat 与异步调度）压缩独立分支。实际限制是关键路径的深度：任何顺序依赖链都是延迟下界。

对于延迟敏感的工作流，架构原则是最小化顺序深度。一个平坦的 5 路并行扇出加一个归约步骤，延迟模型是 max(5 × 并行 LLM 调用) + 1 次归约调用，而非 sum(5 次顺序调用)。

### 可观测性开销

追踪多 Agent 系统比追踪单模型调用成本更高。一个包含 50 次 LLM 调用的 10 节点 LangGraph 执行的 LangSmith 追踪，生成约 100 KB 的追踪数据。在每小时 1,000 次执行下，每小时 100 MB 的追踪数据——可以管理但不可忽视。LangSmith 的基于采样的追踪（只追踪部分执行）以降低对不频繁失败的调试可见性为代价减少了这一数据量。

OpenTelemetry 追踪（AutoGen 0.4+ 和 LangGraph 中间件）与现有可观测性基础设施集成。span 创建和导出的开销每次 LLM 调用增加约 1–5 ms——与 LLM 延迟相比可忽略，但在紧密循环中可见。

## 工程权衡

| 框架 | 状态管理 | 多 Agent 支持 | 可调试性 | 学习曲线 | 最佳使用场景 |
|---|---|---|---|---|---|
| LangGraph | 显式类型化状态字典；Postgres/Redis/SQLite 检查点器；支持恢复 | 一流的子图组合；监督者模式；并行 `Send` | 优秀：图结构 + LangSmith 追踪树；条件边逻辑可检查 | 高：图、状态、检查点器、条件边各有独立 API | 复杂的有状态工作流；人工审批门控；必须在重启后存活的长时运行 Agent |
| AutoGen | 无内置持久化；对话历史仅在内存中 | 带轮询或 LLM 选择器的 GroupChat；支持嵌套 Agent | 中等：消息日志可读，但非确定性对话流难以推理 | 中等：对话模型直观；GroupChat 调优需要反复试验 | 代码编写 + 执行循环；灵活的 Agent 协调优先于可调试性的任务 |
| Swarm | 无：每次运行无状态 | 通过返回值简单切换；无并行调度；无子图 | 高：300 行代码库；线性控制流；易于添加调试输出 | 极低：API 面极小 | 带专业 Agent 的线性流水线；学习/原型验证切换模式 |
| DSPy | 不适用（编译时，非运行时） | 不适用 | 编译时优秀（指标驱动）；推理时黑盒（提示文本是优化工件） | 高：新编程模型（模块、签名、优化器、指标） | 系统性提示优化；需要可复现提示版本管理的团队；评估驱动开发 |

## Claude Agent SDK / Anthropic 的方案

Anthropic 的 Claude Agent SDK 暴露了相同的核心循环原语（工具调用、结果注入、多轮上下文管理），同时具备两个独特的基础设施特性。第一，**提示缓存**：SDK 自动在同一会话的调用间缓存系统提示和工具定义，将 Agent 循环中重复 LLM 调用的首 token 延迟从约 1.5–3 s（完整 prefill）降至约 0.2–0.5 s（缓存命中 + 少量新 token）。对于有 2,000 token 系统提示的 10 步 Agent 循环，仅缓存就能节省约 8 × 1.3 s = 10.4 s 的 prefill 时间。第二，SDK 与 MCP 集成用于工具注册和发现，意味着工具模式无需硬编码在 Agent 定义中——在运行时从 MCP 服务器动态发现。

SDK 不强制任何图模型或对话参与者模型，在哲学上更接近 Swarm（极简脚手架），但具备 Swarm 所缺乏的生产特性（缓存、流式传输、批处理）。需要有状态长时运行 Agent 的团队，在 SDK 核心循环之上叠加 LangGraph 风格的状态管理。

## 参考文献

- [1] LangChain. _LangGraph: Building Stateful Multi-Actor Applications with LLMs._ blog.langchain.dev, 2024.
- [2] Wu et al. _AutoGen: Enabling Next-Gen LLM Applications via Multi-Agent Conversation._ arXiv:2308.08155, 2023.
- [3] OpenAI. _Swarm: Educational Framework for Multi-Agent Orchestration._ github.com/openai/swarm, 2024.
- [4] Khattab et al. _DSPy: Compiling Declarative Language Model Calls into Self-Improving Pipelines._ arXiv:2310.03714, 2023.
- [5] Khattab et al. _Optimizing Instructions and Demonstrations for Multi-Stage Language Model Programs._ arXiv:2406.11695, 2024.（MIPRO）
- [6] Anthropic. _Building Effective Agents._ anthropic.com/research/building-effective-agents, 2024.
- [7] Anthropic. _Claude Agent SDK._ docs.anthropic.com, 2025.
