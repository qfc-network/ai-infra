# 工具调用 / 函数调用基础设施

_系统设计条目，涵盖工具调用背后的运行时机制：模式定义、调度、并行执行、流式传输、验证、安全层以及延迟计算。MCP 传输层详见 [../../anthropic/mcp/](../../anthropic/mcp/)；工具调用链的工作流模式详见 [../../anthropic/building-effective-agents/](../../anthropic/building-effective-agents/)；编排工具调用循环的 Agent 框架详见 [../agent-frameworks/](../agent-frameworks/)。_

## 一句话总结

函数调用是一种协议，LLM 通过该协议输出结构化的 token 序列——一个符合声明模式的 JSON 对象——而非（或同时输出）自然语言回应。宿主运行时拦截此输出，执行指定工具，并将结果作为新的用户轮次注入对话。模型本身从不直接执行代码，它只生成结构化的意图；运行时负责执行。这种分离既是核心安全属性，也是主要的延迟来源：每一次工具调用至少增加一次 LLM 往返加上工具自身的执行时间。基础设施设计是否合理，直接决定了工具增强型 Agent 的体验是流畅还是迟滞。

## 背景与动机

### 原始 LLM API 的空白

标准补全 API 返回 token，token 对文本生成有用，但真实世界的任务需要副作用：查找数据库记录、执行计算、调用在线 API、写入文件。朴素解法是提示工程——指示模型以可识别格式输出"LOOKUP: customer_id=42"，然后用正则解析。这在玩具规模下有效，但在模式演化、嵌套结构、多参数调用和并行操作场景下会崩溃。

函数调用将契约形式化：为每个可用工具提供机器可读的模式，模型经过训练（或提示）在意图与模式匹配时输出结构化调用。运行时在调度前对模式进行验证，提供确定性的失败模式，而不是无声的解析错误。

### 协议的起源

OpenAI 于 2023 年 6 月在 gpt-3.5-turbo-0613 和 gpt-4-0613 中引入了函数调用。Anthropic 于 2024 年 3 月在 Claude 3 中引入了 `tool_use` 内容块。Google Gemini API 在模型配置中使用 `functionDeclarations`。三者在结构上是同构的：工具是名称、自然语言描述和描述其参数的 JSON Schema 对象的组合。字段名、信封格式等表面差异不影响底层运行时设计。

MCP（模型上下文协议）位于这一层之上——它定义了工具如何跨进程边界注册、发现和传输——但 LLM API 层面的在线工具调用格式仍然是这里描述的 JSON Schema + 结构化输出模式。

## 核心机制

### 模式定义

工具声明为一个包含三个字段的 JSON 对象：

```json
{
  "name": "get_customer_orders",
  "description": "检索客户在指定日期范围内的所有订单，返回包含 id、status 和 total 的订单对象列表。",
  "parameters": {
    "type": "object",
    "properties": {
      "customer_id": {
        "type": "string",
        "description": "唯一客户标识符（UUID 格式）"
      },
      "start_date": {
        "type": "string",
        "format": "date",
        "description": "ISO 8601 起始日期，含当天"
      },
      "end_date": {
        "type": "string",
        "format": "date",
        "description": "ISO 8601 截止日期，含当天"
      }
    },
    "required": ["customer_id"]
  }
}
```

此模式被注入模型上下文——通常序列化为 JSON 或 XML 后前置于系统提示，或通过专用的 `tools` 参数传递（服务基础设施自动将其编码进提示模板）。模型看到的描述是关于*何时*调用工具的指导；参数模式约束模型*必须*输出什么。

模型决定调用工具时，其输出是类型为 `tool_use` 的内容块（Anthropic）或带有 `tool_calls` 数组的 `function_call` 完成原因（OpenAI）。输出的 JSON 必须符合声明的参数模式。

### 请求-响应循环

单次工具调用的执行过程如下：

```
1. 用户发送消息 M 给 LLM
2. LLM 生成包含 tool_call(name=T, args=A) 的响应 R
3. 运行时验证 A 是否符合 schema(T)          [进程内约 0 ms]
4. 运行时调度调用：result = execute(T, A)   [10 ms – 30 s]
5. 运行时将工具结果作为新用户轮次追加到对话中
6. LLM 生成后续响应 R'
7. R' 可能包含另一个工具调用 → 从步骤 3 重复
   或 R' 是最终文本响应 → 返回给用户
```

单次工具调用轨迹的延迟为：

$$T_{\text{single}} = T_{\text{TTFT}} + T_{\text{tool}} + T_{\text{续接\_TTFT}} + T_{\text{生成}}$$

其中 $T_{\text{TTFT}}$ 是模型首 token 时间（前沿模型通常为 0.5–3 秒，取决于提示长度和负载），$T_{\text{tool}}$ 是工具执行时间（快速数据库查询约 100 ms，慢速网络搜索或代码执行约 5–30 s），$T_{\text{续接\_TTFT}}$ 是第二次模型调用的首 token 时间（比第一次略长，因为上下文已包含工具结果）。

### 并行工具调用

OpenAI 和 Anthropic API 均支持在单次模型响应中返回多个工具调用。模型输出工具调用对象列表而非单个对象：

```json
{
  "tool_calls": [
    {"id": "call_1", "function": {"name": "get_weather", "arguments": "{\"city\": \"Tokyo\"}"}},
    {"id": "call_2", "function": {"name": "get_weather", "arguments": "{\"city\": \"London\"}"}},
    {"id": "call_3", "function": {"name": "get_exchange_rate", "arguments": "{\"from\": \"JPY\", \"to\": \"GBP\"}"}}
  ]
}
```

运行时并发调度三个调用。延迟从 $\sum_i T_{\text{tool}_i}$ 压缩为 $\max_i T_{\text{tool}_i}$。三个各 500 ms 的工具，串行需要 1,500 ms，并行只需 500 ms——在运行器并发管理开销之外没有额外成本，实现 3 倍加速。

模型必须能够识别调用之间的独立性（没有调用依赖另一个调用的输出）才能在一批中输出它们。这是一种随模型规模和指令跟随能力提升的能力；较小的模型经常将可并行化的调用串行化。

运行时实现需要：首先并行调度所有调用（异步 IO 或线程池）；然后按 `call_id` 收集所有结果；最后在下一次 LLM 调用前将所有结果作为单个多部分用户轮次注入。若某个调用失败，错误结果替代成功结果注入，由模型决定是重试、使用备选方案，还是上报失败。

### 流式工具调用

在流式模式下，模型逐步输出 token。工具调用参数作为部分 JSON 字符串到达：

```
delta: {"index": 0, "delta": {"tool_calls": [{"index": 0, "function": {"arguments": "{\""}}]}}
delta: {"index": 0, "delta": {"tool_calls": [{"index": 0, "function": {"arguments": "city"}}]}}
delta: {"index": 0, "delta": {"tool_calls": [{"index": 0, "function": {"arguments": "\": \""}}]}}
delta: {"index": 0, "delta": {"tool_calls": [{"index": 0, "function": {"arguments": "Tokyo"}}]}}
delta: {"index": 0, "delta": {"tool_calls": [{"index": 0, "function": {"arguments": "\"}"}}]}}
```

运行时必须缓冲参数字符串直到流关闭（finish_reason = `tool_calls`）再调度，因为 JSON 在完整之前无效。这意味着工具执行无法在模型完全生成参数之前开始。

对于延迟敏感的应用，影响显著：一个 2,000 token 的提示加上 100 token 的工具调用参数，运行器需要等待约 100 token 的生成时间（以 100–500 tokens/秒的解码速度约为 100–500 ms）再调度。减少等待时间的策略包括：**约束参数模式**（更简短的模式生成更快、token 更少）；**投机性调度**（对幂等的只读工具，当足够多的结构可解析时尝试提前调度，但对有副作用的工具存在风险）；**乐观预取**（对确定性工具序列，在模型生成完调用之前预取前一个工具的结果）。

### 模式验证

调度工具调用前，运行时对模型输出的参数进行 JSON Schema 验证，捕获缺失的必要字段、类型错误、枚举违规和格式违规。

验证失败时的标准处理方式：一是**带错误注入的重试**，将验证错误作为系统消息追加后重新调用模型，有效但额外消耗一次 LLM 往返（500 ms–3 s）；二是**模式修复**，尝试强制转换无效输出（例如将整数转为字符串），对语义字段危险（错误的日期格式可能转换为不同的日期）；三是**硬失败**，向用户返回错误，适合生产系统中静默失败比显式错误代价更高的场景。

模式复杂度与模型可靠性成反比。包含 15 个可选字段、嵌套对象和跨字段约束的模式产生的错误率，远高于只有 2 个必要字符串字段的模式。工程原则是：工具模式应尽量精简。`description` 字段的影响大于模式复杂度：精确、富含示例的描述比严格的类型约束更能降低语义错误率（调用了正确的工具但传入了错误的意图）。

### 工具调用轨迹与上下文增长

在多步 Agent 任务中，对话随每次工具调用循环增长：

```
系统提示：     S tokens
用户消息：     U tokens
模型响应 1：   R1 tokens（包含工具调用）
工具结果 1：   T1 tokens
模型响应 2：   R2 tokens（可能包含另一个工具调用）
工具结果 2：   T2 tokens
...
模型响应 N：   RN tokens（最终答案）

步骤 N 时的总上下文：S + U + sum(Ri) + sum(Ti)
```

工具结果往往冗长。网络搜索结果可能返回 2,000 token 的文档文本，宽表上的 SQL 查询结果可能返回 5,000 token。经过 5 次工具调用，上下文很容易增长至 15,000–30,000 token。这带来两个后果：其一，**TTFT 随上下文增加**，注意力计算的序列长度复杂度为 $O(n^2)$（无 FlashAttention 分块和 KV cache 复用时），每次后续 LLM 调用都比上一次慢；其二，**上下文窗口耗尽**，128k token 的上下文窗口看似很大，但调用冗长工具 20 次的 Agent 会将其耗尽。

工具结果的截断策略：**硬截断**，将结果裁剪到固定 token 预算（如前 1,000 tokens），快速简单但丢失尾部内容；**结构化摘要**，调用较便宜的模型对结果进行摘要后再注入，增加一次 LLM 调用延迟但压缩效果显著；**语义分块 + 检索**，将完整结果存入向量数据库，只注入最相关的 top-k 块，对文档搜索结果效果好，嵌入 + 检索开销约 50–200 ms。

工具结果的 token 预算应在 Agent 设计时显式规划。实用启发式规则：任何单个工具结果不超过上下文窗口的 20%，所有累积工具结果合计不超过窗口总量的 60%，为增长的模型响应链留有余地。

## 安全层

工具调用是代码执行接口，模型的输出会触达真实系统。生产环境需要多层安全防护：

**模式级约束**：缩小操作空间。接受任意 `path` 字符串的 `write_file` 工具远比接受允许路径枚举的工具危险。声明最小必要参数集。

**人工审批中断**：对高后果操作（发送邮件、修改数据库记录、执行金融交易），在调度前要求明确的人工批准。中断/恢复模式（LangGraph 中使用，详见 [../agent-frameworks/](../agent-frameworks/)）是标准实现方式：Agent 暂停，向用户呈现待执行的工具调用，仅在批准后恢复。

**沙箱化**：代码执行工具（Python REPL、shell exec）必须在隔离环境（容器、虚拟机、Firecracker 等 microVM）中运行。可访问网络的工具（网络抓取、API 调用）应在出口过滤下运行，防止 SSRF 和数据外泄。文件系统工具应进行 chroot 或使用虚拟文件系统层。

**工具级速率限制**：限制每分钟、每会话、每用户的调用次数，约束失控的 Agent 循环。导致 `send_email` 循环的故障应在第 5 次调用时触发速率限制，而非第 5,000 次。

**最小权限原则**：每个工具只应拥有其功能所需的权限。读取客户记录的工具不应有写权限；向沙箱目录写入的工具不应有网络访问权限。这在模型被操控（提示注入）或系统性出错时限制了爆炸半径。

## 工程权衡

| 方案 | 优势 | 劣势 | 适用场景 |
|---|---|---|---|
| 并行工具调用（每模型轮次多调用） | 总延迟降至 max(Ti) 而非 sum(Ti)；独立操作减少 LLM 往返次数 | 需要模型具备识别独立调用的能力；运行器并发调度与结果合并更复杂 | 任何在一个逻辑步骤中需要两次以上独立数据获取或操作的 Agent 任务 |
| 串行工具调用（每轮次一个） | 运行器简单；调试容易；模型在下一次调用前能看到每个结果 | 延迟 = 所有工具时间之和 + LLM 时间；上下文增长更快（更多模型响应 token） | 调用之间存在数据依赖；调试 Agent 行为时 |
| 严格 JSON Schema 验证（拒绝无效输出） | 确定性失败；无静默数据损坏；强制执行契约 | 验证失败需额外一次 LLM 往返；模型顽固出错时可能循环 | 正确性高于可用性的生产系统；任何有副作用的工具 |
| 宽松验证（强制转换或跳过验证） | 不会因模式错误而阻塞；运行器更简单 | 静默失败；强制转换可能损坏语义内容；下游难以调试 | 低风险输出的只读工具；原型开发 |
| 流式工具调用传递 | 用户立即看到部分响应文本；感知延迟更低 | 必须等参数完全生成才能调度；运行器状态机更复杂 | 混合响应（文本与工具调用同时出现）；文本响应在工具调用前的聊天 UX |
| 沙箱化执行（容器 / microVM） | 隔离副作用；限制提示注入的爆炸半径 | 每次工具调用冷启动 50–500 ms；资源开销；网络出口过滤增加延迟 | 任何执行用户影响代码的工具；文件写入；shell exec；网络抓取 |

## 实际延迟计算

一个真实的多步 Agent 轨迹展示了累积成本：

| 步骤 | 操作 | 耗时 |
|---|---|---|
| 第 1 轮 | 用户消息 → LLM（首 token，2k token 提示） | ~800 ms |
| 第 1 轮 | LLM 生成工具调用参数（50 tokens） | ~150 ms |
| 第 1 轮 | 模式验证 | ~1 ms |
| 第 1 轮 | 工具执行（数据库查询） | ~80 ms |
| 第 2 轮 | LLM 处理 3k token 上下文（首 token） | ~1,100 ms |
| 第 2 轮 | LLM 生成第 2 次工具调用（并行，共 2 个） | ~200 ms |
| 第 2 轮 | 工具执行（并行：网络搜索 + 数据库） | ~1,200 ms（max(1,200 ms, 90 ms)） |
| 第 3 轮 | LLM 处理 6k token 上下文（首 token） | ~1,800 ms |
| 第 3 轮 | LLM 生成最终响应（200 tokens） | ~600 ms |
| **合计** | | **~5.9 s** |

若第 2 轮不使用并行工具调用，网络搜索和数据库查询串行执行：1,200 + 90 = 1,290 ms 而非最大值 1,200 ms——此处差异不大，因为网络搜索占主导。但对三个各 300 ms 的等延迟工具，并行化将工具时间从 900 ms 节省至 300 ms，节省 600 ms。各轮次 TTFT 的增长（800 ms → 1,100 ms → 1,800 ms）反映了上下文增长时 KV cache prefill 的 $O(n)$ 成本，即使有前缀缓存（[../prefix-caching/](../prefix-caching/)）也是如此。

实践工程洞察：减少往返次数（使用并行调用），减少上下文增长（激进截断冗长结果），保持工具执行快速（异步 IO、连接池、对幂等读取结果进行缓存）。

## 协议比较

三大主流 API 收敛于相同的底层模型，表面差异仅属管理层面：

| API | 工具定义字段 | 调用输出格式 | 多调用支持 | 流式传输 |
|---|---|---|---|---|
| OpenAI | `tools[].function` | 消息中的 `tool_calls[]` 数组 | 是（parallel_tool_calls=true） | index+delta 模式 |
| Anthropic | `tools[]` | `tool_use` 内容块 | 是（一次响应中多个块） | input_json_delta 事件 |
| Gemini | `tools[].functionDeclarations[]` | `functionCall` parts | 是（多个 parts） | 流式函数调用 parts |

MCP 增加了传输和注册层：工具声明为 MCP 资源，MCP 服务器通过协议（stdio、SSE 或 HTTP）暴露它们，LLM 客户端动态发现。每个工具的 JSON Schema 定义不变；MCP 是管道，而非模式本身。

## 参考文献

- [1] OpenAI. _Function Calling._ platform.openai.com/docs/guides/function-calling, 2023.
- [2] Anthropic. _Tool Use._ docs.anthropic.com/en/docs/build-with-claude/tool-use, 2024.
- [3] Google. _Function Calling with the Gemini API._ ai.google.dev/gemini-api/docs/function-calling, 2024.
- [4] Anthropic. _Model Context Protocol._ modelcontextprotocol.io, 2024.（MCP 传输层）
- [5] Yao et al. _ReAct: Synergizing Reasoning and Acting in Language Models._ ICLR 2023.（工具调用循环的基础模式）
- [6] Anthropic. _Building Effective Agents._ anthropic.com/research/building-effective-agents, 2024.
