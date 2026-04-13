# Model Context Protocol (MCP)

- **作者 / 机构**：Anthropic
- **发表时间**：2024-11（首发）
- **链接**：[官网](https://modelcontextprotocol.io) · [规范](https://spec.modelcontextprotocol.io) · [参考 server](https://github.com/modelcontextprotocol/servers)

## 一句话总结

MCP 是一个**开放协议**，标准化 LLM 应用如何从任意 server 发现和使用外部上下文——工具、资源、prompt。可类比为 **LSP（Language Server Protocol）的 LLM 版**：不再是每个 LLM 应用实现每个集成，而是应用作者写**客户端**、集成作者写**server**，MCP 是它们之间的线协议。把 M×N 集成问题（M 应用 × N 工具 = M×N 份胶水）归约为 M+N。2024 年底 Anthropic 发布，Claude Desktop / Claude Code 首先采用，随后 OpenAI、Google 和几乎所有主要 agent 工具在 2025 年相继接入。如今 MCP 是给语言模型送上下文的事实标准。

## 背景与动机

MCP 之前，每个 LLM 应用的工具集成都是专属代码：

- ChatGPT 插件——专有格式、寿命有限。
- LangChain 工具包装——库特定、只 Python。
- 每个 API 每个应用自定义 JSON tool call——无复用。

每个新集成意味着每个客户端应用里都要写一份自定义代码。工具作者要为 N 个客户端维护 N 份实现。扩展性差、生态碎片。

历史上可比问题是 LSP 之前的代码编辑器：每个编辑器要为每种语言各出一份 parser/analyzer。LSP 解耦了这件事——语言 server 写一次，任何 LSP 兼容编辑器都能用。MCP 把同样的解耦套到 LLM 工具集成上。

## 核心方法

### 架构

```
┌────────────────────┐           ┌────────────────────┐
│  LLM 应用 (Host)    │           │   MCP Server       │
│                    │           │                    │
│  ┌──────────────┐  │   MCP     │  暴露：             │
│  │  MCP Client  │◄─┼───────────┼──► tools           │
│  └──────────────┘  │           │      resources     │
│  ┌──────────────┐  │           │      prompts       │
│  │  LLM         │  │           │                    │
│  └──────────────┘  │           │                    │
└────────────────────┘           └────────────────────┘
```

三个角色：

- **Host** —— 用户运行的应用（Claude Desktop、Claude Code、IDE 插件）。持有 LLM、提供 UX。
- **Client** —— host 内的 MCP client，每连一个 server 一个 client。处理协议。
- **Server** —— 独立进程或远程服务，通过 MCP 暴露能力。可以是本地二进制（文件系统）、桌面应用（Slack、Notion）、或远程服务（GitHub、数据库）。

一个 host 通常跑多个 client，每个 client 连一个 server。

### 三种能力类型

Server 宣告三种东西：

1. **Tools** —— 模型可*调用*的函数。用 JSON Schema 描述输入、任意输出。模型驱动：LLM 决定什么时候调。
2. **Resources** —— 模型可*读*的数据。以 URI 标识（如 `file://path`、`slack://channel/id`）。应用或用户驱动：应用或用户决定把什么拉进上下文。
3. **Prompts** —— server 提供的可复用 prompt 模板。用户驱动：用户以命名方式调用（如在 host 里通过斜杠命令）。

tools（模型调用）与 resources（应用/用户调用）的区分很重要：**不是所有集成都该成工具**。做成资源意味着**用户掌控模型看到什么**。

### 传输

协议是 **JSON-RPC 2.0**，可跑在：

- **stdio** —— 本地 server 由 host 以子进程拉起。最简部署；桌面应用的主流。
- **HTTP + Server-Sent Events (SSE)** —— 远程 / 网络 server。
- **Streamable HTTP**（2025 年加入）—— 更现代的变体，逐步替换 SSE。

JSON-RPC 是刻意选择：广泛实现、工具成熟、对写过 LSP server 的人熟悉。

### 能力协商

连接时 client 和 server 交换**能力**——各自支持什么特性（tools、resources、prompts、sampling、roots、logging）。特性检测显式；server 不必都实现。

### Sampling（罕用）

协议允许 **server 请求 client 代为调用 host 的 LLM**。让 server 在自己逻辑里嵌 LLM 调用，无需自带模型。实际少见——多数 server 是"哑"数据/工具适配器。

### Roots

Client 可宣告 **roots**——允许 server 操作的 URI 范围。对文件系统 server 就是它能访问的目录集。让最小权限可执行。

## 工程 Tradeoff

| 决策 | 得到什么 | 放弃什么 |
|---|---|---|
| 客户端-server 拆分 | 集成数 M+N 而非 M×N；发布周期独立 | 传输 + 协议开销；多一个动件 |
| JSON-RPC over stdio/HTTP | 熟悉、工具好、易实现 | 不如二进制协议紧凑；热路径有 JSON 解析成本 |
| 三类能力（tools / resources / prompts） | 用户掌控上下文；不是什么都成工具 | 概念多一些；应用 UX 表面变大 |
| 能力协商 | 特性检测；server 可部分实现 | 多一次握手 |
| 本地 stdio 子进程 | 零网络配置；进程隔离 | 每 server 自成进程；每 server 启动成本 |
| 开放规范、宽松许可证 | 全行业采纳成为可能 | Anthropic 放弃控制换生态 |

## 实验与结果

- 发布后约 6 个月内，GitHub、Slack、Notion、文件系统、PostgreSQL、SQLite、Google Drive、Figma 等数百种集成的 MCP server 已经存在。
- 采用：先是 Claude Desktop、Claude Code 原生，2025 年中 OpenAI（在 Assistants / Responses API 中作为受支持协议）、Google（Gemini 工具）、GitHub Copilot、Cursor、Zed 等陆续接入。
- 官方 SDK：Python、TypeScript、Kotlin、Rust、C#、Swift、Java。
- 无明显竞争标准。专有格式（OpenAI 插件、Google 扩展）被废弃或围绕 MCP 重构。

## 复现要点

- 规范完全开放（Apache 2.0）。参考 SDK 和 server 在 GitHub `modelcontextprotocol` 下。
- 写一个基础 server：用 SDK 约 100 行 Python。写一个基础 client 类似。
- 社区 server 生态大且在增长；自己写之前先查有没有。
- 生产部署要注意：沙箱（MCP server 常以本地子进程跑，拥有完整用户权限——要谨慎隔离）、日志（MCP 流量容易在可观测性里被漏掉）、远程 server 的认证（规范明示此事但实现交给 server）。

## 个人评注

MCP 是罕见的"**事后看显然**"型协议：它存在之后，大家都想为什么来得这么慢。LSP 类比完全贴合，每个经历过 LSP 迁移的编辑器团队一看就认出它的形状。不显然的只是时机——Anthropic 2023 年任何时候都能发，但"agent 用工具"这件事普及到足以支撑协议，窗口实际到 2024 年中才真正打开。

MCP 真正的承重能力在**能力类型区分**（tools vs resources vs prompts）。ChatGPT 插件把一切当函数调用；MCP 承认不是每个集成都该让模型*选*着调——有些是用户触发（通过 UI 选择资源、通过斜杠命令触发 prompt）。这匹配生产里 agent 工具的真实用法：最好的 agent 从"模型主动调用 + 用户精选上下文"混合里拉东西。

OpenAI 的采纳是 MCP 胜利最强的信号。OpenAI 自己有过格式（plugins → function calling → Assistants API tools），让它转向一个 Anthropic 发起的协议，MCP 必须既技术上更好、又已然普及。2025 Q3 这一步确实发生了。

对 infra 人员，实务问题是**如何把你的系统用 MCP 暴露出来**。如果你跑内部工具（知识库、构建系统、可观测性仪表板、运维 runbook），默认答案是：**写一个 MCP server**，而不是 ChatGPT 插件或 LangChain 工具。你们组织里任何 LLM 客户端都能用上，换客户端（2025–2026 多数组织会换）时什么都不坏。这正是 LSP 迁移在重复上演。

未来 12 个月的开放问题：**MCP 如何与 agent-to-agent 协议组合**（A2A、ACP、以及正在涌现的一批）。MCP 是 模型↔工具；A2A 是 agent↔agent。两者合起来就是工具标准化前进的方向。

## 参考

- [1] MCP 官网：https://modelcontextprotocol.io
- [2] MCP 规范：https://spec.modelcontextprotocol.io
- [3] 参考 server：https://github.com/modelcontextprotocol/servers
- [4] Python SDK：https://github.com/modelcontextprotocol/python-sdk
- [5] Anthropic 发布公告（2024-11）：https://www.anthropic.com/news/model-context-protocol
- [6] Language Server Protocol（灵感来源）：https://microsoft.github.io/language-server-protocol/
