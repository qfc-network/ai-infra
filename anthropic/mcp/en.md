# Model Context Protocol (MCP)

- **Authors / Org**: Anthropic
- **Published**: 2024-11 (initial release)
- **Links**: [official site](https://modelcontextprotocol.io) · [spec](https://spec.modelcontextprotocol.io) · [reference servers](https://github.com/modelcontextprotocol/servers)

## TL;DR

MCP is an **open protocol** that standardizes how LLM applications discover and use external context — tools, resources, prompts — from arbitrary servers. Think **LSP (Language Server Protocol) for LLMs**: instead of every LLM app implementing every integration, app authors write **clients**, integration authors write **servers**, and MCP is the wire format between them. It solves the M×N integration problem (M apps × N tools = M×N pieces of glue) by reducing it to M+N. Released by Anthropic in late 2024, adopted by Claude Desktop / Claude Code, then by OpenAI, Google, and essentially every other major agent-building tooling in 2025. MCP is now the de facto standard way to ship context to a language model.

## Context & Motivation

Pre-MCP, every LLM application did tool integration as bespoke code:

- ChatGPT plugins — proprietary format, limited lifespan.
- LangChain tool wrappers — library-specific, Python-only.
- Custom tool-call JSON per API per app — no reuse.

Every new integration meant writing custom code in each client app. Tool authors had to maintain N implementations for N clients. This scaled poorly and fragmented the ecosystem.

The comparable historical problem is pre-LSP code editors: each editor had to ship a language-specific parser/analyzer per language. LSP decoupled this — write a language server once, any LSP-capable editor can use it. MCP applies the same separation to LLM tool integration.

## Core Method

### Architecture

```
┌────────────────────┐           ┌────────────────────┐
│  LLM App (Host)    │           │   MCP Server       │
│                    │           │                    │
│  ┌──────────────┐  │   MCP     │  exposes:          │
│  │  MCP Client  │◄─┼───────────┼──► tools           │
│  └──────────────┘  │           │      resources     │
│  ┌──────────────┐  │           │      prompts       │
│  │  LLM         │  │           │                    │
│  └──────────────┘  │           │                    │
└────────────────────┘           └────────────────────┘
```

Three roles:

- **Host** — an app a user runs (Claude Desktop, Claude Code, an IDE plugin). Holds the LLM and presents the UX.
- **Client** — an MCP client inside the host, one per connected server. Handles protocol.
- **Server** — an independent process or remote service exposing capabilities over MCP. Could be a local binary (filesystem access), a desktop app (Slack, Notion), or a remote service (GitHub, databases).

One host typically runs many clients, each connected to one server.

### Three capability types

Servers advertise three kinds of things:

1. **Tools** — functions the model can *call*. Defined with JSON Schema inputs and arbitrary outputs. Model-driven: the LLM decides when to call.
2. **Resources** — data the model can *read*. Identified by URIs (e.g., `file://path`, `slack://channel/id`). App-driven or user-driven: the app or user decides what to pull in.
3. **Prompts** — reusable prompt templates the server offers. User-driven: the user invokes a named prompt (e.g., via a slash command in the host).

The distinction between tools (model-invoked) and resources (app/user-invoked) is important: **not every integration should be a tool**. Making something a resource means the user stays in control of what the model sees.

### Transport

The protocol is **JSON-RPC 2.0** over:

- **stdio** — for local servers spawned by the host as subprocesses. Simplest deployment; dominant for desktop apps.
- **HTTP with Server-Sent Events (SSE)** — for remote / networked servers.
- **Streamable HTTP** (added 2025) — a more modern variant replacing SSE for new deployments.

JSON-RPC was deliberately chosen for compatibility — it's widely implemented, has good tooling, and is familiar to anyone who's written LSP servers.

### Capabilities negotiation

On connection, client and server exchange **capabilities** — what features each supports (tools, resources, prompts, sampling, roots, logging). Feature detection is explicit; servers need not implement everything.

### Sampling (rarely implemented)

The protocol allows a **server to request the client to invoke the host's LLM** on its behalf. Lets a server embed LLM calls in its logic without shipping its own model. Rarely used in practice — most servers are "dumb" data/tool adapters.

### Roots

Clients can advertise **roots** — URIs the server is allowed to operate within. For a filesystem server, this is the set of directories it can access. Keeps least-privilege enforceable.

## Engineering Tradeoffs

| Decision | Gained | Gave up |
|---|---|---|
| Client-server split | M+N integrations instead of M×N; independent release cycles | Transport + protocol overhead; one more moving part |
| JSON-RPC over stdio/HTTP | Familiar, well-tooled, easy to implement | Not as compact as binary protocols; JSON parsing cost on hot paths |
| Three capability types (tools / resources / prompts) | User control over what's in context; not everything is a tool | More concepts to learn; app UX surface grows |
| Capabilities negotiation | Feature detection; servers can be partial | One more handshake round |
| Local stdio subprocesses | Zero network config; process isolation | Each server is its own process; startup cost per server |
| Open spec, permissive license | Industry-wide adoption possible | Anthropic gave up control to gain ecosystem |

## Experiments & Results

- Within ~6 months of release, MCP servers existed for GitHub, Slack, Notion, filesystem, PostgreSQL, SQLite, Google Drive, Figma, and hundreds of other integrations.
- Adopted by Claude Desktop (first), Claude Code (native), then by OpenAI (as a supported protocol in the Assistants / Responses API), Google (Gemini tooling), GitHub Copilot, Cursor, Zed, and others by mid-2025.
- Official SDKs available in Python, TypeScript, Kotlin, Rust, C#, Swift, Java.
- No significant competing standard emerged. Proprietary formats (OpenAI plugins, Google extensions) were retired or refactored around MCP.

## Reproducibility Notes

- The spec is fully open (Apache 2.0). Reference SDKs and servers at `modelcontextprotocol` on GitHub.
- Writing a basic server: ~100 lines in Python using the SDK. Writing a basic client: similar.
- The ecosystem of community servers is large and growing; before writing one, check if it exists.
- For production deployments, watch for: sandboxing (MCP servers often run as local subprocesses with full user privileges — isolate carefully), logging (MCP traffic is easy to forget in observability setups), and authentication for remote servers (the spec is explicit about this but leaves implementation to servers).

## Commentary

MCP is the **rare "obvious in retrospect" protocol**: after it existed, everyone wondered why it took so long. The LSP analogy is exact, and every editor team who had lived through the LSP transition recognized the shape immediately. The non-obvious part was timing — Anthropic could have shipped this any time in 2023, but the window where "agents using tools" was universal enough to justify a protocol only really opened in mid-2024.

What makes MCP load-bearing is **the capability-type distinction** (tools vs resources vs prompts). ChatGPT plugins treated everything as a function call; MCP recognizes that not every integration is something the model should *choose* to invoke — some things are user-invoked (resources via UI selection, prompts via slash commands). This matches how agentic tools actually work in production: the best agents pull from a mix of model-initiated calls and user-curated context.

The adoption by OpenAI in particular is the strongest signal MCP won. OpenAI had its own format (plugins, then function calling, then Assistants API tools), and for it to concede to an Anthropic-originated protocol required MCP to be both technically better and already ubiquitous. By Q3 2025 that's what happened.

For infra people, the practical question is **how to expose your systems via MCP**. If you run internal tools (knowledge bases, build systems, observability dashboards, ops runbooks), the default answer is: **write an MCP server**, not a ChatGPT plugin or a LangChain tool. Any LLM client in your org can then use it, and when your org switches clients (as many will across 2025–2026), nothing breaks. This is exactly the LSP migration pattern repeating itself.

The open question for the next 12 months: **how MCP composes with agent-to-agent protocols** (A2A, ACP, and the emerging set). MCP is model↔tool; A2A is agent↔agent. Together they're the shape of where tooling standardization is heading.

## References

- [1] MCP official site: https://modelcontextprotocol.io
- [2] MCP specification: https://spec.modelcontextprotocol.io
- [3] Reference servers: https://github.com/modelcontextprotocol/servers
- [4] Python SDK: https://github.com/modelcontextprotocol/python-sdk
- [5] Anthropic announcement post (2024-11): https://www.anthropic.com/news/model-context-protocol
- [6] Language Server Protocol (the inspiration): https://microsoft.github.io/language-server-protocol/
