# Tool Use / Function Calling Infrastructure

_A system design entry covering the runtime machinery behind tool calls: schema definition, dispatch, parallel execution, streaming, validation, safety, and latency arithmetic. The MCP transport layer is covered in [../../anthropic/mcp/](../../anthropic/mcp/); workflow patterns that chain tool calls are in [../../anthropic/building-effective-agents/](../../anthropic/building-effective-agents/); agent frameworks that orchestrate tool-call loops are in [../agent-frameworks/](../agent-frameworks/)._

## TL;DR

Function calling is a protocol by which an LLM emits a structured token sequence — a JSON object conforming to a declared schema — instead of (or in addition to) a natural language response. The host runtime intercepts this output, executes the nominated tool, and injects the result back into the conversation as a new user turn. The model never executes code directly; it only produces the structured intent. The runtime owns execution. This separation is both the key safety property and the primary source of latency: every tool call adds at minimum one LLM round-trip plus the tool's own execution time. Getting the infrastructure right determines whether tool-augmented agents feel responsive or sluggish.

## Context & Motivation

### The gap raw LLM APIs leave open

A standard completion API returns tokens. Tokens are useful for text, but real-world tasks require side effects: looking up a database record, running a calculation, querying a live API, writing a file. The naive solution is prompt engineering — instruct the model to output "LOOKUP: customer_id=42" in a recognizable format, then parse that with regex. This works at toy scale. It breaks under schema evolution, nested structures, multi-argument calls, and parallel operations.

Function calling formalizes the contract: the model is given a machine-readable schema for each available tool. It is trained (or prompted) to emit a structured call when the schema matches its intent. The runtime validates against the schema before dispatching, giving deterministic failure modes instead of silent parse errors.

### Who defined the protocol

OpenAI introduced function calling in June 2023 (gpt-3.5-turbo-0613, gpt-4-0613). Anthropic introduced `tool_use` content blocks in Claude 3 (March 2024). Google's Gemini API uses `functionDeclarations` in the model config. All three are structurally isomorphic: a tool is a name, a natural-language description, and a JSON Schema object describing its parameters. The surface differences (field names, envelope format) do not affect the underlying runtime design.

MCP (Model Context Protocol) sits above this layer — it defines how tools are registered, discovered, and transported across process boundaries — but the on-wire tool-call format at the LLM API level is still the JSON Schema + structured output pattern described here.

## Core Mechanism

### Schema definition

A tool is declared as a JSON object with three fields:

```json
{
  "name": "get_customer_orders",
  "description": "Retrieve all orders for a customer within a date range. Returns a list of order objects with id, status, and total.",
  "parameters": {
    "type": "object",
    "properties": {
      "customer_id": {
        "type": "string",
        "description": "The unique customer identifier (UUID format)"
      },
      "start_date": {
        "type": "string",
        "format": "date",
        "description": "ISO 8601 start date, inclusive"
      },
      "end_date": {
        "type": "string",
        "format": "date",
        "description": "ISO 8601 end date, inclusive"
      }
    },
    "required": ["customer_id"]
  }
}
```

This schema is injected into the model's context — typically serialized as JSON or XML and prepended to the system prompt, or passed via a dedicated `tools` parameter that the serving infrastructure encodes into the prompt template automatically. The model sees the description as guidance for *when* to call the tool; the parameter schema constrains *what* it must emit.

The model's output when it decides to call a tool is a content block of type `tool_use` (Anthropic) or a `function_call` finish reason with a `tool_calls` array (OpenAI). The emitted JSON must conform to the declared parameter schema.

### The request-response loop

A single tool call executes as follows:

```
1. User sends message M to LLM
2. LLM generates response R containing tool_call(name=T, args=A)
3. Runtime validates A against schema(T)  [~0 ms if in-process]
4. Runtime dispatches call: result = execute(T, A)  [10 ms – 30 s]
5. Runtime appends tool result to conversation as a new user turn
6. LLM generates continuation response R'
7. R' may contain another tool call → repeat from step 3
   or R' is a final text response → return to user
```

The latency of a single-tool-call trajectory is:

$$T_{\text{single}} = T_{\text{TTFT}} + T_{\text{tool}} + T_{\text{continuation\_TTFT}} + T_{\text{generation}}$$

Where $T_{\text{TTFT}}$ is the model's time to first token on the initial turn (typically 0.5–3 s for frontier models depending on prompt length and load), $T_{\text{tool}}$ is the tool execution time (100 ms for a fast database lookup, 5–30 s for a slow web search or code execution), and $T_{\text{continuation\_TTFT}}$ is the second model call's time to first token (slightly longer than the first because the context now includes the tool result).

### Parallel tool calls

Both OpenAI and Anthropic APIs support returning multiple tool calls in a single model response. The model emits a list of tool call objects rather than a single one:

```json
{
  "tool_calls": [
    {"id": "call_1", "function": {"name": "get_weather", "arguments": "{\"city\": \"Tokyo\"}"}},
    {"id": "call_2", "function": {"name": "get_weather", "arguments": "{\"city\": \"London\"}"}},
    {"id": "call_3", "function": {"name": "get_exchange_rate", "arguments": "{\"from\": \"JPY\", \"to\": \"GBP\"}"}}
  ]
}
```

The runtime dispatches all three concurrently. The latency collapses from $\sum_i T_{\text{tool}_i}$ to $\max_i T_{\text{tool}_i}$. For three 500 ms tools, that is the difference between 1,500 ms and 500 ms of tool execution time — a 3× reduction at no cost beyond the concurrency management in the runner.

The model must be capable of identifying that the calls are independent (no call depends on another's output) to emit them in one batch. This is a capability that improves with model scale and instruction-following quality; smaller models often serialize calls that could be parallelized.

The runtime implementation requires:
1. Dispatch all calls in parallel (async IO or thread pool).
2. Collect all results, matching by `call_id`.
3. Inject all results as a single multi-part user turn before the next LLM call.

If any individual call fails, the error result is injected rather than the success result. The model then decides whether to retry, use a fallback, or report failure.

### Streaming tool calls

In streaming mode, the model emits tokens incrementally. Tool call arguments arrive as partial JSON strings:

```
delta: {"index": 0, "delta": {"tool_calls": [{"index": 0, "function": {"arguments": "{\""}}]}}
delta: {"index": 0, "delta": {"tool_calls": [{"index": 0, "function": {"arguments": "city"}}]}}
delta: {"index": 0, "delta": {"tool_calls": [{"index": 0, "function": {"arguments": "\": \""}}]}}
delta: {"index": 0, "delta": {"tool_calls": [{"index": 0, "function": {"arguments": "Tokyo"}}]}}
delta: {"index": 0, "delta": {"tool_calls": [{"index": 0, "function": {"arguments": "\"}"}}]}}
```

The runtime must buffer the argument string until the stream closes (finish_reason = `tool_calls`) before dispatching, because JSON is not valid until complete. This means tool execution cannot begin until the model has fully generated the arguments.

For latency-sensitive applications, the implication is significant: with a 2,000-token prompt and a tool call whose arguments are 100 tokens, the runner waits for ~100 tokens of generation time (roughly 100–500 ms at 100–500 tokens/second decode speed) before dispatching. Strategies to reduce this:
- **Constrain argument schemas**: smaller, simpler argument schemas (short strings, enums) generate faster and with fewer tokens.
- **Speculative dispatch**: for idempotent read-only tools, some runtimes attempt to dispatch on partial JSON when enough structure is parseable (e.g., the function name and a required string argument are complete). This is risky for mutating tools.
- **Optimistic prefetch**: for deterministic tool sequences (always calls tool A before tool B), pre-fetch tool A's result before the model has finished generating the call.

### Schema validation

Before dispatching a tool call, the runtime validates the model's emitted arguments against the declared JSON Schema. Validation catches:
- Missing required fields
- Wrong types (model emits a number where a string is expected)
- Enum violations (model emits a value not in the declared enum)
- Format violations (malformed date strings, invalid UUIDs)

On validation failure, the standard approaches are:
1. **Retry with error injection**: append the validation error as a system message and call the model again. Works but costs one additional LLM round-trip (500 ms–3 s).
2. **Schema repair**: attempt to coerce the invalid output (e.g., cast integer to string). Dangerous for semantic fields (a wrong date format might coerce to a different date entirely).
3. **Hard failure**: return an error to the user. Appropriate for production systems where silent failures are worse than explicit errors.

Schema complexity is inversely correlated with model reliability. A schema with 15 optional fields, nested objects, and cross-field constraints produces higher error rates than one with 2 required string fields. The engineering discipline is to keep tool schemas as minimal as the use case allows. The `description` field matters more than schema complexity: a precise, example-rich description reduces the rate of semantic errors (calling the right tool with wrong intent) more than strict typing reduces syntactic errors.

### Tool call trajectories and context growth

In multi-step agent tasks, the conversation grows with each tool call cycle:

```
System prompt:           S tokens
User message:            U tokens
Model response 1:        R1 tokens (contains tool call)
Tool result 1:           T1 tokens
Model response 2:        R2 tokens (may contain another tool call)
Tool result 2:           T2 tokens
...
Model response N:        RN tokens (final answer)

Total context at step N: S + U + sum(Ri) + sum(Ti)
```

Tool results are often verbose. A web search result might return 2,000 tokens of document text. A SQL query result over a wide table might return 5,000 tokens. After 5 tool calls, the context can easily grow to 15,000–30,000 tokens. This has two consequences:

1. **TTFT increases with context**: attention is O(n²) in sequence length (without FlashAttention chunking and KV cache reuse). Each subsequent LLM call is slower than the last as context grows.
2. **Context window exhaustion**: a 128k-token context window sounds large, but an agent that calls a verbose tool 20 times will exhaust it.

Truncation strategies for tool results:
- **Hard truncation**: cut results to a fixed token budget (e.g., first 1,000 tokens). Fast and simple; loses tail content.
- **Structured summarization**: call a cheaper model (or the same model) to summarize the result before injecting. Adds one LLM call latency but compresses aggressively.
- **Semantic chunking + retrieval**: store the full result in a vector store, inject only the top-k relevant chunks. Works well for document search results; overhead is ~50–200 ms for embedding + retrieval.

The token budget for tool results should be planned explicitly at agent design time. A useful heuristic: allocate no more than 20% of the context window to any single tool result, and no more than 60% of the total window to accumulated tool results, leaving room for the growing model response chain.

## Safety Layers

Tool calls are a code execution interface. The model's output reaches real systems. Several layers of safety are required in production:

**Schema-level constraints**: narrow the action space. A `write_file` tool that accepts an arbitrary `path` string is more dangerous than one that accepts an enum of allowed paths. Declare the minimum-necessary parameter set.

**Human-in-the-loop interrupts**: for high-consequence actions (sending email, modifying database records, executing financial transactions), require explicit human approval before dispatch. The interrupt/resume pattern (used in LangGraph, discussed in [../agent-frameworks/](../agent-frameworks/)) is the standard implementation: the agent pauses, presents the pending tool call to the user, and resumes only on approval.

**Sandboxing**: code execution tools (Python REPL, shell exec) must run in isolated environments (containers, VMs, microVMs like Firecracker). Network-accessible tools (web fetch, API calls) should run with egress filtering to prevent SSRF and data exfiltration. File system tools should be chroot'd or use virtual file system layers.

**Rate limiting per tool**: limit calls per minute, per session, per user. This bounds runaway agent loops. A malfunction that causes a loop of `send_email` calls should hit a rate limit at call 5, not call 5,000.

**Principle of least privilege**: each tool should have only the permissions necessary for its function. The tool that reads customer records should not have write permissions. The tool that writes to a sandbox directory should not have network access. This limits blast radius when the model is manipulated (prompt injection) or makes systematic errors.

## Engineering Tradeoffs

| Approach | Pros | Cons | When to use |
|---|---|---|---|
| Parallel tool calls (multi-call per model turn) | Reduces total latency to max(T_i) instead of sum(T_i); fewer LLM round-trips for independent operations | Requires model capability to identify independent calls; runner complexity for concurrent dispatch + result merging | Any agent task with 2+ independent data fetches or actions in one logical step |
| Sequential tool calls (one per model turn) | Simple runner; easier to debug; model sees each result before deciding next call | Latency = sum of all tool times + LLM times; context grows faster (more model response tokens) | Tasks with data dependencies between calls; when debugging agent behavior |
| Strict JSON Schema validation (reject invalid outputs) | Deterministic failures; no silent data corruption; enforces contract | Failed validation costs one extra LLM round-trip; can loop on stubborn model errors | Production systems where correctness > availability; any tool with side effects |
| Lenient validation (coerce or skip validation) | Never blocks on schema error; simpler runner | Silent failures; coercion can corrupt semantic content; hard to debug downstream | Read-only tools with low-stakes outputs; prototyping |
| Streaming tool call delivery | User sees partial response text immediately; lower perceived latency | Cannot dispatch tool until arguments fully generated; more complex runner state machine | Hybrid responses (text + tool call together); chat UX where response text precedes tool call |
| Sandboxed execution (container / microVM) | Isolates side effects; limits blast radius of prompt injection | 50–500 ms cold start per tool call; resource overhead; network egress filtering adds latency | Any tool that executes user-influenced code; file write; shell exec; web fetch |

## Latency Arithmetic in Practice

A realistic multi-step agent trajectory illustrates the compounding costs:

| Step | Operation | Time |
|---|---|---|
| Turn 1 | User message → LLM (TTFT, 2k token prompt) | ~800 ms |
| Turn 1 | LLM generates tool call arguments (50 tokens) | ~150 ms |
| Turn 1 | Schema validation | ~1 ms |
| Turn 1 | Tool execution (DB lookup) | ~80 ms |
| Turn 2 | LLM processes 3k token context (TTFT) | ~1,100 ms |
| Turn 2 | LLM generates 2nd tool call (parallel, 2 calls) | ~200 ms |
| Turn 2 | Tool execution (parallel: web search + DB) | ~1,200 ms (max of 1,200 ms, 90 ms) |
| Turn 3 | LLM processes 6k token context (TTFT) | ~1,800 ms |
| Turn 3 | LLM generates final response (200 tokens) | ~600 ms |
| **Total** | | **~5.9 s** |

Without parallel tool calls in turn 2, the web search and DB lookup would serialize: 1,200 + 90 = 1,290 ms instead of 1,200 ms max — marginal here because web search dominates. But for three equal-latency 300 ms tools, parallelism saves 600 ms out of 900 ms tool time. The TTFT growth across turns (800 ms → 1,100 ms → 1,800 ms) reflects the O(n) KV cache prefill cost as context grows, even with prefix caching ([../prefix-caching/](../prefix-caching/)).

The practical engineering insight: minimize round-trips (use parallel calls), minimize context growth (truncate verbose results aggressively), and keep tool execution fast (async IO, connection pooling, result caching for idempotent reads).

## Protocol Comparison

All three major APIs converge on the same underlying model. Surface differences are administrative:

| API | Tool definition field | Call output format | Multi-call support | Streaming |
|---|---|---|---|---|
| OpenAI | `tools[].function` | `tool_calls[]` array in message | Yes (parallel_tool_calls=true) | index+delta pattern |
| Anthropic | `tools[]` | `tool_use` content blocks | Yes (multiple blocks in one response) | input_json_delta events |
| Gemini | `tools[].functionDeclarations[]` | `functionCall` parts | Yes (multiple parts) | Streaming function call parts |

MCP adds a transport and registry layer: tools are declared as MCP resources, the MCP server exposes them over a protocol (stdio, SSE, or HTTP), and the LLM client discovers them dynamically. The JSON Schema definition of each tool is unchanged; MCP is the plumbing, not the schema.

## References

- [1] OpenAI. _Function Calling._ platform.openai.com/docs/guides/function-calling, 2023.
- [2] Anthropic. _Tool Use._ docs.anthropic.com/en/docs/build-with-claude/tool-use, 2024.
- [3] Google. _Function Calling with the Gemini API._ ai.google.dev/gemini-api/docs/function-calling, 2024.
- [4] Anthropic. _Model Context Protocol._ modelcontextprotocol.io, 2024. (MCP transport layer)
- [5] Yao et al. _ReAct: Synergizing Reasoning and Acting in Language Models._ ICLR 2023. (Foundational pattern for tool-call loops)
- [6] Anthropic. _Building Effective Agents._ anthropic.com/research/building-effective-agents, 2024.
