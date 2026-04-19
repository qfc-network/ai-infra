# Agent Framework Landscape

_A system design entry covering the four major agent frameworks — LangGraph, AutoGen, OpenAI Swarm, and DSPy — with focus on state management, context window pressure, latency characteristics, and observability. The underlying tool-call primitive is covered in [../tool-use-infra/](../tool-use-infra/); MCP as the tool/resource registry layer is in [../../anthropic/mcp/](../../anthropic/mcp/); the workflow taxonomy these frameworks implement is in [../../anthropic/building-effective-agents/](../../anthropic/building-effective-agents/); SGLang's DSL for multi-call LLM programs is a lower-level alternative described in [../../foundational/sglang/](../../foundational/sglang/)._

## TL;DR

A raw LLM API gives you one-shot completions. Real agent tasks require loop control: retry on tool failure, branch on intermediate results, maintain state across multiple model calls, and coordinate between specialized sub-agents. Agent frameworks provide this scaffolding. The choice of framework is not primarily about feature surface; it is about which failure modes you are willing to manage. LangGraph makes state explicit and graph structure debuggable, at the cost of verbose setup. AutoGen models agents as conversational actors, which scales to complex coordination but produces non-deterministic flows that are hard to trace. OpenAI Swarm is deliberately minimal — useful for understanding the handoff pattern, not for production multi-agent systems. DSPy is categorically different: it is a prompt compiler, not a runtime, and belongs in the evaluation pipeline rather than the serving loop. The infrastructure implications — state persistence size, context window multiplication, p99 latency amplification, and observability overhead — are the practical differentiators that matter at production scale.

## Context & Motivation

### What the raw loop looks like without a framework

Without a framework, the developer writes the agent loop manually:

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

This works for simple linear chains. It breaks when you need: parallel subgraph execution, conditional branching based on intermediate results, persistent state across sessions (the agent was interrupted, now resuming), human-in-the-loop approval gates, coordination between multiple specialized agents, error retry with backoff, and context window management as the message list grows. Frameworks handle these concerns; the tradeoff is the learning curve and operational complexity they introduce.

## LangGraph

### Design model

LangGraph (part of the LangChain ecosystem, released 2024) models an agent workflow as a **directed graph** where nodes are Python functions and edges are transitions. Each node reads from and writes to a **shared state dict** (a typed TypedDict in Python). The graph has a defined start node and one or more terminal nodes; execution follows edges until a terminal is reached or the graph raises.

```python
from langgraph.graph import StateGraph, END
from typing import TypedDict

class AgentState(TypedDict):
    messages: list
    tool_results: dict
    iteration_count: int

def call_llm(state: AgentState) -> AgentState:
    response = llm.invoke(state["messages"])
    return {"messages": state["messages"] + [response]}

def route(state: AgentState) -> str:
    if state["messages"][-1].tool_calls:
        return "execute_tools"
    return END

graph = StateGraph(AgentState)
graph.add_node("llm", call_llm)
graph.add_node("execute_tools", tool_executor)
graph.add_edge("execute_tools", "llm")
graph.add_conditional_edges("llm", route)
graph.set_entry_point("llm")
app = graph.compile()
```

Conditional edges take a router function that inspects the current state and returns the name of the next node. This is where branching logic lives — the router decides whether to continue the tool loop, hand off to a specialized subgraph, or terminate.

### State persistence and checkpointing

LangGraph's defining production feature is its **checkpointer** abstraction. At each node transition, the full state dict is serialized and written to a persistence backend. Supported backends: in-memory (default, for testing), SQLite (single-process), Postgres (distributed), and Redis.

The checkpoint serves two purposes: resumability (an interrupted agent can resume from the last checkpoint rather than restarting) and human-in-the-loop (the graph can be configured to pause before a specified node, wait for external input, then resume — the interrupt/resume pattern).

The memory cost of checkpoints is proportional to the state dict size. In agentic workflows where tool results are stored in state (the common pattern for passing results between nodes), state grows with each iteration. A workflow that accumulates 10 tool results of 2,000 tokens each has a 20,000-token state that must be serialized and written to the checkpointer on every node transition. At 100 concurrent agent sessions with 10 nodes each, the Postgres backend must absorb 1,000 writes per session execution, each up to several MB. This is not a problem at low scale; it becomes a throughput bottleneck at high concurrency.

**Mitigation strategies**: store tool results as external references (store the actual data in object storage or a database, keep only an ID in state); define explicit state compaction nodes that summarize prior turns before they are checkpointed.

### Multi-agent patterns in LangGraph

LangGraph supports subgraph composition: a node can itself be a compiled LangGraph, allowing hierarchical agent architectures. A supervisor graph coordinates specialist subgraphs (a research agent, a writing agent, a code review agent), routing tasks based on current state. Each subgraph maintains its own state dict, with results surfaced to the parent graph.

The latency implication is significant: each subgraph is a separate sequence of LLM calls. A supervisor calling three specialist agents sequentially adds 3× the per-agent latency to the total pipeline. LangGraph supports parallel node execution (using `Send` to fan out work to multiple nodes simultaneously) to reduce this.

### Observability

LangGraph integrates natively with **LangSmith** — Anthropic's LangChain's hosted tracing platform. Every graph execution produces a trace tree: each node invocation is a span with input state, output state, and timing. The trace makes it possible to see exactly which state triggered which LLM call, what tools were invoked, and where latency was spent. This is the primary debugging advantage over the manual loop: the trace tree makes the execution history inspectable without adding logging boilerplate.

## AutoGen

### Design model

AutoGen (Microsoft Research, 2023) takes a different approach: agents are **conversational actors** that communicate by sending and receiving messages. The base class `ConversableAgent` wraps an LLM and defines an auto-reply mechanism: when it receives a message, it calls the LLM with its conversation history, optionally executes tools, and sends back the result.

```python
from autogen import ConversableAgent, GroupChat, GroupChatManager

planner = ConversableAgent("planner", llm_config={"model": "gpt-4o"}, 
                            system_message="You decompose tasks into steps.")
coder = ConversableAgent("coder", llm_config={"model": "gpt-4o"},
                          system_message="You write Python code to complete tasks.")
executor = ConversableAgent("executor", human_input_mode="NEVER",
                             code_execution_config={"work_dir": "/tmp/sandbox"})

groupchat = GroupChat(agents=[planner, coder, executor], messages=[],
                       speaker_selection_method="auto")
manager = GroupChatManager(groupchat=groupchat, llm_config={"model": "gpt-4o"})

planner.initiate_chat(manager, message="Build a CSV parser that handles malformed rows.")
```

`GroupChat` coordinates multiple agents. The `speaker_selection_method` controls who speaks next: `"round_robin"` cycles through agents in order; `"auto"` uses an LLM-based selector to choose the most relevant agent for each turn; `"manual"` requires human selection.

### Conversation model and non-determinism

AutoGen's conversation model produces emergent behavior that is difficult to reason about statically. The LLM-based speaker selector introduces non-determinism: the same task can produce different agent conversation patterns on different runs. This is a feature (more flexible than a fixed graph) and a bug (non-deterministic flows are hard to debug, hard to test, and hard to bound for latency).

The context window management challenge is especially pronounced in AutoGen: each agent maintains its own conversation history, but in a GroupChat the full conversation is visible to all agents. At turn N, the GroupChat manager feeds all N prior messages to the speaker selector LLM. For a complex task that takes 20 turns among 4 agents, the selector LLM at turn 20 has a 20-message context (each message potentially containing code or tool output). This context blowup accelerates with the number of agents and turns.

AutoGen's mitigation: the `max_consecutive_auto_reply` parameter limits how many turns an agent will auto-reply before requiring human input, bounding runaway conversations. Per-agent conversation history can be trimmed with `compress_history` (summarization) or `clear_history` (full reset) at configurable intervals.

### Code execution loop

AutoGen's designed strength is the code-writing + execution loop: the `coder` agent writes Python, the `executor` agent runs it in a sandbox and returns stdout/stderr, the `coder` sees the output and iterates. This pattern maps directly to the auto-reply conversation model. The sandbox is a Docker container by default (`docker_code_execution_config`), giving file system and process isolation without manual container management.

The latency of this loop is dominated by container startup time (0.5–2 s per execution for Docker cold starts), Python execution time, and the LLM call per iteration. For a task that requires 5 code iterations, expect 5 × (LLM call + container exec) ≈ 5 × (2 s + 1 s) = 15 s before reaching a correct result. Persistent container execution (reusing a warm container across iterations) reduces this to 5 × (2 s + 0.1 s) = 10.5 s — a meaningful improvement.

### Observability

AutoGen provides **AutoGen Studio** (a web UI for constructing and running agent workflows) and structured logging of message exchanges. The challenge is that the conversation log is a flat sequence of messages; reconstructing which agent decision caused a downstream failure requires careful log analysis. AutoGen 0.4+ includes OpenTelemetry trace export, which maps agent conversations to standard spans and enables integration with Jaeger, Honeycomb, or any OTel-compatible backend.

## OpenAI Swarm

### Design model

Swarm (OpenAI, experimental, 2024) is intentionally minimal. Agents are defined by a system prompt and a list of available functions (tools). An agent can return a handoff to another agent by returning an `Agent` object from a function — the runner switches context to that agent and continues. There is no persistent state, no graph, no conversation manager:

```python
from swarm import Swarm, Agent

triage_agent = Agent(
    name="Triage",
    instructions="Determine if this is a billing or technical issue. Hand off accordingly.",
    functions=[transfer_to_billing, transfer_to_technical]
)

billing_agent = Agent(
    name="Billing",
    instructions="You handle billing questions. Use lookup_invoice and process_refund tools.",
    functions=[lookup_invoice, process_refund]
)

def transfer_to_billing():
    return billing_agent  # return value triggers handoff

client = Swarm()
response = client.run(agent=triage_agent, messages=[{"role": "user", "content": "I was double charged."}])
```

The Swarm runner calls the current agent's LLM, executes any tool calls (including handoffs), and loops. State is not persisted between `client.run()` calls. The entire conversation history is passed in-memory through the run.

### What Swarm is for

Swarm is best understood as a reference implementation of the handoff pattern, not a production framework. It demonstrates that agent handoffs require no special machinery beyond the standard tool-call protocol: an agent "hands off" by returning a new agent object as a tool result. The orchestrator sees this, swaps in the new agent's system prompt, and continues.

The limitations are deliberate: no persistence, no branching, no parallel execution, no checkpointing. For any agent system that must survive process restarts, support human-in-the-loop, or run sub-agents in parallel, Swarm is the wrong tool. Its value is pedagogical: the Swarm source code is ~300 lines, making it a readable demonstration of the minimal loop.

### When to use it

Swarm is appropriate for linear pipelines where each agent has a clear specialty and the handoff sequence is deterministic (always triage → billing or technical → never loops back). It is also appropriate as a starting point for teams learning the agent pattern before introducing framework complexity.

## DSPy

### Design model

DSPy (Stanford NLP, 2023–2024) is categorically different from the preceding three. It is a **prompt compiler and optimizer**, not an agent runtime. DSPy programs define modules — `Predict`, `ChainOfThought`, `ReAct`, `MultiChainComparison` — that specify the *structure* of an LLM call without writing the prompt text. An optimizer (`BootstrapFewShot`, `MIPRO`, `BayesianSignatureOptimizer`) then searches the prompt space (demonstrations, instructions) to maximize a metric on a development set.

```python
import dspy

class RAGPipeline(dspy.Module):
    def __init__(self):
        self.retrieve = dspy.Retrieve(k=5)
        self.generate = dspy.ChainOfThought("context, question -> answer")

    def forward(self, question):
        context = self.retrieve(question).passages
        return self.generate(context=context, question=question)

# Optimize against a metric
optimizer = dspy.BootstrapFewShot(metric=answer_exact_match)
optimized_rag = optimizer.compile(RAGPipeline(), trainset=train_examples)
```

The optimizer's `compile` step runs the pipeline on training examples, collects successful demonstrations, and injects the best ones as few-shot examples in the final prompt. `MIPRO` goes further: it uses a meta-LLM to propose instruction variations and evaluates them against the metric, effectively performing gradient-free prompt gradient descent.

### Separation of concerns

DSPy's core insight is that prompt text is a hyperparameter, not a hand-crafted artifact. Treating it as such enables systematic optimization rather than manual tuning. The separation between program logic (`RAGPipeline.forward`) and prompt content (optimized by the compiler) is analogous to the separation between algorithm and implementation in traditional software.

The infrastructure consequence: DSPy programs are not deployed directly. The compiler runs offline (the `compile` step may make hundreds or thousands of LLM calls on the training set), and the output is an optimized program object whose prompts are fixed at compile time. The serving infrastructure sees a standard LLM call with a fixed prompt; DSPy is invisible at inference time.

### Where DSPy belongs in the stack

DSPy is not a replacement for LangGraph, AutoGen, or Swarm. It operates in the evaluation-and-compilation pipeline:

```
DSPy (offline optimization) → optimized prompt artifacts
    ↓
LangGraph / AutoGen / Swarm (online serving) → uses the optimized prompts
```

Teams building agentic systems benefit from using DSPy to optimize the prompts inside each agent node (the system prompt, the few-shot demonstrations) while using LangGraph or AutoGen to manage the agent loop at runtime.

### Computational cost of compilation

The `BootstrapFewShot` optimizer makes approximately `num_bootstraps × trainset_size` LLM calls during compilation. For 10 bootstrap iterations and 100 training examples, that is 1,000 LLM calls — at GPT-4o pricing (~$5/M tokens), a 500-token prompt per call costs $2.50 per compilation run. `MIPRO` with instruction search can require 10–50× more calls. This is affordable for periodic re-optimization (weekly prompt refreshes tied to new evaluation data) but prohibitive for per-request optimization at serving time.

## Infrastructure Implications Across Frameworks

### State persistence sizing

LangGraph checkpointers serialize state on every node transition. A 3-hour agent session with a graph that transitions 200 times, each with a 5,000-token state (~20 KB), writes 4 MB of checkpoint data per session. At 1,000 concurrent sessions, that is 4 GB/hr of checkpoint writes. Postgres can handle this; the bottleneck is usually connection pool exhaustion under high concurrency (use pgBouncer or an equivalent proxy).

AutoGen has no built-in persistence — the conversation history lives in memory. Session recovery requires re-running from scratch or implementing a custom serialization layer on top of the `messages` list.

### Context window management under multi-agent load

In multi-agent systems, context pressure multiplies. A supervisor calling 3 specialist agents sequentially, each consuming 5,000 tokens of context per turn, and the supervisor feeding all results back into its own context, can saturate a 128k context window in fewer than 10 supervisor turns. The mitigation is **agent boundaries as context boundaries**: each agent maintains its own context; information crossing the boundary is summarized rather than passed verbatim. LangGraph enforces this naturally through state compaction nodes; AutoGen requires explicit `compress_history` calls.

### Latency amplification in deep chains

Each LLM call in an agent chain adds 1–5 s of TTFT. A 5-node LangGraph pipeline with no parallelism has a p50 latency of 5 × 2 s = 10 s and a p99 latency that scales with tail LLM call latencies — easily 25–40 s. Parallelism (LangGraph's `Send`, AutoGen GroupChat with async dispatch) collapses independent branches. The practical limit is the depth of the critical path: any sequential dependency chain is a latency floor.

For latency-sensitive workflows, the architecture principle is to minimize sequential depth. A flat 5-way parallel fan-out with a reduce step has a latency profile of max(5 × parallel LLM calls) + 1 reduce call, not sum(5 sequential calls). The framework must support parallel execution; LangGraph does via `Send`; AutoGen GroupChat with `async_max_wait` does partially; Swarm does not.

### Observability overhead

Tracing multi-agent systems is more expensive than tracing single-model calls. A LangSmith trace for a 10-node LangGraph execution with 50 total LLM calls generates ~100 KB of trace data. At 1,000 executions/hour, that is 100 MB/hr of trace data — manageable but non-trivial. LangSmith's sampling-based tracing (trace a fraction of executions) reduces this at the cost of reduced debugging visibility for infrequent failures.

OpenTelemetry tracing (AutoGen 0.4+, and available as middleware in LangGraph) integrates with existing observability infrastructure. The overhead of span creation and export adds ~1–5 ms per LLM call — negligible compared to LLM latency but visible in tight loops.

## Engineering Tradeoffs

| Framework | State management | Multi-agent support | Debuggability | Learning curve | Best use case |
|---|---|---|---|---|---|
| LangGraph | Explicit typed state dict; Postgres/Redis/SQLite checkpointers; resumable | First-class subgraph composition; supervisor patterns; parallel `Send` | Excellent: graph structure + LangSmith trace tree; conditional edge logic is inspectable | High: graph + state + checkpointer + conditional edges all have separate APIs | Complex stateful workflows; human-in-the-loop gates; long-running agents that must survive restarts |
| AutoGen | No built-in persistence; conversation history in memory only | GroupChat with round-robin or LLM selector; nested agents supported | Moderate: message log is readable but non-deterministic conversation flows are hard to reason about | Medium: conversational model is intuitive; GroupChat tuning requires trial and error | Code-writing + execution loops; tasks where flexible agent coordination outweighs debuggability |
| Swarm | None: stateless per run | Simple handoffs via return values; no parallel dispatch; no subgraphs | High: 300-line codebase; linear control flow; easy to add print statements | Very low: minimal API surface | Linear pipelines with specialist agents; learning/prototyping the handoff pattern |
| DSPy | Not applicable (compile-time, not runtime) | Not applicable | Excellent at compile time (metric-driven); black-box at inference time (prompt text is optimized artifact) | High: new programming model (modules, signatures, optimizers, metrics) | Systematic prompt optimization; teams that need reproducible prompt versioning; evaluation-driven development |

## Claude Agent SDK / Anthropic's Approach

The Anthropic Claude Agent SDK exposes the same core loop primitives (tool calls, result injection, multi-turn context management) with two distinguishing infrastructure features. First, **prompt caching**: the SDK automatically caches system prompts and tool definitions across calls in the same session, reducing the TTFT cost of repeated LLM calls in an agent loop from ~1.5–3 s (full prefill) to ~0.2–0.5 s (cache hit + short new tokens). For a 10-step agent loop with a 2,000-token system prompt, caching alone saves roughly 8 × 1.3 s = 10.4 s of prefill time. Second, the SDK integrates with MCP for tool registration and discovery, meaning tool schemas don't need to be hardcoded in the agent definition — they are discovered at runtime from MCP servers.

The SDK does not impose a graph model or a conversation actor model. It is closer to Swarm in philosophy (minimal scaffolding) but with production features (caching, streaming, batching) that Swarm lacks. Teams building on Claude who need stateful long-running agents layer LangGraph-style state management on top of the SDK's core loop.

## References

- [1] LangChain. _LangGraph: Building Stateful Multi-Actor Applications with LLMs._ blog.langchain.dev, 2024.
- [2] Wu et al. _AutoGen: Enabling Next-Gen LLM Applications via Multi-Agent Conversation._ arXiv:2308.08155, 2023.
- [3] OpenAI. _Swarm: Educational Framework for Multi-Agent Orchestration._ github.com/openai/swarm, 2024.
- [4] Khattab et al. _DSPy: Compiling Declarative Language Model Calls into Self-Improving Pipelines._ arXiv:2310.03714, 2023.
- [5] Khattab et al. _Optimizing Instructions and Demonstrations for Multi-Stage Language Model Programs._ arXiv:2406.11695, 2024. (MIPRO)
- [6] Anthropic. _Building Effective Agents._ anthropic.com/research/building-effective-agents, 2024.
- [7] Anthropic. _Claude Agent SDK._ docs.anthropic.com, 2025.
