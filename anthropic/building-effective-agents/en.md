# Building Effective Agents

- **Authors / Org**: Anthropic Engineering (Schluntz & Zhang)
- **Published**: 2024-12
- **Links**: [blog post](https://www.anthropic.com/engineering/building-effective-agents) · [cookbook examples](https://github.com/anthropics/anthropic-cookbook/tree/main/patterns/agents)

## TL;DR

Not a paper; an **engineering blog** from Anthropic that proposes a vocabulary for building LLM-powered applications. Separates **workflows** (LLM calls composed via predefined code paths) from **agents** (LLM that dynamically chooses its own tools and trajectory), argues **workflows are usually the right answer**, and documents five concrete workflow patterns plus when true agency is warranted. Its lasting impact is definitional: after this post, the industry converged on the **prompt chaining / routing / parallelization / orchestrator-workers / evaluator-optimizer** vocabulary for workflows, and on "agent = loop + tool use + environment feedback" for agents. Read this instead of agent-framework marketing to understand what's actually happening behind most production "agentic" systems.

## Context & Motivation

By late 2024, "agent" had become the most overloaded word in LLM engineering. Anthropic's observation from their customer work: the vast majority of successful production deployments are **structured LLM workflows**, not autonomous agents — but marketing pressure pushed everyone to call everything an "agent," obscuring what actually worked. And most complex agent frameworks (LangChain, LangGraph, AutoGen at the time) added abstraction layers that often hurt more than helped.

The post's stance:

> Consistently, the most successful implementations weren't using complex frameworks or specialized libraries. Instead, they were building with simple, composable patterns.

The goal: **cut through the hype, name the patterns that actually work, and advise when to reach for each**.

## Core Distinctions

### Agentic systems = workflows + agents

- **Workflow**: LLM + tools orchestrated by **predefined code paths**. You write the control flow; the LLM fills in decisions and content.
- **Agent**: LLM **dynamically directs its own process and tool use**, maintaining control of how it accomplishes tasks. Loop + self-chosen actions.

Workflows are predictable, cheap, easy to debug. Agents are flexible, expensive, hard to debug. The post's strong default recommendation: **start with a workflow; escalate to an agent only when task scope genuinely requires it**.

### Starting point: the augmented LLM

Before either workflows or agents, the building block is an **LLM augmented with retrieval, tools, and memory**. Most applications don't need more. If a single augmented LLM call solves the problem, stop there.

## The Five Workflow Patterns

### 1. Prompt chaining

Decompose a task into a fixed sequence of LLM calls, each consuming the previous output. Between steps, **programmatic gates** can validate and short-circuit. Good when the task has a clean linear decomposition (draft → critique → rewrite; extract → translate → format).

```
input → LLM_1 → [gate check] → LLM_2 → [gate] → LLM_3 → output
```

Gain: each step is simpler, each call is cheaper, debugging is localized.
Cost: one more call per step; less flexibility if the task needs to skip.

### 2. Routing

Classify input and route to a specialized prompt / model / workflow. A small fast model (or rule) decides; larger models handle only what matches their specialty.

```
input → router → {workflow_A | workflow_B | workflow_C}
```

Gain: cost (use small models for most inputs); quality (specialized handling).
Cost: classification must be reliable; miss-routing is expensive.

### 3. Parallelization (sectioning and voting)

Run subtasks concurrently and aggregate. Two variants:

- **Sectioning**: split the task into independent pieces, LLM handles each, results merged.
- **Voting**: run the same task multiple times with different prompts or temperatures, aggregate (majority, filter, best-of-N) for better reliability.

Good when subtasks are independent, or when a single call's reliability isn't enough (safety-critical, high-stakes).

### 4. Orchestrator-workers

A central LLM **dynamically decomposes** a task into subtasks, delegates each to a worker LLM, and synthesizes results. The subtask set is **not predefined** — the orchestrator decides it per input.

```
input → orchestrator (plans) → worker_1 ∥ worker_2 ∥ ... → orchestrator (synthesizes) → output
```

Similar to parallelization but with dynamic decomposition. More powerful; more expensive; failure modes (bad decomposition, lost context) are new.

### 5. Evaluator-optimizer

One LLM generates output; another LLM critiques and provides feedback; loop until the evaluator passes (or max iterations). Useful when:

- You can define clear evaluation criteria.
- First-pass outputs usually need refinement.
- Iteration actually improves the result (not every task benefits — some just degrade with "improvement").

```
generator → output → evaluator → [feedback] → generator → ... → final
```

## Agents: when the workflows aren't enough

An agent is an LLM in a loop where it chooses its own tool use and continuation. A well-designed agent has:

- A clear **task**.
- A well-defined set of **tools** with clear docs.
- **Environment feedback** to ground its decisions (tool outputs, code execution results, search hits).
- Some form of **stopping condition** (task complete, budget exceeded, human handoff).

When to use agents (per the post):

- **Open-ended tasks** where the path can't be predefined.
- **Interactive, long-horizon** problems (coding across a codebase, multi-step research, operating a computer).
- When the value of flexibility outweighs the cost of unpredictability.

When NOT to use them:

- Any task a workflow solves adequately — agents cost more and fail in more interesting ways.
- Tasks with low tolerance for wrong actions (an agent will sometimes do the wrong thing).

## Engineering Tradeoffs

| Pattern | When it's the right choice | What it costs |
|---|---|---|
| Augmented LLM (single call) | Simple, one-shot tasks | Limited quality ceiling |
| Prompt chaining | Linear decomposition with natural gates | More latency + cost per output |
| Routing | Mix of input types needing different handling | Requires classification accuracy |
| Parallelization | Independent subtasks or reliability via voting | Compute multiplier |
| Orchestrator-workers | Dynamic decomposition needed | Complex failure modes |
| Evaluator-optimizer | Iterative refinement helps | Loop bounds, iteration cost |
| Agents | Open-ended, long-horizon, flexible | Expensive; unpredictable; hard to debug |

## Commentary

This post is the clearest public articulation of **what "agent frameworks" should have been**. Every complex framework that preceded it (chains-of-chains abstractions, auto-generated graphs, verbose orchestration DSLs) was trying to solve a problem that, for most tasks, didn't need solving — most production "agent" work is one of the five workflows, and those are **just normal code with LLM calls**. Anthropic's engineering team called this out and the industry quietly aligned.

The definitions also clarify what's different about a *real* agent: the model, not the framework, chooses the trajectory. This means **your investment shifts from orchestration code to tool quality, environment feedback, and observability** — because that's what the model is working with. Anthropic's own product (Claude Code) is an agent in this sense; every design choice is about giving the model useful tools, clear feedback, and the ability to recover from mistakes, not about wrapping it in control-flow scaffolding.

For infra people, the relevant lesson is **instrumentation matters more than framework choice**. Agents without good tracing, replayable runs, and clear SLOs will be uncontrollable in production regardless of which agent library you picked. The cookbook examples are minimal on purpose — if the patterns work in a few hundred lines of Python, everything extra you import is carrying its own bugs.

The broader arc: as of 2026, **most production "agentic" systems are workflows by this definition, not agents**. Real agents — long-horizon, flexible, self-directing — work but are still expensive and finicky. The Anthropic post's framing has aged well: workflows cover 80% of value; agents cover the new frontier but haven't displaced workflows for existing use cases.

## References

- [1] Anthropic Engineering. _Building Effective Agents._ 2024-12. https://www.anthropic.com/engineering/building-effective-agents
- [2] Anthropic Cookbook agent patterns: https://github.com/anthropics/anthropic-cookbook/tree/main/patterns/agents
- [3] Schluntz & Zhang on the Latent Space podcast (2024) — deeper discussion of the same ideas.
- [4] Related: [How we built Claude Code](https://www.anthropic.com/engineering) — applied example.
