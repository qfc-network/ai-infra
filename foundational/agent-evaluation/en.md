# Agent Evaluation Infrastructure

- **Key benchmarks**: SWE-bench (Princeton NLP, 2023), TAU-bench (Sierra AI, 2024), WebArena (CMU, 2023)
- **Links**: [SWE-bench](https://www.swebench.com) · [SWE-bench Verified](https://openai.com/index/introducing-swe-bench-verified/) · [WebArena](https://webarena.dev) · [TAU-bench (arXiv:2406.12045)](https://arxiv.org/abs/2406.12045)

## TL;DR

Evaluating agents is structurally harder than evaluating language models. LM evaluation measures a single response against a known-correct output; agent evaluation measures a multi-step trajectory against an environment that changes with every action. Correctness of the final outcome and correctness of the trajectory are not the same thing — an agent can reach the right answer by accident, or fail despite taking individually reasonable steps. The compounding nature of errors in agentic loops means that a 10% per-step error rate becomes a 65% failure rate over 10 steps. Three benchmarks have emerged as the canonical evaluation surfaces: SWE-bench for code agents, TAU-bench for tool-using dialogue agents, and WebArena for browser-based task completion. Each has distinct infrastructure requirements and distinct failure modes, and none fully covers the production distribution of real-world agent tasks.

## Why Agent Evaluation Is Harder

Standard LM benchmarks (MMLU, HumanEval, MATH) present a static (input, expected output) pair. The model's job is to produce a response, and evaluation is a function of that response alone. This collapses along several dimensions when applied to agents:

**Trajectory dependency**: An agent's action at step 5 depends on the state produced by actions 1–4. The space of possible trajectories grows exponentially with the number of steps. There is no single "correct" trajectory — only correct (and incorrect) final states.

**Compounding error amplification**: If each step has probability `p` of being correct, the probability of completing an N-step task correctly is at most `p^N` under independence, and less under error correlation. For `p = 0.9` and N = 10, success probability is at most 35%. This means that evaluation results are sensitive to individual error modes in ways that single-response benchmarks are not.

**Environment non-determinism**: Real environments change. Web pages update their layout. API endpoints return different data. Network timeouts occur. A task that passes on evaluation day may fail on deployment day for environmental reasons that have nothing to do with the agent's quality. Deterministic environment control is a significant piece of eval infrastructure.

**Partial credit**: For a 20-step task where the agent completes 18 steps correctly before failing, binary pass/fail wastes signal. Graded trajectory metrics are necessary but introduce their own calibration problems.

**Judge reliability**: When the success criterion is not mechanically verifiable (e.g., "write a professional email to this customer"), a second LLM must act as judge. LLM judges have systematic biases — they tend to prefer responses from larger models, penalize unusual formats regardless of quality, and are sensitive to the framing of the evaluation rubric.

## SWE-bench

### What It Measures

SWE-bench (Software Engineering Benchmark) contains **2,294 GitHub issues** from 12 real Python repositories (Django, Flask, sympy, scikit-learn, pytest, astropy, and others). Each instance is a (repository state at a specific commit, issue text, gold patch, test suite) tuple. The agent's task is to produce a code patch that, when applied to the repository at that commit, makes the failing tests pass without breaking previously passing tests.

The "Verified" subset (SWE-bench Verified, 500 instances) was manually reviewed by human software engineers to confirm that the issue description is unambiguous and the gold patch is the only reasonable fix. This subset is the canonical leaderboard because the full 2,294-instance set contains noise: issues where the gold patch is one of several valid solutions, or where the issue description leaves the fix underspecified.

Typical top scores as of early 2026: **48–55% on SWE-bench Verified** for the best publicly reported systems (Claude 3.5 Sonnet + scaffolding, GPT-4o + scaffolding, open-source agents). The upper bound of 100% is not theoretical — some issues genuinely have ambiguous fixes that even expert humans disagree on.

### Infrastructure Requirements

Running SWE-bench requires:

1. **Repository sandbox per instance**: a Docker container with the repository checked out at the specified commit, with all dependencies installed. Repository setup can take 2–10 minutes per unique repo; this is usually pre-baked into a base image. At 2,294 instances across 12 repos, a full evaluation runs hundreds of unique container instances.

2. **Test execution harness**: after the agent produces a patch, apply it (`git apply`), run the test suite (`pytest` with per-repo configuration), and parse the result. The pass/fail verdict is deterministic given the patch. Test execution takes 10 seconds to 10 minutes depending on the repository.

3. **Agent execution sandbox**: the agent itself also runs in a container, with tools to read files (`read_file`), write files (`write_file`), run bash commands (`bash`), and search the codebase (`search`). The agent must not have internet access during evaluation (to prevent retrieval of the gold patch). Context window management is critical — large codebases can have thousands of relevant files; the agent must navigate them efficiently.

4. **Cost budget**: a single full SWE-bench Verified run with a frontier model costs **$200–600 in API calls** depending on the model, scaffolding architecture, and how aggressively context is managed. Teams evaluating weekly at full scale need a dedicated budget line.

The SWE-bench evaluation harness is open source and maintained at the official repository. A minimal self-hosted setup requires ~50 GB of storage for pre-built Docker images and a multi-core machine to run evaluations in parallel (typically 16–32 concurrent containers to complete a full run in under 2 hours).

### What SWE-bench Does Not Measure

SWE-bench measures only code patching. It does not cover: writing new features from a blank-file specification, refactoring without a bug report, multi-repository changes, writing tests, or any software engineering task that does not resolve an existing issue against an existing test suite. It also measures a snapshot of the Python ecosystem; agents may learn to game specific repository idioms.

## TAU-bench

### What It Measures

TAU-bench (Tool-Agent-User benchmark) simulates **customer service scenarios**. Each episode involves three components:

1. **User simulator**: an LLM playing a customer with a specific request and a private "oracle" state (what the customer actually wants, which is not told to the agent directly).
2. **Policy document**: a plain-text document describing the rules the agent must follow (e.g., "refunds are allowed within 30 days of purchase"; "flight changes require a $150 fee unless the customer has Platinum status").
3. **Tool set**: structured API calls the agent can make to look up orders, modify reservations, issue refunds, and so on. Tools have realistic latencies and can return errors.

The agent's task is to resolve the user's request correctly according to policy, using only the available tools, across a multi-turn dialogue. TAU-bench has two domains: **retail** (order management, returns, product questions) and **airline** (flight booking, changes, cancellations, upgrades).

Success is measured by two criteria: whether the agent's final action correctly resolved the user's request (task completion) and whether all actions taken during the episode were consistent with the policy document (policy adherence). A high-completing, low-adherence agent is gaming the benchmark; a high-adherence, low-completing agent is too conservative. Both dimensions are required.

Typical scores: top agents achieve **70–80% task completion on retail**, somewhat lower on airline (more complex policies). Policy adherence rates are typically 85–95% for frontier models but drop sharply on edge cases and rare policy provisions.

### Infrastructure Requirements

TAU-bench requires a live LLM for the user simulator throughout the evaluation. This doubles the API cost relative to a single-model evaluation. The user simulator must be a model capable of realistic multi-turn dialogue and consistent maintenance of the oracle state; GPT-4 class is standard. The policy document and tool set are deterministic; the user simulator introduces stochasticity. Reproducibility requires either fixing the user simulator's temperature to zero or running multiple seeds and reporting mean ± variance.

## WebArena

### What It Measures

WebArena deploys **six real web applications locally** in Docker containers: an e-commerce site (based on Magento), a forum (Reddit clone), a code repository (GitLab), a collaborative wiki (Confluence clone), a map application, and an admin dashboard. Tasks are natural-language instructions that require multi-step browser interaction to complete: "find the cheapest red dress under $50 and add it to the cart", "open a pull request to fix the typo in the README of the linux-kernel repo", "find all posts by user 'alice' that mention Python".

Success is measured by functional state checks: did the cart contain the correct item? Was the pull request opened with a diff that fixes the typo? The checks are programmatic and deterministic given a fixed application state.

The evaluation contains **812 tasks** across the six applications, with varying difficulty. Typical top-agent scores: **30–45%** for frontier models with browser-automation scaffolding. The gap between SWE-bench scores (~50%) and WebArena scores (~35%) for the same models reflects the additional difficulty of visual UI navigation versus structured code editing.

### Infrastructure Requirements

WebArena requires running six containerized web applications and a browser automation environment simultaneously. Total memory footprint: approximately 8–16 GB RAM per evaluation machine. The application state is reset between episodes via database snapshots; reset time is 5–30 seconds per episode depending on the application. Full evaluation with 812 tasks takes 4–8 hours on a single machine with one concurrent episode; parallel execution across multiple machines reduces this proportionally.

Browser automation for WebArena uses Playwright or Selenium; the agent receives observations as the page's accessibility tree (HTML elements + ARIA attributes) or as a screenshot, depending on the scaffolding. Accessibility-tree agents are faster (no vision encoding cost) but miss visual information; screenshot agents are slower but more robust to pages that have poor accessibility markup.

## Evaluation Infrastructure Requirements Summary

Across all three benchmarks, shared infrastructure requirements are:

**Sandboxed execution per episode**: each episode must start from a clean, deterministic state. Docker containers with snapshotted databases or git repositories are standard. State leakage between episodes (e.g., a previous agent's actions persisting into the next episode) invalidates the benchmark.

**Trajectory logging**: all tool calls, LLM inputs and outputs, intermediate states, and timing data must be logged per episode. This is the primary debugging artifact. Without trajectory logs, a 40% success rate is an opaque number; with trajectory logs, it decomposes into "15% failed because the agent couldn't find the relevant file", "10% failed because the agent's patch had a syntax error", "15% failed for other reasons" — each with a clear remediation path.

**Automatic success detection**: the success criterion must be evaluated programmatically. Human evaluation at scale (2,294 SWE-bench instances × multiple agent runs per week) is not practical. For SWE-bench, test suite pass/fail is the detector. For WebArena, database state assertions are the detector. For TAU-bench, a combination of tool call log inspection and user simulator satisfaction score is used.

**Cost tracking per run**: API costs must be tracked per episode, not just per run. An agent that solves 50% of SWE-bench at $2/episode has a different cost-efficiency profile than one that solves 50% at $10/episode.

## Trajectory-Level Metrics

Binary pass/fail is the primary reported metric on all three benchmarks, but it discards signal. Supplementary trajectory metrics include:

**Step efficiency**:

```
efficiency = success_rate / mean_steps_per_successful_episode
```

An agent that solves a task in 5 steps is more efficient than one that solves it in 15 steps, even if the success rate is identical. Step efficiency penalizes verbose, hedge-betting agents.

**Tool call precision**: the fraction of tool calls that were valid (schema-correct) and produced useful information (not an error or an irrelevant result). A high-precision agent wastes fewer context window tokens on tool calls that contribute nothing.

**Partial progress**: for multi-step tasks, a task completion score that awards partial credit for reaching intermediate milestones. WebArena's harder tasks (those with 5+ required steps) are difficult to differentiate on binary pass/fail; partial credit scores are more sensitive.

**Recovery rate**: for episodes where the agent made at least one recoverable error (e.g., called the wrong tool, misread a value), what fraction of the time did the agent detect the error and successfully recover? This measures robustness to imperfect perception rather than just accuracy under ideal conditions.

## LLM-as-Judge for Open-Ended Tasks

For tasks where the success criterion cannot be fully specified mechanically — particularly in TAU-bench and in custom production evals — a separate LLM acts as judge over the trajectory. The judge receives the task specification, the full trajectory, and a scoring rubric, and outputs a score.

LLM judges introduce several failure modes:

- **Length bias**: judges tend to rate longer responses higher, all else equal.
- **Self-enhancement bias**: a judge from the same model family as the agent being evaluated scores that agent's outputs higher.
- **Rubric sensitivity**: changing the framing of the rubric (e.g., "rate from 1–5" vs. "rate from 1–10") changes scores without changing the underlying quality.
- **Calibration drift**: LLM judges are not calibrated against human labels without explicit calibration data. A judge that correlates 0.85 with human agreement on in-distribution tasks may drop to 0.60 on out-of-distribution task types.

The mitigations are: collect a calibration set of (trajectory, human label) pairs; fine-tune or prompt-engineer the judge against this calibration set; measure inter-judge agreement as a proxy for reliability; never use the judge model's own family for judging that model's agent.

LLM-as-judge for agent trajectories is structurally analogous to Process Reward Models (PRMs) applied to reasoning chains: the judge is scoring the quality of intermediate steps, not just the final answer (see [../process-reward-models/](../process-reward-models/)). The difference is that PRMs are typically fine-tuned on step-level human labels and output calibrated probabilities; LLM judges are typically prompted and output natural language scores. PRM-style fine-tuning of trajectory judges is an active research direction in 2025–2026.

## Benchmark Saturation and Its Consequences

SWE-bench Verified scores have climbed rapidly: from ~20% (late 2023 baseline) to ~50% (early 2026 frontier). As scores approach the ceiling (estimated at ~70–80% given human engineer performance), benchmark utility decreases — it no longer differentiates between top systems. The consequences are:

- **Distribution shift**: systems optimized for SWE-bench Python repositories may generalize poorly to other languages (TypeScript, Rust, Java), other repository structures, or real production codebases with different conventions.
- **Benchmark overfitting**: training data and scaffolding choices that were tuned against SWE-bench may not reflect genuine software engineering capability.
- **Need for harder splits**: SWE-bench++ and domain-specific variants (SWE-bench for TypeScript, for infrastructure-as-code) are emerging to maintain signal as the standard benchmark saturates.

The same dynamic applies to WebArena, where scores have risen from ~10% to ~40% since release. The community response is typically to add harder task variants, add new application environments, or shift to private holdout sets that cannot be directly trained against.

## Engineering Tradeoffs

| Evaluation Approach | Task Type | Environment Complexity | Reproducibility | Cost per Run | Production Relevance |
|---|---|---|---|---|---|
| SWE-bench Verified | Code patching from bug reports | Medium (sandboxed Python repos) | Very high (deterministic test suites) | $200–600 (500 instances) | High for code agents; limited to Python OSS |
| TAU-bench | Customer service, policy following, tool use | Low (simulated tools + LLM user) | Medium (user simulator adds variance) | $50–150 (retail + airline) | High for enterprise dialogue agents |
| WebArena | End-to-end browser task completion | High (6 live web apps, full browser) | Medium (app state resets, page rendering variance) | $100–300 (812 tasks) | High for browser automation; limited to standard web apps |
| Custom internal eval | Production task distribution | Variable | High (if deterministic fixtures) | Variable | Highest — matched to actual deployment |

The practical recommendation is to use SWE-bench and WebArena as external calibration signals (to compare against published baselines) and to build a custom internal eval for the production task distribution. Published benchmarks measure generic capability; internal evals measure deployment-specific capability. The gap between the two is the most important diagnostic of whether a capable agent will actually work in production.

## Cross-References

- [../inference-time-scaling/](../inference-time-scaling/) — agent evaluation benchmarks are effectively measuring the effectiveness of inference-time compute in an agentic loop; higher compute budgets per episode → higher success rates, with diminishing returns
- [../process-reward-models/](../process-reward-models/) — PRMs score reasoning steps; LLM-as-judge for agent trajectories is the same structure applied to multi-step action sequences
- [../agent-frameworks/](../agent-frameworks/) — the scaffolding frameworks that structure the agent's tool use, context management, and recovery logic
- [../tool-use-infra/](../tool-use-infra/) — the underlying tool call execution infrastructure that agent benchmarks exercise
