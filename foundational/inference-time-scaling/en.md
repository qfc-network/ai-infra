# Inference-Time Scaling — Test-Time Compute

- **Authors / Org**: Snell et al. (UC Berkeley / Google DeepMind); concurrent work by Brown et al. (Scale AI); foundational sampling work by Wang et al. (Google Brain)
- **Published**: 2024-08 (Snell arXiv:2408.03314); 2024-07 (Brown arXiv:2407.21787); 2023-03 (Wang arXiv:2203.11171)
- **Links**: [Snell et al. (arXiv:2408.03314)](https://arxiv.org/abs/2408.03314) · [Brown et al. "Large Language Monkeys" (arXiv:2407.21787)](https://arxiv.org/abs/2407.21787) · [Wang et al. Self-Consistency (arXiv:2203.11171)](https://arxiv.org/abs/2203.11171) · [Lightman et al. PRM800K (arXiv:2305.20050)](https://arxiv.org/abs/2305.20050)

## TL;DR

Training compute — total FLOPs spent before deployment — has been the dominant axis of LLM capability for five years. Inference-time scaling (also called test-time compute scaling) introduces a second axis: **compute spent per query at inference time**. Given a fixed training budget, you can trade inference FLOPs for higher per-query accuracy. The key mechanisms are **parallel search** (generate many candidates, pick the best via a verifier) and **sequential refinement** (chain-of-thought, extended thinking, iterative revision). Snell et al. (2024) show these two strategies have complementary strengths: parallel search wins on problems that are "hard for the model but verifiable"; sequential refinement wins on problems that require multi-step reasoning the model cannot compress into a single forward pass. The most visible embodiment of inference-time scaling is the o1/R1 family of reasoning models, whose "thinking tokens" are a learned, adaptive form of sequential refinement trained via GRPO.

## Context & Motivation

### The Chinchilla wall and what comes after

Chinchilla (Hoffmann et al., 2022) established that training FLOPs should be split roughly equally between model parameters and training tokens. Scaling beyond Chinchilla-optimal requires proportionally more data, which is increasingly scarce for internet-scale text. The community has been searching for a third lever.

Inference-time compute is that lever. The intuition: a 7B model that generates 64 candidate answers and picks the best one via a process reward model may solve problems that a 70B model cannot solve in a single pass. The 7B model is cheaper to train and cheaper per token — the cost is paid at inference time, where it can be amortized differently (on hard queries only, on latency-insensitive workloads, etc.).

### Why this is different from just "sample more"

Naive best-of-N sampling (generate N answers, take the majority vote) was known from Self-Consistency (Wang et al., 2023) to improve accuracy on reasoning benchmarks. But:

1. Best-of-N scales poorly: accuracy improves as `log(N)` or `N^α` depending on the task. You need a very large N for hard problems.
2. You need a reliable way to pick the correct answer from N candidates. Majority voting works when the correct answer is the most common; it breaks down when the model rarely gets the right answer at all.
3. For hard multi-step problems, the model's failure mode is often structural — it consistently makes the same mistake at the same step, regardless of sampling temperature. More samples don't help.

The key advance is combining sampling with **process reward models (PRMs)** — verifiers that score each reasoning *step*, not just the final answer — and moving from flat parallel search to **structured search** (beam search, MCTS) guided by the PRM.

## Core Method

### Strategy 1 — Parallel search with a verifier

Generate multiple complete or partial reasoning traces and use a verifier to select the best:

**Best-of-N (outcome-supervised):** Sample N completions independently, score each with an outcome reward model (ORM), return the highest-scoring answer. Works when the model solves the problem at least occasionally. Brown et al. (2024) show that on MATH, a 7B model with N=10,000 matches or exceeds a 70B model with N=1 — "large language monkeys" at the keyboard.

**Best-of-N with PRM:** Instead of scoring completed answers, use a PRM to score intermediate reasoning steps. Lightman et al. (PRM800K) trained a step-level verifier on 800K human-labeled reasoning steps; the PRM finds errors mid-chain that the ORM misses. PRM-guided best-of-N consistently beats ORM best-of-N at the same N.

**Beam search over reasoning steps:** Extend the reasoning chain one step at a time. At each step, maintain a beam of B partial chains, score each next-step with the PRM, and prune to the top B. This concentrates compute on the most promising reasoning paths rather than exploring uniformly. Snell et al. show this dominates flat best-of-N on hard MATH problems at the same FLOP budget.

The key equation for PRM-guided search: given a chain of K steps `s₁, s₂, ..., s_K`, the PRM assigns a score:

```
PRM(s₁..K) = ∏ᵢ p_PRM(sᵢ is correct | s₁..ᵢ₋₁)
```

This multiplicative structure means an early error collapses the score even if subsequent steps are locally coherent — correctly modeling error propagation in reasoning chains.

### Strategy 2 — Sequential refinement (thinking tokens)

Rather than searching over parallel completions, allocate more tokens to a single chain that reasons before committing to an answer. This is the "extended chain-of-thought" or "thinking" paradigm.

The model learns (via GRPO or similar RL) to emit a `<think>...</think>` block before the answer. The content of this block is not constrained by supervision — the model develops its own scratchpad notation. What emerges is a mix of:

- **Deliberate multi-step decomposition**: breaking hard problems into solvable sub-problems
- **Self-verification**: the model checks its own intermediate steps (a weak internal PRM)
- **Backtracking**: the model detects a wrong path and restarts from an earlier state

The compute cost is proportional to thinking-token length. For a problem that takes T thinking tokens, inference cost is approximately `(T + A)` forward passes worth of tokens where A is the answer length — versus `A` tokens for a non-thinking model. The speedup hypothesis: a model that spends T tokens thinking can solve problems it would never solve in A tokens alone.

### Which strategy when?

Snell et al.'s central empirical finding, formalized as the "compute-optimal" inference strategy:

| Problem difficulty (relative to model capability) | Optimal strategy |
|---|---|
| Easy — model nearly always correct | No extra compute needed; best-of-1 suffices |
| Medium — model correct ~10–40% of the time | Parallel search with PRM; correctness is verifiable and density is sufficient |
| Hard — model correct <5% even with many samples | Sequential refinement; parallel search can't find a needle that barely exists |
| Beyond model capability | Neither helps; need a stronger model or training |

The key insight: parallel search works when the model has a reasonable probability of a correct trace — you're amplifying signal that exists. Sequential refinement works when the model needs to construct a fundamentally longer reasoning path than it would produce in one pass — you're enabling a qualitatively different computation.

## Engineering Tradeoffs

| Decision | Gained | Gave up |
|---|---|---|
| Best-of-N (parallel) | Trivially parallelizable; no architectural changes; works at test time without training | N× memory for KV caches; N× throughput cost; needs reliable verifier; scales as log(N) |
| Beam search + PRM | Better than flat best-of-N at same FLOP budget; finds correct traces the model produces rarely | PRM inference at every step; beam management complexity; PRM quality bottlenecks the gains |
| Thinking tokens (sequential) | Qualitatively harder problems; emergent self-verification; adaptive — spends more on hard queries | Not parallelizable per-query; higher latency; difficult to bound output length; requires GRPO/RL training |
| Adaptive compute (vary N or think-length by difficulty) | Efficient: easy queries cheap, hard queries expensive | Requires difficulty estimation at query time (a non-trivial classifier) |
| PRM vs ORM as verifier | PRM catches step-level errors; higher precision | PRM training requires step-level labels (expensive); PRM can be gamed by models that learn to produce PRM-pleasing but incorrect reasoning |

## Experiments & Results

**Self-Consistency (Wang et al., 2023):**
- Majority voting over 40 samples on GSM8K: 74.4% → 83.0% (PaLM 540B). A 3-shot CoT model with 40 samples matches a much larger model with 1 sample.
- First demonstration that "sample more" is a reliable scaling axis for reasoning.

**Large Language Monkeys (Brown et al., 2024):**
- Llama-3 8B on MATH: pass@1 = 4%; pass@10,000 ≈ 78%. A small model, given enough samples and a verifier, achieves near-state-of-the-art.
- Highlights the gap between "can the model produce the answer at all" and "does it produce it reliably in one shot" — two very different questions.

**PRM800K (Lightman et al., 2023):**
- GPT-4 with PRM-guided best-of-N on MATH: 78.2% → 90.1% vs ORM at the same N.
- Process supervision (step-level feedback) outperforms outcome supervision (final-answer feedback) at every N tested.

**Snell et al. "Scaling LLM Test-Time Compute Optimally" (2024):**
- On MATH 500 problems, a 70B model with optimally-allocated inference compute matches GPT-4 in accuracy.
- Beam search with a PRM outperforms flat best-of-N by 5–10% absolute on hard subsets at the same total inference FLOP budget.
- Compute-optimal inference strategy varies by problem difficulty quantile — validating the three-regime framing above.

**DeepSeek-R1 (2025):**
- 671B MoE with GRPO-trained thinking tokens. AIME 2024: 79.8% (matches o1). The thinking traces average 4,000–8,000 tokens for hard competition math — 10–20× the answer length.

## Reproducibility Notes

**Parallel search:**
- Straightforward to implement with any inference stack. `vllm generate(n=N)` runs N samples in a single batched forward; no code changes required.
- ORM: standard LM trained on (problem, answer, correct/incorrect) pairs. Use the hidden state of a separator token as the reward head.
- PRM: harder. Lightman et al.'s PRM800K dataset is public. Training requires step-level labels; the dataset covers MATH only. For other domains, automated step labeling via Monte Carlo rollouts (test if this step leads to a correct answer in k completions) is a practical substitute.

**Sequential refinement (thinking models):**
- The GRPO training recipe is described in the GRPO entry and the DeepSeek-R1 tech report.
- Open re-implementations: `trl.GRPOTrainer` (HuggingFace), OpenRLHF, veRL.
- Key infrastructure challenge: rollout generation during training produces thinking traces of variable and often very long length. Use `max_new_tokens` capping during early training to prevent runaway trace length.
- Inference: thinking models require serving with generous `max_tokens` budgets. vLLM and SGLang both support dynamic sequence lengths; TensorRT-LLM requires setting a static max.

**Key public artifacts:**
- PRM800K dataset: [github.com/openai/prm800k](https://github.com/openai/prm800k)
- DeepSeek-R1 weights and training code: [github.com/deepseek-ai/DeepSeek-R1](https://github.com/deepseek-ai/DeepSeek-R1)

## Commentary

Inference-time scaling is the first structural shift in LLM capability since the pre-training scaling laws. For five years, the answer to "how do you make the model smarter?" was "train it longer on more data." Now there are two answers: train more, or think more.

The practical stakes are significant. A thinking model at 7B scale solves problems a 70B non-thinking model cannot. The inference cost is higher per query, but the training cost is an order of magnitude lower. For many use cases — latency-tolerant tasks like code generation, math, research synthesis — this is the right tradeoff.

Three tensions that aren't fully resolved:

**1. The verifier bottleneck.** Parallel search is only as good as the verifier. For math and code, there are ground-truth verifiers (symbolic evaluation, unit tests). For open-ended tasks — summarization, analysis, creative work — there is no ground truth. PRMs for open-ended domains either don't exist or are learned models prone to Goodharting. This limits the parallel-search regime to verifiable domains.

**2. Latency vs quality.** Thinking tokens add latency proportional to thinking length. A model that spends 8,000 tokens thinking before answering takes 8–20× longer to respond than a direct-answer model. For conversational or interactive applications, this is often unacceptable. Adaptive compute (decide *how much* to think based on query difficulty) is the right engineering answer but requires a good difficulty classifier that is itself non-trivial to build.

**3. Training compute vs inference compute substitutability.** Snell et al. show a 70B model with optimized inference can match GPT-4 on MATH. But this doesn't mean inference compute fully substitutes for training compute — the 70B model still needs to have seen enough math to produce correct reasoning traces with reasonable probability. Below some training threshold, no amount of inference compute closes the gap.

For practitioners: if you have a verifiable task (math, code, structured extraction with ground-truth validation), try PRM-guided best-of-N before reaching for a larger model. If the task requires multi-hop reasoning the model can't do in one pass, fine-tune with GRPO or use a pre-trained thinking model. If the task is open-ended, inference-time scaling currently has no clean story — this is an open research area.

## References

- [1] Snell et al. _Scaling LLM Test-Time Compute Optimally._ arXiv:2408.03314, 2024.
- [2] Brown et al. _Large Language Monkeys: Scaling Inference Compute with Repeated Sampling._ arXiv:2407.21787, 2024.
- [3] Wang et al. _Self-Consistency Improves Chain of Thought Reasoning in Language Models._ arXiv:2203.11171, 2023.
- [4] Lightman et al. _Let's Verify Step by Step._ arXiv:2305.20050, 2023.
- [5] Hoffmann et al. _Training Compute-Optimal Large Language Models (Chinchilla)._ arXiv:2203.15556, 2022.
- [6] DeepSeek-AI. _DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning._ arXiv:2501.12948, 2025.
- [7] OpenAI. _Learning to Reason with LLMs._ blog.openai.com/learning-to-reason-with-llms, 2024.
- [8] Shao et al. _DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models._ arXiv:2402.03300, 2024.
