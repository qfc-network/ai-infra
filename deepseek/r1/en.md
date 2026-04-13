# DeepSeek-R1

- **Authors / Org**: DeepSeek-AI
- **Published**: 2025-01
- **Links**: [paper (arXiv:2501.12948)](https://arxiv.org/abs/2501.12948) · [models](https://huggingface.co/deepseek-ai/DeepSeek-R1)

## TL;DR

An open recipe for turning a strong base model (DeepSeek-V3-Base) into a frontier **reasoning** model rivaling OpenAI o1 on math / coding / logic benchmarks, **without relying on a large human-curated SFT dataset**. Two results are paired: **R1-Zero** — pure reinforcement learning from the base model with a simple rule-based reward, no SFT at all — which develops long chain-of-thought, self-verification, and reflection from scratch; and **R1** — a practical version that prepends a small "cold-start" SFT stage for readability, then repeats RL and SFT. The paper also introduces **GRPO** (Group Relative Policy Optimization), a PPO variant that drops the value network. Open weights and distilled student models released.

## Context & Motivation

Pre-R1, reasoning-LLM recipes were proprietary (o1) and the SFT-heavy open recipes plateaued below frontier. Two unknowns in the open literature:

1. **Can long-chain reasoning emerge from RL alone**, without imitation data?
2. **What's the minimal scaffolding** around RL to make it trainable at scale (reward hacking, readability, language mixing)?

R1 addresses both. R1-Zero is the scientific result (yes, it can emerge); R1 is the engineering result (here's the minimal scaffolding that ships).

## Core Method

### R1-Zero — pure RL from base

Start from V3-Base. No SFT. Apply RL with:

- **Prompts**: reasoning tasks with verifiable answers (math with known solutions, code with unit tests).
- **Reward**: purely rule-based.
  - **Accuracy reward**: does the final answer match ground truth?
  - **Format reward**: is the chain-of-thought enclosed in the expected `<think>...</think>` tags?
- **No learned reward model, no preference data.** Eliminates reward hacking by construction — you can only hack a rule you can verify against ground truth.

During training, **response length grows organically**: the model discovers that longer reasoning improves accuracy reward. Self-reflection ("let me verify...", "wait, that's wrong") emerges without being supervised for. Benchmark numbers climb smoothly.

Downside: outputs are unreadable — language mixing (Chinese/English interleaved), poor formatting. Fine for benchmarks, not for products.

### R1 — cold-start SFT + multi-stage RL

Four stages:

1. **Cold-start SFT**: a few thousand long-CoT examples, curated for readability, to give the model a legible "reasoning voice" before RL.
2. **Reasoning RL**: same rule-based reward as R1-Zero, plus a language-consistency reward to discourage code-switching. Train to convergence.
3. **Reject-sample SFT**: use the RL-trained model to generate CoTs, reject low-quality ones, combine with non-reasoning SFT data (writing, self-cognition) to produce a broader-capability SFT dataset (~800k samples). Train from V3-Base again.
4. **Alignment RL**: additional RL pass covering helpfulness and harmlessness, using learned reward models for these non-verifiable dimensions. Rule-based reward retained for reasoning.

Final model matches o1-class benchmarks while being usable in production.

### GRPO — group relative policy optimization

For each prompt, sample a group of `G` responses from the current policy (`G = 8–16`). Use their mean reward as the baseline instead of a learned value network:

```
A_i = (r_i − mean({r_j})) / std({r_j})
```

Apply standard PPO clipping with this advantage. Benefits:

- **No value network** — half the memory and compute of PPO; critical at frontier scale.
- **Group-relative normalization** stabilizes training across prompts of very different difficulty.
- Works well with rule-based rewards, where a value function is arguably unnecessary anyway.

GRPO was introduced in DeepSeekMath earlier; R1 validates it at larger scale.

### Distillation

The 800k SFT dataset is used to distill the reasoning behavior into smaller dense models (Qwen and Llama 1.5B–70B). Distilled students beat direct RL on those same small models — RL at small scale is unstable, imitating a strong teacher is more reliable.

## Engineering Tradeoffs

| Decision | Gained | Gave up |
|---|---|---|
| Pure RL from base (R1-Zero) | Proves reasoning emerges without imitation | Outputs unreadable; not directly useful as a product |
| Cold-start SFT before RL | Usable product; faster convergence | Small dependence on human data, blurs the scientific claim |
| Rule-based rewards only for reasoning | No reward hacking; no reward model compute | Only works where verification is cheap (math, code) |
| GRPO over PPO | ~2× cheaper RL; stable across difficulty | Needs group sampling (`G=8–16` rollouts per prompt) |
| Distill rather than RL small models | Reliable small-model quality | Small models inherit teacher biases; no novel emergence at small scale |
| Multi-stage pipeline | Balanced capability and alignment | Complexity; four stages to reproduce correctly |

## Experiments & Results

- **R1-Zero**: AIME 2024 pass@1 climbs from ~15.6% (V3-Base) to ~71% during RL. Language mixing and readability issues confirmed.
- **R1**: MATH-500 97.3%, AIME 79.8%, Codeforces 96.3 percentile, MMLU 90.8%. Matches or beats o1 on multiple hard reasoning benchmarks; competitive on general tasks.
- **Distilled**: R1-distilled-Qwen-32B beats o1-mini on reasoning benchmarks; distilled-7B is the strongest small reasoning model at release.

## Reproducibility Notes

- Open weights for R1, R1-Zero, and all distilled variants.
- Paper describes the pipeline clearly; training code not released but the recipe has been reproduced (e.g., **Open-R1** from HuggingFace, **simpleRL**, Tsinghua efforts).
- Rule-based-reward RL is the most reproducible part — requires only a base model, a verifiable dataset, and a PPO / GRPO loop.
- Running RL at V3 scale (671B MoE) is still infeasible outside well-resourced labs; small-scale reproductions on Qwen-7B are tractable on a few nodes.

## Commentary

The lasting contribution is **R1-Zero**, not R1. Showing that chain-of-thought, self-verification, and reflection can emerge from **base model + rule-based RL alone** is a meaningful scientific datapoint — it disentangles "reasoning capability" from "reasoning data," suggesting the latter mostly served as a shortcut. R1 itself is the practical version that ships. GRPO deserves its own attention: removing the value network at RLHF-scale is a big efficiency win, and is already being adopted elsewhere. The open question is how far pure-RL scales — R1-Zero works because math and code have cheap verifiers; it's unclear whether the same approach produces frontier reasoning on tasks without ground truth (writing, open-ended problem solving). The next wave of work is already attacking this via **process reward models, tree-search-augmented RL, and self-consistency as a pseudo-reward**. Expect R1's recipe to be the template for open-source reasoning models through 2026.

## References

- [1] DeepSeek-AI. _DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning._ arXiv:2501.12948, 2025.
- [2] Shao et al. _DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models._ arXiv:2402.03300, 2024. (GRPO first appears here)
- [3] OpenAI. _Learning to Reason with LLMs (o1 system card)_, 2024.
- [4] HuggingFace. _Open-R1._ 2025.
