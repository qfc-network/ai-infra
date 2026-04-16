# Process Reward Models (PRMs)

- **Authors / Org**: Lightman et al. (OpenAI) for PRM800K; Luo et al. (DeepSeek) for MCTS+PRM; Wang et al. (UC Berkeley) for MC rollout labeling
- **Published**: 2023-05 (Lightman arXiv:2305.20050); 2024-06 (Wang arXiv:2406.06592); 2024-08 (Snell arXiv:2408.03314)
- **Links**: [Lightman et al. PRM800K (arXiv:2305.20050)](https://arxiv.org/abs/2305.20050) · [Wang et al. MC labeling (arXiv:2406.06592)](https://arxiv.org/abs/2406.06592) · [PRM800K dataset](https://github.com/openai/prm800k)

## TL;DR

An **Outcome Reward Model (ORM)** scores a completed reasoning chain: correct or wrong. A **Process Reward Model (PRM)** scores each reasoning *step*: was this step valid, given what came before? The distinction matters because most reasoning errors are localized — a chain that goes wrong at step 4 still has steps 1–3 that are correct, and a chain that is correct in its final answer may have internally flawed reasoning. PRMs trained on step-level human labels (PRM800K, 800k steps) substantially outperform ORMs on math reasoning benchmarks at every sample count. They are the verifier behind PRM-guided best-of-N and beam search in inference-time scaling, and the missing component that makes GRPO's group-relative advantage a step toward per-step credit assignment. The bottleneck: step-level labels are expensive to collect. Monte Carlo rollout labeling (auto-labeling steps by their empirical probability of leading to a correct answer) makes PRMs trainable without human annotation, at the cost of label noise.

## Context & Motivation

### Why outcome supervision is insufficient

Standard RLHF trains an ORM on (question, answer, correct/incorrect) pairs. The ORM assigns a scalar reward to the complete trajectory. For reasoning tasks this has two failure modes:

1. **Credit assignment is coarse.** A wrong answer might be caused by one bad step out of ten. The ORM penalizes all ten steps equally — the nine correct steps receive the same negative gradient as the one wrong step.

2. **Reward hacking via correct-answer tricks.** A policy can learn to produce plausible-looking steps followed by a guessed correct answer, with no genuine reasoning. The ORM rewards the correct final answer regardless of how it was reached.

PRMs address both: step-level scores enable precise credit assignment, and a PRM that scores each step on its local validity is much harder to game — a model cannot get a high PRM score by reasoning incorrectly through ten steps, even if it guesses the right answer at the end.

### The inference-time connection

As covered in the [inference-time scaling entry](./inference-time-scaling/), the effectiveness of best-of-N search and beam search depends critically on the verifier. For math and code:

- **ORM best-of-N**: generate N chains, return the one with the highest ORM score. Works when the model frequently produces a correct final answer; fails when the model frequently produces a wrong answer that *looks* correct to the ORM.
- **PRM best-of-N**: score each step of each chain; select the chain with the highest PRM score (product or minimum of step scores). The PRM catches errors that the ORM misses because it evaluates the reasoning, not just the conclusion.
- **PRM beam search**: generate reasoning one step at a time; at each step, score partial chains with the PRM; keep the top-B beams and prune the rest. Concentrates inference compute on the most promising reasoning paths.

Lightman et al. (PRM800K) showed PRM-guided best-of-N outperforms ORM best-of-N by 5–15% absolute on MATH at every N tested (N = 4 to N = 1860). At N = 1860 with PRM guidance, GPT-4 reaches 78.2% → 90.1% on MATH 500.

## Core Method

### What a PRM predicts

A PRM is a language model with a scalar head that predicts, for each step `s_i` in a reasoning chain:

```
PRM(s_i | question q, steps s_1, ..., s_{i-1}) → p ∈ [0, 1]
```

where `p` is the probability that step `s_i` is correct given the preceding steps. The model sees the full chain up to `s_i` and outputs a score at a designated token (typically a newline or step separator).

The chain-level PRM score is either the **product** of step scores (penalizes any wrong step) or the **minimum** step score (identifies the weakest link). The product is more discriminative for search; the minimum is more interpretable.

### Human annotation: PRM800K

Lightman et al. collected 800,000 step-level labels by having human contractors grade each step of GPT-4 reasoning chains on MATH problems. Each step is labeled:

- **Positive**: correct and valid
- **Negative**: contains an error
- **Neutral**: not mathematically meaningful (formatting, restating the problem)

This required ~3,000 hours of contractor time for 75,000 unique solutions to ~12,000 MATH problems. The key finding: agreement between contractors is ~80% — reasoning steps are genuinely ambiguous in some cases, especially when a step is not *wrong* but takes a suboptimal approach.

The PRM is then trained as a binary classifier on (chain prefix, step, label) tuples, using a language model backbone with a value head:

```
loss = -[y · log(PRM(s_i)) + (1-y) · log(1 - PRM(s_i))]
```

where `y = 1` for positive labels, `y = 0` for negative.

### Monte Carlo rollout labeling (auto-labeling)

Human annotation is expensive and domain-specific (PRM800K covers MATH only). Wang et al. (2024) proposed **MC rollout labeling**: for each step `s_i` in a training chain, estimate its correctness by sampling K completions from `s_i` to the end of the chain and checking whether they reach the correct answer:

```
p̂(s_i correct) = (# completions from s_i that reach correct answer) / K
```

This auto-label is then used as the training target for the PRM. The intuition: a correct reasoning step should frequently lead to a correct final answer; an incorrect step is unlikely to recover.

MC labeling requires:
- A verifier for the final answer (symbolic math engine for MATH, unit tests for code)
- K rollout completions per step (typically K = 4–16)
- A policy model to generate the completions (can be the model being trained)

**Noise profile**: MC labels are noisy in two directions:
- A correct step may reach a wrong answer (if subsequent steps fail) — under-estimates step quality
- A wrong step may reach a correct answer by accident — over-estimates step quality

In practice, steps early in the chain are under-estimated (small probability of failure from a correct premise) and steps late in the chain are over-estimated (near the answer, even a bad step might not matter). Calibration techniques (temperature on the logit output, label smoothing) help.

### PRM architecture

A PRM shares its backbone with the policy model (same tokenizer, same architecture). Two variants:

1. **Separate PRM**: trained independently from scratch or from the SFT checkpoint, on labeled (chain, step-scores) data. Cleaner separation; can be smaller than the policy (3B–7B PRM for a 70B policy is common).

2. **Value head on the policy**: attach a linear scalar head to the policy, trained jointly. The policy and PRM share representations. More parameter-efficient; can cause interference if the PRM objective conflicts with the language modeling objective.

For serving with beam search or MCTS, the PRM needs to be fast. A separate, smaller PRM is preferred for this reason — PRM inference runs at every beam step, so latency directly multiplies.

### Integration with search

**Best-of-N with PRM:**
```
for each candidate chain c in [c_1, ..., c_N]:
    score(c) = ∏_i PRM(s_i | q, s_{<i})   # or min
return argmax_c score(c)
```

**Beam search:**
```
beams = [([], 1.0)]   # (partial chain, score)
for step 1..T:
    candidates = []
    for (chain, score) in beams:
        for next_step in generate_candidates(chain, q, B):
            new_score = score × PRM(next_step | q, chain)
            candidates.append((chain + [next_step], new_score))
    beams = top-B(candidates)   # prune to beam width B
return beams[0]
```

At each step, `generate_candidates` samples B candidate next-steps from the policy; the PRM scores each; only the top-B survive. This concentrates compute on high-PRM-score prefixes. The cost: B × PRM inference calls per step, plus B × policy calls.

**MCTS with PRM:**
The PRM serves as the value function in MCTS: the value of a node (partial reasoning chain) is estimated by the PRM score. This avoids the need for full rollouts to estimate node value — the PRM provides an instant value estimate. DeepSeek-R1's report discusses MCTS as a search strategy for generating high-quality SFT data; the PRM is the oracle that makes MCTS over reasoning practical.

## Engineering Tradeoffs

| Decision | Gained | Gave up |
|---|---|---|
| Human labels (PRM800K-style) | High-quality, calibrated labels; good signal even for ambiguous steps | ~3k hours per domain; not scalable beyond a few benchmarks; labelers need domain expertise |
| MC rollout auto-labels | Scalable to any verifiable domain; no human annotation; self-improving as policy improves | Label noise (under/over-estimation at chain boundaries); requires K rollouts per step during data generation; noisy early in training when policy is weak |
| Separate PRM (smaller) | Fast inference for beam/MCTS; training isolation from policy | Must re-train if policy changes significantly; two model checkpoints to manage |
| Value head on policy | Parameter-efficient; shared representations | Interference risk; PRM head can degrade LM quality if not carefully balanced; model is larger at inference time |
| Product vs min step score | Product: more discriminative for ranking chains | Product collapses to 0 if any step is wrong — unstable gradient; min: more stable but less expressive |
| Beam width B (1 → 16) | Larger B: finds better reasoning paths | Larger B: B× more PRM inference per step; B× more policy inference per step; memory for B KV caches |

## Experiments & Results

**PRM800K (Lightman et al., 2023):**
- ORM best-of-N on MATH at N=1860: 72.4%
- PRM best-of-N on MATH at N=1860: 78.2%
- Both with GPT-4 base model; 6% absolute gain at maximum N
- At N=8 (practical compute budget): ORM 62%, PRM 68% — 6% absolute gain holds across N

**MC rollout labeling (Wang et al., 2024):**
- Math-Shepherd (their method): 7B model + MC-labeled PRM on MATH: 84.1% (vs ORM best-of-N baseline 78.6%)
- MC labels are noisier than human labels but generalizable to new domains without annotation cost
- The auto-labeled PRM is within 3–4% of the human-labeled PRM on MATH when K=4 rollouts per step

**Inference-time scaling (Snell et al., 2024):**
- PRM beam search on MATH 500: outperforms flat best-of-N by 5–10% absolute at the same total FLOP budget
- The margin is largest on hard problems (AMC, AIME difficulty) — exactly where the model benefits most from step-level guidance

## Reproducibility Notes

- **PRM800K dataset**: publicly released at [github.com/openai/prm800k](https://github.com/openai/prm800k). Covers MATH problems with GPT-4 solutions. Not easily extensible to other domains without new annotation.
- **Math-Shepherd dataset**: released by Wang et al. for MC-labeled math PRMs. Lower quality than PRM800K but broader coverage.
- **Training a PRM**: use any LM with a scalar head. HuggingFace `transformers` + a regression or binary CE head on a step-separator token works. The tricky part is tokenizing step boundaries consistently — the PRM must see the same step separators at training and inference time.
- **Serving a PRM for beam search**: run the PRM as a separate process from the policy; use async batching so PRM calls don't block policy generation. At B=8 and 20 steps, you issue 160 PRM inference calls per query — latency stacks badly if synchronous.
- **MC rollout generation**: compute-intensive. Generating K=4 rollouts per step for a 1000-step training set requires 4000 full completions. Use vLLM for throughput; batch all rollouts for a given prefix together to maximize KV cache reuse.

## Commentary

PRMs are the answer to a question RLHF left open: **where in the reasoning chain did the model go wrong?** An ORM can tell you the model is wrong; a PRM can tell you it went wrong at step 4. This step-level signal changes what's possible: you can search over reasoning steps rather than complete chains, you can provide gradient signal precisely at the error location, and you can build interpretable systems that explain *why* a solution is wrong rather than just *that* it is wrong.

The MC rollout approach is particularly significant. It breaks the dependency on expensive human annotation by using the model itself to estimate step quality — a step is good if it leads somewhere. This is philosophically similar to how GRPO replaced the reward model with rule-based verifiers: both moves reduce the human annotation bottleneck. The catch is the same: you need a verifiable domain. For MATH and code, final-answer verification is cheap. For open-ended reasoning, there is no cheap final-answer verifier, and PRMs trained on MATH don't generalize to other domains.

Three open problems that define the current frontier:

**1. PRM generalization.** PRM800K covers MATH; Math-Shepherd covers grade-school math. There are no well-validated PRMs for coding (unit tests are process verifiers, but not step-level), science reasoning, or multi-hop factual questions. Every lab building reasoning models needs its own domain-specific PRM pipeline.

**2. PRM Goodharting.** Models trained with GRPO or RL against a PRM learn to produce PRM-pleasing reasoning that isn't necessarily correct. The PRM scores high on locally-coherent steps that collectively go wrong. This is the process-level analog of reward hacking, and it's observed empirically — models fine-tuned against PRMs sometimes exhibit stylistic patterns the PRM rewards without improving actual correctness. Regularly re-training the PRM on new policy samples is the standard mitigation; it's expensive.

**3. PRM for long chains.** At 4,000+ token reasoning traces (typical for R1-class models), a step-level PRM must make hundreds of scoring calls per chain. The product aggregation compounds numerical instability. Alternative aggregations (learned pooling over step scores, attention over the step-score sequence) are being explored but are not yet standard practice.

For practitioners: if you have a verifiable task and want to improve reasoning quality, train a small (7B–13B) MC-labeled PRM before investing in GRPO. PRM-guided best-of-N at N=16 is often cheaper and more effective than one round of GRPO, and it requires no online rollout infrastructure. Layer GRPO on top once you have a working PRM signal.

## References

- [1] Lightman et al. _Let's Verify Step by Step._ arXiv:2305.20050, 2023.
- [2] Wang et al. _Math-Shepherd: Verify and Reinforce LLMs Step-by-step without Human Annotations._ arXiv:2312.08935, 2023.
- [3] Wang et al. _Scaling LLM Test-Time Compute with Inference-Time Intervention._ arXiv:2406.06592, 2024.
- [4] Snell et al. _Scaling LLM Test-Time Compute Optimally._ arXiv:2408.03314, 2024.
- [5] DeepSeek-AI. _DeepSeek-R1._ arXiv:2501.12948, 2025.
- [6] Ouyang et al. _InstructGPT._ arXiv:2203.02155, 2022. (ORM baseline; RLHF foundation)
- [7] Shao et al. _DeepSeekMath / GRPO._ arXiv:2402.03300, 2024.
- [8] Silver et al. _Mastering the game of Go with deep neural networks and tree search._ Nature, 2016. (MCTS value function analogy)
