# RLHF / InstructGPT — Aligning LLMs with Human Preferences

- **Authors / Org**: Ouyang et al., OpenAI
- **Published**: 2022-03 (InstructGPT); built on Christiano et al. (2017), Stiennon et al. (2020)
- **Links**: [InstructGPT paper (arXiv:2203.02155)](https://arxiv.org/abs/2203.02155) · [Summarization with RLHF (arXiv:2009.01325)](https://arxiv.org/abs/2009.01325) · [Deep RL from HF (arXiv:1706.03741)](https://arxiv.org/abs/1706.03741)

## TL;DR

The three-stage recipe — **SFT → reward model (RM) → RL against RM with PPO** — that turned raw next-token-prediction LLMs into models people would actually use. InstructGPT showed that a 1.3B model aligned via RLHF is preferred by human raters over a 175B base GPT-3 on instruction-following tasks. This paper set the template for every modern LLM product: GPT-3.5 / 4, Claude, Llama 2 Chat, DeepSeek, and nearly everything else built on variants of it. The infrastructure consequences are as significant as the algorithmic ones: RLHF introduced **rollout-heavy RL loops into LLM training**, forcing the field to build systems for serving the policy during training (actor), running the reward model inline, and coordinating PPO updates across thousands of GPUs. Every major post-training stack (TRL, DeepSpeed-Chat, OpenRLHF, verl) exists because RLHF is operationally harder than pretraining.

## Context & Motivation

GPT-3 was a pretrained base model: given enough prompt engineering, it could follow instructions, but out of the box it would continue prompts rather than respond to them. The core problem:

- **Pretraining loss (next-token prediction) does not align with "be useful and harmless."**
- Collecting enough curated "useful response" data for supervised fine-tuning is possible but limited.
- Human preferences over response pairs are cheaper to collect than gold-standard demonstrations and arguably more informative.

Christiano et al. 2017 had already shown RL from human preferences works for simple RL tasks. Stiennon et al. 2020 applied it to summarization with GPT-2. InstructGPT scaled the recipe to 175B and made it the default alignment method for the next several years.

## Core Method

### Stage 1 — Supervised fine-tuning (SFT)

- Collect ~13k prompt → high-quality response demonstrations from contractor labelers.
- Fine-tune the base model on these examples with standard language modeling loss.
- Output: the "SFT model," a decent but often verbose / over-confident instruction follower.

This stage anchors the distribution — the RL phase needs to stay close to something reasonable.

### Stage 2 — Reward model training

- Sample K (typically 4–9) responses per prompt from the SFT model.
- Contractors rank them from best to worst.
- Train a reward model (same architecture as the LM, different head) on this preference data with the **Bradley-Terry pairwise-comparison loss**:

```
L_RM = -E[(y_w, y_l) ~ D] log σ(r(x, y_w) − r(x, y_l))
```

Higher `r(x, y_w) − r(x, y_l)` ⇒ higher likelihood of matching the human preference. The RM learns to predict a scalar "how good is this response" that matches human rankings.

Output: a reward model that scores (prompt, response) → scalar.

### Stage 3 — PPO against the reward model

Train the policy (starting from SFT) with **Proximal Policy Optimization**, using the RM as reward:

```
objective = E_prompt [ r(x, y) − β · KL(π_θ(y|x) || π_SFT(y|x)) ]
```

Two things to note:

- **KL penalty to SFT** prevents the policy drifting to reward-hacked outputs that score well but diverge from coherent language.
- **PPO** keeps each update close to the previous policy, stabilizing training — important because rewards from an RM are noisy and a single bad update can collapse generation.

The full PPO loop needs four models live in memory (or sharded across devices):

1. Policy (being trained).
2. Reference policy (frozen SFT, for KL computation).
3. Value network / critic (estimates baseline for advantage).
4. Reward model (frozen, scores rollouts).

**This is the expensive part.** For a 175B policy, PPO on GPU clusters is the first time "training" and "inference" had to coexist at frontier scale on the same hardware.

## Engineering Tradeoffs

| Decision | Gained | Gave up |
|---|---|---|
| Three stages (SFT → RM → PPO) | Quality + labeling efficiency | Pipeline complexity; stage failures compound |
| Pairwise preferences instead of direct scores | Cheaper, more reliable labeling | Loses cardinal info; only relative |
| KL penalty to SFT | Prevents reward hacking / mode collapse | Limits how far policy can move |
| PPO (vs simpler RL) | Stable updates; well-studied | Value network doubles memory; rollout/train coordination complex |
| Reward model as scalar | Single objective, easy to optimize | RM limitations become policy limitations (Goodhart's law) |
| Human raters | Grounded in preferences | Slow, expensive, inconsistent |

## Experiments & Results

- **1.3B InstructGPT > 175B GPT-3** on human-preference eval for instruction-following — by a large margin (~85% win rate). The headline result: alignment beats scale at this task.
- Improvements across truthfulness (TruthfulQA), reduced toxicity, better instruction-following.
- Some cost: slight degradation on raw NLP benchmarks ("alignment tax"), mostly recoverable by mixing pretraining data back into RL.
- Widely replicated since: ChatGPT (Nov 2022) was InstructGPT applied at product scale; every major post-trained LLM until ~2023 used essentially the same recipe.

## Reproducibility Notes

- The paper describes the method clearly; the original code and reward data are not released.
- Open implementations: **TRL** (HuggingFace), **DeepSpeed-Chat**, **OpenRLHF**, **verl** (Bytedance, 2024+). Each has nuances but all follow the same three-stage recipe.
- **The expensive part is not code, it's the preference data** — quality pairwise comparisons at scale require a well-run labeling operation. Open datasets (Anthropic HH-RLHF, OpenAssistant, UltraFeedback) have enabled academic replication.
- PPO at scale is still brittle. Seeds, hyperparameters, and RM quality all affect outcomes significantly.

## Commentary

InstructGPT is the **single most important post-training paper in LLM history**. It's not just that the method works; it's that it **reframed what a language model product is**. Before: an LLM is a text completer, and you prompt-engineer it. After: an LLM is a system explicitly aligned to human preferences, and the preferences are part of the artifact.

The infra consequences ripple outward. RLHF forced every serious LLM shop to build or adopt:

- **Distributed rollout infrastructure** — generate completions at scale during training. This is why vLLM / SGLang matter for training, not just serving.
- **Reward model serving** — a second inference path running inline.
- **KL / ref-policy machinery** — a third inference path for the frozen reference.
- **Memory-efficient PPO** — four models co-resident is untenable without sharding, offloading, or the successor methods below.

This operational burden is why the field has spent two years looking for simpler alternatives. **DPO** (next entry) removed the explicit reward model. **GRPO** (see DeepSeek R1) removed the value network. The entire post-RLHF algorithmic landscape is optimizing the infra cost of what InstructGPT started.

Also worth naming explicitly: **the reward model is the load-bearing part, and it's the weakest link**. An RM trained on a few hundred thousand pairs cannot capture the full space of human preferences; policies trained against it learn to exploit its blind spots (Goodhart). Much of the alignment research after 2022 is about making this less bad: process reward models, constitutional AI, AI feedback (RLAIF), rule-based rewards (R1). None of them fully solves the problem, but the direction is away from single-shot scalar RMs.

For anyone doing post-training in 2026: RLHF-with-PPO is mostly a baseline now, not the default. DPO for small-scale / simple preference tuning, GRPO for reasoning tasks with verifiable rewards, full PPO only for large-scale proprietary products where the operational investment is justified. Read InstructGPT for the conceptual foundation; use its successors in practice.

## References

- [1] Ouyang et al. _Training Language Models to Follow Instructions with Human Feedback._ arXiv:2203.02155, 2022.
- [2] Stiennon et al. _Learning to Summarize with Human Feedback._ NeurIPS '20 / arXiv:2009.01325.
- [3] Christiano et al. _Deep Reinforcement Learning from Human Preferences._ NeurIPS '17 / arXiv:1706.03741.
- [4] Schulman et al. _Proximal Policy Optimization Algorithms._ arXiv:1707.06347, 2017.
- [5] Bai et al. _Training a Helpful and Harmless Assistant with RLHF._ arXiv:2204.05862, 2022. (Anthropic's version)
