# GRPO — Group Relative Policy Optimization

- **Authors / Org**: Shao et al., DeepSeek-AI
- **Published**: 2024-02 (arXiv:2402.03300); central to DeepSeek-R1 (2025-01)
- **Links**: [DeepSeekMath paper](https://arxiv.org/abs/2402.03300) · [DeepSeek-R1 tech report](https://arxiv.org/abs/2501.12948) · [code (DeepSeek-R1)](https://github.com/deepseek-ai/DeepSeek-R1)

## TL;DR

Reinforcement learning from human feedback via PPO requires four co-resident models: policy, reference policy, critic (value network), and reward model — totaling 4× the memory of a single inference run and demanding complex, brittle rollout infrastructure. GRPO (Group Relative Policy Optimization) eliminates the critic entirely. Instead of learning a value function that estimates expected future reward, GRPO samples a **group of responses** for each prompt, scores them all, and uses the **within-group rank** of each response's reward as its advantage estimate. No critic training, no critic memory, no value function bootstrapping. Combined with rule-based rewards that eliminate the reward model (as in DeepSeek-R1-Zero), GRPO reduces the training footprint to two models (policy + reference) while achieving state-of-the-art performance on mathematical reasoning. GRPO is the training algorithm behind DeepSeekMath, DeepSeek-R1, and most of the open-source reasoning model wave of 2025.

## Context & Motivation

### PPO's memory and infrastructure burden

PPO (Proximal Policy Optimization, Schulman et al. 2017) is the dominant RL algorithm for LLM post-training. A standard PPO-RLHF setup requires:

1. **Policy model (π_θ)**: the model being trained. Updated each gradient step.
2. **Reference policy (π_ref)**: a frozen copy of the initial SFT model. Used to compute the KL divergence penalty that prevents the policy from deviating too far from the SFT baseline.
3. **Reward model (RM)**: trained separately on human preference data; scores completions during rollouts.
4. **Critic / value network (V_φ)**: a second neural network (typically the same size as the policy, with a value head) that estimates V(s) — the expected cumulative reward from state s. Required to compute the PPO advantage A(s,a) = Q(s,a) − V(s).

With a 7B policy, this means 4 models × ~14 GB each ≈ 56 GB minimum, before activations and optimizer states. For 70B-scale models, the memory requirement becomes staggering. Beyond memory, the rollout phase — generating samples from the policy, scoring them with the RM, running the critic forward pass, computing advantages, and then running the policy backward pass — involves five distinct neural network operations per training step, with careful sequencing to avoid stale values.

The critic is particularly problematic. Training a value network alongside a policy that is rapidly changing is notoriously unstable (the "moving target" problem). Value function estimation error compounds into noisy advantage estimates, which destabilize policy gradients. Much of PPO's reputation for being brittle comes from critic instability.

### The insight: group-relative advantage without a critic

GRPO asks: do we actually need a learned value function to estimate advantage? The advantage function exists to reduce variance in policy gradient estimates — it tells the policy "this action was better (or worse) than average, by how much." 

A simpler alternative: instead of learning V(s) to estimate the baseline, just **sample multiple completions for the same prompt** and use the group's reward distribution as a natural baseline. The advantage of response i is how much better its reward is than the average reward in the group. No critic needed.

This idea has precedent in contextual bandits and in REINFORCE with baseline, but applying it cleanly to LLM post-training with PPO-style clipping and KL regularization is GRPO's contribution.

## Core Method

### Algorithm

For each training step:

1. **Sample G responses** from the current policy for each prompt q in the batch:
   ```
   {o₁, o₂, ..., o_G} ~ π_θ_old(· | q)
   ```

2. **Score each response** with a reward signal {r₁, r₂, ..., r_G}.

3. **Compute group-relative advantages**:
   ```
   Â_i = (r_i − mean({r₁,...,r_G})) / std({r₁,...,r_G})
   ```
   This is a simple z-score normalization within the group. The advantage is positive for responses that score above the group mean, negative for below.

4. **Update the policy** using a PPO-clip-style objective with KL regularization:

```
L_GRPO(θ) = E_{q, {o_i}} [
    (1/G) Σ_i  min(ρ_i · Â_i,  clip(ρ_i, 1-ε, 1+ε) · Â_i)
] − β · KL(π_θ || π_ref)
```

where ρ_i = π_θ(o_i | q) / π_θ_old(o_i | q) is the importance ratio (probability under current policy vs the policy that generated the sample), ε is the clip ratio (typically 0.2), and β is the KL penalty coefficient.

The KL term keeps the policy close to the reference (frozen SFT model), preventing reward hacking while allowing meaningful updates. In practice the KL can be estimated token-by-token using the reference model's log probabilities; this adds one forward pass of the reference model per training step (unavoidable — this is why two models remain even when the critic and RM are eliminated).

### Why it works: variance reduction via group normalization

The standard REINFORCE gradient estimate suffers from high variance because the raw reward signal is noisy and un-centered. PPO with a critic reduces variance by subtracting the value baseline V(s), but the critic itself introduces bias and instability.

GRPO's group-relative baseline achieves variance reduction without a learned value function:
- By normalizing within the group, the advantage estimates have zero mean and unit variance by construction.
- The group serves as a local, online estimate of the prompt-conditional reward baseline — much like a learned critic, but without gradient instability.
- Larger group size G reduces estimation variance (more samples → more stable mean/std), at the cost of more rollout compute.

### Rule-based rewards: eliminating the reward model

In DeepSeek-R1-Zero, GRPO is combined with **verifiable, rule-based rewards** rather than a learned reward model:

- **Correctness reward**: 1 if the final answer is correct (verified by symbolic comparison or code execution), 0 otherwise.
- **Format reward**: small bonus for following the expected output format (e.g., wrapping reasoning in `<think>` tags, answer in `<answer>` tags).

Rule-based rewards are binary and deterministic — no ambiguity, no distribution shift, no RM training required. This eliminates the fourth model entirely, reducing the training footprint to policy + reference. The tradeoff: rule-based rewards only work for verifiable domains (math, code, logic puzzles). Open-ended tasks (writing, summarization, general helpfulness) still require a learned RM or human feedback.

## Engineering Tradeoffs

| Decision | Gained | Gave up |
|---|---|---|
| GRPO vs PPO (no critic) | 2 models instead of 4 in memory; eliminates critic instability; simpler training loop | Critic provides richer per-token, per-state value estimates; GRPO advantage is response-level only (no sub-response credit assignment) |
| Group size G (4 → 64) | Larger G: lower variance advantage estimates; policy gets clearer signal | Larger G: G× more rollout compute per prompt; memory for G completions simultaneously |
| Rule-based vs learned reward | Stable, calibrated, no RM training; no reward hacking from RM distributional shift | Hard constraint to verifiable domains; can't reward nuanced quality, tone, or open-ended creativity |
| KL penalty β (0 → 0.1) | Prevents reward hacking; anchors to SFT behavior; training stability | Limits how far policy can drift from reference; may suppress novel strategies the policy should explore |
| GRPO vs DPO | Online RL: can use non-differentiable rewards, rule-based signals, live exploration | DPO is simpler (offline, no rollouts); GRPO requires rollout infrastructure and is more compute-intensive |

## Experiments & Results

**DeepSeekMath (Shao et al., 2024):**
- Base: DeepSeek-Coder-v1.5-7B fine-tuned on 120B math tokens.
- GRPO applied on top of SFT checkpoint, using a reward model trained on math problem/solution pairs.
- Results: 51.7% on MATH benchmark (vs 34.9% for the SFT-only baseline), 88.2% on GSM8K. Outperforms all models of comparable size including proprietary models (Gemini Ultra at 53.2% on MATH for comparison, though different eval settings).
- GRPO provides ~15% absolute improvement over SFT-only on MATH, compared to ~12% for PPO — while using half the memory and simpler code.

**DeepSeek-R1-Zero (DeepSeek-AI, 2025):**
- Policy: DeepSeek-V3-Base (671B MoE). Training: GRPO with rule-based correctness + format rewards only — no RM, no SFT warm-up.
- Emergent behavior: the model spontaneously develops extended chain-of-thought reasoning, self-verification, and backtracking strategies — purely from RL with binary rewards. No human-curated chain-of-thought data used.
- AIME 2024: 71.0% pass@1 (vs OpenAI o1 at 79.2%). MATH-500: 97.3%.
- R1-Zero demonstrates that GRPO + rule-based rewards can teach LLMs to reason without any labeled reasoning data — a qualitatively different training paradigm.

**DeepSeek-R1 (DeepSeek-AI, 2025):**
- Adds cold-start SFT data and iterative RL rounds on top of R1-Zero's GRPO backbone. AIME 2024: 79.8% (matches OpenAI o1). Distilled variants (1.5B–70B) from R1 achieve strong results, confirming the reasoning capability transfers via distillation.

## Reproducibility Notes

The original GRPO training code for DeepSeek-R1 was released at https://github.com/deepseek-ai/DeepSeek-R1 (training scripts and model weights). The community has also produced several high-quality open re-implementations:

- **TRL (HuggingFace)**: `GRPOTrainer` added in v0.13 (2025). Works with any model on the HuggingFace hub; reward can be a callable function (enabling rule-based rewards). `pip install trl` and use `GRPOConfig`.
- **OpenRLHF**: full-featured RLHF framework with GRPO support, designed for multi-node training. Handles vLLM-based rollout generation for large models.
- **veRL (Volcano Engine RL)**: ByteDance's RL framework, also supports GRPO; optimized for large-scale distributed training.

Key implementation details:
- **Rollout generation is the bottleneck**: for G=8 and batch size 64, you generate 512 completions per step. Use vLLM or TGI for fast rollouts; do not use HuggingFace `generate()` naively.
- **Group size G**: G=4 is a minimum; G=8 gives a good variance-compute tradeoff; G=16–64 used for very noisy reward signals (e.g., process reward models).
- **KL estimation**: the reference model needs to score every token of every rollout. This is embarrassingly parallel but doubles memory — keep reference on separate GPU(s) or use CPU offload.
- **Reward normalization**: normalize rewards per group (as described) and also per batch to prevent scale drift across training.

## Commentary

GRPO is the clearest recent example of the principle: **the critic in RL is a means to an end, not the end itself**. The critic exists to reduce variance in policy gradient estimates. If you have a better, cheaper way to reduce variance — like sampling multiple responses for the same prompt — you should use that instead.

The rule-based reward insight (R1-Zero) is arguably more significant than the algorithmic simplification. The traditional RLHF story was: you need human preferences, you need a reward model, you need to manage RM distributional shift and reward hacking. R1-Zero showed that for verifiable domains, you need none of that — binary correctness is a sufficient training signal to produce world-class reasoning. This reframes RLHF as one point in a design space, not an inevitable component.

The practical consequence: GRPO with rule-based rewards is now the standard recipe for reasoning model training. The open-source ecosystem reproduced DeepSeek-R1's results at 7B and 32B scale within weeks of the paper release (Qwen-2.5-Math-GRPO, Sky-T1, STILL-3, etc.), demonstrating that the technique is robust and accessible.

Remaining open questions:

- **Credit assignment**: GRPO's advantage is response-level — it rewards or penalizes the entire completion uniformly. For long chain-of-thought responses, the mistake might be in step 5 of 20, but all tokens get the same gradient signal. Process Reward Models (PRMs) that give step-level credit exist but are expensive to train and haven't yet cleanly integrated with GRPO.
- **Non-verifiable tasks**: for open-ended generation (creative writing, nuanced instruction following), rule-based rewards don't exist. GRPO still needs a learned RM here, reintroducing one of the models it eliminated.
- **Exploration-exploitation in large G**: very large group sizes improve variance reduction but also push the policy toward the average of the group rather than toward the best response. Optimal G depends on the reward signal's noise level and the model's current capability — not a universal constant.

For practitioners building reasoning systems: start with GRPO + rule-based rewards if your task is verifiable. If you need open-ended quality, add a lightweight RM (can be 3B–7B scale, much smaller than the policy) and keep GRPO for the optimizer. PPO is defensible only if you have the infrastructure investment already amortized.

## References

- [1] Shao et al. _DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models._ arXiv:2402.03300, 2024.
- [2] DeepSeek-AI. _DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning._ arXiv:2501.12948, 2025.
- [3] Schulman et al. _Proximal Policy Optimization Algorithms._ arXiv:1707.06347, 2017.
- [4] Rafailov et al. _Direct Preference Optimization._ NeurIPS '23 / arXiv:2305.18290.
- [5] TRL GRPOTrainer. https://github.com/huggingface/trl
- [6] OpenRLHF. https://github.com/OpenRLHF/OpenRLHF
- [7] Lightman et al. _Let's Verify Step by Step (PRM800K)._ arXiv:2305.20050, 2023. (Process reward models)
