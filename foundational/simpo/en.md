# SimPO — Simple Preference Optimization with a Reference-Free Reward

- **Authors / Org**: Yu Meng, Mengzhou Xia, Danqi Chen — Princeton University
- **Published**: 2024-05 (NeurIPS '24)
- **Links**: [paper (arXiv:2405.14734)](https://arxiv.org/abs/2405.14734) · [code](https://github.com/princeton-nlp/SimPO) · [blog](https://huggingface.co/papers/2405.14734)

## TL;DR

SimPO removes the reference model from DPO entirely and replaces the per-token log-ratio reward with the **average log-probability of the response under the current policy** — a length-normalized, reference-free reward. A target reward margin γ enforces a minimum gap between chosen and rejected rewards, preventing reward collapse. SimPO outperforms DPO and IPO on AlpacaEval 2 and Arena-Hard across Llama-3 and Mistral backbones, while using roughly 10% less peak GPU memory (no reference model forward pass) and requiring fewer than five meaningful lines of change from a standard DPO implementation.

## Context & Motivation

DPO simplified the RLHF pipeline dramatically: one supervised loss, no rollouts, no explicit reward model. But DPO introduced two engineering burdens of its own that compound badly at scale.

**Burden 1 — The reference model.** DPO requires computing `log π_ref(y|x)` for every training sample. This means keeping a frozen copy of the base model resident in memory for the entire training run. At 70B scale on BF16, that's roughly 140 GB just for the reference — over half a DGX H100 node. The reference is not updated, not learned, purely overhead. The only role it plays is providing a KL anchor to prevent the policy from drifting too far from SFT.

**Burden 2 — Length bias.** The DPO reward for a response y is proportional to the **sum** of per-token log-ratio terms:

```
r_DPO(x, y) = β · Σ_{t=1}^{|y|} [log π_θ(y_t | x, y_{<t}) − log π_ref(y_t | x, y_{<t})]
```

This is a sum over `|y|` terms. A 200-token mediocre response has roughly 2× the reward magnitude of a 100-token excellent response, simply by virtue of being longer. DPO training consequently biases the policy toward verbosity — a known empirical failure mode that produces padded, overly hedged outputs.

Prior attempts to patch the length problem (explicit length penalties, response filtering by length bracket) are ad-hoc. SimPO redesigns the reward from scratch to eliminate both problems simultaneously.

The alignment of the field at this point: DPO sits between PPO-RLHF (powerful but heavy) and GRPO (online, rollout-based, rule-reward-compatible). SimPO pushes the offline preference optimization direction further — lighter memory footprint, no reference, same data format.

## Core Method

### The SimPO Reward

SimPO replaces the DPO log-ratio reward with the **average log-probability under the policy alone**:

```
r_SimPO(x, y) = (β / |y|) · Σ_{t=1}^{|y|} log π_θ(y_t | x, y_{<t})
              = (β / |y|) · log π_θ(y | x)
```

This is the mean per-token log-likelihood of the response under the current model — equivalently, the negative of the per-token cross-entropy loss on y given x. No reference model, no log-ratio, no subtraction. Division by `|y|` eliminates the length bias: a 200-token response and a 100-token response are evaluated on the same per-token scale.

The average log-probability has a concrete semantic: it measures how fluent and coherent the model finds the response on a per-token basis. High average log-prob means the model assigns high confidence uniformly across the response, not just at a few easy tokens.

### The SimPO Objective with Target Margin γ

The full training loss is:

```
L_SimPO = −E_{(x, y_w, y_l) ~ D} [ log σ( r_SimPO(x, y_w) − r_SimPO(x, y_l) − γ ) ]
```

where `y_w` is the chosen response, `y_l` is the rejected response, and γ > 0 is the **target reward margin**. The sigmoid σ ensures probabilities stay in [0, 1].

Without γ, the optimal solution of the loss is any policy where `r(y_w) > r(y_l)` by an infinitesimal margin — the chosen and rejected rewards could both drift to −∞ as long as the chosen is marginally higher. γ prevents this by demanding a strict minimum gap. In practice γ = 0.5–1.5; the authors use γ = 0.5 for most tasks.

**Why this works without a reference:** DPO's reference model provides an implicit floor — the policy can't get too far from `π_ref` without incurring high KL cost. SimPO doesn't have that floor, but the target margin γ and the length normalization together provide sufficient implicit regularization. The average log-probability of a coherent response converges to a stable value as training proceeds; margin γ keeps chosen responses well-separated from rejected ones without needing to anchor to any fixed reference distribution.

### Implementation Delta from DPO

The code change from a TRL DPOTrainer is minimal:

1. Remove the reference model initialization and the reference log-prob forward pass.
2. Compute `log π_θ(y_w|x)` and `log π_θ(y_l|x)` from the policy's forward pass (this was already done for the policy side in DPO).
3. Divide each by the response length to get average log-prob.
4. Subtract γ from the margin before passing to `log σ`.
5. Delete the `log π_ref` terms from the loss computation.

No new data format, no new infrastructure, no new hyperparameter tuning surfaces beyond γ (and γ is straightforward — tune on a validation win rate).

## Engineering Tradeoffs

| Decision | Gained | Gave up |
|---|---|---|
| No reference model | ~50% fewer GPU-hours for parameter loading; simpler setup; single-model forward pass per step | Reference model provides explicit KL regularization; without it, policy drift is harder to bound theoretically |
| Length-normalized reward | Eliminates verbosity bias; average log-prob is a stable, interpretable signal per token | Per-token KL ratio in DPO is a richer signal that encodes where the policy diverges from SFT; SimPO loses this locality |
| Target margin γ | Prevents reward collapse and degenerate solutions where both chosen/rejected drift toward −∞ | Introduces one more hyperparameter; γ too large causes training instability as the sigmoid saturates |
| Offline preference data | No rollout infrastructure needed; compatible with existing DPO datasets | Cannot incorporate online or rule-based rewards (verifiable tasks); GRPO is superior in that regime |
| Simpler reward signal | Lower implementation complexity; easier to audit and debug | Weaker theoretical grounding for KL control compared to DPO; harder to reason about policy divergence bounds |

## Experiments & Results

**Evaluation benchmarks:** AlpacaEval 2 (LC win rate vs GPT-4-turbo) and Arena-Hard (GPT-4 judge). Both are head-to-head win-rate metrics, less gameable by verbosity than length-correlated metrics.

**Llama-3-8B-Instruct as base:**
- SimPO: 44.7% LC win rate on AlpacaEval 2
- DPO: 40.5% (same base, same preference data — UltraFeedback-binarized)
- IPO: 41.2%
- Gain of ~4 points with no additional data, no architectural change, less memory.

**Mistral-7B-Instruct-v0.2 as base:**
- SimPO: 72.4% win rate on Arena-Hard
- DPO: 67.3%
- State-of-the-art among 7B models on Arena-Hard at time of publication.

**Ablations (Llama-3-8B):**
- Remove γ (set to 0): −2.3 points on AlpacaEval 2.
- Remove length normalization (use sum instead of mean): −4.8 points — confirming length normalization carries most of the empirical gain.
- Remove both: −6.1 points, worse than vanilla DPO.

**Memory profile:** At batch size 8 with a 7B model in BF16, removing the reference model saves approximately 14 GB of GPU VRAM (7B × 2 bytes × overhead). This is ~10–15% of typical peak memory at that batch size, enabling one additional gradient accumulation step or a larger per-device batch.

**Training speed:** ~1.15× faster per step (one fewer forward pass per batch), not accounting for reduced data movement overhead from the simpler loss.

## Reproducibility Notes

The full training code is at [github.com/princeton-nlp/SimPO](https://github.com/princeton-nlp/SimPO), built on HuggingFace TRL with a lightweight modification to `DPOTrainer`. Key configuration details for replication:

- **Data**: UltraFeedback-binarized (standard DPO benchmark dataset, ~60K preference pairs). No special data pipeline needed.
- **Hardware**: 4× A100-40GB for 7B models; 8× A100-80GB for 13B and 70B.
- **β = 2.0**: Same conceptual role as DPO's β (controls reward scale); higher than DPO's typical 0.1–0.5 because there's no reference model denominator pulling the signal down.
- **γ = 0.5**: Default for instruction-following tasks; try γ ∈ {0.5, 1.0, 1.5} and select on validation win rate.
- **Learning rate**: 5e-7, cosine schedule, 1 epoch on UltraFeedback-binarized. Lower LR than SFT because preference data is noisier.
- **Max sequence length**: 2048 tokens (prompt + response).

Failure modes to watch: if γ is too large (> 2.0), the sigmoid saturates and gradients vanish — you'll see the loss stall near zero without any improvement in win rate. If β is too small relative to γ, the reward gap can't physically reach γ and training becomes meaningless. Rule of thumb: ensure β · (expected avg log-prob gap) > γ.

## Commentary

SimPO is a refinement, not a revolution — but the right kind of refinement. DPO had two concrete engineering problems; SimPO identifies both precisely, fixes both with principled changes (length normalization, target margin), and demonstrates empirical gains. This is how good systems work should look: diagnose, redesign, measure.

The length-normalized reward deserves more attention than the paper's headline numbers might suggest. The average log-probability as a reward signal is deeply natural — it's exactly what language model pretraining optimizes (negative cross-entropy), just applied at inference time to preference ranking. Alignment researchers had been treating DPO's per-token log-ratio as if it were fundamental, when in fact it was an artifact of the derivation path from the RLHF objective. SimPO shows you can cut the derivation path and design a reward from first principles.

The **reference-free** direction matters increasingly as models scale. At 70B, the reference model's memory cost starts to dominate scheduling decisions: you can't fit reference + policy + optimizer states in 8× H100 without careful memory offloading. SimPO's elimination of the reference isn't just a 10% savings — it's a qualitative simplification that unlocks straightforward single-node training for models that would otherwise require careful memory engineering.

Where SimPO doesn't fully replace DPO:

- **KL control**: DPO provides explicit, theoretically grounded control over how far the policy drifts from SFT. SimPO's target margin γ provides implicit regularization, but without the same formal guarantees. For safety-critical applications where you need to certify bounded policy change, DPO's reference anchor is more defensible.
- **Verifiable tasks**: Neither SimPO nor DPO handles rule-based rewards (math, code). GRPO with online rollouts is the right tool there. SimPO has no advantage over DPO in this regime — both fail at the same point.
- **Data quality sensitivity**: SimPO shares DPO's core weakness: both are offline methods sensitive to the distributional quality of the preference dataset. SimPO's better empirics on standard benchmarks may not transfer to narrow domain fine-tuning where UltraFeedback-style data quality assumptions break down.

SimPO's position in the post-training taxonomy as of 2026: **DPO simplified RLHF; SimPO simplifies DPO; GRPO goes orthogonal** (online RL for verifiable tasks). For general instruction following where you have offline preference pairs and no rule-based verifier, SimPO is the pragmatic default. You get DPO-level quality, lower memory overhead, and code that's trivially easier to maintain.

## References

- [1] Meng et al. _SimPO: Simple Preference Optimization with a Reference-Free Reward._ NeurIPS '24 / arXiv:2405.14734.
- [2] Rafailov et al. _Direct Preference Optimization: Your Language Model is Secretly a Reward Model._ NeurIPS '23 / arXiv:2305.18290.
- [3] Azar et al. _A General Theoretical Paradigm to Understand Learning from Human Feedback (IPO)._ arXiv:2310.12036, 2024.
- [4] Ethayarajh et al. _KTO: Model Alignment as Prospect Theoretic Optimization._ arXiv:2402.01306, 2024.
- [5] Hong et al. _ORPO: Monolithic Preference Optimization without Reference Model._ arXiv:2403.07691, 2024.
- [6] Schulman et al. _Proximal Policy Optimization Algorithms._ arXiv:1707.06347, 2017.
- [7] HuggingFace TRL library: https://github.com/huggingface/trl
