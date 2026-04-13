# Speculative Decoding

- **Authors / Org**: Leviathan et al. (Google, ICML '23) and Chen et al. (DeepMind, 2023, concurrent). Later: Medusa (Princeton/Together), EAGLE, Lookahead (Meta).
- **Published**: 2022-11 — 2024
- **Links**: [Leviathan (arXiv:2211.17192)](https://arxiv.org/abs/2211.17192) · [Chen (arXiv:2302.01318)](https://arxiv.org/abs/2302.01318) · [Medusa](https://arxiv.org/abs/2401.10774) · [EAGLE](https://arxiv.org/abs/2401.15077)

## TL;DR

LLM decoding is memory-bandwidth-bound: each step reads the full model weights from HBM to produce one token. Speculative decoding breaks this by using a **cheap draft model** to propose `K` tokens in sequence, then having the **expensive target model verify all `K` in a single forward pass**. A principled rejection-sampling scheme (derived in both papers independently) guarantees the accepted tokens are **exactly distributed** as if they came from the target model — no quality loss. Typical speedups: 2–3× for greedy, similar for temperature sampling. The technique is now standard in vLLM, TensorRT-LLM, and every serious inference stack; downstream variants (Medusa, EAGLE) remove the separate draft model entirely.

## Context & Motivation

Decode-phase transformer inference is dominated by **reading the weight tensors from HBM**, not by compute. One forward produces one token; arithmetic intensity is tiny. But a forward pass over `K` tokens costs roughly the same as one token in wall time — the weights were going to be read anyway, and tensor cores have enormous headroom.

The question: can we turn that headroom into speedup when we don't have a genuinely batched workload? Yes — generate `K` candidate tokens speculatively, then verify them in one batched forward pass. If most candidates are accepted, effective tokens-per-step rises.

## Core Method

### Draft + verify

- **Draft model** `q(x_t | x_<t)`: small, fast; could be a 1B model distilled from a 70B target, or simply a shorter version.
- **Target model** `p(x_t | x_<t)`: the full model whose distribution we want to sample from.

Algorithm at each outer step:

1. Draft model greedily or stochastically generates `K` tokens `x_1, ..., x_K`.
2. Target model does **one forward pass** over the prefix + `K` drafted tokens, producing `p(· | x_<t+i)` for `i = 0, ..., K`.
3. **Verify** each drafted token via rejection sampling:
   - Accept `x_i` with probability `min(1, p_i(x_i) / q_i(x_i))`.
   - On rejection, sample a replacement token from the residual distribution `(p_i − q_i)_+ / norm`, and stop.
4. If all `K` were accepted, sample one extra token from `p_{K+1}` (free, since the target forward already produced it).

Net effect per outer step: you accept between 1 and `K+1` tokens, in one target-model forward. The mathematics guarantees the marginal distribution of accepted tokens equals the target's. Works for greedy, temperature, top-k, top-p; sampler-specific.

### Acceptance rate governs speedup

Speedup is bounded by:

- How often the draft agrees with the target (acceptance rate `α`).
- The cost ratio `c = time(draft) / time(target)`.

Expected speedup ≈ `(1 − α^(K+1)) / ((1 − α) · (K·c + 1))`. Tuning: pick `K` just high enough that additional speculation isn't wasted; typical `K = 4–8`.

### Dropping the separate draft model

- **Medusa**: attach extra LM heads to the target model that predict tokens at positions `+2, +3, +4` directly. Draft is parallel, not autoregressive. No separate weights; minor fine-tuning cost.
- **EAGLE**: predict features of future tokens (not tokens themselves), then feed those features through the target's LM head for verification. Higher acceptance at similar cost; currently leading method on quality-preserving throughput.
- **Lookahead decoding**: uses Jacobi iteration to generate draft trajectories from the target itself, no draft model or extra training. Weaker but zero-setup.

### Batching and serving

In production serving, speculative decoding interacts non-trivially with batching: different requests accept different numbers of tokens per outer step, breaking alignment. vLLM and TensorRT-LLM handle this via padding + masked verification; the wins survive at production batch sizes but are smaller than in single-stream benchmarks.

## Engineering Tradeoffs

| Decision | Gained | Gave up |
|---|---|---|
| Separate draft model | Simple, model-agnostic | Need to train/choose a draft; extra memory; alignment with target's distribution affects `α` |
| Medusa / EAGLE (self-draft) | No extra model; higher `α` | Fine-tuning required; ties the method to specific model weights |
| Lookahead (no training) | Zero setup | Lower `α`, smaller speedup |
| Exact rejection sampling | No quality loss, mathematically clean | More complex than just accepting top-1; small per-step overhead |
| Large `K` | Higher peak speedup | Wasted compute when `α` is low; verification cost grows |
| In batched serving | Still useful at real batch sizes | Smaller effective speedup; scheduler complexity |

## Experiments & Results

- Original Leviathan: 2–3× wall-clock speedup on T5-XXL and PaLM variants.
- Chen: similar gains on Chinchilla.
- Medusa: 2–3× on Vicuna / Llama without draft model.
- EAGLE-2: up to ~4× on Llama 2 / 3 with preserved distributions; current state of the art for lossless decode acceleration.

All methods preserve the target's output distribution exactly (with the specified sampler); this is the key property that distinguishes them from quantization or distillation.

## Reproducibility Notes

- vLLM, TensorRT-LLM, TGI all support speculative decoding in production.
- Medusa / EAGLE publish weights and heads for popular base models.
- Correctness testing is subtle — verifying the distribution is preserved requires careful statistical tests, not just spot-checking outputs.

## Commentary

Speculative decoding is one of the few inference tricks that is **genuinely free** — no quality loss, no extra parameters at inference (for self-draft variants), real wall-clock wins. The conceptual move is lovely: **decoding is bandwidth-bound, so pay compute to buy bandwidth**. The line of work from vanilla speculative → Medusa → EAGLE is a case study in iterating on the acceptance-rate / draft-cost tradeoff. The open frontier is **tree-based verification** (verify branching draft trees, not linear chains), integration with **disaggregated prefill/decode** architectures, and how speculative decoding composes with very long context (where the target forward gets expensive again). For anyone serving LLMs, this is the first optimization to reach for after PagedAttention — the gains are immediate and stack with everything else.

## References

- [1] Leviathan et al. _Fast Inference from Transformers via Speculative Decoding._ ICML '23 / arXiv:2211.17192.
- [2] Chen et al. _Accelerating Large Language Model Decoding with Speculative Sampling._ arXiv:2302.01318, 2023.
- [3] Cai et al. _Medusa: Simple LLM Inference Acceleration Framework with Multiple Decoding Heads._ arXiv:2401.10774, 2024.
- [4] Li et al. _EAGLE: Speculative Sampling Requires Rethinking Feature Uncertainty._ arXiv:2401.15077, 2024.
- [5] Fu et al. _Breaking the Sequential Dependency of LLM Inference Using Lookahead Decoding._ ICML '24.
