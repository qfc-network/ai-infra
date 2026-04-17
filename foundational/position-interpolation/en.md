# Position Interpolation — YaRN, LongRoPE, and NTK-aware Scaling

- **Authors / Org**: Chen et al. (Meta) · Peng et al. (Eleuther AI / Microsoft) · Ding et al. (Microsoft) · kaiokendev (NTK-aware, independent)
- **Published**: PI: 2023-06 (arXiv:2306.15595) · YaRN: 2023-09 (arXiv:2309.00071) · LongRoPE: 2024-02 (arXiv:2402.13753) · NTK-aware: 2023-07 (blog post)
- **Links**: [PI paper](https://arxiv.org/abs/2306.15595) · [YaRN paper](https://arxiv.org/abs/2309.00071) · [LongRoPE paper](https://arxiv.org/abs/2402.13753)

## TL;DR

RoPE (Rotary Position Embeddings) encodes position by rotating query/key vectors at frequencies that were calibrated during training. Beyond the training context length, these frequencies produce rotations the model has never seen, causing perplexity to spike and coherence to collapse. Position Interpolation (PI) addresses this by linearly rescaling all position indices so they fit within the trained range. NTK-aware scaling improves on PI by changing the RoPE base frequency instead of the position indices, leaving high-frequency (local) dimensions less distorted — and crucially, it works zero-shot without any fine-tuning. YaRN further refines this with non-uniform per-frequency scaling and an attention temperature correction, achieving near-lossless perplexity at 128k context with only a few hundred fine-tuning steps. LongRoPE pushes the state of the art to 2M context via evolutionary search over per-frequency scale factors.

## Context & Motivation

### The RoPE extrapolation problem

As established in the [RoPE entry](../rope/en.md), RoPE encodes position m by rotating the d-dimensional query/key vector at d/2 frequencies:

```
θ_i = base^{-2i/d},   i = 0, 1, ..., d/2 - 1   (base = 10,000 by default)
```

At training context length L_train, the model sees rotation angles m · θ_i for m ∈ [0, L_train). All rotation values in this range are thoroughly covered by the training distribution.

At inference with m > L_train:
- **High-frequency dimensions** (small θ_i, complete many full rotations over L_train): the angle m · θ_i is only modestly outside the training range — these dimensions extrapolate reasonably well.
- **Low-frequency dimensions** (large θ_i, complete less than one full rotation over L_train): the angle m · θ_i at m = 2 × L_train is 2× the maximum training angle — these dimensions are catastrophically out-of-distribution.

The result: perplexity at 2× the training length can increase by 10–100×, and generative coherence collapses. This is not a theoretical concern — it was empirically observed when practitioners tried to serve longer contexts than the training length.

### Why not just train longer?

Training at 128k context from scratch requires:
1. **KV cache memory**: at 128k tokens, KV cache for a 7B model with 32 layers is ~32 GB per sequence — not feasible per-request at training batch sizes.
2. **Attention compute**: attention is O(L²) in sequence length; 128k context is 32× the compute of 4k context per token.
3. **Difficulty of curating long-context data**: documents of 128k+ tokens are rare; the training distribution is dominated by short sequences regardless.

The practical solution: train at a manageable context length (4k–8k), then extend via post-training techniques that require only a small number of fine-tuning steps, if any.

## Linear Position Interpolation (Chen et al., 2023)

### The idea

Instead of using position indices 0, 1, ..., m-1 for a sequence of length m > L_train, scale them down:

```
m → m · (L_train / L_target)
```

This maps positions [0, L_target) linearly into [0, L_train), ensuring all rotations seen at inference lie within the training distribution. For L_target = 2 × L_train, each position index is halved.

**What this costs**: the minimum distinguishable position difference doubles. At L_train = 4096 and L_target = 8192, positions that were 1 step apart now appear (to the model) as 0.5 steps apart — below the resolution the model was trained to distinguish. This is the "position confusion" or "interpolation blur" problem.

**Fine-tuning requirement**: with only position interpolation (no fine-tuning), perplexity improves compared to naive extrapolation but remains degraded. Chen et al. found that 1,000 fine-tuning steps on long-context documents are sufficient to adapt the model. With fine-tuning, the model learns to distinguish positions at the new finer granularity.

**Results**: LLaMA 7B extended from 2k to 32k context with 1k fine-tuning steps. Passkey retrieval accuracy at 32k: ~100%. Long-document benchmarks: competitive with models trained natively at longer context.

## NTK-aware Interpolation (kaiokendev, 2023)

### The neural tangent kernel insight

A key observation from the neural tangent kernel (NTK) literature: when you compress a signal (by scaling position indices), you lose the ability to represent high-frequency components. Uniform interpolation across all frequencies is analogous to low-pass filtering — it destroys fine-grained local position information that high-frequency dimensions encode.

The solution: instead of scaling positions, change the RoPE **base frequency** so that each dimension's wavelength stretches proportionally:

**Key equation — NTK-aware base change:**

```
θ'_i = (base · s^(d/(d-2)))^{-2i/d}
```

equivalently:

```
new_base = base · s^(d/(d-2)),   where s = L_target / L_train
```

For example, with base = 10,000, L_train = 4096, L_target = 32768 (s = 8), d = 128:

```
new_base = 10,000 · 8^{128/126} ≈ 10,000 · 8.13 ≈ 81,300
```

**Why this works**: changing the base stretches the wavelength of every frequency dimension, allowing each dimension's rotation to remain within a reasonable range at the target context length. Crucially, high-frequency dimensions (which encode local, fine-grained position information) are **less compressed** than with linear interpolation, because their wavelengths are short to begin with and can absorb the stretching with less relative distortion.

**Zero-shot capability**: NTK-aware scaling works at inference time without any fine-tuning. You simply change the `rope_theta` parameter in the model config. This is the reason it became widely used in production inference servers — you can extend the context of an existing model without a fine-tuning run.

**Limitation**: at very long context (>4× training length), NTK-aware scaling without fine-tuning shows quality degradation, especially for tasks requiring precise long-range attention patterns. Fine-tuning is still recommended for production long-context use.

## YaRN: Non-Uniform Frequency Scaling (Peng et al., 2023)

### Motivation

Both linear PI and NTK-aware scaling apply a single scale factor — either to positions (PI) or to the base (NTK). But different frequency dimensions have fundamentally different properties:

- **High-frequency dimensions** (fast rotation, short wavelength): encode local token order and relative position within short spans. These should not be interpolated — they are correct within their range.
- **Low-frequency dimensions** (slow rotation, long wavelength): encode global position within the document. These are the dimensions that go out-of-distribution beyond L_train and need interpolation.
- **Mid-frequency dimensions**: a blend.

YaRN applies frequency-dependent scaling:

```
For dimension i:
  if λ_i < √(L_train) / (2π):      # high frequency: no interpolation
      γ = 1.0
  elif λ_i > √(L_target) / (2π):   # low frequency: full interpolation
      γ = s = L_target / L_train
  else:                              # ramp between them
      γ = 1 - (β - λ_i) / (β - α)  # where α, β are freq boundaries
```

where λ_i = 2π / θ_i is the wavelength of dimension i.

**Attention temperature correction**: when position indices are scaled, the inner product q^T R_{n-m} k changes magnitude. YaRN applies a temperature factor `1/√t` to the attention logits:

```
Attention score → Attention score / √t
```

where t is empirically set to ≈ 0.1 × log(s) + 1. This corrects the attention distribution to maintain the same effective "attention temperature" as at the training context length.

### Practical performance

YaRN achieves near-lossless perplexity at 32k–128k context for LLaMA 2 7B and 13B models trained at 4k context, using only 400 fine-tuning steps. This is dramatically fewer steps than linear PI requires. The attention temperature correction is critical — without it, perplexity at long context is significantly worse.

YaRN is now the dominant practical method for context extension, available as `rope_scaling = {"type": "yarn", ...}` in HuggingFace Transformers.

## LongRoPE (Ding et al., 2024)

LongRoPE extends YaRN's non-uniform scaling idea via **evolutionary search** over the per-frequency scale factors, optimizing directly for perplexity on a held-out long-context dataset. Rather than using the analytic ramp function in YaRN, it searches for the best scale factor for each of the d/2 frequency dimensions independently.

**Two-stage approach**:
1. Search for optimal per-frequency scale factors on a target context length (e.g., 512k).
2. Fine-tune with these factors for a small number of steps; then rescale further to 2M context using a second search stage.

**Results**: achieves 2M context length on Phi-2 and LLaMA models with perplexity within 0.5 bits/token of models trained natively at those lengths. The evolutionary search takes ~1 GPU-day.

**Used in**: Phi-3-mini, Phi-3-small (Microsoft, 2024).

## Memory and Compute Implications

The context extension methods themselves add essentially zero overhead to the forward pass — they are just a change to how position indices or the base frequency is computed. The dominant cost of long-context serving is the KV cache:

```
KV cache per sequence = 2 × L × n_layers × n_kv_heads × d_head × bytes_per_element
```

For a 7B model (32 layers, 32 KV heads, head dim 128, bf16) at 128k context:

```
= 2 × 131,072 × 32 × 32 × 128 × 2 = 68 GB
```

This is the GPU memory budget consumed by a single sequence at 128k. This is why Ring Attention (see [ring-attention entry](../ring-attention/en.md)) and chunked prefill (see [chunked-prefill entry](../chunked-prefill/en.md)) are necessary companions to position interpolation in production systems — you need to distribute the KV cache across devices.

At the same time, prefill compute at 128k context is also 1024× the compute at 4k context (quadratic scaling). For a 7B model, prefilling 128k tokens takes ~10–15 seconds on a single A100.

## Engineering Tradeoffs

| Decision | Gained | Gave up |
|---|---|---|
| Linear PI (Chen et al.) | Simple; zero new parameters; works with brief fine-tuning | Compresses all frequencies uniformly; high-freq dims lose local position resolution |
| NTK-aware (zero-shot) | No fine-tuning needed; works out-of-the-box; production deployable immediately | Quality degrades at >4× training length; not suitable for precision long-context tasks without fine-tuning |
| NTK-aware + fine-tuning | Better quality than PI at equivalent fine-tuning budget | Requires choosing new_base value; slightly more complex than PI |
| YaRN | Near-lossless quality at 32k–128k; far fewer fine-tuning steps than PI | More hyperparameters (α, β frequency boundaries, temperature t); analytic ramp is an approximation |
| LongRoPE (evolutionary search) | Best known quality at 512k–2M context; per-dimension optimality | 1 GPU-day search cost; not easily reproducible; harder to adapt to new models |
| Large θ base at pretraining (Llama 3: 500k) | Native long-context without post-training extension | Requires full pretraining rerun; slightly reduces short-context position discrimination |

## Experiments & Results

**Linear PI (Chen et al. 2023)**: LLaMA 7B–65B extended to 32k context with 1,000 fine-tuning steps. Passkey retrieval at 32k: ~100% accuracy. SCROLLS benchmark (long-document summarization, QA): competitive with models trained natively at longer lengths. Position resolution degradation visible at >16k for tasks requiring precise local ordering.

**NTK-aware (kaiokendev 2023)**: zero-shot perplexity on sequences up to 8k (2× LLaMA 2's 4k training length) within 0.3 bits/token of training-length perplexity. No fine-tuning. At 16k (4×), degradation becomes visible without fine-tuning (~1 bit/token).

**YaRN (Peng et al. 2023)**: LLaMA 2 7B trained at 4k context → extended to 128k. After 400 fine-tuning steps, perplexity at 128k context is within 0.1 bits/token of a model trained natively at 128k. Passkey retrieval at 128k: 99.8% accuracy. YaRN at 64k with 400 steps outperforms PI at 32k with 1,000 steps on SCROLLS. Temperature correction alone accounts for ~0.3 bits/token improvement.

**LongRoPE (Ding et al. 2024)**: Phi-2 (2.7B) extended to 2M context. At 512k context, perplexity within 0.5 bits/token of oracle. RULER benchmark (long-context retrieval tasks) at 128k: outperforms YaRN by 3–4 points.

## Commentary

Position interpolation illustrates a pattern common in LLM infrastructure: a theoretically minor change (how position indices are computed) has surprisingly large second-order consequences (whether a model can serve long contexts at all). The progression from PI → NTK-aware → YaRN → LongRoPE tracks a disciplined engineering refinement: each iteration identified which component of the previous method was the bottleneck and addressed it specifically. PI compressed everything uniformly; NTK-aware acknowledged the non-uniformity of frequency importance; YaRN formalized it; LongRoPE solved it optimally.

The practical recommendation for production systems in 2024–2025 is: train with a large θ base (500k–1M) and use YaRN for any further extension needed. The zero-shot NTK-aware trick is useful for quick experiments or for situations where fine-tuning is not feasible. LongRoPE is only worth the search cost for applications requiring contexts beyond 256k.

The open question is whether any of these methods actually provides true long-context capability, or whether they just reduce perplexity degradation. Evidence from "lost in the middle" research (Liu et al. 2023) suggests that even models with low long-context perplexity may fail to attend to information in the middle of long documents. Position interpolation solves the frequency distribution problem; it does not solve the attention pattern problem (where the model learns to focus on beginnings/ends of documents regardless of content). Streaming LLM (see [streaming-llm entry](../streaming-llm/en.md)) approaches this from a completely different angle — bounding the context rather than extending it.

## References

- [1] Chen et al. 2023, "Extending Context Window of Large Language Models via Positional Interpolation," arXiv:2306.15595.
- [2] Peng et al. 2023, "YaRN: Efficient Context Window Extension of Large Language Models," arXiv:2309.00071.
- [3] Ding et al. 2024, "LongRoPE: Extending LLM Context Window Beyond 2 Million Tokens," arXiv:2402.13753.
- [4] kaiokendev 2023, "Things I'm learning while training SuperHOT," blog post. (NTK-aware scaling)
- [5] Su et al. 2021, "RoFormer: Enhanced Transformer with Rotary Position Embedding," arXiv:2104.09864. See also: [RoPE entry](../rope/en.md).
- [6] Liu et al. 2023, "Lost in the Middle: How Language Models Use Long Contexts," arXiv:2307.03172.
- [7] Dubey et al. 2024, "The Llama 3 Herd of Models," arXiv:2407.21783. See also: [Llama 3 entry](../../meta/llama3/en.md).
- [8] [Ring Attention entry](../ring-attention/en.md) — for KV cache distribution across devices at long context.
- [9] [Streaming LLM entry](../streaming-llm/en.md) — bounded-memory alternative to full context extension.
