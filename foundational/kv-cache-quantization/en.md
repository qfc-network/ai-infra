# KV Cache Quantization — KIVI and KVQuant

- **Papers**:
  - **KIVI** — Zirui Liu et al., Rice University. 2024-02.
  - **KVQuant** — Coleman Hooper et al., UC Berkeley. 2024-01.
- **Links**: [KIVI (arXiv:2402.02750)](https://arxiv.org/abs/2402.02750) · [KVQuant (arXiv:2401.18079)](https://arxiv.org/abs/2401.18079) · [KIVI code](https://github.com/jy-joy/KIVI) · [KVQuant code](https://github.com/SqueezeAILab/KVQuant)

## TL;DR

KV cache is the dominant memory consumer at long context and large batch. KIVI quantizes keys to INT2 and values to INT4 per-channel without retraining, cutting KV cache memory 2.6× while matching FP16 quality. KVQuant goes further — per-channel non-uniform quantization with outlier handling — achieving INT4 keys and INT2 values with similar quality degradation. Together they complete the quantization arc: weights (GPTQ/AWQ) → activations (SmoothQuant) → KV cache (KIVI/KVQuant).

The practical upshot: at batch size 64 and sequence length 32k, LLaMA-2-13B's KV cache alone exceeds 330 GB in FP16. INT4 KV reduces this to ~85 GB. KIVI's asymmetric 2/4-bit scheme gets it below 50 GB. These are the margins that determine whether long-context serving is economically viable.

## Context & Motivation

Weight quantization (GPTQ/AWQ) compresses the static part of LLM memory — the model parameters. But during serving, a second, dynamic memory pool dominates: the **KV cache**, which stores the key and value tensors from every attention layer for every active token in every in-flight request.

Concretely, for LLaMA-2-13B (40 layers, 40 heads, d_head = 128, FP16):

```
KV cache = 2 (K+V) × 40 layers × 40 heads × 128 d_head × 2 bytes × B × S
         = 819,200 bytes per (batch entry × token)
```

At B=64, S=4096: ~42 GB — larger than the 26 GB model weights. At S=32k, this scales to ~330 GB. GQA (Grouped Query Attention) helps by reducing the number of KV heads, but even at G=8 the pressure remains significant. Unlike weights, the KV cache cannot be quantized offline — it must be quantized online during the forward pass, per request, in the hot path.

The statistical challenge is real: KV tensors are not like weight matrices. Key vectors exhibit large, channel-wise outliers that dominate the attention dot-product computation. Value vectors are more uniformly distributed but must preserve precision because they are weighted-summed with softmax coefficients that can be peaky. Naive per-tensor quantization misses these statistical differences and causes visible accuracy degradation even at INT8.

Both KIVI and KVQuant take the lesson from SmoothQuant — that outliers are channel-specific and must be handled channel-by-channel — and apply it aggressively to KV tensors in the INT2–INT4 range.

## Core Method

### KIVI: Asymmetric Per-Channel Quantization with a Residual Cache

KIVI's central observation is that keys and values have different quantization sensitivities, so they should get different bit-widths:

- **Keys → INT2 per-channel**: Each key vector dimension (channel) has its own scale and zero-point computed across the sequence dimension. Keys enter the attention score computation via a dot product with query vectors; because the resulting logits are passed through softmax, only their *relative* magnitudes matter. Absolute precision is less critical, which makes INT2 viable for keys.

- **Values → INT4 per-channel**: Values are multiplied directly by the post-softmax attention weights and summed to form the context vector. The accumulated error propagates through the output linearly, so values need more precision than keys.

The quantization formula for both is standard uniform quantization per channel:

```
q(x) = clamp(round((x − z) / s), 0, 2^b − 1)
s = (max(x) − min(x)) / (2^b − 1),   z = min(x)
```

where `s` and `z` are computed independently per channel, with `x` ranging over the sequence dimension for that channel. At dequantization, `x̂ = q(x) · s + z`.

**Residual FP16 cache**: The most recently generated R tokens (e.g., R = 128) are kept in full FP16. Recent tokens are heavily attended to (attention sink + recency bias), and their quantization error has an outsized effect on generation quality. The residual cache is negligible in size for large S — at S = 4096, R = 128 represents 3% overhead — but substantially recovers quality at INT2.

At inference, the combined KV for a position is: dequantize the compressed historical portion on-the-fly and concatenate with the residual. This requires a custom fused attention kernel — a non-trivial implementation burden.

**Memory reduction**: KIVI's average bit-width across K (INT2) and V (INT4) is 3 bits, versus 16 bits for FP16. Reduction factor: 16/3 ≈ 5.3×. In practice, accounting for per-channel scales (stored as FP32, one per channel per layer) and the residual cache, the effective reduction is ~2.6×.

### KVQuant: Non-Uniform Quantization, Pre-RoPE Keys, and Outlier Isolation

KVQuant extends KIVI's ideas with four engineering refinements:

**1. Per-channel non-uniform quantization (NUQ)**: Instead of uniform INT bins, KVQuant fits the quantization levels to the actual distribution of each channel using a technique analogous to NF4 (Normal Float 4): bins are placed at the quantiles of a calibrated distribution, so more bins fall in the high-density region. For INT2 keys, this effectively gives better coverage of the common values without changing the bit-width.

**2. Pre-RoPE key quantization**: Standard transformers apply RoPE (Rotary Position Embedding) to queries and keys before the attention dot product. KVQuant quantizes keys *before* RoPE is applied. The intuition: RoPE applies a rotation in 2D subspaces of the head dimension, and a small quantization error in the original domain becomes a rotated error post-RoPE. Quantizing pre-RoPE ensures the rotation is applied to exact (integer) values, making the final error more predictable and smaller.

**3. Outlier channel isolation**: Some channels of the key matrix have extreme outliers that a uniform 2-bit range cannot represent well even with per-channel scales. KVQuant identifies the top-k outlier channels per layer (typically k ≈ 1% of channels) using a one-time calibration pass and stores them at FP4 or FP8 while quantizing the rest to INT2. This mirrors the SmoothQuant/LLM.int8() insight applied to KV vectors.

**4. Sensitivity-weighted bit allocation**: Not all attention layers and heads are equally sensitive to quantization error. KVQuant runs a lightweight sensitivity analysis (gradient-based or activation-statistics-based) and allocates additional bits to sensitive heads/layers, while being more aggressive elsewhere.

## Engineering Tradeoffs

| Decision | Gained | Gave up |
|---|---|---|
| Per-channel vs per-tensor quantization | Handles channel-wise outliers; viable INT2/INT4 quality | Per-channel scale storage (one FP32 per channel per layer per request — small but non-zero) |
| INT2 keys vs INT4 keys | Further 2× memory reduction for keys | Higher quantization error; residual cache required to stay competitive with FP16 |
| Residual FP16 cache for recent tokens | Preserves quality for high-attended recent tokens; cheap at large S | R × full precision bytes per request; adds code complexity to the attention kernel |
| Pre-RoPE key quantization (KVQuant) | RoPE applied to exact values; better INT2 key quality in practice | Requires architectural plumbing change — quantize at a different point in the forward pass |
| KV quantization vs GQA | Orthogonal and composable — INT4 KV quantization stacks on top of GQA for multiplicative gains | GQA reduces #KV heads; KV quantization reduces bits per value; independent knobs with no interaction penalty |
| Custom dequantization kernels | On-the-fly dequantization hides latency; no storage expansion at rest | Significant kernel engineering effort; INT2 fused attention not available in stock vLLM (as of 2024) |

## Experiments & Results

**KIVI** evaluates on LLaMA-2-7B and LLaMA-2-13B:

- INT2 keys + INT4 values: Wikitext-2 perplexity within 0.1 of FP16 baseline for 7B; within 0.15 for 13B. On downstream benchmarks (MMLU, GSM8K, HumanEval): consistently within 1% of FP16. No retraining required.
- Throughput: at the same GPU memory budget, KIVI allows up to 2.35× higher batch size than FP16 KV cache. The throughput gain scales with sequence length — longer sequences make KV cache the dominant bottleneck, and KIVI's advantage grows.
- The residual cache is important: without it (pure INT2 everywhere), perplexity increases by ~0.5–1.0 on 7B models.

**KVQuant** evaluates on models up to 70B parameters:

- INT4 average bit-width: perplexity gap vs FP16 is < 0.1 on LLaMA-2 across 7B/13B/70B. Stronger than KIVI at matching FP16 due to non-uniform bins.
- INT2 keys with outlier isolation: perplexity gap < 0.5 on LLaMA-2-7B. Pre-RoPE quantization alone recovers ~0.2 perplexity points compared to post-RoPE INT2.
- Sensitivity analysis reveals that layer 0 (the first attention layer, which handles raw token embeddings) and the final few layers are consistently more sensitive to quantization and benefit from additional bits.
- KVQuant's non-uniform quantization is particularly effective for key channels with long-tailed distributions — a common pattern in the first and last layers.

Both papers report no degradation on long-context tasks (LongBench, SCROLLS) at their recommended configurations, which is the key practical claim: memory reduction without compromising long-context reasoning.

## Reproducibility Notes

**KIVI**: fully open source at [github.com/jy-joy/KIVI](https://github.com/jy-joy/KIVI). Integrates with HuggingFace `generate` through a monkey-patched attention implementation. Supports LLaMA and Mistral families. Requires a custom CUDA kernel for INT2 dequantization inside the attention loop; the repo provides this. Calibration is not required — KIVI is truly training-free. Typical overhead: 5–10% latency increase per token vs FP16 due to dequantization, which is negligible compared to the memory-bound savings at large batch.

**KVQuant**: code at [github.com/SqueezeAILab/KVQuant](https://github.com/SqueezeAILab/KVQuant) from the same Berkeley group behind SmoothQuant and SpAtten. Requires a one-time calibration pass (128–512 samples) to compute per-channel quantization levels and identify outlier channels. The pre-RoPE quantization requires modifying the model's attention forward function. Non-uniform quantization levels are stored as a lookup table (256 entries for INT8 indexing into FP16 levels); the storage overhead is about 0.1% of the KV cache.

**Integration status**: as of early 2024, neither method is in vLLM or TGI mainline. Both require running the custom attention implementation, which limits easy deployment on standard serving stacks. Active integration work is ongoing in the vLLM community. For production use, INT8 KV cache (available in vLLM natively) is the pragmatic first step; KIVI/KVQuant offer the next level of compression for teams willing to maintain custom kernels.

## Commentary

KV cache quantization completes the quantization story for LLM serving:

- **GPTQ/AWQ** handle weights — offline, before serving, once per model checkpoint.
- **SmoothQuant** handles activations — at inference time, keeping weights and activations both in INT8.
- **KIVI/KVQuant** handle the KV cache — at inference time, for the memory that grows proportionally with context and batch size.

The practical order of operations for a deployment engineer: apply GQA at training time (free quality-preserving compression of the KV head count), then INT4 KV quantization at serving time (2–4× further compression, minimal quality cost). These are orthogonal knobs — a model with GQA and INT4 KV quantization gets multiplicative benefits.

The asymmetric design choice in KIVI — INT2 for keys, INT4 for values — is worth internalizing. It follows from how keys and values are used: keys enter a dot product whose output is squashed by softmax (error-tolerant), while values are weighted-summed directly into the output (error-sensitive). The same differential precision logic appears in mixed-precision training (forward vs backward passes have different precision requirements) and in speculative decoding (draft tokens are cheap; verification is exact).

The residual cache idea is also broadly applicable: whenever you have a quantized buffer that is frequently accessed at the recent end, keeping the recent portion in full precision recovers most of the quality at trivial memory cost. This pattern appears in KV cache management (KIVI), in ring-attention implementations (last chunk full precision), and in attention sink literature (keep first + recent tokens in full precision).

Looking ahead: FP8 KV cache is already supported in Blackwell-generation hardware, making INT8 KV the trivial case and INT4 achievable in hardware. KIVI and KVQuant are the research foundation for what will likely become INT4 or INT2 KV cache in production serving stacks as hardware support matures.

## References

- [1] Liu et al. _KIVI: A Tuning-Free Asymmetric 2bit Quantization for KV Cache._ arXiv:2402.02750, 2024.
- [2] Hooper et al. _KVQuant: Towards 10 Million Context Length LLM Inference with KV Cache Quantization._ arXiv:2401.18079, 2024.
- [3] Ainslie et al. _GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints._ arXiv:2305.13245, 2023.
- [4] Xiao et al. _SmoothQuant: Accurate and Efficient Post-Training Quantization for Large Language Models._ ICML '23 / arXiv:2211.10438.
- [5] Dettmers et al. _LLM.int8(): 8-bit Matrix Multiplication for Transformers at Scale._ NeurIPS '22 / arXiv:2208.07339.
- [6] Kwon et al. _Efficient Memory Management for Large Language Model Serving with PagedAttention._ SOSP '23 / arXiv:2309.06180.
