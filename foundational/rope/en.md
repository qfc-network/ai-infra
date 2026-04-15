# Rotary Position Embeddings (RoPE)

- **Authors / Org**: Jianlin Su, Yu Lu, Shengfeng Pan, Bo Wen, Yunfang Wu (Zhuiyi Technology)
- **Published**: 2021-04 / RoFormer paper, arXiv:2104.09864
- **Links**: [Paper](https://arxiv.org/abs/2104.09864) · [Code](https://github.com/ZhuiyiTechnology/roformer)

## TL;DR

RoPE encodes position by rotating query and key vectors in 2D subspaces before the attention dot product. The rotation is designed so that the inner product of a rotated query and a rotated key depends only on the **relative position** between them, not on their absolute positions. No extra parameters. No positional embedding lookup table. No changes to the value path. The operation is element-wise and kernel-friendly: FlashAttention 2+ fuses RoPE into the attention kernel. Every major open LLM — LLaMA, Mistral, Gemma, DeepSeek, Qwen — uses RoPE. Long-context extensions (Position Interpolation, YaRN) adjust the rotation frequency to extrapolate beyond the training length without retraining from scratch. RoPE is infrastructure: invisible in the forward pass, zero-cost in parameters, and a key hyperparameter axis (θ base) in modern model design.

## Context & Motivation

### Transformers and position

Vanilla Transformer attention is permutation-invariant: given the same set of tokens in different orders, the attention scores are identical (ignoring masking). Position information must be injected explicitly. The question is how.

### Prior approaches and their failure modes

**Learned absolute embeddings (BERT, GPT-2)**
Add a learned vector e_m to the token embedding at position m. Simple and effective at training length. Problems: the embedding table has fixed size — the model cannot generalize to positions it has never seen during training. Position m=4097 is out-of-distribution for a model trained at max length 4096.

**Sinusoidal absolute embeddings (original Transformer)**
Add fixed sinusoidal functions of position to the token embedding. No learned parameters; slightly better extrapolation than learned embeddings. Still absolute: the model must learn to infer relative positions from absolute ones, which is indirect and imperfect.

**Relative position bias (T5, Shaw et al.)**
Add a learned scalar bias to attention logits based on the relative offset (m − n) between query position m and key position n. Encodes relative position directly; works well in practice. Problem: the bias is added after the QK dot product and interacts poorly with FlashAttention's tiled SRAM computation. Each tile needs access to all relative offsets, breaking the tile-local computation that makes FlashAttention efficient.

**ALiBi (Press et al. 2021)**
Subtract a linear function of relative distance from attention logits: score(m, n) → score(m, n) − λ · |m − n| where λ is head-specific. Simple, no parameters, better length generalization than absolute embeddings. Problems: ALiBi modifies the softmax numerically in a way that is outside the QK dot product — it cannot be fused into the standard attention kernel path that FlashAttention exploits. Also, ALiBi's linear decay is a strong prior that harms quality on some tasks compared to learned relative position.

**The requirements**

A good position encoding for large LLMs needs all of:
1. Encodes relative position (not just absolute) in the QK dot product.
2. Zero learned parameters — position info should not occupy parameter budget.
3. Kernel-friendly — must be fusable into FlashAttention's tiled computation.
4. Exact attention values — no approximation in the attention distribution.
5. Natural length generalization or a clear extension path.

RoPE satisfies all five.

## Core Method

### The rotation idea

For position m, RoPE defines a rotation matrix R_{Θ,m} and applies it to both the query and key vectors before the dot product:

```
f(q, m) = R_{Θ,m} · q
f(k, n) = R_{Θ,n} · k
```

The attention score between query at position m and key at position n becomes:

```
score(m, n) = f(q, m)^T · f(k, n) = q^T · R_{Θ,m}^T · R_{Θ,n} · k = q^T · R_{Θ,(n-m)} · k
```

This is the key identity: **R_{Θ,m}^T · R_{Θ,n} = R_{Θ,(n-m)}**. The dot product depends only on the relative position (n − m), satisfying requirement (1) without adding any per-position parameters, satisfying requirement (2).

### The rotation matrix

R_{Θ,m} is a block-diagonal matrix of d/2 independent 2×2 rotation matrices:

```
R_{Θ,m} = block_diag([R(m·θ_1), R(m·θ_2), ..., R(m·θ_{d/2})])
```

where each 2×2 block R(φ) is:

```
R(φ) = [[cos(φ), -sin(φ)],
         [sin(φ),  cos(φ)]]
```

and the rotation frequencies are:

```
θ_i = 10000^{-2(i-1)/d}   for i = 1, 2, ..., d/2
```

This is the same geometric progression as sinusoidal embeddings (base 10000), but applied as rotations to Q and K rather than as additive offsets to the token embedding.

**Geometric interpretation**: each pair of dimensions (2i-1, 2i) in the d-dimensional vector lives in its own 2D plane. RoPE rotates that plane by angle m·θ_i when the token is at position m. Low-frequency dimensions (large i, small θ_i) rotate slowly — they encode coarse position. High-frequency dimensions (small i, large θ_i) rotate fast — they encode fine-grained local position.

### Implementation

In practice, the block-diagonal matrix multiply is never materialized. Instead, split the d-dimensional query vector q into d/2 adjacent pairs [q_{2i-1}, q_{2i}] and apply:

```
q'_{2i-1} = q_{2i-1} · cos(m·θ_i) - q_{2i} · sin(m·θ_i)
q'_{2i}   = q_{2i-1} · sin(m·θ_i) + q_{2i} · cos(m·θ_i)
```

This is purely element-wise (interleaved cos/sin multiply-add). No matrix multiply. Applies identically to keys. Values are unchanged. FlashAttention 2 applies RoPE inside the tiled kernel before computing scores — no extra memory bandwidth required.

### The θ base as a hyperparameter

The default θ base is 10,000 (from the original paper). But this determines the effective wavelength of each frequency:

- The longest wavelength (dimension pair i=d/2, slowest rotation) completes one cycle over 2π / θ_{d/2} = 2π · 10000^{(d-2)/d} positions.
- For d=128: longest wavelength ≈ 2π · 10000^{126/128} ≈ 62,000 — the model can distinguish positions up to ~62,000 before frequencies wrap around.
- For context lengths beyond ~8,000–16,000, the high-frequency dimensions wrap around multiple times, causing position confusion.

Setting a larger θ base extends the effective wavelength. Llama 3 uses θ = 500,000 for its 8k–128k context variants; many long-context models use θ = 1,000,000 (1M). A larger θ base requires retraining or fine-tuning but provides native long-context capability.

### Long-context extensions

**Position Interpolation (PI, Chen et al. 2023)**

Instead of position indices 0, 1, ..., L'-1 for a sequence of length L' > L, scale them down to 0, L/L', 2L/L', ..., (L'-1)·L/L'. This maps any position in the extended window back into the trained position range [0, L]. The scaling factor is L/L'.

Pro: trivial to implement — just one line of code multiplying position indices. Zero new parameters.
Con: scales all frequencies uniformly. High-frequency dimensions (small θ_i, fast rotation) are the most important for local token order. Scaling down the high frequencies means they now resolve coarser position differences. Quality degrades at long context, especially for tasks requiring precise local token ordering.

**YaRN (Peng et al. 2023)**

YaRN (Yet another RoPE extensioN) uses frequency-dependent scaling: apply different scale factors to different frequency bands.

- High frequencies (large θ_i): interpolate less aggressively — these encode local position and should be preserved.
- Low frequencies (small θ_i): interpolate more aggressively — these encode global position and can tolerate scaling.
- Mid frequencies: blend linearly between the two regimes.

YaRN also applies a temperature correction to the attention softmax to compensate for the change in the norm of RoPE-modified vectors at extended lengths. The result: near-lossless perplexity at 32k–128k context on models trained at 4k, with only fine-tuning (not full retraining) at the extended context length.

**LongRoPE (Microsoft, 2024)**

Further refinements: separately optimize the scale factors for different frequency bands using evolutionary search on a perplexity objective. Used in Phi-3. Achieves better quality than YaRN at 128k+ context.

## Engineering Tradeoffs

| Decision | Gained | Gave up |
|---|---|---|
| Absolute position embeddings | Simple implementation; best quality within training length | No length generalization; embedding table size is a hard cap on context |
| Sinusoidal absolute | No learned parameters; slightly better extrapolation than learned | Still absolute; relative position must be inferred indirectly |
| T5 relative bias | True relative position encoding; effective for medium context | Incompatible with FlashAttention tiling; adds memory overhead proportional to seq_len² |
| ALiBi | Simple (just add a scalar bias); parameter-free; linear decay is an inductive bias | Outside QK dot product — cannot be fused into FlashAttention; linear decay can hurt quality |
| RoPE (default, θ=10k) | Relative position in QK dot product; zero parameters; FlashAttention-fusable; exact attention | Wraps around at context lengths >> 8k without extensions |
| Large θ base (500k–1M, Llama 3) | Longer native effective wavelength; better long-context without extension tricks | Requires retraining; very slightly worse absolute position discrimination at short context |
| Position Interpolation | Trivial to implement; zero new parameters; enables context extension with fine-tuning | Uniformly compresses all frequencies; quality degrades at very long context |
| YaRN per-frequency scaling | Near-lossless quality at 32k–128k; better than PI; only fine-tuning required | More hyperparameters (scale factor per frequency band, temperature); careful calibration needed |

## Experiments & Results

**RoFormer (Su et al. 2021)**: evaluated on Chinese NLP benchmarks (text classification, question answering, NLI). RoFormer with RoPE outperforms sinusoidal and learned absolute embeddings across tasks. The relative position property is particularly useful for longer sequences where absolute positions are less informative.

**Position Interpolation (Chen et al. 2023)**: extends LLaMA 7B–65B to context length 32k with only 1,000 fine-tuning steps. Perplexity on long-context documents improves from chance level (at 32k, model trained at 2k) to near-lossless (within 5% of a model trained at 32k). At 100k context, PI with 1,000 fine-tuning steps is competitive with models trained natively at 100k.

**YaRN (Peng et al. 2023)**: extends LLaMA 2 7B and 13B from 4k to 128k context. Perplexity at 128k context is within 0.1 bits/token of a model trained at 128k natively. Passkey retrieval accuracy (a synthetic long-context task) reaches near-100% at 128k. PI at the same length shows perplexity degradation of 2–5 bits/token.

**Mistral 7B**: uses θ = 1,000,000 with sliding window attention for native long-context. No post-training context extension required.

**Llama 3**: uses θ = 500,000 with a gradual fine-tuning schedule across context lengths (8k → 32k → 128k). Context extension is part of the training recipe, not an afterthought.

## Reproducibility Notes

HuggingFace Transformers exposes RoPE through the `rope_theta` config parameter. Setting `rope_theta=500000` and fine-tuning on long sequences is the standard recipe for context extension.

```python
from transformers import LlamaConfig
config = LlamaConfig(
    rope_theta=500000,          # θ base; default is 10000
    max_position_embeddings=131072,  # extended context length
)
```

The `apply_rotary_pos_emb` utility function in HuggingFace Transformers applies RoPE to query and key tensors:

```python
from transformers.models.llama.modeling_llama import apply_rotary_pos_emb
cos, sin = self.rotary_emb(value_states, position_ids)
query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)
```

**FlashAttention 2+**: RoPE is fused into the attention kernel. `flash_attn_with_kvcache` accepts `rotary_cos` and `rotary_sin` tensors and applies them inside the kernel — no separate pass required, no extra memory bandwidth.

**Implementation from scratch**: approximately 20 lines of PyTorch. Precompute cos/sin tables for position indices, then apply element-wise to Q and K using the interleaved pattern. The only subtlety is whether to interleave pairs as [q0, q1, q2, q3, ...] → rotate (q0, q1), (q2, q3), ... or to split the head dimension at d/2. Different implementations use different conventions; ensure Q, K, and the KV cache use the same convention.

**YaRN in practice**: available in the `transformers` library via `rope_scaling` config:

```python
config = LlamaConfig(
    rope_theta=10000,
    rope_scaling={
        "type": "yarn",
        "factor": 32.0,              # target_length / training_length
        "original_max_position_embeddings": 4096,
    }
)
```

## Commentary

RoPE is infrastructure, not a technique. It is invisible in the forward pass — it adds no computation visible to the optimizer, modifies no loss landscape directly, and occupies no parameter budget. Yet it is foundational: the choice of θ base is now a primary design hyperparameter for large LLMs, determining the model's native context range as decisively as hidden size or number of layers.

The θ base story is instructive about how seemingly minor choices have large downstream effects:

- θ = 10,000 (original): adequate for 2k–4k context, wraps around at ~8k.
- θ = 500,000 (Llama 3): native good behavior up to ~128k without interpolation.
- θ = 1,000,000 (Mistral): pushes the effective range further; some quality loss at short context.

YaRN is the practical upgrade path for models that were trained with small θ and need to serve longer contexts. It requires only fine-tuning (not retraining) and is now mature enough to be a one-command flag in HuggingFace. The combination of a large θ base at training time plus YaRN for extension is the current best practice for long-context LLMs.

The one unsolved tension: RoPE is inherently sequential — it assigns positions based on the order of tokens in the input. This works for autoregressive text but becomes awkward for modalities with 2D or 3D structure (images, video). Extensions like 2D-RoPE (used in some vision-language models) apply different rotation rates along spatial axes. This is an active area as multimodal models push beyond text-only assumptions.

## References

- [1] Su et al. 2021, "RoFormer: Enhanced Transformer with Rotary Position Embedding," arXiv:2104.09864.
- [2] Chen et al. 2023, "Extending Context Window of Large Language Models via Positional Interpolation," arXiv:2306.15595.
- [3] Peng et al. 2023, "YaRN: Efficient Context Window Extension of Large Language Models," arXiv:2309.00071.
- [4] Press et al. 2021, "Train Short, Test Long: Attention with Linear Biases Enables Input Length Extrapolation (ALiBi)," arXiv:2108.12409.
- [5] Ding et al. 2024, "LongRoPE: Extending LLM Context Window Beyond 2 Million Tokens," arXiv:2402.13753.
- [6] Dubey et al. 2024, "The Llama 3 Herd of Models," arXiv:2407.21783.
