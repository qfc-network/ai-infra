# Jamba / Hybrid SSM-Transformer

- **Key papers**:
  - **Jamba** — Lieber et al., AI21 Labs. 2024-03.
  - **Nemotron-H** — NVIDIA. 2025.
- **Links**: [Jamba (arXiv:2403.19887)](https://arxiv.org/abs/2403.19887) · [Nemotron-H (arXiv:2507.00509)](https://arxiv.org/abs/2507.00509) · [Jamba-1.5 (arXiv:2408.12570)](https://arxiv.org/abs/2408.12570) · [code](https://huggingface.co/ai21labs)

## TL;DR

Pure Mamba/SSM has linear prefill complexity and constant-memory decode, but underperforms attention on tasks requiring exact in-context retrieval. Pure attention has quadratic prefill and an O(S × layers) KV cache that becomes the dominant memory cost at long context. Hybrid architectures interleave SSM layers and attention layers in a fixed ratio — roughly 1 attention per 4–8 SSM layers — exploiting SSM's efficiency for the majority of layers while letting attention handle the minority of positions where precise token recall matters. **Jamba** (AI21 Labs, 2024) is the first production-deployed large hybrid, combining Mamba layers, Transformer (attention) layers, and MoE within a single 52B-total / 12B-active model; it demonstrated 3× higher throughput than Mixtral-8×7B at 256K context. **Nemotron-H** (NVIDIA, 2025) extends the approach to frontier scale on NVIDIA hardware with FP8 and TransformerEngine integration. The infrastructure consequences are concrete: KV cache is reduced in proportion to the attention layer fraction, decode is faster for SSM-dominant models, prefill is dominated by the fewer attention layers, and the serving stack must manage two distinct memory pools simultaneously.

## Context: Why Neither Pure Architecture Wins

### The Pure Attention Problem

Standard multi-head attention has cost `O(S²·d)` per layer at sequence length `S` and model dimension `d`. FlashAttention removes the `O(S²)` memory materialization and makes the kernel IO-efficient, but it does not change the FLOPs — attention FLOPs still scale quadratically. More critically for inference: every attention layer accumulates a KV cache. For a model with `L_attn` attention layers, `H` heads, head dimension `d_h`, and `S` tokens in context, the KV cache is:

```
KV_cache = S × L_attn × 2 × H × d_h × bytes_per_element

Example (dense 32-layer transformer, S=256k, H=32, d_h=128, BF16):
  = 256,000 × 32 × 2 × 32 × 128 × 2 bytes
  ≈ 134 GB
```

This is larger than the model weights themselves for many frontier models at 256K context. The KV cache balloons with context, constrains batch size, and ultimately limits how many concurrent long-context sessions a deployment can serve.

### The Pure SSM Problem

Mamba and its derivatives evolve a hidden state `h_t ∈ R^d_state` per layer via a discretized recurrence. At steady state the memory cost per layer is `O(d_state)` regardless of sequence length — constant, not growing. Prefill uses a parallel scan (O(S·d_state) FLOPs), not a quadratic operation. Decode advances the state in O(d_state) per step per layer, with no growing cache.

The cost is informational: the SSM state is a **lossy, fixed-size compression** of all prior context. The model must compress the entire prompt history into a vector of fixed dimension. For tasks requiring retrieval of a specific token or value buried in a long context — needle-in-haystack, associative recall, verbatim copy — this compression loses information. Empirical results are consistent: pure Mamba models show significant degradation on hard recall benchmarks versus attention models of similar scale. Attention's softmax selection is exact; given the correct key, it can retrieve the associated value with zero information loss. SSM state compression cannot match this for arbitrary recall targets.

### The Hybrid Insight

The solution is architectural triage: use SSM for the majority of layers where context compression is sufficient for the task, and use attention for a minority of layers where exact retrieval is needed. This gives:

- The KV cache cost of only `L_attn / L_total` fraction of attention layers.
- Prefill dominated by the attention layers (quadratic) but those are rare — most layers are linear.
- Decode state that is mostly constant-size SSM state plus a small, bounded KV cache for the attention layers.
- Quality matching or exceeding pure attention models because the attention layers handle retrieval-sensitive positions.

## Jamba Architecture

Jamba (2024) is the production realization of this hybrid design at scale.

### Block Structure

Jamba interleaves Mamba blocks, Transformer attention blocks, and MoE layers in a fixed repeating pattern:

```
[Mamba, Mamba, Attention, MoE-FFN] × N_repeat

Concrete Jamba 52B configuration:
  Total layers: ~72
  Attention layers: ~18  (1 per 4 total layers)
  Mamba layers:    ~54
  MoE layers:      subset of the FFN positions
  Total params:    52B
  Active params:   12B (MoE sparsity)
```

The Mamba layers use the standard Mamba-2 architecture (selective SSM, d_state=16 or 64 depending on configuration, input-dependent B, C, Δ parameters). The attention layers use standard multi-head attention with GQA and RoPE positional encoding. The MoE layers replace dense FFNs at certain positions with top-2 routing over a small expert pool.

### KV Cache Reduction

Only the attention layers accumulate KV cache. The Mamba layers maintain a fixed-size SSM state:

```
SSM state per layer = batch_size × d_state × d_model × bytes

Example (d_state=16, d_model=4096, BF16, batch=1):
  = 1 × 16 × 4096 × 2 = 131 KB per layer
  × 54 Mamba layers = ~7 MB total SSM state

KV cache for 18 attention layers at S=256k context, H=32 heads, d_h=128, BF16:
  = 256,000 × 18 × 2 × 32 × 128 × 2 bytes
  ≈ 75 GB

vs. equivalent dense 72-layer transformer KV cache:
  = 256,000 × 72 × 2 × 32 × 128 × 2 bytes
  ≈ 301 GB
```

The reduction factor is `L_attn / L_total = 18/72 = 25%` — Jamba's KV cache is approximately 4× smaller than a dense transformer with equivalent layer count at the same context length. SSM state adds only ~7 MB regardless of context length, which is negligible.

AI21 Labs reports 3× higher throughput than Mixtral-8×7B at 256K context. Mixtral has full attention across all its attention layers; the KV cache at that context length saturates memory, constraining batch size to 1. Jamba can batch more requests because its KV cache is smaller.

### Prefill Efficiency

During prefill (processing the full prompt), each layer type contributes differently:

```
Attention layer prefill cost: O(S² × d)     — quadratic in sequence length
Mamba layer prefill cost:     O(S × d_state × d)  — linear via parallel scan

Total prefill:
  ≈ L_attn × S² × d  +  L_mamba × S × d_state × d
  = 18 × S² × d  +  54 × S × d_state × d

At S=256k, S² = 6.5×10^10; S×d_state ≈ 256k×16 = 4.1×10^6
Attention dominates above moderate S; Mamba layers are cheap.
```

The break-even point where Mamba prefill cost equals attention prefill cost scales as `S ~ d_state × L_mamba / L_attn`. For Jamba parameters, this is around `S ~ 16 × 3 = 48` tokens — for any practical prompt, Mamba layers are dominated by attention layers in FLOPs.

### Decode Efficiency

During autoregressive decode, each token advances the model state:

```
Attention layer decode step:   O(S × d)   — query attends over full KV cache
Mamba layer decode step:       O(d_state × d)  — recurrent state advance, constant

At S=256k:
  Attention: 256,000 × d operations per layer
  Mamba:     16 × d operations per layer (for d_state=16)
  Ratio: 16,000× cheaper for Mamba at this context length
```

The attention layers still require O(S·d) per step per layer because they must query the full KV cache (or use FlashDecoding which parallelizes the query but does not reduce total work). At 256K context, the per-token decode cost at each attention layer is substantial. The Mamba layers are effectively free by comparison, requiring only a simple matrix-vector multiply to advance the state. In Jamba with 18 attention layers out of 72, the attention layers dominate decode compute but their fraction (25%) means the total cost per decode step is roughly `(18/72) × S×d + (54/72) × d_state×d`, substantially less than a full 72-layer attention model.

## SSM State as Lossy Compression

The fundamental asymmetry between attention and SSM in memory at long context deserves a precise framing:

**Attention KV cache**: stores the exact key and value projections for every prior token at every attention layer. Given a query, the answer is computed from the exact stored representations. No information about prior tokens is lost (up to floating point precision). The cost is O(S × L_attn) memory, growing without bound.

**SSM hidden state**: a fixed-size vector (O(d_state) per layer) that summarizes all prior context via the learned recurrence. The state is updated at each step by `h_t = A·h_{t-1} + B·x_t`; information about early tokens is progressively overwritten unless the learned `A` happens to preserve it. The compression is lossy and irreversible — you cannot reconstruct the original sequence from the state.

This is why hybrid architectures place attention at a minority of layers: those layers can exactly retrieve any prior token their KV cache covers, while the SSM layers handle the "background context" that doesn't require exact recall. The typical needle-in-haystack failure of pure SSM models is because the needle's key-value association gets overwritten in the SSM state before the output needs to reference it.

One practical consequence: **SSM state cannot be composed for prefix caching** in the same way KV cache can. Concatenating two KV caches is exact; extending an SSM state requires re-running the recurrence from the beginning or from a saved checkpoint. Serving stacks that use prefix caching (like vLLM's RadixAttention) get reduced benefit from SSM layers — you can cache SSM states at prefixes, but the states are not compositional and cannot be shared across requests with overlapping but non-identical prefixes in the general case.

## Nemotron-H (NVIDIA, 2025)

Nemotron-H extends the hybrid SSM-Transformer approach to frontier scale with NVIDIA-specific optimizations:

- Uses a similar Mamba + attention interleaving, tuned for H100/H200 hardware.
- Integrates **TransformerEngine FP8** for the attention layers, achieving near-peak tensor core utilization on Hopper.
- Mamba-2 kernels benefit from the same WGMMA / async TMA infrastructure as FA3.
- Training stack is Megatron-Core with custom Mamba sequence parallelism extensions.

Nemotron-H demonstrates that hybrid architectures are viable at the scale where frontier labs compete, not just at mid-scale. The combination of FP8 attention, linear-scan SSM layers, and MoE (where applicable) gives a serving profile that outperforms pure transformer baselines on long-context throughput benchmarks.

## Serving Stack Implications

A hybrid model requires the inference system to manage two distinct memory pools:

1. **KV cache** for the attention layers: paged or contiguous blocks, can use PagedAttention / RadixAttention strategies, grows with context.
2. **SSM state** for the Mamba layers: fixed-size per sequence, does not grow with context, must be saved/restored for preempted sequences.

vLLM's Jamba support allocates both pools and multiplexes them per sequence. The SSM state is small enough (MBs per sequence) that it is typically not the binding constraint; the KV cache remains the dominant memory consumer even after the 4× reduction.

**Batching**: because KV cache is smaller, more sequences can be batched simultaneously. This is the primary throughput gain at long context. At shorter contexts (< 8K tokens), the KV cache advantage is less pronounced and the two architectures are roughly equivalent in serving throughput.

**Sequence parallelism**: for very long sequences, the attention layers still need sequence parallelism (e.g., Ring Attention) to distribute the O(S²) prefill compute. The Mamba layers need their own sequence-parallel scan implementation. Nemotron-H's training stack (Megatron-Core) implements both.

**Preemption and paging**: SSM states, unlike KV blocks, are single dense tensors per layer per sequence. They cannot be paged at sub-sequence granularity. If a sequence is preempted and evicted from memory, the full SSM state stack (all Mamba layers' states) must be written to host memory and restored on continuation — a larger atomic unit than a KV page.

## Engineering Tradeoffs

| Architecture | Prefill Complexity | Decode Memory | Recall Accuracy | Hardware Efficiency | Serving Complexity |
|---|---|---|---|---|---|
| Pure Mamba/SSM | O(S·d) linear; parallel scan | Constant O(d_state·L); no KV cache | Degraded on hard recall tasks; lossy compression limits retrieval | Good throughput; custom scan kernel required; no standard infra | Simpler; no KV cache management; prefix caching limited |
| Pure Transformer (dense attention) | O(S²·d) quadratic; FlashAttention helps kernel efficiency | O(S·L_attn·d_h·H); grows without bound | Exact recall; softmax selection is lossless | Near-peak on FlashAttention; KV cache limits batch size at long context | Mature; PagedAttention, prefix caching, chunked prefill all apply |
| Jamba hybrid (SSM + attn, 1:3 ratio) | Dominated by attention layers; 4–8× fewer attn ops than dense | ~1/4 to 1/8 of dense transformer KV cache; + small fixed SSM state | Near-parity with attention; few attn layers handle retrieval-critical positions | Good; 3× throughput over Mixtral at 256K | Moderate; two cache types; SSM state must be managed on preemption |
| Linear attention variants (RetNet, GLA) | O(S·d) | Constant | Moderate; better than pure SSM, worse than softmax attention on hard recall | Comparable to SSM; no custom scan needed for some variants | Similar to SSM; prefix caching limited |

## Practical Guidance

**When to use a hybrid model**: workloads with 64K+ context where pure transformer KV cache is the binding memory constraint. At shorter contexts, the latency and quality characteristics of pure transformers are usually preferable given their more mature serving infrastructure.

**Ratio selection**: 1 attention per 4 Mamba layers is the Jamba default and appears to be close to the quality-efficiency frontier for current benchmarks. More attention improves recall quality at the cost of more KV cache; fewer attention layers reduce KV cache but degrade recall. Empirically, models with fewer than 1 attention per 8 SSM layers show measurable recall degradation on long-document tasks.

**KV cache estimation**: for a hybrid with `f_attn` attention fraction:
```
KV_cache = S × L_total × f_attn × 2 × H × d_h × bytes
```
This scales exactly as the pure-attention case but with a `f_attn` multiplier. At `f_attn = 0.25` and `S=256K`, a 72-layer hybrid uses the same KV memory as an 18-layer pure transformer.

**SSM state management**: ensure your serving framework allocates SSM state outside the KV cache pool and handles preemption correctly (full-state eviction to host). Missing this causes silent corruption on sequence preemption.

## Cross-References

- `../mamba-ssm/` — Mamba architecture; prerequisite for understanding SSM state, parallel scan, and selectivity
- `../flash-attention/` — attention layers in hybrids still use FlashAttention; KV cache calculations reference FA3 performance numbers
- `../ring-attention/` — long-context serving of hybrid models still needs sequence parallelism for the attention layers
- `../../nvidia/megatron-core/` — Nemotron-H training stack; Mamba sequence parallelism and TransformerEngine FP8 integration

## References

- [1] Lieber et al. _Jamba: A Hybrid Transformer-Mamba Language Model._ arXiv:2403.19887, 2024.
- [2] Team AI21. _Jamba-1.5: Hybrid Transformer-Mamba Models at Scale._ arXiv:2408.12570, 2024.
- [3] NVIDIA. _Nemotron-H: A Family of Accurate and Efficient Hybrid Mamba-Transformer Models._ arXiv:2507.00509, 2025.
- [4] Gu & Dao. _Mamba: Linear-Time Sequence Modeling with Selective State Spaces._ arXiv:2312.00752, 2023.
- [5] Dao & Gu. _Transformers are SSMs: Generalized Models and Efficient Algorithms Through Structured State Space Duality._ ICML '24 / arXiv:2405.21060.
- [6] Dao et al. _FlashAttention-3: Fast and Accurate Attention with Asynchrony and Low-precision._ arXiv:2407.08608, 2024.
