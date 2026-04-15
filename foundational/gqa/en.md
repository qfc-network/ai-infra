# Grouped Query Attention (GQA)

- **Authors / Org**: Joshua Ainslie, James Lee-Thorp, Michiel de Jong, Yury Zemlyanskiy, Federico Lebrón, Sumit Sanghai (Google Research)
- **Published**: 2023-05 / arXiv:2305.13245
- **Links**: [Paper](https://arxiv.org/abs/2305.13245)

## TL;DR

GQA sits between Multi-Head Attention (MHA) and Multi-Query Attention (MQA). MHA gives every query head its own K/V head — maximum quality, maximum KV cache. MQA collapses all K/V heads into one — minimal cache, noticeable quality drop. GQA splits the H query heads into G groups, with each group sharing one K/V pair. The KV cache shrinks by H/G while quality tracks MHA to within noise. Critically, an existing MHA checkpoint can be **uptrained** to GQA in a small number of steps by mean-pooling the K/V heads within each group — no training from scratch required. Every major open LLM released after mid-2023 uses GQA: Llama 2, Llama 3, Mistral, Gemma, DeepSeek. GQA is now the default attention variant.

## Context & Motivation

### The KV cache bottleneck

Transformer inference has two distinct phases: prefill (processing the prompt, compute-bound) and decode (generating tokens one at a time, memory-bandwidth-bound). During decode, the model reads the entire KV cache for every generated token. With MHA, the KV cache per token per layer is:

```
KV cache (MHA) = 2 × H × d_head × dtype_bytes
```

For a model with H=64 heads, d_head=128, and BF16 (2 bytes): 2 × 64 × 128 × 2 = 32,768 bytes per token per layer. At 80 layers (Llama 3 70B), a single token occupies ~2.6 MB of KV cache. A batch of 32 sequences at 4,096 tokens each requires 32 × 4096 × 2.6 MB ≈ 340 GB — more than four A100-80GB GPUs. KV cache size directly caps throughput: smaller cache means smaller batches and lower GPU utilization.

### MQA: the aggressive baseline

Shazeer (2019) proposed Multi-Query Attention: all H query heads share a **single** K/V head. KV cache drops to 1/H of MHA. Throughput on long sequences improves by 2–3×. But MQA degrades quality, especially on long-context tasks and reasoning benchmarks — the single shared K/V head must represent all the information that H independent heads previously could. PaLM and Falcon used MQA but it was recognized as a quality-compute tradeoff, not a clean win.

### The need for a middle ground

Labs needed a variant that:
1. Shrinks the KV cache substantially (to enable large batches and long contexts).
2. Preserves MHA-level quality (no task-specific regressions).
3. Can be obtained from existing MHA checkpoints without full retraining (most large models are expensive to train from scratch).

GQA satisfies all three.

## Core Method

### Three variants

**Multi-Head Attention (MHA)**
- H query heads, H K/V heads (one K/V per query head).
- KV cache per token per layer: `2 × H × d_head × dtype_bytes`

**Multi-Query Attention (MQA)**
- H query heads, 1 K/V head (all queries share one K/V).
- KV cache per token per layer: `2 × 1 × d_head × dtype_bytes`
- Reduction: H× vs MHA.

**Grouped Query Attention (GQA)**
- H query heads, G K/V groups where G ∈ [1, H].
- Each group of H/G query heads shares one K/V pair.
- KV cache per token per layer: `2 × G × d_head × dtype_bytes`
- Reduction: H/G× vs MHA.
- MQA is GQA with G=1; MHA is GQA with G=H.

### Memory formula

For batch size B, sequence length S, L layers, G K/V groups, head dimension d_head, and dtype size b bytes:

```
KV cache memory = B × S × L × 2 × G × d_head × b
```

**Llama 3 70B concrete example**: H=64 query heads, G=8 K/V groups, d_head=128, BF16 (b=2), L=80 layers.

- MHA KV cache per token: 2 × 64 × 128 × 2 = 32,768 bytes
- GQA KV cache per token: 2 × 8 × 128 × 2 = 4,096 bytes
- Reduction: **8× smaller KV cache** vs MHA.

At a batch of 32 sequences × 8,192 tokens: GQA uses ~42 GB vs ~340 GB for MHA — the difference between fitting on two A100s and needing four.

### Uptraining from MHA checkpoints

The practical key to GQA's adoption is that MHA models can be converted without full retraining:

1. Take a pretrained MHA checkpoint with H K/V heads.
2. Divide the H K/V heads into G groups of H/G heads each.
3. Within each group, **mean-pool** the H/G K/V weight matrices into a single K/V head.
4. Fine-tune the resulting model for a small number of steps (roughly 5% of the original training compute) with standard language modeling loss.

Quality recovers close to MHA after uptraining. The mean-pool initialization gives a better starting point than random initialization. The paper shows that uptraining to GQA is substantially better than training GQA from scratch with the same limited compute budget — the pretrained MHA weights provide strong initialization for the query projections and FFN layers.

### Attention computation

During a forward pass, each K/V group's single key and value are broadcast across the H/G query heads in that group. The attention score computation within a group:

```
Attention(Q_g, K_g, V_g) for each query head q in group g:
  score = softmax(Q_q · K_g^T / sqrt(d_head))
  output = score · V_g
```

K_g and V_g are the same tensor for all H/G query heads in group g. This broadcast is free in most attention kernels — FlashAttention handles GQA natively.

## Engineering Tradeoffs

| Decision | Gained | Gave up |
|---|---|---|
| MQA (G=1) vs MHA | Maximum KV cache reduction (H×); highest throughput on long sequences | Noticeable quality drop, especially on long-context reasoning and complex tasks |
| GQA with G=8 (Llama 3 style) | ~8× KV cache vs MHA; throughput nearly matches MQA; quality matches MHA within noise | Slightly more K/V compute than MQA (8 K/V heads vs 1) |
| GQA with G=H/2 (moderate grouping) | Moderate cache reduction (2×) with very small quality impact | Less throughput improvement than G=8 |
| Uptraining from MHA checkpoint | Avoids full retraining; reuses all pretrained weights; quality recovery in ~5% compute | Sub-optimal initialization vs training GQA from scratch with full compute |
| Smaller KV cache (GQA) | Higher throughput, larger batch sizes, longer contexts at same GPU memory | None materially — GQA is a strict improvement over MHA at any given memory budget |
| GQA vs MLA (DeepSeek-V2/V3) | Simpler implementation; no latent projection overhead; compatible with all attention kernels | MLA achieves further KV compression via low-rank latent projection; GQA is the baseline MLA improves upon |

### Choosing G in practice

The choice of G is a hardware-aware decision:

- **G=H (MHA)**: Only if KV cache is not a bottleneck (short sequences, small batches).
- **G=1 (MQA)**: When KV cache is the dominant constraint and quality degradation is acceptable.
- **G=8 (Llama 3 70B style)**: The sweet spot for large models — 8× cache reduction with near-MHA quality. G should divide H evenly.
- **Alignment with tensor parallelism**: with TP degree T, set G ≥ T so each device has at least one K/V head. Llama 3 70B uses G=8 with TP=8 — exactly one K/V head per device.

## Experiments & Results

The original paper uptrains T5-XXL (11B) and PaLM (540B) checkpoints to GQA and evaluates on SuperGLUE and a suite of generation tasks.

**Quality**: GQA with G ≥ 2 matches MHA within noise on all tasks. MQA (G=1) shows 0.5–2 point drops on SuperGLUE tasks, with the largest gaps on tasks requiring precise token-level attention (e.g., WiC, MultiRC). GQA closes this gap at G=8.

**Throughput**: 2–3× improvement over MHA on long-sequence decoding (sequence length ≥ 2,048). The improvement is memory-bandwidth-bound: GQA reduces the data volume the GPU must read per decode step.

**Uptraining efficiency**: uptraining to GQA recovers within 1% of MHA quality after 5% of the original training compute. Training GQA from scratch with the same 5% compute budget underperforms uptraining — the MHA weights provide strong initialization.

## Reproducibility Notes

GQA is fully supported in HuggingFace Transformers via the `num_key_value_heads` config parameter. Setting `num_key_value_heads < num_attention_heads` automatically enables GQA in all models that support it (Llama, Mistral, Gemma, Falcon, etc.).

```python
from transformers import LlamaConfig
config = LlamaConfig(
    num_attention_heads=64,      # H query heads
    num_key_value_heads=8,       # G K/V groups
    hidden_size=8192,
    num_hidden_layers=80,
)
```

**vLLM**: the paged KV cache allocator uses `num_key_value_heads` for block sizing. GQA models use proportionally fewer KV cache blocks, enabling larger effective batch sizes automatically.

**FlashAttention 2+**: natively supports GQA. The kernel broadcasts the K/V tensors within each group without materializing the expanded tensor — memory-efficient.

**Implementation complexity**: adding GQA to a custom attention kernel is approximately 10 lines of code — reshape K/V from `[batch, seq, G, d_head]` to `[batch, seq, H, d_head]` by repeating each K/V group H/G times before the attention computation, or handle the grouping directly in the einsum.

## Commentary

GQA is the new MHA — the default, not an optimization. Every major open model released after mid-2023 uses it. The uptraining recipe was the decisive factor in adoption: it meant that labs could convert their existing MHA checkpoints rather than committing to a full retraining run. The quality story is clean: at G=8, GQA is indistinguishable from MHA on downstream benchmarks while using 8× less KV cache memory.

The next step beyond GQA is Multi-Head Latent Attention (MLA), introduced in DeepSeek-V2. MLA further compresses the KV cache by projecting keys and values into a low-rank latent space and decomposing RoPE separately. MLA can achieve KV compression ratios of 10–20× with no quality degradation — substantially better than GQA at G=8. GQA is the floor; MLA is the current ceiling. GQA remains the practical default because MLA's low-rank projection adds implementation complexity and breaks standard FlashAttention kernels.

For a practitioner in 2026: if you are training a new model, use GQA with G chosen so that G/H ≈ 1/8 (e.g., G=8 for a 64-head model). If you have an existing MHA checkpoint that needs to serve at longer context or larger batch, the uptraining recipe is the fastest path to meaningful KV cache reduction.

## References

- [1] Ainslie et al. 2023, "GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints," arXiv:2305.13245.
- [2] Shazeer 2019, "Fast Transformer Decoding: One Write-Head is All You Need," arXiv:1911.02150.
- [3] Touvron et al. 2023, "Llama 2: Open Foundation and Fine-Tuned Chat Models," arXiv:2307.09288.
- [4] Dubey et al. 2024, "The Llama 3 Herd of Models," arXiv:2407.21783.
- [5] Liu et al. 2024, "DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts Language Model," arXiv:2405.04434.
