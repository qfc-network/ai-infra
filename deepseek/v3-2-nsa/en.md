# DeepSeek V3.2 — Native Sparse Attention (NSA / DSA)

- **Authors / Org**: DeepSeek-AI
- **Published**: 2025-09 (experimental release)
- **Links**: [DeepSeek-V3.2-Exp (GitHub)](https://github.com/deepseek-ai/DeepSeek-V3.2-Exp) · [DeepGEMM V3.2 indexer (PR #200)](https://github.com/deepseek-ai/DeepGEMM)

## TL;DR

DeepSeek V3.2 extends the 671B V3 MoE base by replacing standard full attention with **Native Sparse Attention (NSA)**, also called **DeepSeek Sparse Attention (DSA)** in the DeepGEMM codebase. The core change: every attention layer now runs three parallel sub-computations — a compressed global path, a learned sparse selection path, and a local sliding-window path — whose outputs are merged by a per-head weighting. At 128k context the selected path attends to roughly 1024 tokens per query (k=16 blocks of 64) rather than 128k, reducing per-layer attention memory by four orders of magnitude versus full attention while preserving near-lossless quality on long-context benchmarks. This entry closes the arc opened by the V3 tech report and the DeepGEMM walkthrough, which both reference the V3.2 indexer without explaining the architectural mechanism behind it.

## Context and Motivation

The V3 base architecture already solved two out of three long-context bottlenecks:

1. **KV cache size**: MLA compresses the KV cache in the **head dimension** — at inference, cache per token drops to roughly 1/10 of standard MHA. Detailed in `../mla/`.
2. **Sequence parallelism overhead**: Ring Attention (covered in `../../foundational/ring-attention/`) distributes the sequence across devices, keeping per-GPU activation memory constant. V3 uses this at the 128k context extension stage.
3. **Per-device attention compute and memory**: at S=128k, full attention's O(S²·d) cost dominates even with FlashAttention's IO efficiency. This is the gap NSA fills.

To see why full attention fails at 128k, consider a single attention layer with standard parameters:

$$
\text{attention matrix memory} = S^2 \cdot n_\text{heads} \cdot 2\;\text{bytes (BF16)}
$$

For S=128k, n_heads=128: 128000² × 128 × 2 ≈ **4.3 TB per layer**. Even with FlashAttention's tiled execution, the effective working set that must transit HBM for a single forward pass through one layer vastly exceeds any single GPU's HBM (80 GB on H100). Sequence parallelism divides this by the number of CP ranks, but each rank still pays O((S/P)²) compute per layer. The only way to bring attention cost down to something proportional to S rather than S² is to attend to fewer tokens — which requires making that selection smartly so quality holds.

Approximate sparse attention approaches had existed for years (Longformer, BigBird, fixed local windows) but they either used fixed patterns that do not capture task-specific relevance, or required separate retrieval systems not trained jointly with the model. NSA's design goal is learned, end-to-end sparse selection where the pattern adapts per query and is trained with the model's primary loss from scratch.

## NSA Architecture: Three Parallel Paths

Every NSA attention layer computes three sub-attentions over the same query and produces a weighted sum:

```
Output = w_c · Attn_compressed(Q, KV_compressed)
       + w_s · Attn_selected(Q, KV_selected)
       + w_w · Attn_window(Q, KV_local)
```

The per-head scalar weights `(w_c, w_s, w_w)` are learned and applied after the three attention outputs are computed. This is not a hard switch — all three paths are active; the model can learn to up-weight the selected path for retrieval-heavy queries and up-weight the window path for locally-coherent text.

### Path 1 — Compressed (Coarse Global Context)

The entire KV sequence is divided into fixed-size blocks. Within each block, a **learned linear projection** maps the block's key vectors to a single representative key (and similarly for values). With block size B=64 and S=128k, this produces 2000 representative KV pairs — a coarse but global summary of the sequence.

The compressed path attends Q against these 2000 representative keys. Cost: O(S/B) per head, roughly O(2000) — effectively constant relative to S. This path provides a low-frequency global signal: themes, long-range dependencies, very early context. The projection weights are trained, so the model learns what to summarize.

Memory: 2 × (S/B) × d_head × 2 bytes = 2 × 2000 × 128 × 2 = ~1 MB per layer, per batch element. Negligible.

### Path 2 — Selected (Fine Sparse Retrieval)

This is the mechanically interesting path. Before computing attention, a **lightweight gating network** scores each key block against the current query and selects the top-k blocks. DeepSeek uses k=16 blocks of 64 tokens = 1024 tokens per query. The gating score for block j against query i is:

$$
g_{ij} = \text{ReLU}\!\left(\sum_{h} \mathbf{q}_i^{(h)} \cdot \mathbf{k}_j^{(h)}\right) \cdot w_i
$$

where the sum is over heads, `k_j` is the block's representative key (same one computed for the compressed path), `w_i` is a learned per-query weight vector, and the output is a scalar relevance score. The top-k block indices by score are passed to the sparse attention kernel.

This is what the DeepGEMM entry calls the **V3.2 Lightning Indexer**: the `fp8_mqa_logits` and `fp8_paged_mqa_logits` kernels implement exactly this scoring. The kernel shape is `[Q_tokens, n_heads, d_head] × [KV_blocks, d_head]` with a fused ReLU-and-sum epilogue — a very narrow GEMM (large K/N, small M) that produces one scalar per (query, block) pair.

Once top-k blocks are identified, the **V3.2 indexer** materializes the sparse block indices as a compact list of (query_idx, key_block_idx) pairs. DeepGEMM's `m_grouped_fp8_gemm_*` kernels then dispatch each selected block's attention as a grouped GEMM — the same MoE-layout grouped GEMM that handles expert dispatch in V3 training. This is the architectural reason NSA and DeepGEMM are tightly coupled: the MoE grouping infrastructure was already built; NSA's block selection uses the same dispatch mechanism.

Attention memory for the selected path: k × B × d_head × n_heads × 2 bytes = 16 × 64 × 128 × 128 × 2 ≈ **34 MB per layer** at BF16 — versus 4.3 TB for full attention. A reduction of roughly 130,000×.

Attention compute: O(k·B·d_head) per head = O(1024 × 128) = O(131k) per head per query, versus O(128k × 128) = O(16M) for full attention. About 120× reduction in attention FLOPs per head.

### Path 3 — Window (Local Coherence)

A standard causal sliding window over the w most recent tokens (w typically 512–1024). No selection, no gating. This path handles local syntactic coherence, sentence-level context, and recent state — things that deterministically require proximity. This path is dense attention but over a small fixed window, so its cost is O(w) per query regardless of total sequence length.

FlashAttention's tiled kernel is used here without modification; the window mask is applied as a standard causal-with-window mask pattern that FA2 already supports.

## The V3.2 Indexer: Kernel-Level View

The `fp8_mqa_logits` kernel lives in `csrc/jit_kernels/impls/smxx_fp8_mqa_logits.hpp` (DeepGEMM repo, added in PR #200). Its role:

1. **Prefill (non-paged)**: `Q ∈ [T_q, n_h, d_h]` × `KV ∈ [T_kv, d_h]` → `logits ∈ [T_q, n_h, T_blocks]` after block-level pooling and ReLU-weighted sum. KV is stored contiguously in memory.

2. **Decode (paged)**: `fp8_paged_mqa_logits` handles the paged KV cache variant, where KV blocks may not be contiguous. The paged variant iterates over the block table to gather the required key blocks before scoring.

In both cases, the result is a per-query, per-head scalar relevance for each KV block. Top-k selection is done in Python/host code on this small matrix (`T_q × n_blocks`), and the resulting sparse index list is passed back to the grouped GEMM kernel for the actual attention computation.

The ReLU in the scoring function is not arbitrary — it enables the scores to be zero for clearly irrelevant blocks (hard exclusion) rather than merely small. It also makes the gating function non-negative, allowing the weights `w_i` to be interpreted as attention amplitudes rather than having cancellation.

## Training Considerations

### Differentiability of Block Selection

Top-k selection is not differentiable in the standard sense: a small change in a score that flips which block is in the top-k causes a discontinuous change in the output. Two standard approaches exist:

- **Straight-through estimator (STE)**: forward pass uses hard top-k; backward pass treats the selection as if it were soft (passes gradients through as if all blocks were selected). Simple but introduces gradient noise.
- **Gumbel-softmax relaxation**: replace the hard selection with a softmax with a temperature parameter that is annealed toward hard selection during training. Differentiable throughout; computationally more expensive.

V3.2 uses a form of STE with auxiliary losses to manage gradient quality. The gating function (the `fp8_mqa_logits` scoring) receives gradients only through the selected blocks, which means unselected blocks get no signal about how their scores should change. Without intervention, this leads to routing collapse where a small subset of blocks gets perpetually selected.

### Load Balancing: The Sparse Analog of MoE Routing Losses

The problem mirrors MoE expert load balancing (covered in `../v3-tech-report/`). V3.2 adds an **auxiliary coverage loss**:

$$
\mathcal{L}_\text{cover} = \lambda \sum_j \left( f_j - \frac{1}{N_\text{blocks}} \right)^2
$$

where `f_j` is the empirical fraction of queries that select block j. This penalizes concentration of selections on a small number of "always-selected" blocks (typically the most recent window or highly salient early context). Without this loss, blocks in the beginning (BOS context) and end (recent window) tend to monopolize selections, and blocks in the middle go undertrained.

The `λ` coefficient is a significant hyperparameter: too large and it forces uniform selection that destroys the relevance signal; too small and collapse resumes. V3.2 reports training from scratch with NSA, avoiding the retrofit problem where a pretrained dense model must relearn its attention patterns.

### Training from Scratch vs Retrofitting

Training NSA from scratch is substantially cleaner than retrofitting to a dense model. A pretrained dense model distributes information across all attention positions in ways that assume full attention; introducing sparsity collapses many of those information paths and requires full relearning. V3.2's approach — initialize with sparse attention from the first training step — means the model never learns to rely on positions that will later be masked out.

The cost: the first ~100B training tokens are harder for the model because sparse attention provides less signal per step. In practice, V3.2 compensates by using a warm-up schedule for the coverage loss `λ` and starting with wider windows in the window path.

## Memory and Compute Summary

| Path | Tokens attended per query | Attention memory (per layer, S=128k) | Attention FLOPs per head |
|---|---|---|---|
| Full (baseline) | 128,000 | ~4.3 TB | O(16.4M) |
| Compressed | ~2,000 (S/B) | ~1 MB | O(256k) |
| Selected (k=16, B=64) | 1,024 | ~34 MB | O(131k) |
| Window (w=512) | 512 | ~17 MB | O(65k) |
| NSA total | ~3,536 | ~52 MB | ~O(450k) |

The NSA total is roughly **83× less attention memory** than full attention at this context length, and this ratio grows with sequence length: at S=256k it would be ~330× less, since full attention scales quadratically while NSA's selected path scales linearly with k·B (fixed) plus S/B (linear in S) for the compressed path.

The indexer kernel adds overhead: scoring `T_q × N_blocks` = `128k × 2000` logits per layer per head. This is a large matrix multiply but at lower precision (FP8) and with aggressive fusion; DeepGEMM's `fp8_mqa_logits` kernel is designed to be bandwidth-bound, not compute-bound, and runs at a fraction of the cost of the main attention.

## Comparison with Other Sparse Attention Approaches

| Approach | Pros | Cons | When to use |
|---|---|---|---|
| **NSA (V3.2)** | Learned, end-to-end trained; composes with FlashAttention within selected blocks; integrates with paged KV and FP8 | Complex training (gradient, load-balance); requires CUDA kernel for indexer; 3-path architecture increases code surface | Long-context MoE models trained from scratch; when task distribution is known and consistent enough for gating to learn |
| **Longformer / BigBird (fixed pattern)** | Simple; no training overhead; predictable memory pattern | Pattern not task-adaptive; global tokens are fixed (CLS, SEP) not learned; quality degrades for retrieval-heavy tasks | Applications with well-defined local + global structure (document classification, QA over fixed schema) |
| **FlashAttention (dense + IO-optimal)** | Exact; no approximation; widely available; works for all tasks | O(S²) compute and memory; cannot extend to 128k+ without sequence parallelism | S ≤ 32k or with strong CP support; when quality on every position is required |
| **Sliding Window only** | Very simple; O(S·w) memory | No long-range retrieval; misses cross-document dependencies entirely | Text generation where recent context dominates (next-token completion, streaming generation) |
| **Ring Attention (sequence parallel)** | Exact; distributes memory across GPUs linearly; orthogonal to other techniques | Requires multiple GPUs even per sequence; latency limited by ring communication; O(S²/P) per device still quadratic | Multi-GPU serving or training where adding devices to a sequence is acceptable; complements NSA (outer-loop parallelism + inner-loop sparsity) |

## Results

V3.2 reports near-lossless quality on RULER (a long-context synthetic benchmark testing retrieval, tracking, and reasoning) and LongBench (real long-document QA and summarization) compared to a full-attention baseline trained on the same data. The key claims:

- **Perplexity** on held-out text is within 0.1–0.2 nats of full attention across context lengths from 4k to 128k.
- **RULER**: selective retrieval and multi-hop tasks, which are the hardest for fixed-pattern sparse methods, show minimal regression. This validates that the learned gating is picking up the right blocks.
- **128k native context** without RoPE interpolation. Full-attention V3 used RoPE with extended base frequency (YaRN-style) to reach 128k; V3.2's selected path only attends within 1024 tokens at a time, so the queries and keys it uses for that path never need to handle position-IDs more than 1024 apart. The position encoding pressure is removed for the dominant compute path.

The load-balancing auxiliary loss is critical to these results — early ablations without it showed 3–5 point RULER regression on multi-hop tasks, exactly the regime where block coverage matters.

## Cross-References

- `../v3-tech-report/` — V3 base architecture (671B MoE, DualPipe, FP8 training, MLA, aux-loss-free MoE balancing). NSA is an extension; V3.2 retains all of V3's training infra.
- `../open-source-week/deep-gemm/` — The `fp8_mqa_logits` and `fp8_paged_mqa_logits` kernels implement the V3.2 indexer scoring. The `m_grouped_fp8_gemm_*_contiguous` and masked variants dispatch the sparse GEMMs for the selected path.
- `../../foundational/ring-attention/` — Complementary approach: Ring Attention reduces per-GPU sequence length (outer parallelism); NSA reduces per-GPU attention compute (inner sparsity). The two compose: each ring shard runs NSA internally.
- `../../foundational/flash-attention/` — NSA's selected path and window path both use FlashAttention for the actual tiled attention computation within each selected block or window. NSA selects which tokens to attend; FlashAttention handles how to attend to them efficiently.
- `../mla/` — MLA compresses KV in the **head dimension** (fewer bits per KV pair). NSA compresses in the **sequence dimension** (fewer pairs attended). They are orthogonal: V3.2 uses both simultaneously, and their memory savings multiply.

## Commentary

NSA is the third pillar of DeepSeek's long-context stack alongside MLA and Ring Attention. Each operates on a different axis of the `S × n_heads × d_head` attention tensor: MLA on d_head, NSA on S, Ring on S/P. What makes NSA industrially significant rather than just academically interesting is the kernel co-design — the V3.2 indexer is not a toy scoring mechanism but a production FP8 kernel integrated into the same grouped-GEMM dispatch infrastructure that V3 uses for MoE. The decision to build NSA on top of the MoE dispatch machinery rather than writing a new sparse attention kernel from scratch is the kind of systems insight that is easy to miss in the paper but obvious when you read the DeepGEMM PR.

The load-balancing problem is also worth taking seriously: the analogy to MoE routing collapse is not superficial. In both cases you have a learned discrete selection under a non-differentiable top-k that will collapse to a small attractor set without explicit regularization. The aux coverage loss for NSA plays the same role as the per-expert bias term in V3's aux-loss-free MoE balancing — and both are necessary to get the trained model to actually use the capacity you designed in.

## References

- [1] DeepSeek-AI. _DeepSeek-V3.2-Exp._ GitHub, 2025. https://github.com/deepseek-ai/DeepSeek-V3.2-Exp
- [2] DeepSeek-AI. _DeepGEMM V3.2 Lightning Indexer._ GitHub PR #200, 2025. https://github.com/deepseek-ai/DeepGEMM
- [3] DeepSeek-AI. _DeepSeek-V3 Technical Report._ arXiv:2412.19437, 2024.
- [4] DeepSeek-AI. _DeepSeek-V2._ arXiv:2405.04434, 2024. (MLA, DeepSeekMoE)
- [5] Beltagy et al. _Longformer: The Long-Document Transformer._ arXiv:2004.05150, 2020.
- [6] Zaheer et al. _Big Bird: Transformers for Longer Sequences._ NeurIPS 2020.
- [7] Liu et al. _Ring Attention with Blockwise Transformers._ arXiv:2310.01889, 2023.
- [8] Dao et al. _FlashAttention-2._ arXiv:2307.08691, 2023.
- [9] Hao et al. _RULER: What's the Real Context Size of Your Long-Context Language Models?_ arXiv:2404.06654, 2024.
