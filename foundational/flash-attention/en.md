# FlashAttention (1 / 2 / 3)

- **Authors / Org**: Tri Dao et al. (Stanford → Princeton → Together AI)
- **Published**: FA1 — 2022-05 · FA2 — 2023-07 · FA3 — 2024-07
- **Links**: [FA1 (arXiv:2205.14135)](https://arxiv.org/abs/2205.14135) · [FA2 (arXiv:2307.08691)](https://arxiv.org/abs/2307.08691) · [FA3 (arXiv:2407.08608)](https://arxiv.org/abs/2407.08608) · [code](https://github.com/Dao-AILab/flash-attention)

## TL;DR

Attention is **memory-bound, not compute-bound** — the full `N×N` attention matrix is materialized to HBM and read back, and HBM bandwidth is the bottleneck. FlashAttention reorganizes the computation into **tiled, fused, IO-aware kernels** that never materialize the full matrix: tiles of Q, K, V are streamed through on-chip SRAM, softmax is computed in a numerically-stable online fashion, and the backward pass trades recomputation for memory. FA1 made long-context training tractable; FA2 improved work partitioning and parallelism; FA3 rebuilt the kernel around Hopper's async TMA/WGMMA and added FP8 support. Together, these kernels are now the attention implementation of choice across essentially every frontier lab.

## Context & Motivation

Standard attention:

```
S = Q · K^T        # [N, N]
P = softmax(S)     # [N, N]
O = P · V          # [N, d]
```

The intermediate `S` and `P` each take `O(N²)` memory and, more importantly, `O(N²)` HBM reads/writes. On modern GPUs the compute/bandwidth ratio is so skewed that attention runs at a tiny fraction of peak FLOPs — on A100, standard attention was ~3% of peak.

Previous "efficient attention" work (sparse, linear, low-rank) changed the math to reduce FLOPs. That was the wrong problem. FlashAttention keeps the math exact and attacks the memory hierarchy.

## Core Method

### FlashAttention 1 (2022)

- **Tile Q, K, V into blocks** that fit in SRAM (shared memory / register file).
- For each Q tile, iterate over K/V tiles; compute attention over the tile in SRAM; use **online softmax** (Milakov & Gimelshein 2018) to correctly combine partial results across K/V tiles without ever materializing full `S` or `P`.
- **Backward pass**: rather than store `P` (large), store `O` and the per-row softmax statistics `(m, ℓ)` (small), and recompute `S, P` block-by-block during backward. Extra FLOPs, far less HBM traffic.

Net effect: exact attention, `O(N)` HBM accesses instead of `O(N²)`. At `N=8k` on A100, 3× speedup and 10–20× memory reduction; the speedup grows with `N`.

### FlashAttention 2 (2023)

FA1 under-utilized GPUs on long sequences because its parallel axis was `batch × heads`, which is too coarse when either is small. FA2 adds:

- **Parallelism over the sequence dimension**, not just batch/heads.
- **Better work partitioning between warps** — one warp owns a Q tile and sweeps K/V, eliminating inter-warp softmax reductions.
- **Reduced non-matmul ops** by rearranging where scaling and masking happen.

Result: ~2× faster than FA1 on A100/H100, reaching 50–70% of peak FP16 FLOPs.

### FlashAttention 3 (2024)

Rebuilt from scratch around Hopper-specific features:

- **Async TMA** (Tensor Memory Accelerator) overlaps global-to-shared copies with math.
- **WGMMA** (warpgroup matmul) replaces MMA; larger tile shapes, better async behavior.
- **Ping-pong scheduling**: overlap softmax of one tile with matmul of the next — softmax is on CUDA cores, matmul is on tensor cores, they are disjoint hardware.
- **FP8 support** with incoherent processing to manage outliers.

Result: 75% of H100 peak for FP16, ~1.2 PFLOPs/s for FP8 — roughly 1.5–2× over FA2 on H100.

## Engineering Tradeoffs

| Decision | Gained | Gave up |
|---|---|---|
| Tile + online softmax | Exact attention, `O(N)` HBM | Kernel complexity; careful numerics for softmax across tiles |
| Recompute in backward | Dramatic memory savings, long-context training feasible | ~2× attention FLOPs in backward (still bandwidth-bound, so "free" in wall time) |
| Hand-tuned per-arch kernels | Near-peak utilization | Major engineering rewrite per GPU generation; FA3 is Hopper-only initially |
| FP8 (FA3) | Another ~2× throughput | Outlier handling required; accuracy verification non-trivial |
| Fused single kernel | Best performance | Rigid shape support; head dims, mask types, dropout variants each need a code path |

## Experiments & Results

- FA1: exact BERT attention 3× faster, GPT-2 2.4× faster end-to-end; enabled 16k-context training on single A100.
- FA2: ~2× over FA1 across shapes; 72% of A100 peak FP16.
- FA3: ~75% of H100 peak FP16 (≈740 TFLOPs/s), ~1.2 PFLOPs/s FP8.
- Adoption: default attention in PyTorch (`scaled_dot_product_attention`), vLLM, TensorRT-LLM, essentially all major training stacks.

## Reproducibility Notes

- Fully open source. FA2 is production-grade on Ampere and Hopper; FA3 targets Hopper and is still evolving.
- Integration is straightforward via PyTorch SDPA; custom masks/bias require dropping to the lower-level API.
- On AMD/MI300, analogous work is underway (Composable Kernel FlashAttention, others).

## Commentary

FlashAttention is the cleanest example in recent ML systems of **"the algorithm is fine; the implementation was wrong."** The quadratic memory of attention was treated as a mathematical problem for years — hence linear/sparse attention, Performer, Linformer, etc. — when the actual bottleneck was that no one had written the kernel properly. The lesson generalizes: whenever a workload is running well below roofline, ask what the memory hierarchy is doing before you change the math. FA1→FA2→FA3 is also a case study in **kernel software needing to be rewritten per GPU generation**; expect FA4 for Blackwell, and expect the work to continue compounding rather than settling. Anyone doing serious LLM infra should read the FA2 paper in full at least once — it is the clearest single piece of writing on GPU kernel design for transformers.

## References

- [1] Dao et al. _FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness._ arXiv:2205.14135, 2022.
- [2] Dao. _FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning._ arXiv:2307.08691, 2023.
- [3] Shah et al. _FlashAttention-3: Fast and Accurate Attention with Asynchrony and Low-precision._ arXiv:2407.08608, 2024.
- [4] Milakov & Gimelshein. _Online normalizer calculation for softmax._ arXiv:1805.02867, 2018.
