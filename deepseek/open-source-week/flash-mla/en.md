# FlashMLA — Source Walkthrough

- **Repo**: [deepseek-ai/FlashMLA](https://github.com/deepseek-ai/FlashMLA)
- **First released**: 2025-02 (Open Source Week, Day 1)
- **Landmarks**: 2025-04 new kernel ("seesaw" scheduling), 2025-09 FP8 sparse decode (DSA)
- **Companion**: see [`deepseek/mla/`](../../mla/) for the MLA paper write-up

## What this repo is

Production-grade CUDA kernels for **MLA attention on Hopper (SM90) and Blackwell (SM100)**, powering DeepSeek-V3 and V3.2 inference. Four kernel families:

| Kernel | Stage | Mode | KV Format | Arch |
|---|---|---|---|---|
| Dense Decoding | decode | MQA-absorbed | BF16 | SM90 |
| Sparse Decoding | decode | MQA-absorbed | FP8 | SM90 / SM100 |
| Dense Prefill | prefill | MHA | — | SM100 |
| Sparse Prefill | prefill | MQA-absorbed | — | SM90 / SM100 |

Reported performance on H800 SXM5 (CUDA 12.8): **3000 GB/s** memory-bound, **660 TFLOPs** compute-bound for dense decode; **410 TFLOPs** for FP8 sparse decode.

Two deep-dive blogs ship with the repo:
- [`docs/20250422-new-kernel-deep-dive.md`](https://github.com/deepseek-ai/FlashMLA/blob/main/docs/20250422-new-kernel-deep-dive.md) — "seesaw" scheduling
- [`docs/20250929-hopper-fp8-sparse-deep-dive.md`](https://github.com/deepseek-ai/FlashMLA/blob/main/docs/20250929-hopper-fp8-sparse-deep-dive.md) — FP8 sparse

Much of what follows paraphrases them; read the originals for the canonical explanation.

## Repository layout

```
FlashMLA/
├── flash_mla/
│   ├── __init__.py
│   └── flash_mla_interface.py     # Python API: get_mla_metadata, flash_mla_with_kvcache
├── csrc/
│   ├── api/                       # C++/CUDA entry points bound to Python
│   ├── kerutils/                  # Shared kernel utilities
│   ├── sm90/
│   │   ├── decode/dense/          # SM90 dense MLA decode (BF16)
│   │   ├── decode/sparse_fp8/     # SM90 sparse FP8 decode
│   │   ├── prefill/               # SM90 sparse prefill
│   │   └── helpers.h
│   ├── sm100/                     # Blackwell kernels
│   ├── smxx/                      # Arch-agnostic pieces
│   ├── cutlass                    # submodule
│   ├── params.h, defines.h, utils.h
├── benchmark/
├── docs/                          # Deep-dive blogs + assets
├── tests/
└── setup.py
```

Key observation: kernels are **arch-partitioned at the directory level** (`sm90/`, `sm100/`) rather than `#ifdef`-gated. Each kernel family is a separate translation unit, templated on shape parameters but not on architecture.

## Python surface

`flash_mla/flash_mla_interface.py` exposes two functions:

```python
tile_scheduler_metadata, num_splits = get_mla_metadata(
    cache_seqlens, s_q * h_q // h_kv, h_kv, h_q, is_fp8, topk,
)

o_i, lse_i = flash_mla_with_kvcache(
    q_i, kvcache_i, block_table, cache_seqlens, dv,
    tile_scheduler_metadata, num_splits,
    is_causal, is_fp8_kvcache, indices,
)
```

`get_mla_metadata` is called **once per decode step** (not per layer) to produce a tile-scheduler plan that all layers reuse. `flash_mla_with_kvcache` runs the actual kernel. The metadata decouples request-level load balancing from the hot path.

## Why MLA decode is compute-bound

Counter-intuitive given that decode-phase attention is usually memory-bound. The deep-dive's arithmetic (paraphrased):

- FLOPs per request ≈ `2 · h_q · s_q · s_k · (d_k + d_v)`
- HBM bytes ≈ `2 · s_k · d_k` (KV cache dominates; `s_k ≫ h_q · s_q`)
- FLOPs/byte ≈ `h_q · s_q · (d_k + d_v) / d_k ≈ 2 · h_q · s_q`

On H800 with throttled peak ~865 TFLOPs and 3.35 TB/s HBM, the crossover is at `h_q · s_q ≈ 128`. DeepSeek's inference system **doesn't use TP for decode instances** (see the DeepSeek inference overview), so `h_q = 128` even without MTP, pushing MLA decode firmly into the compute-bound regime.

This flips the kernel design problem: you're no longer fighting bandwidth, you're fighting to keep Tensor Cores busy.

## Seesaw scheduling (the 2025-04 kernel)

FlashAttention-3's **ping-pong scheduling** overlaps softmax (CUDA cores) with matmul (Tensor cores) by having two warpgroups each own an output tile and alternate roles. MLA decode can't do this: the per-warpgroup output `[64, 512]` needs 32,768 registers — exactly half of an SM's 65,536 32-bit register file. **Two full output matrices per SM is infeasible.**

The workaround: keep a **single output**, but split it **vertically** into `O_L` and `O_R` (each `[64, 256]`), one per warpgroup. Each step processes two K/V blocks (`K_0, V_0` and `K_1, V_1`), and the two warpgroups cooperate on interleaved updates:

```
0. m ← -inf, O_L ← 0, O_R ← 0   (running max shared; outputs per-WG)
1. [WG0]  p_0 = q · K_0^T / scale
2. [WG1]  p_1 = q · K_1^T / scale
3. [WG0]  scale_0 = exp(m_new - m); m ← max(m, max(p_0))
4. [WG0]  p_0 = exp(p_0 - m_new)
5. [WG0]  O_L = O_L · scale_0 + p_0 · V_{0L}
6. [WG1]  scale_1 = exp(m_new - m); m ← max(m, max(p_1))
7. [WG1]  p_1 = exp(p_1 - m_new)
8. [WG1]  O_R = O_R · (scale_0 · scale_1) + p_1 · V_{1R}
9. [WG0]  p_0 = p_0 · scale_1
10. [WG1] O_R = O_R + p_0 · V_{0R}
11. [WG0] O_L = O_L · scale_1 + p_1 · V_{1L}
```

Mathematically identical to FlashAttention's online softmax. Operationally, CUDA core ops (softmax, scaling) on one warpgroup overlap with Tensor core ops (WGMMA) on the other, and TMA loads for the next K/V block launch as soon as the current block is consumed.

Result: up to **80% of throttled Tensor Core peak** sustained; ~3 TB/s HBM; ~660 TFLOPs on H800. The 2025-04 deep-dive blog walks through the full diagram.

## Memory pipelining tricks

Even in the compute-bound regime, latency matters. Two techniques from the blog:

1. **Fine-grained TMA / GEMM pipelining.** A `[64, 576]` K block is moved by **9 TMA copies of [64, 64]**, not one big copy. GEMM can start on the first sub-block as soon as its TMA completes.
2. **Cache eviction hints.** `cute::TMA::CacheHintSm90::EVICT_FIRST` improves L2 hit rate — counter-intuitively, marking data for early eviction keeps L2 working set tighter.

## Tile scheduler and split-KV

`get_mla_metadata` returns two things: tile-scheduler metadata and `num_splits`. The design mirrors **Flash-Decoding**'s split-KV idea: long KV sequences are split across SMs, each SM computes a partial attention, and a second `combine` kernel merges them via LSE.

The `splitkv_mla` → `combine` pair is overlapped via **Programmatic Dependent Launch** (CUDA 12 feature) so the combine can start preparing as soon as the split finishes, rather than after a full sync.

The tile scheduler balances (request, block) pairs across SMs — decode batches have highly variable `cache_seqlens`, and naive round-robin leaves SMs idle. Metadata is request-level, layer-invariant, so the cost is amortized across all 61 MLA layers in V3.

## FP8 sparse decode (2025-09)

Powers **DeepSeek Sparse Attention (DSA)** introduced with V3.2. Two noteworthy properties:

- **KV cache stored FP8 E4M3**; matmul performed in **BF16**. The kernel dequantizes per-128-element groups using per-group FP32 scales on the way into shared memory.
- **Per-token KV layout is 656 bytes**: 512 bytes FP8 NoPE + 16 bytes (4 × FP32) group scales + RoPE and metadata — all in one contiguous record.
- **Top-k indices** passed in per query to select which KV positions to attend; the kernel gathers sparsely from the paged KV cache.

FP8 KV ≈ halves the effective cache footprint vs BF16 MLA; combined with sparsity, DSA dramatically lowers the memory wall for long-context decode.

## What's reusable outside DeepSeek models

- **MQA-absorbed MLA decode** generalizes to anyone using MLA-style compressed KV (shape `d_k = 576, d_v = 512`).
- **The seesaw schedule** is a concrete recipe for compute-bound decode kernels where the output matrix is too large to duplicate. The pattern applies beyond MLA — any attention variant with `d_v ≥ 256` hits the same register-pressure wall.
- **Tile scheduler + split-KV + PDL** composition is reusable as a template for variable-length-batch decode.
- **FP8 KV with grouped scales** is portable to other models willing to accept that format.

## What's not

The kernels are **specialized to MLA's head dimensions** (`d_k = 576 / 192 / 128`, `d_v = 512 / 128`) and to V3/V3.2 inference patterns (no TP for decode, `h_q = 128`). Dropping into a model with different MLA dims or TP=8 decode would require re-tuning schedule parameters, potentially restructuring the warpgroup split.

## Commentary

FlashMLA is the cleanest example I know of a kernel **co-designed with the inference system**. The "no-TP-for-decode → `h_q = 128` → compute-bound regime" chain is infrastructure reasoning driving the kernel design, not the reverse. The seesaw schedule is a delight: the register-pressure constraint looks like a dead end, and the fix is to split the *output* not the operands — a move that preserves FlashAttention's online-softmax invariant while unlocking the overlap that ping-pong provides. Read this alongside the MLA paper and the V3 technical report to see the full stack: **architecture (MLA) chosen to enable compute-bound decode → system decision (no TP) that puts you in that regime → kernel (FlashMLA) that extracts near-peak utilization**. Every link is non-obvious; the combination is what makes V3 inference fast.

## References

- FlashMLA repo: https://github.com/deepseek-ai/FlashMLA
- New kernel deep dive: `docs/20250422-new-kernel-deep-dive.md`
- FP8 sparse deep dive: `docs/20250929-hopper-fp8-sparse-deep-dive.md`
- FlashAttention-3 (ping-pong): arXiv:2407.08608
- Flash-Decoding: https://crfm.stanford.edu/2023/10/12/flashdecoding.html
- DeepSeek V3.2-Exp (DSA): https://github.com/deepseek-ai/DeepSeek-V3.2-Exp
- DeepSeek inference system overview: https://github.com/deepseek-ai/open-infra-index/blob/main/202502OpenSourceWeek/day_6_one_more_thing_deepseekV3R1_inference_system_overview.md
