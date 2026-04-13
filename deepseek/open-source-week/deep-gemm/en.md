# DeepGEMM — Source Walkthrough

- **Repo**: [deepseek-ai/DeepGEMM](https://github.com/deepseek-ai/DeepGEMM)
- **First released**: 2025-02 (Open Source Week, Day 3)
- **Companion**: [`deepseek/v3-tech-report/`](../../v3-tech-report/) — V3's FP8 fine-grained scaling recipe
- **Companion**: [`deepseek/open-source-week/deep-ep/`](../deep-ep/) — feeds DeepGEMM MoE kernels in inference

## What this repo is

A **JIT-compiled GEMM library** for Hopper (SM90) and Blackwell (SM100), specialized for FP8 (with BF16 kernels added in 2025) and for the matrix shapes DeepSeek actually ships. Deliberately **not** a CUTLASS replacement — the README states the goal is "clean and efficient," with only a limited number of core kernels.

Performance claim: **up to 1550 TFLOPS on H800** (see PRs #74, #78, #81, #86). Matches or beats expert-tuned libraries on the shapes it targets.

Three GEMM flavors cover the entire V3 training + inference stack:

| API | Used for |
|---|---|
| `fp8_gemm_{nt,nn,tn,tt}` | Dense layers (attention projections, MLA up/down, LayerNorm-adjacent matmuls) |
| `m_grouped_fp8_gemm_*_contiguous` | MoE training / prefill — M-axis-grouped, experts concatenated |
| `m_grouped_fp8_gemm_nt_masked` | MoE inference decode — CUDA-graph-friendly, mask-based, consumes DeepEP low-latency output |
| `k_grouped_fp8_gemm_tn_contiguous` | MoE weight backward |
| `fp8_mqa_logits`, `fp8_paged_mqa_logits` | V3.2 Lightning Indexer (DSA) scoring |

## Why "clean" matters here

CUTLASS is the canonical NVIDIA matmul library, but it is also ~1M lines of heavily templated C++ covering thousands of shape combinations. Using it in production means either (a) accepting the default templates (leaves performance on the table) or (b) becoming a CUTLASS expert (long learning curve). DeepGEMM's README is explicit:

> DeepGEMM leverages some concepts from CUTLASS and CuTe, it avoids heavy reliance on their templates or algebras. Instead, the library is designed for simplicity, with only a limited number of core kernel functions. This makes it a clean and accessible resource for learning NVIDIA GPU kernel optimization techniques.

The design choice is **"fewer kernels, hand-tuned, read top-to-bottom"** rather than "every shape covered." DeepSeek can afford this because they only need the shapes their models use.

## Repository layout

```
DeepGEMM/
├── deep_gemm/
│   ├── __init__.py
│   ├── include/deep_gemm/
│   │   ├── common/                   # device-side primitives
│   │   └── impls/                    # device-side kernel implementations
│   ├── legacy/                       # pre-2025-07 refactor
│   ├── testing/
│   └── utils/
├── csrc/
│   ├── python_api.cpp                # PyTorch binding surface
│   ├── apis/                         # host-side entry points per kernel family
│   ├── jit/                          # JIT cache, NVCC/NVRTC drivers
│   ├── jit_kernels/
│   │   ├── heuristics/               # shape → config selection
│   │   └── impls/                    # host-side kernel launchers
│   │       ├── sm90_fp8_gemm_1d1d.hpp
│   │       ├── sm90_fp8_gemm_1d2d.hpp        # the V3 fine-grained recipe
│   │       ├── sm100_fp8_gemm_1d1d.hpp
│   │       ├── sm90_bf16_gemm.hpp
│   │       ├── sm100_bf16_gemm.hpp
│   │       ├── smxx_fp8_mqa_logits.hpp       # V3.2 indexer (non-paged)
│   │       ├── smxx_fp8_paged_mqa_logits.hpp # V3.2 indexer (paged)
│   │       ├── sm{90,100}_bmk_bnk_mn.hpp     # batched variants
│   │       ├── epilogue.hpp
│   │       └── runtime_utils.hpp
│   ├── indexing/
│   └── utils/
├── third-party/                      # CUTLASS submodule for CuTe primitives
├── scripts/, tests/
└── setup.py
```

Notable: `sm90_fp8_gemm_1d1d.hpp` vs `sm90_fp8_gemm_1d2d.hpp` are separate files, not a single templated kernel. Reading either top-to-bottom is tractable.

## The 1d1d vs 1d2d split (V3's fine-grained scaling, in code)

FP8 training requires scaling to contain outliers. V3's recipe is **per-tile scaling on activations (1×128), per-block scaling on weights (128×128)**. DeepGEMM encodes this as two separate kernel variants:

- **`fp8_gemm_1d1d`** — both operands scaled per 1-D tile. Simpler, used in some shapes / contexts.
- **`fp8_gemm_1d2d`** — LHS (activations) scaled per 1-D tile, RHS (weights) scaled per 2-D block. This is the V3 paper's recipe.

The scale data layout differs by arch:

- **SM90**: scales in FP32. Requires TMA-aligned, transposed layout (`get_mn_major_tma_aligned_tensor`).
- **SM100**: scales packed as **UE8M0** (4-per-`int32`), using Blackwell's native micro-scaling format.

UE8M0 is a Blackwell micro-scaling format — an 8-bit exponent, zero mantissa, essentially a power-of-two multiplier — hardware-supported as the scale in FP8 matmuls. Packing four scales per `int32` amortizes load cost.

## JIT compilation — the unusual part

Unlike cuBLAS / CUTLASS / FlashAttention, **DeepGEMM does not ship pre-compiled kernels**. There is no `.so` of kernels. Instead:

1. User calls `fp8_gemm_nt(A, B, ...)`.
2. Python side inspects shapes, selects a config via heuristics (`csrc/jit_kernels/heuristics/`).
3. The C++ JIT module (`csrc/jit/`) emits the kernel source and compiles it with NVCC (or NVRTC if `DG_JIT_USE_NVRTC=1`).
4. The compiled kernel is cached on disk (`$HOME/.deep_gemm` by default, overridable via `DG_JIT_CACHE_DIR`).
5. Subsequent calls with the same shape hit the cache.

Why JIT? Three reasons visible from the code and news:

1. **Shape-specific kernels**: each (M, N, K) block tiling, pipeline depth, TMA layout gets its own compiled binary. AOT would require building thousands of variants.
2. **Installation simplicity**: no long-running build on `pip install`; first-call latency is paid once per shape per machine.
3. **Per-cluster compiler tuning**: `DG_JIT_NVCC_COMPILER` lets you point at a specific NVCC; this matters because compiler choice affected generated SASS quality before NVCC 12.9.

NVRTC was added in 2025-05 (PR #94) for "up to 10× compilation speedup," though with "potential performance loss for some cases." Users can trade compile time for runtime performance.

## The FFMA interleaving story

One of DeepGEMM's most-cited optimizations was **FFMA instruction interleaving**, a post-compilation SASS pass that reordered fused-multiply-add instructions to improve latency hiding. The 2025-04 announcement ("up to 1550 TFLOPS on H800") referenced this.

The 2025-07 news item closes the loop:

> As NVCC 12.9 will automatically do the FFMA interleaving, all post optimizations will be no longer supported.

Two things worth noting:

- DeepGEMM shipped a SASS-level post-processor. This is extremely rare in open-source GPU code — you are literally rewriting the assembler's output.
- NVCC caught up within a few months, making the hack unnecessary. **A lot of what looks like permanent kernel magic is transient compiler lag.** Worth remembering when reading any bleeding-edge GPU code.

## Three MoE layouts

V3 at 256 routed experts forces every MoE kernel to support grouping. DeepGEMM exposes three:

### Contiguous layout — training / prefill

Experts have **the same N and K** (same weight shape), but different `M_i` (token count varies per expert). DeepGEMM concatenates all experts' tokens along M and runs one kernel that sees the boundaries via per-group offsets:

```
[tokens for expert 0 | tokens for expert 1 | ... | tokens for expert 255]
```

Each group's `M_i` must be aligned to the GEMM M block size (exposed via `get_mk_alignment_for_contiguous_layout()`). The kernel processes the whole concatenated tensor in one launch; N and K are fixed per launch, matching the MoE assumption that all experts share shape.

### Masked layout — inference decode

Decode is CUDA-graph-ified for latency. The CPU doesn't know `M_i` at launch time (it's determined by routing at runtime). DeepGEMM's answer: **run the kernel at a worst-case `M_max` and mask off invalid portions**. A mask tensor tells the kernel which M positions per group are valid.

This composes directly with **DeepEP's low-latency output**: DeepEP delivers routed tokens with a mask; DeepGEMM consumes that mask and produces expert outputs in the same CUDA graph. The two libraries were co-designed.

### K-grouped — MoE weight backward

Weight gradient for MoE needs `dW_e = X_e^T @ dY_e` per expert `e`. Here M and N are fixed (weight shape) but K varies (token count per expert). `k_grouped_fp8_gemm_tn_contiguous` is the dual of the M-contiguous kernel for this case.

## V3.2 Lightning Indexer kernels

Added 2025-09 (PR #200), separate from the main GEMM family. The V3.2 paper introduces **DeepSeek Sparse Attention (DSA)**: a lightweight indexer scores token-pair relevance, top-k is selected, and only the selected KV is attended to. The scoring has its own kernel shape:

```
# pseudo-code from README
kv_j = kv[0][j, :] * kv[1][j]          # dequant with scale
out_ij = q[i, :, :] @ kv_j             # per-head logit
out_ij = out_ij.relu() * weights[i, :]  # weighted ReLU
out_ij = out_ij.sum()                  # scalar logit
```

`fp8_mqa_logits` (prefill, non-paged) and `fp8_paged_mqa_logits` (decode, paged KV) implement this. The ReLU activation is **fused into the matmul epilogue** — cheap, critical for the indexer's low-compute-per-byte profile.

## Heuristics and tuning

`csrc/jit_kernels/heuristics/` picks block sizes, pipeline depth, TMA multicast shape based on (M, N, K, arch). The roadmap item

> Better `get_best_configs` modeling

signals this is still improving. For now, setting `DG_PRINT_CONFIGS=1` dumps the chosen config per shape — useful for debugging performance regressions.

Three utility functions interact with the rest of the stack:

- **`set_num_sms(n)`** / `get_num_sms()` — cap SMs used by GEMM (symmetric with DeepEP's `Buffer.set_num_sms`; the two libraries share an SM budget).
- **`set_tc_util(r)`** — set an approximated Tensor Core utilization target for heuristics; lets you trade peak throughput for lower power / better co-location.
- **`transform_sf_into_required_layout`** — convert user-held scaling factors into the TMA-aligned transposed layout the kernel expects. Explicitly "may be slower" — the library signals that real fusion should happen upstream.

## What's reusable outside DeepSeek

- **The 1d1d / 1d2d FP8 scaling recipe** is the clearest open implementation of V3-style fine-grained FP8. Directly usable by any training stack willing to adopt that layout.
- **The masked grouped GEMM + DeepEP-low-latency pairing** is a template for **CUDA-graph-compatible MoE decode**. Anyone building MoE inference at scale will end up reimplementing this pair; DeepGEMM's version is short enough to port.
- **JIT GEMM with on-disk cache** is a reasonable pattern for any domain where shape space is large but each call site is shape-stable. Inference servers, training with dynamic shapes.
- **UE8M0 scaling on Blackwell** — DeepGEMM is one of the first open libraries with production paths using it; useful reference for SM100 adopters.

## What's not

- Ships only **FP8 and BF16**. No FP16, no INT8. A deliberate choice — V3 doesn't need them — but means the library isn't drop-in for older stacks.
- Shape coverage is **what DeepSeek needs**, not universal. Oddly-shaped matmuls may compile but underperform.
- JIT adds **first-call latency per shape**. For serving latency-sensitive workloads with dynamic shapes, plan for cache warmup.
- Some SASS-level tricks relied on pre-NVCC-12.9 compilers. On current toolchains you get equivalent SASS without the post-processor.
- Ampere is on the roadmap but not delivered; SM80 users are out of scope for now.

## Commentary

DeepGEMM is the clearest "we only need our shapes, so our library only handles our shapes" stance in open GPU code. Read alongside the V3 technical report, you can map **every claim in the report about FP8 scaling to a specific kernel variant here** — 1d2d is the recipe, the `transform_sf_into_required_layout` util is the bookkeeping, and the UE8M0 path shows where Blackwell fits in. The co-design with DeepEP (masked grouped GEMM ↔ DeepEP LL output) is the part worth studying most — **MoE inference is not a GEMM problem or an all-to-all problem, it's one pipeline**, and DeepGEMM + DeepEP are the two ends of it. The FFMA interleaving story is the lesson to carry forward: **some kernel optimizations are real algorithmic wins, others are compiler-lag exploits with a 6-month half-life**. Knowing which is which takes reading the commit history, not the README.

## References

- DeepGEMM repo: https://github.com/deepseek-ai/DeepGEMM
- V3 technical report (FP8 scaling recipe): arXiv:2412.19437
- V3.2 paper (DSA, indexer): https://github.com/deepseek-ai/DeepSeek-V3.2-Exp
- CUTLASS: https://github.com/NVIDIA/cutlass
- NVIDIA PTX UE8M0 docs: https://docs.nvidia.com/cuda/parallel-thread-execution/#alternate-floating-point-data-formats
