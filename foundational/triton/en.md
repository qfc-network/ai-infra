# Triton — A Language and Compiler for GPU Kernels

- **Authors / Org**: Philippe Tillet et al., Harvard → OpenAI
- **Published**: MAPL '19 (original); modern Triton released by OpenAI 2021 onward
- **Links**: [paper (MAPL '19)](https://www.eecs.harvard.edu/~htk/publication/2019-mapl-tillet-kung-cox.pdf) · [code](https://github.com/triton-lang/triton) · [docs](https://triton-lang.org)

## TL;DR

Triton is **a Python-embedded DSL plus an MLIR-based compiler** for writing GPU kernels at the **block level** instead of the thread level. You describe a kernel as operations on tiles of memory — load this block, do this matmul, store that block — and the compiler handles thread mapping, shared-memory allocation, software pipelining, and instruction selection. The result: you can write a competitive FP16 matmul or fused attention in **dozens of lines of Python** instead of hundreds of lines of CUDA. Triton powers FlashAttention's PyTorch reference, vLLM's custom kernels, Liger Kernels, Unsloth, large parts of `torch.compile`, and most modern open-source GPU kernel work outside the strict CUTLASS / CUDA C++ camp.

## Context & Motivation

Pre-Triton, writing a competitive GPU kernel meant choosing among:

1. **CUDA C++** with hand-coded thread layouts, shared memory tiling, and warp shuffles. Maximum control, slow to write, painful to retarget.
2. **CUTLASS** templates. Faster than raw CUDA for matmul-shaped problems, but a steep learning curve and not easy to extend to non-standard ops (fused attention, custom epilogues).
3. **Vendor libraries** (cuBLAS, cuDNN). Black boxes; no extension; no insight.
4. **PyTorch eager** with kernel fusion via `torch.compile` / TVM / XLA. Auto-fusion gets you ~70% of peak; the last 30% requires hand-tuning.

The gap: **a productive language for a researcher to write a kernel that competes with vendor libraries, in an afternoon.** Triton fills it. Tillet's MAPL '19 paper introduced the block-level abstraction; the OpenAI rewrite (2021+) made it practical and aggressive about real-world performance.

## Core Method

### Block-level programming model

A Triton kernel looks like:

```python
@triton.jit
def matmul_kernel(A, B, C, M, N, K,
                  stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn,
                  BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    a_ptrs = A + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = B + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, K, BLOCK_K):
        a = tl.load(a_ptrs)
        b = tl.load(b_ptrs)
        acc += tl.dot(a, b)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    c_ptrs = C + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    tl.store(c_ptrs, acc.to(tl.float16))
```

Key conceptual moves:

- **Each kernel instance owns a *block* of the output**, not a single thread.
- **Memory ops (`tl.load`, `tl.store`) operate on tiles**, not scalars. The compiler vectorizes and coalesces.
- **Computation (`tl.dot`)** is on tiles too. The compiler chooses Tensor Core MMA shapes, lays out shared memory, and assigns warps.
- **Control flow** (the `for k` loop) drives the K-axis tiling; the compiler software-pipelines it for latency hiding.

The programmer never says "thread 17 owns row 4 of the tile." That mapping is the compiler's job.

### What the compiler does

Triton is built on **MLIR**. The compilation pipeline (simplified):

1. Triton IR — Python AST → high-level tiled IR with `triton.dot`, `triton.load`, etc.
2. Optimization passes: layout selection, software pipelining, masked-load coalescing, redundant-load elimination.
3. Lower to GPU dialect (NVIDIA: PTX → ptxas → SASS; AMD: ROCDL → LLVM AMDGPU).
4. Fall through to vendor assemblers.

The non-trivial passes are:

- **Layout selection** — for a `tl.dot`, choose between different MMA instruction variants (m16n8k16, m16n8k8, etc. on Ampere; WGMMA on Hopper) and the corresponding tile layouts in registers / shared memory.
- **Software pipelining** — overlap `tl.load`s of the next iteration with `tl.dot` of the current iteration. Equivalent to hand-written cp.async pipelines, but generated.
- **Shared memory allocation** — the compiler decides what gets staged through SMEM and computes addresses with bank-conflict avoidance.

You can think of Triton as: **the boring, error-prone parts of CUDA kernel writing, automated; the interesting parts, exposed**.

### Autotuning

Triton ships a built-in autotuner:

```python
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 32}, num_warps=8, num_stages=4),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=4, num_stages=3),
        # ...
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def matmul_kernel(...): ...
```

You declare a search space; Triton benchmarks each config the first time a (M, N, K) signature is seen, caches the winner. This trades cold-start latency for sustained perf — the same pattern DeepGEMM uses but built into the language.

## Engineering Tradeoffs

| Decision | Gained | Gave up |
|---|---|---|
| Block-level abstraction | Kernels in dozens of lines; fast iteration | Some warp-level / instruction-level tricks not expressible without dropping to PTX inline |
| Python embedding | Researchers can use it; integrates with PyTorch | Compile time per kernel signature (mitigated by cache) |
| Compiler autoschedules | No manual thread/warp/SMEM bookkeeping | When the compiler's schedule is wrong, debugging is hard |
| MLIR backend | Multiple GPU targets (NVIDIA, AMD, Intel) | Compiler complexity; backend bugs visible to kernel authors |
| Built-in autotune | Easy way to cover shape variation | Wall-clock first-call cost; need representative shapes during dev |
| JIT only (no AOT) | Same shape-specialization story as DeepGEMM | Production deploys need cache warming or persistent disk cache |

## Experiments & Results

- Original MAPL '19 paper: hand-written Triton matmuls within ~10% of cuBLAS on V100; FFTs and convolutions competitive with cuDNN on a subset of shapes.
- Modern Triton (2024 vintage): FlashAttention-2's **reference implementation is in Triton**; runs at 70%+ of cuBLAS-equivalent peak on Hopper for many shapes. FA3 is partly Triton, partly hand-written CUTLASS.
- vLLM, SGLang, FlashInfer all ship Triton kernels for paged attention, sampling, RMSNorm, RoPE, etc.
- Triton matmul on Hopper reaches comparable performance to cuBLAS at sufficient M/N/K. The gap closes year over year.

## Reproducibility Notes

- Fully open source. `pip install triton` for the OpenAI release.
- Active development: `triton-lang/triton` ships frequent releases; major NVIDIA contributions for Hopper / Blackwell support; AMD CDNA support (rocBLAS-competitive in 2025).
- Debugging tools: `TRITON_INTERPRET=1` runs a kernel in pure Python for correctness checks; `triton.testing.do_bench` for clean perf benchmarks.
- Performance varies sharply by shape and Triton version. Pin the version in production.

## Commentary

Triton is the **single biggest productivity multiplier** in modern GPU kernel work. The "FlashAttention reproduction in Triton in an afternoon" pattern that became standard around 2023 was unthinkable with CUDA C++ alone. The block-level abstraction is the right one — it matches how performant kernels are actually structured (tiled, software-pipelined, MMA-based) without making the programmer write the bookkeeping.

The honest limitations: Triton trails hand-written CUTLASS / CUDA at the bleeding edge by maybe **5–15%** on the most-tuned shapes. For production matmul where every percent matters, you still drop to CUTLASS / CuTe. For everything else — fused attention, sampling, normalizations, custom op research — Triton is the default. **DeepGEMM uses CUTLASS for matmul, but anything DeepGEMM doesn't cover, the rest of the open-source community writes in Triton.**

The deeper lesson is about **abstractions that win**: Triton picked the right level (blocks, not threads), embedded in the right language (Python), with the right escape hatches (PTX inline). Many would-be GPU DSLs failed because they overshot (Halide's polyhedral model) or undershot (kernel autogen with no expressivity). Triton landed in the productive middle. For comparison, **PyTorch's `torch.compile` lowers many ops to Triton internally** — meaning the compiler / DSL distinction is now blurry, and Triton has effectively become PyTorch's GPU IR.

For anyone building infra in 2026: if you can express your kernel in Triton, do — you ship in days, not months, and the perf is usually within striking distance of expert-tuned. Drop to CUTLASS / CUDA only when profiling proves Triton left meaningful performance on the table.

## References

- [1] Tillet et al. _Triton: An Intermediate Language and Compiler for Tiled Neural Network Computations._ MAPL '19.
- [2] Triton repo: https://github.com/triton-lang/triton
- [3] Triton docs: https://triton-lang.org
- [4] FlashAttention Triton reference: in the FlashAttention repo
- [5] OpenAI blog post introducing Triton (2021): https://openai.com/blog/triton/
