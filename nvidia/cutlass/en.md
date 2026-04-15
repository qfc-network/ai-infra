# CUTLASS — CUDA Templates for Linear Algebra Subroutines

- **Authors / Org**: Andrew Kerr, Duane Merrill, Julien Demouth, John Tran et al. (NVIDIA)
- **Published**: 2017-11 (initial); continuously updated through CUTLASS 3.x (2023–2024)
- **Links**: [code](https://github.com/NVIDIA/cutlass) · [blog](https://developer.nvidia.com/blog/cutlass-linear-algebra-cuda/) · [docs](https://github.com/NVIDIA/cutlass/blob/main/media/docs/cutlass_3x_design.md)

## TL;DR

CUTLASS is NVIDIA's open-source C++ template library for high-performance GEMM (General Matrix Multiply) and related operations on CUDA GPUs. It exposes the same hierarchical decomposition used inside cuBLAS — threadblock tiles, warp tiles, instruction-level fragments — as composable C++ templates, so you can customize epilogues, fuse operations, and target Hopper-specific hardware instructions without reinventing the underlying machinery. Triton provides productivity (Python DSL, compiler-managed memory); CUTLASS provides control (C++ templates, hand-scheduled memory pipelines, WGMMA/TMA instructions). FlashAttention-2/3, DeepGEMM, and most production inference kernels are either built on CUTLASS directly or share its design patterns so closely that understanding CUTLASS is prerequisite reading.

## Context & Motivation

cuBLAS delivers peak FLOP utilization for standard matrix sizes but is a black box — no customization of epilogues, no fusion of custom activations, no control over memory access patterns. Writing raw CUDA GEMM from scratch to compete with cuBLAS requires deep, simultaneous mastery of:

- Shared memory tiling to hide HBM latency (tiles must fit in SMEM; tile size drives occupancy)
- Register allocation to avoid spilling (large accumulator tiles consume most of the register file)
- Warp-level matrix instructions (WMMA on Volta, MMA on Ampere, WGMMA on Hopper)
- Double-buffering (software pipelining) to overlap compute and memory fetch
- Instruction scheduling to keep tensor cores saturated between memory operations

The failure mode is a kernel that gets one of these right and misses the others, landing at 40–60% of cuBLAS instead of 95%+. CUTLASS solves this by factoring the complexity into reusable, composable C++ templates organized around the same hierarchical decomposition that cuBLAS uses internally — except you can read, extend, and modify every layer.

CUTLASS 2.x (Ampere era, 2020–2022) established the four-level template hierarchy that became the dominant pattern for production GEMM customization. CUTLASS 3.x (Hopper, 2023) rearchitected around **CuTe** — a new tensor algebra DSL embedded in C++ — to make the index arithmetic correct by construction rather than by convention. The 3.x redesign also added first-class support for Hopper's asynchronous hardware units (WGMMA, TMA) that are essential to reaching H100 peak throughput.

## Core Method

### Hierarchical GEMM decomposition

The fundamental problem is: compute **C = A × B + C** where A ∈ ℝ^{M×K}, B ∈ ℝ^{K×N}, C ∈ ℝ^{M×N}.

CUTLASS decomposes this across four hardware levels:

1. **Grid level** — partition (M, N) across threadblocks. Each threadblock owns an output tile of size (BlockM × BlockN) and iterates over the full K dimension.
2. **Threadblock level** — load a (BlockM × BlockK) tile from A and a (BlockK × BlockN) tile from B into shared memory. Iterate over K in steps of BlockK, accumulating into a (BlockM × BlockN) fragment in registers.
3. **Warp level** — each warp owns a (WarpM × WarpN) fragment of the threadblock's C tile. Issues tensor core instructions (WMMA on Volta, MMA on Ampere, WGMMA on Hopper).
4. **Thread / instruction level** — individual tensor core operations on small tiles, e.g., 16×8×16 elements for FP16 on Ampere (the `m16n8k16` MMA instruction).

The critical path for throughput is feeding the tensor cores fast enough to keep them saturated. At each level, the tile size controls the tradeoff between parallelism, register pressure, and shared memory usage. CUTLASS templates these as compile-time constants — changing (BlockM, BlockN, BlockK, WarpM, WarpN) changes the occupancy, register file usage, and SMEM footprint in one step.

### Software pipelining (double-buffering)

The canonical trick for hiding HBM latency is double-buffering: while the warpgroup computes on tile K, asynchronously prefetch tile K+1 from HBM into a second shared memory buffer. In pseudocode:

```
allocate smem_buf[2][BlockM][BlockK]  // two ping-pong buffers
issue_async_copy(A_tile[k=0] → smem_buf[0])
wait_for_copy()

for k in range(0, K, BlockK):
    issue_async_copy(A_tile[k+1] → smem_buf[(stage+1) % 2])
    compute_mma(smem_buf[stage % 2])          // tensor cores read SMEM
    wait_for_copy()
    stage += 1
```

This requires 2× shared memory for the A tile (and similarly for B), but it hides HBM latency behind compute. CUTLASS 2.x implements this as `PipelinedGemmKernel`; CUTLASS 3.x generalizes it to `PipelineAsync` with configurable stage depth (1–5+ stages depending on SMEM budget and desired overlap).

### CuTe: layout algebra for correct index math

CUTLASS 3.x introduces **CuTe** as the foundation layer for tensor layout description and manipulation. A `Layout` in CuTe is a function from a multi-dimensional integer coordinate to a linear memory offset:

```
Layout = (Shape, Stride)
// e.g., a row-major M×K matrix:
auto layout = make_layout(make_shape(M, K), make_stride(K, Int<1>{}));

// Partition layout for threadblock tiling:
auto blk_layout = zipped_divide(layout, make_shape(BlockM, BlockK));
// blk_layout(tile_m, tile_k, inner_m, inner_k) → correct offset
```

CuTe layouts compose via `coalesce`, `zipped_divide`, `tiled_divide`, and `logical_divide` — algebraic operations that partition tensors for each level of the hierarchy without manual index arithmetic. The payoff: the same layout operations that describe A's global memory layout also describe how its tile lands in shared memory and how a warp's fragment reads from it, enforced by type composition rather than by convention.

This eliminates an entire class of bugs: mismatched strides between global memory, shared memory, and register layouts. In CUTLASS 2.x, such mismatches required careful documentation and were a common source of correctness issues when customizing kernels.

### Hopper-specific: WGMMA + TMA

Hopper (H100) introduces two hardware units that require CUTLASS 3.x patterns to exploit:

**WGMMA (Warpgroup Matrix Multiply Accumulate)** — an asynchronous tensor core instruction that spans all 4 warps in a warpgroup (128 threads). Unlike Ampere's synchronous MMA which reads from registers, WGMMA reads directly from shared memory, allowing the register file to remain dedicated to accumulator tiles. This roughly doubles the achievable accumulator size per warpgroup for a given register budget.

**TMA (Tensor Memory Accelerator)** — a hardware DMA unit that copies rectangular tiles from HBM to shared memory with address generation fully offloaded from the warp. Instead of `cp.async` instructions (which still consume warp issue bandwidth), TMA is triggered with a single descriptor and runs independently. Warps issue a TMA instruction, do other work, then wait on a barrier. CUTLASS 3.x wraps TMA as `SM90_TMA_LOAD` and `SM90_TMA_STORE` and pipelines it with WGMMA to achieve near-theoretical FP8/FP16 throughput on H100.

## Engineering Tradeoffs

| Decision | Gained | Gave up |
|---|---|---|
| C++ templates vs cuBLAS black box | Full customization: epilogue fusion, custom layouts, FP8 per-block scaling, arbitrary output types | Long compile times (minutes for a CUTLASS kernel suite); template error messages are notoriously opaque — a type mismatch can produce 100+ lines of errors |
| CUTLASS vs Triton | Fine-grained memory control; Hopper WGMMA/TMA access; peak throughput; ability to hand-schedule instruction sequences | Triton is far more productive for research; CUTLASS requires C++ expertise and awareness of PTX-level hardware details |
| CuTe layout algebra (3.x) | Composable, correct index math enforced by type system; eliminates manual stride arithmetic across hierarchy levels | Steep learning curve; CuTe types (`Tensor<Engine, Layout>`, layout functors) are unfamiliar to most CUDA programmers; documentation is the source code |
| Double-buffering (2× SMEM per buffer) | Full HBM latency hiding; near-roofline throughput on most matrix sizes | Halves available SMEM for tile sizing; limits achievable BlockM × BlockN, which can reduce occupancy or force smaller tiles |
| WGMMA async (Hopper only) | Overlaps compute and SMEM loads at warpgroup granularity; unlocks H100 FP8 peak | Not portable to Ampere without a separate code path; warpgroup-synchronous programming model adds complexity vs per-warp MMA |
| Header-only C++ library | No separate library compilation; easy to vendor or modify | Users pay compile cost; no pre-built binaries to link against; heavy use of CUDA device code in headers creates friction with some build systems |

## Experiments & Results

**Standard GEMM throughput**: CUTLASS achieves 95–98% of cuBLAS throughput on well-tuned shapes (M=N=K=4096, FP16, A100). The remaining 2–5% gap reflects cuBLAS's ability to use un-documented PTX intrinsics and offline profiling at NVIDIA.

**FlashAttention-2**: Uses CUTLASS MMA abstractions for its Ampere attention kernel. FA2 achieves 70%+ MFU on A100 for typical sequence lengths, with the CUTLASS-based kernel accounting for the majority of that performance. The custom epilogue (softmax normalization across the K dimension) is the part cuBLAS cannot provide.

**FlashAttention-3**: The Hopper version uses CUTLASS 3.x WGMMA and TMA explicitly. It achieves 75%+ MFU on H100 FP16 and near-peak for FP8 attention, compared to ~35% MFU for a naive attention implementation.

**DeepGEMM** (DeepSeek): Uses CUTLASS 3.x as the foundation for FP8 GEMM with per-block scaling. Achieves >90% of H100 FP8 peak (roughly 1.9 PFLOP/s on a single H100) by combining CUTLASS's WGMMA pipelines with JIT-compiled tile configurations.

**Epilogue fusion**: CUTLASS's epilogue framework allows bias addition, activation functions (ReLU, GELU), and output scaling to execute in the same kernel pass as the GEMM, eliminating 1–2 separate memory-bound kernel launches per layer. At transformer scale (billions of parameters, thousands of layers per training step), this is a meaningful fraction of end-to-end time.

## Reproducibility Notes

CUTLASS is fully open source at [github.com/NVIDIA/cutlass](https://github.com/NVIDIA/cutlass). It is a **header-only C++ library** — there is no separate compilation of the library itself; you include headers and compile your kernel with `nvcc`.

**Requirements**:
- CUTLASS 2.x: CUDA 11.4+, sm_70 (Volta) or newer
- CUTLASS 3.x / CuTe: CUDA 12.0+, sm_90 (Hopper) for WGMMA/TMA; sm_80 (Ampere) for the rest of 3.x

**Profiler tool**: `tools/profiler` benchmarks any kernel configuration (problem size, data types, tile shapes) without writing host code. This is the fastest way to find the right (BlockM, BlockN, BlockK) for a given (M, N, K) on a specific GPU.

**Python bindings**: `cutlass-py` (`pip install nvidia-cutlass`) provides a Python interface for emitting CUTLASS kernels programmatically, useful for rapid exploration without writing C++ templates directly.

**Build**: CMake with `-DCUTLASS_NVCC_ARCHS=90a` (or 80, 89, etc.) to target specific GPU generations. Debug builds use `-DCUTLASS_ENABLE_TESTS=ON`.

**Testing**: The `test/unit/` directory has exhaustive correctness tests. `test/perf/` has reference performance numbers per shape per GPU. These are the ground truth for "did my change break anything."

## Commentary

CUTLASS occupies a unique position: it is the **documented, open-source implementation of techniques that cuBLAS uses but doesn't expose**. Every non-trivial GPU kernel for LLM training and inference eventually either uses CUTLASS directly, reinvents CUTLASS's patterns, or builds on a library (FlashAttention, DeepGEMM) that does one of the first two.

The right mental model is a **spectrum of abstraction**:
- `torch.compile` → Triton → CUTLASS 3.x + CuTe → raw PTX / SASS

Each step down gives you ~5–15% more performance potential and multiplies the engineering cost by 3–10×. CUTLASS sits at the level where you control memory access patterns at the granularity of individual cache lines and instruction pipelines, but you don't have to write assembler.

Triton (covered separately) is the higher-productivity alternative — and for operations that research kernels need to prototype quickly, Triton wins on time-to-correct-code. But for operations where Hopper-specific hardware is critical — WGMMA feeding from SMEM, TMA for address-generation-free DMA, FP8 with per-block E4M3/E5M2 scaling — CUTLASS 3.x is the reference implementation. The relationship is **complementary, not competitive**: Triton for research kernels, CUTLASS for production kernels where the last 10% of throughput matters and where the operation shape is stable enough to justify the engineering investment.

The CuTe rearchitecture in 3.x is a genuine improvement in software engineering, not just a performance story. The old CUTLASS 2.x had many implicit conventions about how strides composed across levels; CuTe makes those conventions explicit in the type system. Future hardware-targeting — Blackwell, post-Hopper — will be easier to add correctly because the abstraction layer is cleaner.

For anyone building production AI infra in 2026: if you are writing custom attention, quantized GEMM, mixture-of-experts routing, or any other kernel that cuBLAS/cuDNN doesn't cover, you will interact with CUTLASS. Invest in learning CuTe's layout model; it is the part that takes the longest to understand and rewards the most once internalized.

## References

- [1] Kerr et al. 2017. "CUTLASS: Fast Linear Algebra in CUDA C++." NVIDIA Developer Blog. https://developer.nvidia.com/blog/cutlass-linear-algebra-cuda/
- [2] NVIDIA 2023. "CUTLASS 3.0: NVIDIA CUTLASS Design Guide." https://github.com/NVIDIA/cutlass/blob/main/media/docs/cutlass_3x_design.md
- [3] NVIDIA 2023. "CuTe: A Domain-Specific Language for CUDA Tensor Operations." In CUTLASS repo, `include/cute/`.
- [4] Dao et al. 2023. "FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning." arXiv:2307.08691.
- [5] Shah et al. 2024. "FlashAttention-3: Fast and Accurate Attention with Asynchrony and Low-precision." arXiv:2407.08608.
- [6] DeepSeek 2025. "DeepGEMM: Clean and Efficient FP8 GEMM Kernels with Fine-grained Scaling." https://github.com/deepseek-ai/DeepGEMM
