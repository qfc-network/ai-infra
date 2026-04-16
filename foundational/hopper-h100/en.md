# Hopper / H100 Architecture Primer

_A reference primer, not a paper write-up. The H100 (Hopper microarchitecture, SM90) is the hardware assumed by FlashAttention-3, DeepGEMM, DualPipe, DeepEP, and the FP8 training recipes in DeepSeek-V3. This entry names the key primitives and explains why they matter for each of those systems._

- **Org**: NVIDIA
- **Announced**: 2022-03; volume shipping 2023
- **Links**: [Hopper Architecture Whitepaper](https://resources.nvidia.com/en-us-tensor-core/gtc22-whitepaper-hopper) · [H100 Datasheet](https://www.nvidia.com/en-us/data-center/h100/) · [CUDA PTX ISA (SM90)](https://docs.nvidia.com/cuda/parallel-thread-execution/)

## TL;DR

H100 (SM90 / Hopper) introduces four primitives that collectively change how high-performance CUDA kernels are written: **wgmma** (warpgroup matrix multiply-accumulate, asynchronous), **TMA** (Tensor Memory Accelerator, asynchronous DMA with tensor descriptors), **Thread Block Clusters** (shared memory across SMs in the same GPC), and **native FP8 Tensor Cores** (E4M3 and E5M2). Together they enable a programming model where data movement, compute, and synchronization proceed concurrently — a fundamental shift from Ampere's synchronous mma.sync. FlashAttention-3's warp specialization, DeepGEMM's JIT tiles, and DeepEP's fine-grained dispatch all depend on these primitives. Understanding them makes ~6 entries in this repo legible without consulting external documentation.

## Context: what changed from Ampere (A100)

| Feature | A100 (SM80) | H100 (SM90) | Impact |
|---|---|---|---|
| Tensor Core instruction | `mma.sync` (per warp, blocking) | `wgmma.mma_async` (per warpgroup, async) | Enables compute/data overlap within a kernel |
| Async data movement | `cp.async` (no descriptor) | TMA (tensor descriptor, no register pressure) | Eliminates index arithmetic overhead and register spill |
| SM coordination scope | Thread block only | Thread Block Clusters (shared SMEM across SMs) | Allows fine-grained load sharing within a GPC |
| FP8 support | None | E4M3 + E5M2 Tensor Cores | Enables fine-grained per-tile FP8 GEMM at high throughput |
| HBM bandwidth | 2.0 TB/s (HBM2e) | 3.35 TB/s (HBM3) | Higher compute-to-bandwidth ratio; more register reuse amortizes |
| NVLink bandwidth | 600 GB/s | 900 GB/s (NVLink 4) | All-to-all cost per step drops for MoE expert parallelism |
| SMs | 108 | 132 | Raw throughput increase |

The design philosophy shift: A100 was "give programmers powerful synchronous building blocks." H100 is "give programmers powerful *asynchronous* building blocks with explicit synchronization points." The performance ceiling rises — but the programming model is harder, and naive ports of A100 kernels to H100 often achieve only 50–70% of peak.

## The SM Hierarchy

Understanding Hopper requires understanding the three levels of parallelism inside the chip:

```
H100 GPU
└── GPC (GPU Processing Cluster) × 8
    └── TPC (Texture Processing Cluster) × ~8 per GPC
        └── SM (Streaming Multiprocessor) × 132 total
            ├── Warpgroup (4 warps = 128 threads)
            │   └── Warp (32 threads)
            ├── Shared Memory: 228 KB (configurable split with L1)
            ├── Register File: 256 KB
            └── 4th-gen Tensor Cores (FP8 / FP16 / BF16 / TF32 / INT8)
```

**Thread Block Clusters** are a new abstraction above thread blocks. A cluster groups up to 8 thread blocks scheduled to SMs *within the same GPC*. Blocks in a cluster can access each other's shared memory via distributed shared memory (`cluster.map_shared_memory`, `cp.async.mbarrier` with cluster scope). This is the hardware mechanism behind cooperative loads: one block can prefetch data into its SMEM while another block computes on data in its own SMEM — across SM boundaries.

## wgmma — Warpgroup Matrix Multiply-Accumulate

### What it is

`wgmma.mma_async` is a new PTX instruction targeting the entire **warpgroup** (4 warps, 128 threads) rather than a single warp. It issues an asynchronous matrix multiply-accumulate into the Tensor Core pipeline and returns immediately — the warpgroup threads can proceed to issue the next wgmma or perform other work while the hardware executes the multiply in parallel.

Tile shapes are larger than Ampere's mma:
- For FP16/BF16: 64×16×16 per warp → 64×64×16 per warpgroup minimum; up to 64×256×16
- For FP8: 64×64×32 per warpgroup minimum; effectively doubles arithmetic intensity vs FP16

The asynchrony is controlled by **wgmma fences**: `wgmma.fence.sync.aligned` marks the boundary before issuing wgmmas; `wgmma.commit_group.sync.aligned` commits a group; `wgmma.wait_group.sync.aligned N` waits until at most N groups are in flight. This lets kernels pipeline multiple tiles.

### Why this matters

On A100, `mma.sync` is **blocking**: the warp stalls until the Tensor Core result is in registers. The programmer can interleave memory loads between mma.sync calls, but the compute and the load still contend for warp execution slots. The result: kernels spend significant time in software-managed instruction-level parallelism.

On H100, wgmma is **non-blocking**: the Tensor Core pipeline executes independently of the warpgroup's instruction stream. A kernel can issue wgmma, then issue TMA loads for the next tile, then wait on both — compute and data movement truly overlap.

This is what FlashAttention-3's "warp specialization" exploits: some warps in the warpgroup are designated as "producers" (issuing TMA loads) and others as "consumers" (issuing wgmma), communicating via shared memory barriers. The hardware runs them concurrently within the same SM.

## TMA — Tensor Memory Accelerator

### What it is

TMA is an on-chip DMA engine that moves tiles between global memory (HBM) and shared memory. The programmer creates a **tensor descriptor** (a struct describing the tensor's shape, strides, element type, and the tile size to copy) and issues a copy with a single PTX instruction:

```ptx
cp.async.bulk.tensor.2d.shared::cluster.global [smem_ptr], [desc_ptr], [coords], [barrier];
```

One thread issues this. The TMA engine executes the copy asynchronously, then signals the `mbarrier` (memory barrier) when done. The issuing thread — and the rest of the warpgroup — can do other work immediately.

### Why this matters

On A100, loading a tile from HBM to shared memory requires:
- Calculating per-element global addresses (uses registers)
- Issuing `cp.async` for each element or vector
- Managing register pressure from address computations

At large tile sizes (say, 128×128 BF16), address computation consumes significant register file capacity, leading to register spill. Register spill drops occupancy, which is the primary lever for hiding memory latency.

TMA eliminates all of this. The descriptor is set up once (on CPU or once per kernel launch); the PTX instruction is a single call with coordinates. No per-element address arithmetic, no register pressure from the load path. This is why DeepGEMM can use very large tiles (128×128 or 128×256) with high occupancy, and why FlashAttention-3 can pipeline Q/K/V loads with wgmma computation without register pressure limiting tile size.

### mbarrier — the synchronization primitive

TMA and wgmma both signal `mbarrier` objects (memory-side barriers) on completion. An mbarrier has a phase (0 or 1) and an arrival count; it flips phase when the arrival count reaches its threshold. Producers (TMA loads, cluster peers) arrive at the barrier; consumers (wgmma computations reading the loaded data) wait on it. This decoupled producer-consumer model, managed through mbarriers, is the foundation of every async H100 kernel.

## FP8 Tensor Cores

H100 natively supports two FP8 formats:
- **E4M3**: 4 exponent bits, 3 mantissa bits. Range: ±448. Better precision; used for forward-pass activations and weights where the value range is more constrained.
- **E5M2**: 5 exponent bits, 2 mantissa bits. Range: ±57,344. Better range; used for gradients in backward passes where large values appear during training.

The wgmma instruction supports FP8 operands:
```ptx
wgmma.mma_async.sync.aligned.m64n128k32.f32.e4m3.e4m3
```
Operand A (from registers) and operand B (from SMEM) are both E4M3; accumulator is F32. The Tensor Core hardware handles the format conversion internally.

**Why per-tile scaling is necessary**: FP8's limited dynamic range (~8 bits total) cannot represent the full range of a weight matrix in a single global scale factor. DeepSeek-V3 uses per-tile (1×128 for activations, 128×128 for weights) scaling factors. Each tile is scaled to fit within FP8 range before the wgmma; the per-tile scale factors are multiplied back during accumulation. TMA's tile-level granularity maps directly to this: each TMA copy moves one tile, and the scale factor for that tile is fetched alongside. DeepGEMM implements this in under 300 lines by exploiting this alignment between tile size, TMA, and wgmma.

## Shared Memory Configuration

Each SM has **228 KB** of combined L1 data cache and shared memory (up from 192 KB on A100). The programmer configures the split; typical high-performance kernels use 164–228 KB as shared memory with minimal L1.

228 KB is enough to double-buffer two 128×128 BF16 tiles (2 × 128 × 128 × 2 bytes = 64 KB), leaving room for accumulators and metadata. Double-buffering is the canonical Hopper pattern: while wgmma computes on buffer A, TMA fills buffer B; then swap. The larger SMEM budget is what makes the double-buffer without register spill feasible at H100's tile sizes.

## L2 Cache and HBM

- **L2 cache**: 50 MB (vs 40 MB on A100). Large enough to hold the weight tiles for a single attention layer in most configurations, enabling true weight reuse across a sequence.
- **HBM3 bandwidth**: 3.35 TB/s (vs 2.0 TB/s on A100). Attention and decode-phase inference remain memory-bandwidth-bound; this 1.67× bandwidth increase directly translates to throughput for KV cache reads in serving.
- **HBM capacity**: 80 GB (SXM5) — same as A100 80 GB, but higher bandwidth and faster CPU-GPU transfers via PCIe 5 / NVLink 4.

## NVLink 4 and Fabric

Full coverage in the [GPU Interconnect primer](./gpu-interconnect/). Key H100 numbers:
- **NVLink 4**: 900 GB/s bidirectional per GPU (vs 600 GB/s on A100)
- **NVSwitch 3rd gen**: supports NVL72 (72 H100s in a rack, all-to-all at full NVLink speed)
- **IBGDA**: InfiniBand GPU Direct Async — GPUs issue RDMA over InfiniBand without CPU involvement; DeepEP's low-latency path (`notify_dispatch`) uses this for fine-grained expert dispatch timing

The 50% NVLink bandwidth increase over A100 directly reduces the per-step all-to-all cost for MoE expert parallelism. For V3's 256-expert configuration, each token's dispatch touches multiple nodes; the lower per-byte cost per NVLink hop translates to lower pipeline stall time in DualPipe.

## How These Primitives Compose: the Hopper Kernel Pattern

Every high-performance H100 kernel follows the same pattern:

```
Launch kernel with Thread Block Cluster
│
├── Producer warps:
│   ├── Create TMA descriptors (once)
│   └── Loop:
│       ├── cp.async.bulk.tensor [TMA load tile A → smem_A]
│       ├── cp.async.bulk.tensor [TMA load tile B → smem_B]
│       └── mbarrier.arrive (signal tiles are loaded)
│
└── Consumer warps:
    └── Loop:
        ├── mbarrier.wait (wait for tiles)
        ├── wgmma.mma_async [smem_A × smem_B → reg accumulators]
        ├── wgmma.commit_group
        ├── wgmma.wait_group 1   (keep 1 group in flight)
        └── [epilogue: scale, store result via TMA or registers]
```

The key invariant: **producer and consumer run concurrently**. The mbarrier decouples them. The TMA engine and Tensor Core pipeline execute in parallel with the warp instruction stream. Correct use of `wgmma.fence`, `wgmma.commit_group`, `wgmma.wait_group`, and `mbarrier.wait` determines whether computation is actually overlapped or accidentally serialized.

FlashAttention-3 names its producer warps "load warps" and its consumer warps "math warps." DeepGEMM uses the same pattern but generates the kernel via JIT (Python → PTX) with tile-size parameters selected at runtime. DualPipe's compute-communication overlap uses the same async principle at a coarser granularity: while one pipeline stage computes, the NVLink/IB transfer for the next stage proceeds in parallel.

## Engineering Tradeoffs

| Decision | Gained | Gave up |
|---|---|---|
| wgmma (async, warpgroup-level) | Compute/memory overlap; larger effective tiles; higher peak TFLOPS utilization | Programming complexity; incorrect fence placement silently serializes; debugging async compute is hard |
| TMA (descriptor-based DMA) | Zero register pressure from loads; enables large tiles with high occupancy | Descriptor setup overhead (amortized per kernel); limited to contiguous tensor layouts (stride restrictions) |
| FP8 Tensor Cores | ~2× throughput vs FP16/BF16; enables per-tile scaling for fine-grained quantization | 8-bit range requires careful scaling; accumulator must be F32 (no FP8 accumulation); custom kernels needed — cuBLAS FP8 is less flexible than hand-tuned |
| 228 KB shared memory | Large double-buffer tiles; reduced HBM traffic per tile | Limits occupancy (fewer thread blocks per SM) if all SM resources are claimed by one block |
| Thread Block Clusters | Fine-grained shared memory sharing across SMs; better load balancing within GPC | Restricts block placement (must co-locate in same GPC); limited to 8 blocks per cluster; hardware support varies by problem size |

## What This Means for the Repo's Existing Entries

| Entry | H100 primitives it uses | How |
|---|---|---|
| [FlashAttention-3](../flash-attention/) | wgmma, TMA, mbarrier | Warp specialization: producer warps TMA-load Q/K/V tiles; consumer warps wgmma-compute attention; mbarrier decouples them |
| [DeepGEMM](../../deepseek/open-source-week/deep-gemm/) | wgmma (inline PTX), TMA, FP8 Tensor Cores | JIT-generates PTX using wgmma.mma_async.e4m3; per-tile FP8 scaling maps to TMA tile granularity |
| [DualPipe](../../deepseek/open-source-week/dualpipe/) | Async CUDA streams, NVLink 4 | Bidirectional pipeline hides compute behind NVLink transfers; NVLink 4's 900 GB/s reduces per-transfer stall |
| [DeepEP](../../deepseek/open-source-week/deep-ep/) | IBGDA, NVLink 4, async streams | Low-latency path uses IBGDA to issue RDMA without CPU; normal path overlaps IB transfers with compute via async streams |
| [Mixed Precision Training](../mixed-precision/) | FP8 Tensor Cores, E4M3/E5M2 | H100's per-format accumulation + F32 accumulator is the hardware that makes V3's FP8 training viable |
| [Triton](../triton/) | wgmma (via `tl.dot`), TMA (via `tl.load` with `cache_modifier`) | Triton targets SM90 in its MLIR backend; wgmma is exposed through the block-level `tl.dot` abstraction |

## Reproducibility Notes

- H100 SXM5 (data-center) and H100 PCIe are the same Hopper die with different memory and connectivity configurations. SXM5 has full NVLink 4 + 900 GB/s; PCIe variant has no NVLink.
- H800 (export-controlled variant for China): same SM90 die, same compute primitives (wgmma, TMA, FP8), but NVLink bandwidth capped at ~400 GB/s. DeepSeek's V3 and OSW kernels were developed on H800 — the on-chip compute primitives are identical; the inter-node communication is where H800 suffers.
- **CUDA 12.0+** required for wgmma and TMA. Most production frameworks (PyTorch 2.1+, vLLM, SGLang) require CUDA 12 on H100 for full performance.
- Learning resources: NVIDIA's [Hopper Architecture Whitepaper](https://resources.nvidia.com/en-us-tensor-core/gtc22-whitepaper-hopper), the [CUDA PTX ISA guide (SM90 section)](https://docs.nvidia.com/cuda/parallel-thread-execution/), and the FlashAttention-3 paper appendix (most readable explanation of wgmma/TMA in a real kernel).

## References

- [1] NVIDIA. _NVIDIA H100 Tensor Core GPU Architecture._ Whitepaper, 2022.
- [2] NVIDIA. _Parallel Thread Execution ISA Version 8.x (SM90)._ developer.nvidia.com, 2024.
- [3] Shah et al. _FlashAttention-3: Fast and Accurate Attention with Asynchrony and Low-precision._ arXiv:2407.08608, 2024.
- [4] DeepSeek-AI. _DeepGEMM._ github.com/deepseek-ai/DeepGEMM, 2025.
- [5] DeepSeek-AI. _DeepEP._ github.com/deepseek-ai/DeepEP, 2025.
- [6] DeepSeek-AI. _DeepSeek-V3 Technical Report._ arXiv:2412.19437, 2024. (FP8 training recipe)
- [7] Luo et al. _CUTLASS._ github.com/NVIDIA/cutlass, 2024. (CuTe layout algebra underlying SM90 kernels)
