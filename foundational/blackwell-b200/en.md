# Blackwell / B200 Architecture Primer

- **Authors / Org**: NVIDIA
- **Published**: GTC March 2024 (architecture announcement) · GB200 NVL72 system: late 2024 / 2025 availability
- **Links**: [NVIDIA Blackwell Architecture Whitepaper](https://resources.nvidia.com/en-us-blackwell-architecture) · [GB200 NVL72 product page](https://www.nvidia.com/en-us/data-center/gb200-nvl72/)

## TL;DR

Blackwell is NVIDIA's GPU architecture following [Hopper](../hopper-h100/en.md) (H100), introduced at GTC 2024. The B200 GPU and GB200 Superchip (two B200 GPUs + one Grace CPU on one package) are the flagship compute elements. The headline additions over Hopper are **FP4 Tensor Cores** (doubling dense FLOPS again), **NVLink 5** at 1.8 TB/s bisection bandwidth, and the **GB200 NVL72** rack-scale unit — 36 Grace-Blackwell Superchips connected in a fully non-blocking NVLink fabric, presenting as a 130 TB/s flat memory system to software. This primer covers the compute primitives (FP4, new GEMM tile shapes), memory hierarchy (HBM3e, NVLink 5, NVSwitch 4), and the system-level GB200 NVL72 architecture, then maps each to the inference and training workloads that care about them.

## Context: The Progression from Hopper

To understand what Blackwell changes, start from [Hopper's](../hopper-h100/en.md) strengths and limits:

- **Hopper's compute peak** (H100 SXM): 989 TFLOPS FP16, 1,979 TFLOPS FP8 (with sparsity).
- **Hopper's memory**: HBM2e/HBM3, 80 GB, 3.35 TB/s.
- **Hopper's interconnect**: NVLink 4, 900 GB/s total NVLink bandwidth per GPU.

Blackwell's design decisions follow directly from where H100 bottlenecks in production LLM workloads:

1. **Compute/memory bandwidth gap**: inference decode is bandwidth-bound; any additional FLOPS don't help unless bandwidth grows proportionally.
2. **FP8 adoption**: H100 FP8 showed major gains in training and inference; the next step is FP4, pushing models into 4-bit where memory bandwidth is halved again.
3. **Scale-up vs scale-out**: NVLink 4 at 900 GB/s is sometimes the bottleneck for all-reduce in 8-GPU TP; NVLink 5 at 1.8 TB/s addresses this.
4. **Rack-scale coherency**: disaggregated prefill/decode architectures need low-latency cross-node communication; the NVL72 flat fabric directly targets this.

## Compute: FP4 Tensor Cores

### The New Precision

Blackwell introduces **FP4 (E2M1)** as a new compute precision for Tensor Cores:

- 4 bits per value: 1 sign bit, 2 exponent bits, 1 mantissa bit.
- Dynamic range: ±6.0 (vs ±448 for FP8 E4M3).
- Use case: inference weights and KV cache quantization where post-training quantization can absorb the loss.

The FLOPS scaling: same die area as H100 FP8 doubles FLOPS again when operands are FP4.

| Precision | B200 FLOPS (dense) | B200 FLOPS (2:4 sparse) |
|---|---|---|
| FP32 | ~20 TFLOPS | — |
| BF16 / FP16 | ~4.5 PFLOPS | ~9 PFLOPS |
| FP8 | ~9 PFLOPS | ~18 PFLOPS |
| FP4 | ~18 PFLOPS | ~36 PFLOPS |

(Numbers are approximate; consult the official whitepaper for exact specs.)

### Scaling Considerations

FP4 at model weights means 2× memory footprint reduction vs FP8. A 70B model in FP8 requires ~70 GB; in FP4, ~35 GB. A single B200 (192 GB HBM3e) can hold a 70B model in FP4 with substantial KV cache headroom.

The quality loss from FP4 quantization is non-trivial: more sensitive than FP8, requiring either per-group scaling (see [Mixed Precision Training](../mixed-precision/en.md) for the FP8 scaling analog) or activation-aware methods similar to [AWQ](../weight-quantization/en.md). Active research area as of 2025.

### GEMM Tile Shapes

Blackwell retains the `wgmma` async warpgroup GEMM instruction from Hopper (see [Hopper primer](../hopper-h100/en.md)) and adds a new **5th-generation Tensor Core** that natively handles FP4 operands. The tile shapes and scheduling model are consistent with Hopper — existing Triton and CUTLASS kernels targeting H100 wgmma require updates for FP4 but the fundamental programming model is unchanged.

## Memory: HBM3e

B200 GPUs use **HBM3e** (3rd generation High Bandwidth Memory, extended):

- **Capacity**: 192 GB per B200 (vs 80 GB for H100 SXM).
- **Bandwidth**: ~8 TB/s per B200 (vs 3.35 TB/s for H100 SXM).

The capacity jump (80 GB → 192 GB) is the most immediately impactful change for serving. A Llama 3 405B model:

- In BF16: 810 GB → requires 11 H100s (tensor-parallel) or 5 B200s.
- In FP8: 405 GB → requires 6 H100s or 3 B200s.
- In FP4: ~202 GB → fits in **2 B200s**.

This directly affects the minimum tensor-parallel degree required and thus the all-reduce communication overhead.

## Interconnect: NVLink 5 and NVSwitch 4

### NVLink 5

NVLink 5 doubles the per-link bandwidth over NVLink 4:

| Generation | Total NVLink BW per GPU | Notes |
|---|---|---|
| NVLink 4 (H100) | 900 GB/s | 18 links × 50 GB/s |
| NVLink 5 (B200) | 1.8 TB/s | 18 links × 100 GB/s |

For tensor parallelism with all-reduce: in an 8-GPU TP group, each GPU sends/receives `(TP-1)/TP × hidden_size × batch_size × 2 bytes` per all-reduce. Doubling NVLink bandwidth halves the all-reduce latency, directly improving the TP scaling efficiency for large hidden dimensions (e.g., Llama 3 405B's 16384 hidden dim).

### NVSwitch 4

NVSwitch 4 provides the switch fabric connecting multiple GPUs within the NVL72 system. Like its predecessors, it enables **all-to-all at full NVLink bandwidth** — not just point-to-point. This is critical for MoE expert-parallel all-to-all: all 72 GPUs can exchange data simultaneously at full speed.

## GB200 NVL72: Rack-Scale System

The GB200 NVL72 is the key system-level unit:

```
GB200 NVL72
├── 36 Grace-Blackwell Superchips
│   └── Each: 1 Grace CPU (72-core Arm Neoverse V2) + 2 B200 GPUs
│                             connected via NVLink-C2C (900 GB/s, cache-coherent)
│
├── 9 NVSwitch 4 chips (providing the switching fabric)
└── 72 B200 GPUs total, fully non-blocking NVLink fabric
```

### Memory Capacity

Total unified GPU memory: `72 × 192 GB = 13.8 TB`. With the NVSwitch fabric, any GPU can access any other GPU's memory at NVLink bandwidth. From a programming model perspective, the 72 GPUs present as a single **130 TB/s** memory pool (aggregate NVLink bandwidth across all switches).

For reference: a 671B DeepSeek V3 model in FP8 requires ~671 GB. A single NVL72 can hold ~20 copies of it simultaneously.

### NVLink-C2C: CPU-GPU Coherency

The Grace CPU and two B200 GPUs on each Superchip are connected via **NVLink-C2C** — a coherent CPU-GPU interconnect at 900 GB/s bidirectional (450 GB/s each direction). This enables:

- **CPU-GPU unified memory**: the Grace CPU can directly access GPU HBM at NVLink bandwidth, and the GPU can access CPU DRAM at the same speed. Useful for offloading KV cache or optimizer states to CPU memory without the 64 GB/s PCIe bottleneck.
- **Cache coherency**: CPU caches and GPU L2 maintain coherency, enabling fine-grained CPU-GPU collaboration without explicit data copies.

This is the key difference from x86+PCIe-connected systems: in H100 DGX systems, CPU-GPU bandwidth is limited by PCIe 5.0 (~128 GB/s). NVLink-C2C at 900 GB/s makes CPU memory a practical extension of GPU memory.

## Relevance to LLM Workloads

| Workload | Key B200 Advantage | Why |
|---|---|---|
| Decode (latency-critical) | 8 TB/s HBM3e | Decode is bandwidth-bound; 2.4× bandwidth increase directly reduces decode time per token |
| Large model serving (single-node) | 192 GB HBM3e | 405B in FP8 fits in 3 B200s vs 6 H100s; lower TP degree → less all-reduce overhead |
| FP4 inference | 18 PFLOPS FP4 dense | 2× throughput vs FP8 for quantized inference; enables smaller batches to hit MFU targets |
| MoE expert-parallel all-to-all | NVSwitch 4 + NVLink 5 | Full-bisection all-to-all at 1.8 TB/s per GPU; reduces EP communication bottleneck |
| Disaggregated prefill/decode | NVL72 flat fabric | Prefill and decode nodes co-located on same NVL72 fabric; cross-node KV transfer at NVLink speeds |
| Training (large MoE) | Full rack as one system | 671B+ MoE training without cross-node RDMA for within-NVL72 communication |
| KV cache extension to CPU | NVLink-C2C 900 GB/s | Multi-tier KV cache (GPU HBM → Grace DRAM) at bandwidth high enough to be competitive with recompute |

## Engineering Tradeoffs

| Decision | Gained | Gave up |
|---|---|---|
| FP4 Tensor Cores | 2× FLOPS and 2× memory savings vs FP8 | Larger quantization error; requires per-group scaling or activation-aware methods |
| 192 GB HBM3e (vs 80 GB) | More models fit on single GPU; lower TP degree | Higher cost per GPU; greater power draw |
| NVLink 5 at 1.8 TB/s | Faster all-reduce; better TP scaling | Higher NVSwitch cost; H100 software not automatically portable |
| NVLink-C2C CPU-GPU coherency | CPU DRAM as fast memory tier; cache coherency | Arm-only (Grace CPU); locks into NVIDIA's SoC design |
| GB200 NVL72 (72-GPU rack unit) | Rack-scale flat memory fabric; eliminates RDMA within NVL72 | Must buy/deploy as 72-GPU unit; less flexible than individual GPU SKUs |
| Backward compatibility with Hopper SM90 ISA | Existing CUDA/Triton/CUTLASS kernels portable | FP4 and new tile shapes require new kernels; not automatic speedup |

## Comparison: B200 vs H100

| Feature | H100 SXM | B200 |
|---|---|---|
| HBM | 80 GB, 3.35 TB/s | 192 GB, ~8 TB/s |
| Peak FP8 (dense) | 1,979 TFLOPS | ~9,000 TFLOPS |
| Peak FP4 (dense) | — | ~18,000 TFLOPS |
| NVLink BW | 900 GB/s | 1.8 TB/s |
| TDP | 700W | 1,000W |
| System unit | DGX H100 (8 GPUs) | GB200 NVL72 (72 GPUs) |

## Commentary

The GB200 NVL72's most important property is architectural, not numerical: it eliminates the distinction between "intra-node" and "inter-node" for up to 72 GPUs. This directly targets the disaggregated prefill/decode pattern — if prefill and decode are separate processes (as in [DistServe](../distserve/en.md) and [Mooncake](../../moonshot/mooncake/en.md)), the KV transfer between them can now happen at NVLink speeds rather than RDMA InfiniBand speeds.

The 192 GB HBM3e per GPU changes the minimum TP degree for large models. For Llama 3 405B in FP8, dropping from 6 to 3 GPUs for TP cuts the all-reduce count in half — potentially a larger practical speedup than the raw bandwidth increase.

FP4 is the biggest open question: can post-training quantization maintain acceptable quality at 4 bits for frontier-class 70B+ models? Early results are mixed. The answer will determine whether B200 FP4 is a production tool or a benchmark-paper number.

For teams planning infrastructure: the GB200 NVL72's 72-GPU granularity means it is a data-center-level commitment. Individual B200 SXM GPUs in standard DGX-style configurations will follow; the NVL72 is for large-scale inference clusters and training jobs where rack-scale coherency justifies the form factor.

## References

- [1] NVIDIA. _NVIDIA Blackwell Architecture Technical Brief._ GTC 2024.
- [2] NVIDIA. _GB200 NVL72 Product Overview._ 2024.
- [3] NVIDIA. _NVLink 5 and NVSwitch 4 Architecture._ Internal whitepaper, 2024.
- [4] For Hopper predecessor: [Hopper / H100 Architecture Primer](../hopper-h100/en.md).
- [5] For interconnect context: [GPU Interconnect Primer](../gpu-interconnect/en.md).
- [6] For FP4/FP8 quantization context: [Mixed Precision Training](../mixed-precision/en.md), [KV Cache Quantization](../kv-cache-quantization/en.md).
