# DeepEP — Source Walkthrough

- **Repo**: [deepseek-ai/DeepEP](https://github.com/deepseek-ai/DeepEP)
- **First released**: 2025-02 (Open Source Week, Day 2)
- **Companion**: [`deepseek/v3-tech-report/`](../../v3-tech-report/) — V3 paper's all-to-all and group-limited gating
- **Companion**: [`deepseek/moe/`](../../moe/) — DeepSeekMoE architecture

## What this repo is

A GPU-side **expert-parallel all-to-all** library — the "dispatch" and "combine" halves of every MoE forward. Three kernel families targeting three distinct regimes, with FP8 support and hook-based overlap:

| Kernel | Fabric | Target use |
|---|---|---|
| **Intranode** | NVLink only | Single-node EP |
| **Internode (normal)** | NVLink + RDMA, asymmetric forwarding | Training / prefill, high throughput |
| **Internode low-latency (LL)** | Pure RDMA (+ NVLink since 2025-06) via IBGDA | Decode, latency-critical |

Reported H800 + CX7 400 Gb/s IB numbers (V3 pretrain setting, 4096 tokens/batch, 7168 hidden, top-8 experts, FP8 dispatch + BF16 combine):

- Intranode dispatch/combine: **~153–158 GB/s** (near NVLink peak).
- Internode dispatch/combine at EP=32: **~58/57 GB/s** (RDMA-bottlenecked; note the 50 GB/s NIC is the ceiling — they exceed it by overlapping NVLink within the bottleneck).
- Low-latency at EP=8: **77 µs dispatch / 114 µs combine** per batch of 128 tokens.

## Why a dedicated library

MoE at 256 experts + top-8 routing with cross-node EP is a communication problem first, a compute problem second. Off-the-shelf `all_to_all` over NCCL is far from what the V3 paper reports it needs. Two V3-specific constraints shape DeepEP:

1. **Group-limited gating** — each token dispatches to experts within ≤4 nodes, not uniformly across the whole EP group. This means intra-node (NVLink) and inter-node (RDMA) hops are fundamentally asymmetric: a token first fans out on NVLink, then forwards to RDMA only for the subset that needs to cross nodes.
2. **Decode vs training have opposite objectives** — training wants throughput (tolerate latency to maximize batched bytes/s); decode wants latency (minimize µs per hop, accept lower saturation).

DeepEP therefore ships **two normal kernels plus one LL kernel**, not one generic `all_to_all`.

## Repository layout

```
DeepEP/
├── deep_ep/
│   ├── __init__.py
│   ├── buffer.py                     # Python Buffer class: user-facing API
│   └── utils.py
├── csrc/
│   ├── deep_ep.cpp / .hpp            # PyTorch bindings
│   ├── config.hpp                    # Tuning config descriptors
│   ├── event.hpp                     # EventOverlap wrapper
│   └── kernels/
│       ├── intranode.cu              # NVLink dispatch/combine
│       ├── internode.cu              # NVLink+RDMA normal kernel
│       ├── internode_ll.cu           # Pure-RDMA low-latency kernel
│       ├── layout.cu                 # get_dispatch_layout (who goes where)
│       ├── runtime.cu                # Buffer / queue setup, handle lifecycle
│       ├── ibgda_device.cuh          # IBGDA device-side RDMA primitives
│       ├── launch.cuh, buffer.cuh    # Launch machinery, device buffers
│       ├── configs.cuh, utils.cuh    # Tuning constants, warp helpers
│       └── api.cuh
├── third-party/                      # NVSHMEM (required)
├── tests/                            # test_intranode / test_internode / test_low_latency
└── figures/
```

Three `.cu` files for three regimes — deliberately not unified behind a single kernel. Each one is a couple thousand lines of CUDA tuned for its fabric.

## User surface (`deep_ep/buffer.py`)

The library revolves around one class:

```python
from deep_ep import Buffer, EventOverlap

Buffer.set_num_sms(24)                       # global SM budget for comm kernels
buf = Buffer(group, num_nvl_bytes, num_rdma_bytes)
```

Four main operations:

```python
num_tokens_per_rank, num_tokens_per_rdma_rank, num_tokens_per_expert, \
    is_token_in_rank, event = buf.get_dispatch_layout(topk_idx, num_experts, ...)

recv_x, recv_topk_idx, recv_topk_weights, nrecv_per_expert, handle, event = \
    buf.dispatch(x, topk_idx=topk_idx, topk_weights=topk_weights, ...)

# ... run experts on recv_x ...

out_x, out_topk_weights, event = buf.combine(expert_out, handle, ...)

# Backward of dispatch = combine (adjoint); backward of combine = dispatch.
```

Two design decisions worth naming:

- **`handle`** returned by `dispatch` encodes per-rank / per-expert counts and indices. `combine` reuses it — no recomputation of routing, and the two ops are each other's adjoints (this is why `dispatch_backward` literally calls `combine`).
- **`EventOverlap`** wraps a CUDA event and an allocator stream. Passing a `previous_event` into a dispatch/combine makes the comm kernel wait for that event on the communication stream, unlocking comm-compute overlap without stream surgery in user code.

## Normal kernel: asymmetric NVLink + RDMA forwarding

`csrc/kernels/internode.cu` implements the V3-paper-aligned fast path. Intuition:

1. **Layout phase** (`layout.cu::get_dispatch_layout`) inspects `topk_idx` and computes, per token, which of the ≤4 RDMA peers it needs and which local (NVLink) peers within each.
2. **Dispatch**:
   - Each token is sent intra-node over NVLink to a **forwarder rank** co-located with the target node's RDMA NIC.
   - The forwarder then issues one RDMA write per destination node, batching all tokens for that node.
   - On receive, each destination rank pulls its own experts' tokens from the local RDMA buffer over NVLink.
3. **Combine** is the reverse: experts produce outputs → local NVLink collection → RDMA back to origin nodes → NVLink fan-out to original ranks.

Why asymmetric forwarding beats symmetric all-to-all: a naive design would have every rank RDMA-write to every other rank, burning 8× the NIC traffic. The V3 paper's ≤4-nodes-per-token constraint plus DeepEP's forwarder structure keep RDMA volume bounded at **one write per (token, target-node) pair**, while NVLink absorbs the in-node fan-out for free.

**FP8 dispatch, BF16 combine** halves inbound bytes (tokens dispatched are FP8) but keeps enough precision on the return path where gradients will flow. This is baked into the kernel signatures.

**SM budget**: `Buffer.set_num_sms(N)` caps how many SMs the comm kernel consumes. Matters because the same GPUs run the MoE matmul — giving the kernel more SMs speeds dispatch but starves compute.

## Low-latency kernel: IBGDA, pure RDMA (+ NVLink)

`csrc/kernels/internode_ll.cu` + `ibgda_device.cuh` targets decode, where batches are small (128 tokens), latency is the metric, and throughput barely matters.

- Uses **IBGDA** (InfiniBand GPUDirect Async) via NVSHMEM — the GPU posts WRs directly to the NIC without CPU involvement. Critical for microsecond-scale latencies; a CPU round-trip alone would eat ~20 µs.
- **Pure RDMA** initially (no forwarder hop): each rank writes directly to every destination rank that needs its tokens. The extra NIC traffic is acceptable at decode batch sizes.
- **NVLink reintroduced in 2025-06** ([#173](https://github.com/deepseek-ai/DeepEP/pull/173)) for intra-node paths to avoid unnecessary NIC round-trips — hybrid but still latency-first.
- Decouples the backward "combine" into a separate kernel that can be fused differently with decode compute.

The hook-based overlap hooks here are the most interesting piece:

> "a hook-based communication-computation overlapping method that does not occupy any SM resource"

In normal CUDA-stream overlap, comm and compute run on different streams but both consume SMs. DeepEP's LL kernel exposes a **hook** that inserts the comm into the compute stream's tail, using the NIC's async completion rather than an SM-resident kernel. You pay zero SM budget for the comm — useful when every SM is fully subscribed for expert matmul.

## Intranode kernel

`intranode.cu` is the simplest of the three: NVLink-only, no RDMA, no IBGDA. Exists mainly for (a) single-node dev / test and (b) the NVLink portion of the internode kernels, which internally reuses the intranode primitive as the "local fan-out after RDMA receive" stage. Reading this file first is the recommended entry point if you want to understand the library — concepts carry up to the harder kernels.

## The "undefined-behavior PTX" note

The README flags a `DISABLE_AGGRESSIVE_PTX_INSTRS` env var and a warning:

> DeepEP uses an undocumented PTX instruction `ld.global.nc.L1::no_allocate.L2::256B` … not officially documented, but speedups observed empirically. If the compiler / driver rejects it, disable this flag.

This is a small window into how DeepEP was tuned: **by trying PTX instructions that technically aren't documented to exist for Hopper**, because the hardware does implement them and the assembler accepts them. This is the kind of choice a third-party library cannot really make — only someone training on their own cluster, with their own PTX compiler path, can accept the risk. It's also a factor in V3's cost story.

## Configurations and tuning

`csrc/config.hpp` + `configs.cuh` define per-EP-size tuning configs: NVLink buffer size hints, RDMA buffer size hints, number of channels, queue depth. `Buffer.get_dispatch_config(ep_size)` returns the default — but the README notes:

> You may also replace `get_*_config` with your auto-tuned results via all the tests

i.e. retune per cluster. The `tests/` directory is designed to be run as an autotuner, not just a correctness check.

Network-level advice included in the README:

- **Traffic isolation** across InfiniBand virtual lanes: `NVSHMEM_IB_SL` separates normal-kernel traffic from LL-kernel traffic from other workloads. Without this, a training dispatch behind a latency-sensitive decode will starve the decode.
- **Adaptive routing** on for heavy load, off for light — adaptive adds latency, so LL kernels prefer static.
- **Congestion control off** in DeepSeek's environment. Not a universal recommendation; they observed no contention.

## What's reusable outside DeepSeek

- **The asymmetric NVLink+RDMA forwarding pattern** is the right shape for any MoE with top-k that implicitly or explicitly groups experts by node. Not every MoE does — if your routing is uniform, a symmetric all-to-all may be easier.
- **IBGDA + NVSHMEM for decode-phase EP** is portable to any MoE inference system that can pay the NVSHMEM setup cost.
- **SM-budget-controlled comm kernels + hook-based overlap** is a pattern worth copying regardless of MoE: "let the user size comm against compute" rather than assuming default stream overlap works.
- **FP8 dispatch + BF16 combine** is an easy win if your routing can tolerate the quantization noise (most can).

## What's not

- Baked assumptions: **group-limited gating**, specific hidden sizes (7168) for the reported tuning, DeepSeek's IB network shape. Other MoEs (Mixtral top-2, non-grouped routing) will not exercise the fast paths.
- Depends on **NVSHMEM** and on IBGDA support in your fabric. Not every cluster has this operational.
- Uses PTX instructions at the edge of the documented ISA. Acceptable risk inside DeepSeek; a question mark in a regulated production environment.

## Commentary

DeepEP is probably the most load-bearing of the Open Source Week releases. The V3 paper reports infrastructure numbers that require DeepEP; without it, V3's reported training efficiency on H800 is simply not reproducible. What's interesting about the code is **how tightly coupled it is to V3's architectural choices**: group-limited gating isn't a separate design choice that DeepEP "supports," it's the thing that makes DeepEP fast. This is also why DeepEP doesn't try to be a general NCCL replacement — it's co-designed with a specific MoE shape, and that's exactly the point. For anyone building MoE infra, the two lessons are (a) **treat routing shape and comm kernel as one co-designed object**, not separable layers, and (b) **train and decode need different comm kernels** — the same function call, two kernel implementations, chosen at launch. Expect both patterns to become standard in open MoE stacks over 2026.

## References

- DeepEP repo: https://github.com/deepseek-ai/DeepEP
- V3 technical report (group-limited gating + ≤4 nodes/token): arXiv:2412.19437
- NVSHMEM: https://developer.nvidia.com/nvshmem
- IBGDA overview: NVIDIA HPC-SDK docs
- DeepSeek inference system overview: https://github.com/deepseek-ai/open-infra-index/blob/main/202502OpenSourceWeek/day_6_one_more_thing_deepseekV3R1_inference_system_overview.md
