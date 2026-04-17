# Multi-Tenant LoRA Serving — SLoRA, Punica, and CaraServe

- **Authors / Org**: SLoRA: Sheng et al. (UC Berkeley), arXiv:2311.03285, 2023; Punica: Chen et al., arXiv:2310.18547, 2023; CaraServe: 2024
- **Published**: SLoRA: Nov 2023; Punica: Oct 2023; CaraServe: 2024
- **Links**: [SLoRA](https://arxiv.org/abs/2311.03285) | [Punica](https://arxiv.org/abs/2310.18547)

## TL;DR

Serving thousands of fine-tuned LoRA adapters from a single base model is a distinct infrastructure problem from serving a single model. The naive approach — one GPU process per adapter — fails at scale due to GPU utilization collapse. SLoRA solves this by paging adapter weights between CPU DRAM and GPU HBM on-demand, grouping requests by adapter to maximize batch efficiency. Punica solves the orthogonal problem of mixed-adapter batching via the SGMV kernel — a custom CUDA kernel that computes ΔWx for multiple different adapters in one kernel launch. Together they define the two axes of multi-tenant LoRA serving: memory-efficient adapter management and compute-efficient mixed-batch execution.

## Context & Motivation

Fine-tuning with LoRA (see [../lora/en.md](../lora/en.md)) has become the dominant adaptation method for LLMs: freeze the base model, train low-rank matrices ΔW = BA inserted at each attention and FFN projection. A rank-16 adapter on a 70B model adds roughly 200–300 MB of weights; the base model itself is 140 GB at BF16. This structure naturally suggests a serving optimization: **share the base model across many adapters**.

The business case is compelling: a platform serving 1000 different fine-tuned variants of Llama-3-70B would need 1000 GPU servers at 140 GB/adapter in the naive approach. With adapter sharing, you need GPU memory for one base model (140 GB) plus adapter weights that can be paged in and out from CPU DRAM.

The technical challenges are:

1. **Adapter memory management**: 1000 adapters × 200–300 MB = 200–300 GB of adapter weights. This does not fit in GPU HBM alongside the base model. Adapters must be stored in CPU DRAM and loaded on-demand.
2. **Batching efficiency**: if each request uses a different adapter, naive batching serializes all adapter computations — the GPU processes one adapter's ΔW at a time. Throughput collapses to 1/k of a single-adapter baseline for k distinct adapters in a batch.
3. **Cold-start latency**: loading an adapter from CPU DRAM to GPU HBM before the first request can be served adds latency — PCIe bandwidth is ~32 GB/s, so a 300 MB adapter load takes ~10ms.

## LoRA Memory Structure

At each attention and FFN layer, LoRA adds:

```
ΔW = B · A    where B ∈ R^{d×r}, A ∈ R^{r×k}
```

The modified forward pass:

```
output = (W + ΔW) · x = W · x + B · (A · x)
```

`W · x` is the base model computation (shared across all adapters). `B · (A · x)` is the adapter-specific computation (different per adapter, O(r) rank budget).

For a Llama-3-70B with rank-16 LoRA applied to all attention projections (Q, K, V, O) and all FFN projections (gate, up, down) across 80 layers:

```
Adapter size ≈ 80 layers × 7 projections × 2 matrices × rank × dim × dtype_bytes
             ≈ 80 × 7 × 2 × 16 × 4096 × 2 bytes (BF16)
             ≈ ~250 MB
```

At rank 64, this doubles to ~1 GB per adapter. Rank choice is the primary adapter memory tuning knob.

## SLoRA: Paged Adapter Loading

### Core Design

SLoRA treats adapter weights analogously to how PagedAttention (see [../paged-attention/en.md](../paged-attention/en.md)) treats KV cache: as a pool of memory that must be managed under capacity constraints, with eviction and on-demand loading.

**Adapter memory pool**: SLoRA maintains a fixed-size GPU buffer for adapter weights (e.g., 8 GB for adapters on a server with 80 GB HBM for the base model). This buffer holds the currently-loaded adapters. Adapter weights are stored in CPU DRAM when not in use.

**LRU eviction**: when a new adapter is needed and the buffer is full, the least-recently-used adapter is evicted (transferred back or simply dropped if it's not dirty). The eviction policy is the same as any cache system — LRU is simple and performs well under temporal locality.

**PCIe transfer**: adapter loading uses CUDA pinned memory with asynchronous DMA transfers. Loading a 250 MB adapter over PCIe 4.0 (32 GB/s) takes ~8ms. This is the cold-start overhead — the first request for a not-cached adapter pays this cost. Subsequent requests to the same adapter pay nothing.

### Batching Strategy: Adapter-Grouped Batches

SLoRA's scheduler groups requests by adapter: if 10 requests are pending for adapter A and 5 for adapter B, SLoRA forms a batch containing all 15 if both adapters are in the GPU buffer, or a batch of 10 from A if only A is loaded.

This maximizes the ΔW computation reuse within a batch: all requests sharing an adapter A compute `B_A · (A_A · x_batch)` once for the group rather than once per request. The base model computation `W · x_batch` is already batched regardless.

The tradeoff: adapter-grouped batching reduces batching diversity. If the 15 requests have very different output lengths, head-of-line blocking from the shorter sequences waiting for the longer ones to finish reduces GPU utilization. SLoRA partially mitigates this with continuous batching — short sequences complete and are replaced within the batch — but the fundamental tension between adapter locality and sequence-length diversity remains.

### Adapter Scheduling

SLoRA adds an adapter-aware scheduling layer above the base inference scheduler:

1. **Warm adapter check**: when a request arrives, check if its adapter is already in GPU HBM.
2. **Pre-load decision**: if the adapter is not warm, initiate an async PCIe transfer. The request is held in the queue until its adapter is loaded. Requests for warm adapters are dispatched immediately.
3. **Eviction decision**: when buffer is full and a new adapter must be loaded, evict the LRU adapter whose requests are not currently in-flight.

The system maintains an adapter heat map — a running estimate of request arrival rate per adapter — to preemptively load adapters before their requests arrive. This reduces cold-start latency from ~8ms to near-zero for high-traffic adapters.

## Punica: SGMV — Mixed-Adapter Batching via Custom CUDA Kernel

### The Batching Problem

SLoRA's adapter-grouped approach sidesteps a fundamental efficiency problem: what if you *want* to batch requests from different adapters simultaneously? In a multi-tenant system with 1000 adapters and requests arriving at Poisson intervals, grouping by adapter means some adapters always have thin batches (low throughput) and some accumulate large queues (high latency).

Punica solves this with a different approach: **allow arbitrary mixed-adapter batches** and make the ΔW computation efficient across heterogeneous adapters in a single kernel call.

### SGMV: Segmented Gather Matrix-Vector Multiply

The key insight: in a batch of N requests each using a different adapter, the ΔW contributions can be computed as a segmented batched matrix multiply. Each request `j` uses adapter `i_j` with matrices `B_{i_j}` and `A_{i_j}`. The full batch computation is:

```
For each request j in batch:
    delta_output_j += B_{i_j} · (A_{i_j} · x_j)
```

This looks like N independent matrix multiplications, but Punica observes that when adapters share the same rank `r`, the `A` matrices all have shape `r × k` and the `B` matrices have shape `d × r`. We can pack all `A` matrices into a 3D tensor `A_stack[N, r, k]` and all `B` matrices into `B_stack[N, d, r]`, then compute the full batch with a single segmented batched GEMM:

```
hidden = batched_mm(A_stack, x_batch)     # [N, r, 1] = [N, r, k] × [N, k, 1]
delta_output = batched_mm(B_stack, hidden) # [N, d, 1] = [N, d, r] × [N, r, 1]
```

The SGMV kernel implements this directly in CUDA with segment offsets — the "gather" in the name refers to gathering the correct `A` and `B` matrices for each request's adapter from a pre-populated index. The kernel avoids the overhead of N separate kernel launches (each with CUDA kernel launch overhead of ~5–10 µs), replacing them with one launch.

At batch size 32 with 32 distinct adapters, SGMV is ~8× faster than sequential single-adapter ΔW computation. At smaller batch sizes the advantage diminishes; below batch size 4, per-request kernel launches may be faster due to parallelism overhead in SGMV.

### Memory Layout for SGMV

Punica keeps all adapter `A` and `B` matrices in GPU HBM simultaneously — it does not page them to CPU. This limits the number of concurrently-served adapters to HBM capacity / (adapter size × num_adapters_in_pool). For a 80B model on 4× A100 80GB serving rank-16 adapters at 250 MB each:

```
Available for adapters ≈ (4 × 80 GB) - base model (140 GB) - KV cache (~60 GB) = 120 GB
Max concurrent adapters ≈ 120 GB / 250 MB ≈ 480 adapters
```

For deployments requiring more than 480 adapters simultaneously, Punica's full-HBM approach breaks. SLoRA's paging approach handles this case at the cost of cold-start latency.

## CaraServe: Combining Paging and Mixed Batching

CaraServe (2024) synthesizes the SLoRA and Punica approaches:

- Paged adapter loading from CPU DRAM (SLoRA-style) for handling more adapters than fit in HBM.
- SGMV-style mixed-adapter batching for the adapters that are warm in HBM.
- A scheduler that separates "warm" requests (adapter in GPU HBM, can be SGMV-batched) from "cold" requests (adapter on CPU, must wait for PCIe transfer).

The key insight is that SLoRA and Punica are solving orthogonal problems: SLoRA handles memory capacity (more adapters than fit in GPU), Punica handles compute efficiency (heterogeneous adapter batching). CaraServe handles both simultaneously.

## Engineering Tradeoffs

| Decision | Gained | Gave Up |
|---|---|---|
| SLoRA paging (adapter weights in CPU DRAM) | Serve unlimited adapters from one base model; no GPU HBM limit on adapter count | PCIe cold-start latency (~8ms per adapter load); CPU DRAM bandwidth contention under high eviction rate |
| Adapter-grouped batching (SLoRA) | Maximum ΔW reuse within batch; simple implementation | Reduced batching diversity; head-of-line blocking within adapter group; thin batches for low-traffic adapters |
| SGMV mixed-adapter batching (Punica) | True mixed-adapter batching; near-base-model throughput with heterogeneous adapters | All active adapters must fit in GPU HBM simultaneously; ~480 adapter limit on 4× A100 80GB |
| LRU adapter eviction | Simple, low-overhead cache policy; performs well under temporal locality | Poor performance for workloads with uniform adapter popularity (all adapters equally cold) |
| Low adapter rank (r=8 or r=16) | Small adapter memory footprint; faster PCIe load time | Reduced adapter expressiveness; may need higher rank for complex task adaptation |
| Continuous batching with adapter awareness | GPU stays occupied; short sequences complete without blocking | Requires adapter state tracking per in-flight request; scheduling complexity grows with adapter count |

## Experiments & Results

**SLoRA** (from the paper):
- 1000 adapters, Llama-7B base, A100 40GB server: achieves 4× higher throughput than vLLM with individual adapter processes; p99 latency 2× better than single-adapter serving with request queuing.
- Cold-start latency: 7–12ms per adapter (PCIe 4.0 measured); warm-serving latency matches base model serving.

**Punica** (from the paper):
- Mixed-adapter batch of 32 on A100: SGMV achieves 2.5–4× higher throughput vs. SLoRA-style serial ΔW computation.
- At batch size 1, overhead ~10% vs. single-adapter serving; at batch size 32, near-parity with single-adapter throughput on same hardware.

**CaraServe** (combined):
- Handles 2000+ adapters with 90th percentile cold-start latency < 20ms on an 8× A100 server.
- Throughput within 5% of single-adapter optimal at steady state.

## Commentary

The multi-tenant LoRA serving problem is a microcosm of the broader challenge in ML infrastructure: systems built for research settings (one model, one GPU, maximize utilization for that model) do not compose well when the operational reality is "many models, shared hardware, heterogeneous load." SLoRA and Punica each took one clean abstraction from adjacent systems work — paged memory management from vLLM, batched GEMM from cuBLAS — and applied it precisely to the LoRA serving case. The result is a pair of systems that together cover the design space far better than either alone.

The SGMV kernel deserves particular attention as a case study in kernel engineering for ML serving. The problem statement — compute ΔWx for a batch where each sample has a different W — appears to require N kernel launches. SGMV reframes this as a segmented batched GEMM with gather semantics, reducing N launches to 1. The key prerequisite is that all adapters share the same rank r, which is almost always true in practice (rank is a training hyperparameter set by the fine-tuning operator). This "homogeneous structure, heterogeneous weights" property is what SGMV exploits.

Looking forward, the multi-tenant LoRA problem is likely to evolve in two directions. First, higher-rank adapters for reasoning-heavy tasks (rank 64–256 for math/code fine-tunes) increase adapter memory footprint and PCIe load time proportionally — paging strategies will need to be more aggressive. Second, MoE base models (Mixtral, Llama 4; see [../../mistral/mixtral/en.md](../../mistral/mixtral/en.md) and [../../meta/llama4/en.md](../../meta/llama4/en.md)) with LoRA adapters add a new dimension: which experts are active for each adapter instance varies by token, making adapter-level grouping much harder. The combination of MoE routing and multi-tenant LoRA is an open systems problem as of 2024.

## References

- [1] Sheng et al. _S-LoRA: Serving Thousands of Concurrent LoRA Adapters._ arXiv:2311.03285, 2023.
- [2] Chen et al. _Punica: Multi-Tenant LoRA Serving._ arXiv:2310.18547, 2023.
- [3] Hu et al. _LoRA: Low-Rank Adaptation of Large Language Models._ See [../lora/en.md](../lora/en.md)
- [4] Kwon et al. _PagedAttention._ See [../paged-attention/en.md](../paged-attention/en.md)
- [5] Yu et al. _Orca: Continuous Batching._ See [../orca/en.md](../orca/en.md)
- [6] Mixtral of Experts: see [../../mistral/mixtral/en.md](../../mistral/mixtral/en.md)
- [7] Llama 4: see [../../meta/llama4/en.md](../../meta/llama4/en.md)
