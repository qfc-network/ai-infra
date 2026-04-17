# NVIDIA Dynamo — Disaggregated Inference Orchestration at Cluster Scale

- **Authors / Org**: NVIDIA
- **Published**: Open-sourced March 2025
- **Links**: [GitHub](https://github.com/ai-dynamo/dynamo) | [Blog: "NVIDIA Dynamo: A Scalable Inference Framework"](https://developer.nvidia.com/blog/nvidia-dynamo-a-scalable-inference-framework/)

## TL;DR

NVIDIA Dynamo is a cluster-level orchestration framework for LLM inference, not a kernel library. It disaggregates prefill and decode into separate worker pools, routes requests to prefill workers that already hold the relevant KV cache, and transfers KV blocks between workers over RDMA/NVLink — translating cache locality into reduced compute waste. Dynamo composes with TensorRT-LLM as its per-GPU compute engine and extends the academic disaggregated inference line (DistServe, Mooncake) to production-grade, hardware-accelerated deployment on H100/B200 clusters.

## Context & Motivation

A production LLM serving cluster has two fundamentally different workload regimes that standard all-in-one serving systems force onto the same GPU:

- **Prefill**: compute-bound. Processing a 1000-token prompt on a single GPU is a large dense matrix multiply — high arithmetic intensity, GPU compute is the bottleneck.
- **Decode**: memory-bandwidth-bound. Generating one token at a time requires loading the full KV cache and all model weights for each step — memory bandwidth is the bottleneck, not compute.

Mixing both on the same GPU forces a bad compromise: the GPU is sized for one regime and underutilized in the other. At scale, this compounds into 30–50% GPU waste according to NVIDIA's internal benchmarks.

The academic framing is in DistServe (see [../../foundational/distserve/en.md](../../foundational/distserve/en.md)) and Mooncake (see [../../moonshot/mooncake/en.md](../../moonshot/mooncake/en.md)): separate the prefill and decode stages into different worker pools. Dynamo is NVIDIA's production implementation of this philosophy, with hardware-level integration (NVLink for intra-node KV transfer, RDMA/InfiniBand for cross-node), a KV-aware router, and an online profiler that dynamically resizes worker pools.

The second motivation is **KV cache reuse across requests**. In a naive serving setup, every request recomputes prefill from scratch even if it shares a long system prompt with a thousand other requests. Dynamo's KV Router tracks which prefill workers hold which prefix KV blocks and routes new requests to workers that already have the matching blocks cached — eliminating duplicate computation.

## Core Abstractions

### Processor

A Processor is a worker process that handles either prefill or decode for a set of requests. Each Processor runs a TensorRT-LLM engine on one or more GPUs. Processors are stateless from Dynamo's perspective — they receive a batch descriptor from the Dynamo layer, execute forward passes, and return KV blocks or generated tokens.

Prefill Processors and Decode Processors are distinct types with different resource profiles:
- **Prefill Processors**: need high compute throughput (H100 SXM preferred), benefit from large L2 cache for weight reuse across long prompts, but have relatively small working KV cache (prefill processes each prompt once and ships the KV out).
- **Decode Processors**: need high HBM bandwidth, benefit from being batched with many concurrent sequences, need larger KV cache to hold active decode state. NVLink-connected multi-GPU setups are preferred to maximize cross-GPU KV bandwidth.

### KV Router

The KV Router is the central scheduler of Dynamo. For each incoming request, it:

1. Tokenizes the input prefix.
2. Queries the distributed KV index to find which Prefill Processor has the most prefix already cached (prefix match length).
3. Routes the request to that Processor — or, if no good match exists, to the least-loaded Prefill Processor.
4. After prefill completes, the KV Router selects a Decode Processor and initiates KV transfer.

This is cache-aware routing: the routing decision is a function of cache state, not just load. The intuition is similar to [SGLang's RadixAttention](../../foundational/sglang/en.md) (match the longest common prefix in the cache tree) but operating at the cluster level across multiple workers rather than within one worker's local cache.

The distributed KV index is maintained by each Prefill Processor reporting its cached prefix set to the router via a gossip-style protocol. The index is eventually consistent; the router may occasionally route to a Processor that has evicted the relevant block between the index update and the request arrival.

### KV Transfer

After prefill, the computed KV blocks must move from the Prefill Processor to the Decode Processor. Dynamo uses:

- **NVLink** for intra-node transfers (within an NVLink-connected multi-GPU server): ~600 GB/s bidirectional. At this bandwidth, transferring the KV cache for a 4096-token prompt in a Llama-70B model (~1.5 GB at FP8) takes ~2.5ms — comparable to one decode step.
- **RDMA/InfiniBand** for cross-node transfers: ~400 Gb/s with RDMA, ~50ms for the same 1.5 GB transfer. This is non-trivial and means cross-node KV transfer adds meaningful latency to time-to-first-token (TTFT).

Dynamo's transfer implementation uses CUDA IPC for intra-GPU and NCCL/UCX for cross-node, with direct GPU-to-GPU peer access bypassing the CPU copy path.

The key constraint this creates: **co-locate prefill and decode workers on the same node when possible** to exploit NVLink bandwidth. Cross-node disaggregation is used when worker pool scaling requires it, accepting the InfiniBand latency penalty.

### Planner

The Planner is Dynamo's online profiler and resource manager. It continuously observes:

- **Prefix hit rate**: fraction of requests that find a significant prefix match in the KV Router. If hit rate drops, the Planner may expand the Prefill Processor pool or adjust eviction policy.
- **Decode length distribution**: p50/p90/p99 of output lengths. Longer decode sequences need more Decode Processor capacity.
- **Queue depths**: per-pool waiting time. If prefill queues are building, add Prefill Processors; if decode queues are building, add Decode Processors.

The Planner makes scaling decisions at the pool level — it does not preempt individual requests. It communicates scale-up/scale-down decisions to the cluster orchestration layer (Kubernetes or NVIDIA Base Command). Scaling a Processor pool involves starting new TensorRT-LLM engine processes, which takes 30–120s for large models (weight loading time), so Planner decisions have significant inertia.

## Integration with TensorRT-LLM

Dynamo handles orchestration; TensorRT-LLM (see [../tensorrt-llm/en.md](../tensorrt-llm/en.md)) handles per-GPU inference. The interface between them:

- Dynamo dispatches a batch to a TRT-LLM engine via gRPC.
- The TRT-LLM engine returns KV blocks for prefill batches, or generated tokens for decode batches.
- Dynamo's KV Transfer layer then moves the KV blocks to the target Decode Processor.

TRT-LLM provides the low-level kernel efficiency (fused attention, FP8 quantization, in-flight batching) that Dynamo relies on for GPU utilization. Dynamo provides the cluster-level routing, KV tracking, and pool management that TRT-LLM does not have.

This separation means Dynamo is, in principle, backend-agnostic. The gRPC interface to the Processor could be served by vLLM or TGI as well, though at launch TRT-LLM is the primary integration target.

## Memory Architecture

In a disaggregated setup, KV cache memory is managed separately in each pool:

**Prefill Processors**: hold KV cache only during active prefill computation, then immediately transfer and evict. KV resident time per request: O(prompt_length / prefill_throughput) — typically 50–200ms. Small working KV footprint; the bottleneck is HBM bandwidth during prefill computation.

**Decode Processors**: hold KV cache for the full duration of decode — from TTFT to last output token. At 100 concurrent sequences of 2048 tokens each in a Llama-70B model: ~30 GB of active KV at FP8. HBM capacity is the primary sizing constraint for decode pools.

The total cluster KV memory is roughly:
```
KV_total ≈ (prefill_batch_size × prompt_len × kv_per_token_prefill)
          + (decode_concurrency × decode_len × kv_per_token_decode)
```
where `kv_per_token` depends on model architecture and quantization precision.

## Engineering Tradeoffs

| Decision | Gained | Gave Up |
|---|---|---|
| Disaggregated prefill/decode pools | Each pool sized for its bottleneck (compute vs. bandwidth); no mixed-regime compromise | KV transfer overhead (2–50ms per request depending on co-location); routing complexity |
| KV-aware routing (cache-aware dispatch) | Eliminates duplicate prefill for shared prefixes; lower average TTFT under high prefix reuse | Router maintains distributed KV index with eventual consistency; stale cache misses under rapid eviction |
| Planner-driven dynamic pool sizing | Adapts to workload shifts without manual intervention | Scaling latency (30–120s to bring new Processor online); Planner can oscillate under bursty load |
| TRT-LLM as compute engine | Best-in-class H100 kernel efficiency; FP8 support; fused operations | Tied to NVIDIA hardware; TRT-LLM engine build time (minutes per model); less flexible than PyTorch backends |
| NVLink-preferred intra-node KV transfer | Near-zero-overhead KV transfer (2–5ms) for co-located workers | Forces tight coupling of prefill/decode within node; cross-node scaling incurs InfiniBand latency penalty |
| gRPC between Dynamo and Processor | Clean separation; backend-agnostic interface | Additional serialization overhead per batch; latency floor higher than in-process batching |

## Production Notes

Dynamo targets H100/B200 cluster deployments at the 10s to 100s of GPUs scale. Key operational characteristics:

- **Minimum useful scale**: disaggregation only pays off when the cluster has enough GPUs to meaningfully separate prefill and decode pools (roughly 8+ GPUs per pool minimum). For smaller deployments, monolithic serving with TRT-LLM directly is simpler.
- **Prefix hit rate sensitivity**: Dynamo's efficiency advantage is strongly dependent on prefix hit rate. For workloads with diverse, non-overlapping prompts (general chatbot traffic), hit rates may be 10–20%; for RAG pipelines with shared document context or coding assistants with shared system prompts, hit rates of 60–80% are achievable.
- **Model update overhead**: when model weights are updated (fine-tune swap, version update), all KV caches across all Processors are invalidated. Dynamo requires a coordinated flush that briefly degrades cache hit rate.
- **Observability**: the Planner's metrics (hit rate, queue depth, decode length distribution) are the primary operational signals. Prometheus/Grafana integration is provided.

## Commentary

Dynamo is best understood as the productization of the DistServe/Mooncake research insight: disaggregating prefill and decode is the right architectural choice for clusters at scale, but doing it properly requires solving five hard systems problems simultaneously — KV routing, KV transfer, pool sizing, cache invalidation, and fault tolerance. The academic papers solved one or two of these cleanly; Dynamo attempts to solve all five with hardware-accelerated infrastructure.

The most technically interesting component is the KV Router. Cache-aware routing at the cluster level is a distributed systems problem that the academic literature understated: maintaining a consistent view of which worker holds which KV prefix blocks across hundreds of workers, under continuous eviction pressure, requires careful consistency tradeoffs. Dynamo's eventual consistency model (gossip-style index updates) is the right call — strong consistency would add intolerable latency to every routing decision — but it means the system must gracefully handle cache misses that result from stale index state.

The long-term question for Dynamo is whether the tight coupling to TRT-LLM and NVIDIA hardware is a feature or a limitation. For enterprises deploying on NVLink clusters with H100s or B200s, the answer is clearly "feature" — the performance gains from FP8 fused kernels and NVLink KV transfer compound. For multi-cloud or heterogeneous deployments, the gRPC abstraction boundary is where the work to support other backends would happen. Watch for Dynamo's KV Router and Planner to evolve toward supporting non-NVIDIA accelerators as the disaggregated inference pattern becomes the dominant serving architecture.

## References

- [1] NVIDIA. _Dynamo: A Scalable Inference Framework._ https://github.com/ai-dynamo/dynamo
- [2] Zhong et al. _DistServe: Disaggregating Prefill and Decoding for Goodput-Optimized LLM Serving._ arXiv:2401.09670. See [../../foundational/distserve/en.md](../../foundational/distserve/en.md)
- [3] Qin et al. _Mooncake: A KVCache-centric Disaggregated Architecture for LLM Serving._ See [../../moonshot/mooncake/en.md](../../moonshot/mooncake/en.md)
- [4] NVIDIA TensorRT-LLM. See [../tensorrt-llm/en.md](../tensorrt-llm/en.md)
- [5] Zheng et al. _SGLang: Efficient Execution of Structured Language Model Programs._ See [../../foundational/sglang/en.md](../../foundational/sglang/en.md)
- [6] Prefix caching. See [../../foundational/prefix-caching/en.md](../../foundational/prefix-caching/en.md)
