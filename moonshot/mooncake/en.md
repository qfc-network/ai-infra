# Mooncake — KVCache-Centric Disaggregated Inference

- **Authors / Org**: Qin et al., Moonshot AI + Tsinghua
- **Published**: 2024-06 (preprint), FAST '25 (USENIX)
- **Links**: [paper (arXiv:2407.00079)](https://arxiv.org/abs/2407.00079) · [code](https://github.com/kvcache-ai/Mooncake)

## TL;DR

Mooncake is the inference architecture behind **Kimi**, Moonshot AI's production LLM service. Its core move: **physically separate prefill from decode** onto different GPU pools, treat the **KV cache as a first-class distributed resource** (with its own pool of CPU DRAM and SSD across the cluster), and route requests through a global scheduler that maximizes cache reuse. The result, per the paper: **75–525% throughput improvement** over a vLLM baseline on production traces, while still meeting SLOs. The paper introduced "**KVCache-centric**" as a design philosophy and "**conditioned PD-disaggregation**" as the production answer to the prefill/decode mismatch — and is now the reference model for serious inference systems (vLLM, SGLang, NVIDIA Dynamo all converged toward similar designs).

## Context & Motivation

Standard LLM serving (vLLM circa 2023, TGI, etc.) co-locates prefill and decode on the same GPUs. Two structural problems:

1. **Prefill is compute-bound** (long input sequences, high arithmetic intensity); **decode is memory-bound** (one token at a time, weights re-read per step). Co-locating them means one always wastes the resource the other needs.
2. **KV cache from one request is throwaway** the moment that request ends — even though the next request might share a long prefix (system prompt, multi-turn chat history, RAG context).

At Kimi-scale (very long contexts, multi-turn, heavy prefix sharing), these inefficiencies dominate. Mooncake's contribution is showing that solving them requires re-architecting around the KV cache, not just optimizing kernels.

## Core Method

### Disaggregated prefill / decode

Two GPU pools:

- **Prefill pool** — runs the prompt forward, produces the initial KV cache, then ships it to the decode pool over RDMA.
- **Decode pool** — receives the KV cache, runs autoregressive generation.

Each pool is sized and tuned independently. Prefill GPUs can be heavily batched (compute-bound likes batching); decode GPUs prioritize low latency per step. Batch sizes, parallelism choices (TP/PP), and even GPU types can differ between the two pools.

The KV cache transfer between pools is the new hot path. Mooncake uses RDMA + careful scheduling to keep transfer time within the SLO budget.

### KVCache pool — distributed cache as a first-class resource

CPU DRAM and SSD across all serving nodes are pooled into a **distributed KV cache store**, indexed by (model, prompt-prefix-hash, position). When a new request arrives:

1. The scheduler hashes its prompt prefix and queries the KV cache pool.
2. If a hit is found, the cache blocks are pulled (RDMA from another node's DRAM, or from SSD) directly into the prefill GPU's HBM.
3. Prefill only computes the *uncached* portion of the prompt.
4. Generated KV is written back to the pool for future reuse.

For workloads with heavy prefix sharing (system prompts, multi-turn chat, RAG), this is dramatic: a 50K-token system prompt computed once can serve thousands of subsequent requests at near-zero prefill cost.

### Conditioned PD-disaggregation and the scheduler

The global scheduler ("Conductor") decides per request:

- Which prefill instance to assign — based on cache hit rate, current load, network proximity to the cached blocks.
- When to disaggregate vs colocate — short prompts may not be worth the transfer overhead.
- How to overlap KV transfer with compute — start streaming KV blocks to decode pool while prefill is still running on later layers.

The scheduling is **SLO-aware**: under load, the system prefers to drop or delay requests that wouldn't meet TTFT (time-to-first-token) anyway, rather than admitting them and degrading everyone. The paper shows this prevents the "everything slows down equally" failure mode of naive admission.

### Chunked prefill, layer-wise transfer, KV streaming

Several engineering details that make the architecture viable:

- **Chunked prefill** processes the prompt in fixed-size token chunks, allowing the scheduler to interleave with decode requests on shared resources during burst loads.
- **Layer-by-layer KV transfer** starts shipping cache for layer 0 to the decode pool while the prefill is still computing layer 1 — overlap rather than staged.
- **Cache eviction** uses a priority based on (recency, prefix length, hit rate) — long shared prefixes are sticky.

## Engineering Tradeoffs

| Decision | Gained | Gave up |
|---|---|---|
| Disaggregated prefill/decode | Each pool tuned for its bottleneck; clear scaling story | KV transfer between pools (RDMA bandwidth, latency, complexity) |
| Distributed KV cache pool | Massive savings on shared-prefix workloads | Cache-coherence and eviction logic; cache miss = expensive |
| Global scheduler with cache awareness | Higher hit rates, better SLO adherence | Single hot scheduler component; complex policy surface |
| SLO-aware admission | Tail latency under load is sane | Some requests are dropped that a simpler system would limp through |
| Chunked prefill + layer-wise transfer | Hides PD transfer behind compute | More state to track per request; harder to reason about |
| Heavy reliance on RDMA | Microsecond-scale transfers | Fabric requirement; not portable to commodity Ethernet without significant degradation |

## Experiments & Results

- On synthetic workloads: throughput improvements of **75% to 525%** over a vLLM baseline depending on prefix-sharing intensity.
- On Kimi production traces (Sept 2023 sample): higher request handling under the same SLO, lower TTFT for cache-warm requests.
- KV cache hit rate: high enough that, in some traces, **prefill is the minority of compute** — the cache is doing the work.

Caveat: the 525% number is workload-dependent. On workloads with no prefix sharing (one-shot independent prompts), the gains are much smaller. The architecture's value is proportional to how much your traffic looks like Kimi's (long contexts, multi-turn, RAG).

## Reproducibility Notes

- The reference implementation [Mooncake](https://github.com/kvcache-ai/Mooncake) is open-sourced and integrates with vLLM and SGLang as a transfer engine + KV cache store backend.
- Hard requirements: RDMA fabric (IB / RoCE), enough host DRAM to make the pool meaningful, and a workload that benefits.
- Several follow-on systems are functionally compatible (NVIDIA Dynamo, SGLang's hierarchical KV cache).
- The full Conductor scheduler with SLO awareness is a significant operational effort to deploy at scale.

## Commentary

Mooncake's lasting contribution is the **framing** more than any one technique. By naming the architecture "KVCache-centric," it forced the field to acknowledge that **KV cache is the working set of LLM inference, and inference systems should be designed around managing it**, not around managing compute. Prefill/decode disaggregation was already in the air (DistServe, Splitwise) but Mooncake showed it works at production scale for a real, profitable service.

What's interesting in retrospect is how fast everyone converged. By 2025, vLLM, SGLang, TensorRT-LLM, NVIDIA Dynamo, and most production stacks have either implemented or planned PD-disaggregation and a distributed KV cache pool. The DeepSeek inference-system overview describes a similar architecture for V3/R1 serving. **Mooncake didn't invent these ideas, but it published them with production data, and that anchored the consensus.**

The open frontier is **what to do when prefix sharing is low** — most of the gains depend on it. For agentic workloads (high tool-call branching, less reuse), the cache-pool win shrinks and the disaggregation win has to carry alone. The next-generation question is **how to make the cache pool useful even without prefix sharing** — speculative cache, partial cache reuse, learned prefetch.

## References

- [1] Qin et al. _Mooncake: A KVCache-centric Disaggregated Architecture for LLM Serving._ FAST '25 / arXiv:2407.00079.
- [2] Patel et al. _Splitwise: Efficient Generative LLM Inference Using Phase Splitting._ ISCA '24.
- [3] Zhong et al. _DistServe: Disaggregating Prefill and Decoding for Goodput-Optimized LLM Serving._ OSDI '24.
- [4] Mooncake reference impl: https://github.com/kvcache-ai/Mooncake
- [5] DeepSeek inference system overview: https://github.com/deepseek-ai/open-infra-index/blob/main/202502OpenSourceWeek/day_6_one_more_thing_deepseekV3R1_inference_system_overview.md
