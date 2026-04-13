# DistServe — Disaggregating Prefill and Decoding for Goodput

- **Authors / Org**: Zhong et al., Peking University + UC San Diego + Microsoft Research
- **Published**: OSDI '24
- **Links**: [paper (arXiv:2401.09670)](https://arxiv.org/abs/2401.09670) · [code](https://github.com/LLMServe/DistServe)

## TL;DR

The academic sibling of Mooncake: a systematic argument and implementation for **separating prefill and decode onto different GPU groups**, plus the first paper to frame the problem in terms of **goodput** — requests/second that meet both TTFT (time-to-first-token) and TPOT (time-per-output-token) SLOs, not raw throughput. DistServe shows the two phases have fundamentally conflicting resource preferences, co-locating them forces compromises that hurt goodput by ~2–4×, and a well-designed disaggregated system can simultaneously raise goodput and tighten latency tails. The framework of the paper (goodput, phase decoupling, placement + parallelism co-design) is now the standard lens for reasoning about LLM serving.

## Context & Motivation

Serving in 2023 was dominated by continuous batching (Orca, vLLM) with prefill and decode interleaved on the same GPUs. This made sense when the two phases were treated as "just forward passes." But under real SLOs — where an API has to promise both "first token in X ms" and "subsequent tokens every Y ms" — the systems showed pathological behavior:

- Under prefill load, decode requests get starved (head-of-line blocking by large compute-bound chunks), blowing TPOT.
- Under decode load, new prefill requests stall waiting for resources, blowing TTFT.
- Different requests within the same batch want different parallelism (TP/PP/batch size), and a single config has to compromise.

The paper's central observation: **prefill is compute-bound, decode is memory-bandwidth-bound, and they should not share resources.**

## Core Method

### Goodput as the right metric

`throughput = requests/s` ignores whether requests met their SLO. A system that handles 100 req/s at 10× the latency target is not doing better than one that handles 40 req/s within target. DistServe formalizes:

```
goodput = requests/s that meet both SLO_TTFT and SLO_TPOT
```

Everything in the rest of the paper optimizes this, not raw throughput. This single reframing is probably the most reused idea from the paper — later work (Mooncake, Splitwise, NVIDIA Dynamo) inherits it.

### Phase decoupling

Two GPU pools:

- **Prefill instance** — takes a prompt, runs forward, produces KV cache, hands it off.
- **Decode instance** — receives KV, generates tokens autoregressively.

The KV transfer between phases is non-trivial — several hundred MB per request for long prompts. DistServe uses NVLink for intra-node transfer when possible, and RDMA or IB for inter-node, with careful overlap between transfer and the target instance's next computation.

### Independent placement and parallelism

Each pool picks its own **TP / PP / number of replicas** based on its own workload characteristics:

- Prefill: benefits from TP (single large matmul per layer, all-reduce costs amortized over long sequences). PP less useful because pipeline fill dominates for short prompts.
- Decode: benefits from PP (per-token latency is dominated by weight-loading; PP pipelines this across stages), less from TP for small batch sizes.

A unified serving system has to pick one config; disaggregation lets each pool be independently optimal.

### Placement optimizer

The paper includes a **placement algorithm** that, given a cluster and a workload distribution, outputs:

- How many prefill vs decode instances.
- What (TP, PP) each should use.
- Which physical nodes they live on (affinity with KV transfer paths).

Formulated as an optimization over an analytical performance model — not online scheduling, but a one-shot cluster-level plan.

### Latency-aware request routing

Incoming requests are routed to the prefill instance whose current queue best matches the request's prompt length (short prompts go to less-loaded instances to hit TTFT; long prompts are batched to maximize throughput-per-FLOP). Generated KV is then shipped to the least-loaded decode instance.

## Engineering Tradeoffs

| Decision | Gained | Gave up |
|---|---|---|
| Phase decoupling | Each phase optimal; clear SLO ownership | KV transfer cost; extra hops |
| Goodput metric | Aligned with real API contracts | Harder to benchmark than raw throughput — need SLO definitions |
| Independent parallelism per pool | Matches per-phase workload shape | Deploy complexity; config surface |
| Offline placement optimization | Near-optimal cluster layout | Requires workload characterization; adapts slowly |
| Analytical performance model | Fast search over placements | Model accuracy limits the optimizer |
| KV over high-bandwidth interconnect | Transfer within SLO budget | Dependency on fabric (NVLink / IB / RoCE) |

## Experiments & Results

- On production-realistic traces (ShareGPT, LongBench), DistServe achieves **2.0–4.48× higher goodput** vs vLLM and DeepSpeed-MII at the same latency SLOs.
- More striking: under tight SLOs where vLLM drops goodput to near-zero, DistServe still hits ~70% of peak — because vLLM's co-located design simply cannot meet tight TTFT + TPOT simultaneously.
- KV transfer overhead: single-digit milliseconds for NVLink-connected instances; ~10–20ms over RDMA for typical shapes. Small relative to TTFT budget for any non-trivial prompt.

## Reproducibility Notes

- Reference implementation open-sourced. Paper-quality on a research cluster; not a drop-in production system (no admission control, no disk KV pool, no multi-model support).
- Workload traces used in experiments are public (ShareGPT, synthetic distributions).
- The **ideas** have been re-implemented in every major production serving stack since. vLLM has a prefill-decode-disaggregation mode. SGLang natively supports it. TensorRT-LLM added it. NVIDIA Dynamo is built around it.

## Commentary

DistServe and Mooncake arrived within weeks of each other (early 2024), with similar core ideas — and the field treats them as complementary rather than competing. The division of labor:

- **DistServe** is the *academic* framing: goodput, formal placement optimization, clean ablations. Read this for the argument structure.
- **Mooncake** is the *production* framing: cache pools, SLO-aware admission, real traces from a profitable service. Read this for "what does disaggregation actually look like under pressure."

Together they established that **PD-disaggregation is not a niche optimization but the correct default architecture for LLM serving at scale**. Every major production stack adopted some form of it within 12 months.

The open frontier is exactly what DistServe didn't address: **how to dynamically re-disaggregate under shifting workload**. The placement is offline; if your workload shifts from short-prompt chat to long-prompt RAG mid-day, you need to re-plan. The next generation of systems (NVIDIA Dynamo especially) is working on online re-planning. Also: **how disaggregation composes with speculative decoding, prefix caching, and MoE expert parallelism** — the combinatorics are nontrivial, and the right unified framing is still unsettled.

For anyone designing an LLM serving system in 2026: **start from the disaggregated architecture and justify deviations**, not the other way around. DistServe is the paper that makes this the default.

## References

- [1] Zhong et al. _DistServe: Disaggregating Prefill and Decoding for Goodput-Optimized LLM Serving._ OSDI '24 / arXiv:2401.09670.
- [2] Qin et al. _Mooncake._ FAST '25 / arXiv:2407.00079. (Companion read)
- [3] Patel et al. _Splitwise._ ISCA '24. (Contemporary, complementary)
- [4] Agrawal et al. _Sarathi-Serve: Taming Throughput-Latency Tradeoff in LLM Inference._ OSDI '24. (Chunked prefill within a unified pool — contrast)
- [5] DistServe reference impl: https://github.com/LLMServe/DistServe
