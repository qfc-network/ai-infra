# Orca — Continuous Batching for LLM Serving

- **Authors / Org**: Gyeong-In Yu et al., Seoul National University / NAVER
- **Published**: 2022-11 (OSDI '22)
- **Links**: [paper](https://www.usenix.org/conference/osdi22/presentation/yu) · [code (not open-sourced; vLLM is the canonical open impl)](https://github.com/vllm-project/vllm) · [blog (Anyscale explainer)](https://www.anyscale.com/blog/continuous-batching-llm-inference)

## TL;DR

Before Orca, every LLM serving system batched requests at the **request level**: collect a batch, pad all sequences to the longest one, run until every sequence finishes, then admit new requests. GPU cycles were burned on padding, and latency for short requests was held hostage by long ones. Orca's answer is **iteration-level scheduling** — also called continuous batching or in-flight batching — where the scheduler makes a new decision at every decode step. Finished sequences exit immediately; new sequences enter without waiting for the rest of the batch to complete. The result on OPT-30B was a **36.9× throughput improvement** over FasterTransformer, the prior state of the art. Orca does not solve KV cache memory fragmentation (that waits for PagedAttention / vLLM), but it establishes the scheduling primitive on which every modern serving system is built.

## Context & Motivation

### The request-level scheduling bottleneck

In 2022, production LLM inference systems — FasterTransformer, HuggingFace TGI (early versions), DeepSpeed Inference — all shared the same basic serving loop:

1. Collect requests until a batch is full or a timeout fires.
2. Pad all sequences to the length of the longest sequence in the batch.
3. Run the full batch through the model until **all** sequences have emitted their EOS token.
4. Release the batch, admit the next one.

This design has two compounding problems:

**Head-of-line blocking.** If one request in a batch needs 2,000 tokens and the rest need 50, the short requests sit idle for 1,950 extra decode steps. Tail latency for short requests is dominated by the longest request in the batch.

**Padding waste.** Every batch member is padded to the length of the longest member. In practice, sequence-length distributions are highly skewed — a few long sequences force massive padding on the majority. GPU utilization for actual token computation can drop well below 50%.

Together these produced a fundamental trade-off: to improve throughput, choose large batches (which increases head-of-line blocking); to improve latency, choose small batches (which reduces GPU utilization). There was no good answer within request-level scheduling.

### The two-phase structure of LLM inference

A key observation in Orca is that LLM inference has two structurally distinct phases:

- **Prefill** (also called prompt processing): the full prompt is processed in a single forward pass. Attention is over all prompt tokens simultaneously. This phase is **compute-bound** — it amortizes the attention computation over many tokens and keeps the GPU well-utilized even at small batch sizes.
- **Decode**: tokens are generated one at a time autoregressively. At each step, the model attends over the entire KV cache (all past tokens). This phase is **memory-bandwidth-bound** — the KV cache is loaded from HBM at every step, and the single new token doesn't give the GPU enough compute to hide memory latency.

Request-level schedulers conflate these two phases. Orca's scheduler treats them separately.

## Core Method

### Iteration-level scheduling

The central idea: instead of scheduling at the granularity of requests (complete sequences), schedule at the granularity of **iterations** (individual decode steps). After each decode step, the scheduler can:

- Remove sequences that have emitted EOS.
- Add new sequences from the waiting queue.
- Continue running sequences that are mid-generation.

The batch composition changes at every step. There is no waiting for the slowest sequence in the batch. A newly admitted sequence enters immediately after its prefill completes.

The key throughput metric Orca introduces is **goodput**: the number of requests successfully completed per unit time, as opposed to raw GPU FLOP utilization. Goodput aligns the optimization objective with what users care about — request completion — rather than hardware efficiency metrics that may not translate to business value.

### Selective batching

Not every operation in a Transformer can be batched across sequences with different lengths without modification. The main challenge is **attention**: standard batched attention requires sequences to have the same shape, or else padding to the maximum length.

Orca's selective batching identifies which layers can be batched directly across variable-length sequences and which require special handling:

- **Feed-forward layers, layer norms, projections**: these operate independently per token, so tokens from different sequences can be concatenated along the batch dimension and processed together without any padding. Orca does exactly this — the "batch" for these layers is a flat tensor of all live tokens.
- **Attention layers**: KV cache is per-sequence and per-layer; the attention operation must respect sequence boundaries. Orca handles this with a custom attention kernel that processes each sequence's attention separately, then concatenates outputs. This is less efficient than fully fused batched attention but avoids padding.

This distinction is what makes iteration-level scheduling practical: by concatenating tokens from different sequences for the cheap layers, Orca can keep GPU utilization high even with a batch of sequences at different positions.

### The iteration loop

```
while serving:
    # Prefill newly admitted requests (one or more, depending on memory budget)
    for each new_request in admitted:
        KV[new_request] = prefill(new_request.prompt)

    # Decode step: one token per live sequence
    new_tokens = selective_batch_decode(all_live_sequences)

    for seq, token in zip(all_live_sequences, new_tokens):
        seq.append(token)
        if token == EOS or len(seq) >= seq.max_length:
            emit(seq)
            free(KV[seq])
            admit_next_from_queue()
```

The scheduler runs this loop continuously. Because sequences are added and removed mid-flight, the batch is always saturated (up to the memory limit), eliminating the idle time that plagued request-level systems.

### Memory: static KV allocation

Orca predates PagedAttention. KV cache is allocated **statically per request** at admission time, sized to the maximum possible output length. This is the main limitation of the Orca design: it wastes GPU memory on reserved-but-unused space for sequences that terminate early, and it makes it impossible to serve requests whose actual output length exceeds the pre-allocated budget.

The combination of iteration-level scheduling (Orca) + virtual paged KV cache (vLLM/PagedAttention) is what makes modern LLM serving efficient. Orca contributes the scheduling policy; PagedAttention contributes the memory management.

## Engineering Tradeoffs

| Decision | Gained | Gave up |
|---|---|---|
| Iteration-level vs request-level scheduling | Near-zero idle time; short requests no longer blocked by long ones; 36.9× throughput gain | Scheduler runs every decode step — more control-plane overhead; harder to reason about batch composition statically |
| Prefill and decode as distinct phases | Can co-schedule prefill + decode optimally; cleaner accounting of compute vs memory bottlenecks | Prefill preempts ongoing decode iterations, introducing latency spikes for in-flight sequences (TTFT vs TPOT tension) |
| Goodput as the optimization target | Aligns optimization with user-visible latency; exposes real trade-offs in scheduler policy | Goodput can be maximized at the cost of tail latency — a request that is continuously deprioritized never completes |
| Static KV cache allocation per request | Simple allocator; no fragmentation within a single request's lifetime | Up to 60–80% GPU memory wasted on padding and early-termination slack; hard limit on max batch size |
| Centralized scheduler | Easy to reason about global memory budget and scheduling policy | Single point of bottleneck; does not scale naturally to disaggregated prefill/decode (DistServe, Mooncake architecture) |

## Experiments & Results

Orca's paper benchmarks against FasterTransformer (NVIDIA's optimized inference library, the best available baseline in 2022) on OPT-30B and OPT-66B, measuring throughput (requests/second) at varying request rates and output lengths.

Key numbers from the paper:

- **36.9× throughput improvement** over FasterTransformer on OPT-30B at matched latency SLO. This is the headline figure, obtained under a realistic mix of prompt and output lengths (drawn from real trace data).
- At OPT-66B, Orca shows 22× improvement, slightly lower because the larger model is more memory-bandwidth-constrained and less sensitive to scheduling efficiency.
- Mean latency improves proportionally; tail latency (P99) improves more dramatically because head-of-line blocking is eliminated.
- GPU utilization increases from roughly 30–40% (FasterTransformer under load) to 70–90% (Orca), measured as actual compute cycles over total cycles.

The workload used in the evaluation is a real trace from a NAVER production serving system, which gives the results credibility beyond synthetic benchmarks. Output length distributions are long-tailed, which is exactly the regime where request-level scheduling performs worst and iteration-level scheduling gains the most.

## Reproducibility Notes

The original Orca codebase was not open-sourced. However, iteration-level scheduling is now the default in every major open-source serving framework:

- **vLLM**: full open-source implementation combining Orca-style scheduling with PagedAttention. `pip install vllm` gives you both.
- **TGI (HuggingFace Text Generation Inference)**: added continuous batching in v0.9 (2023), explicitly citing Orca.
- **LightLLM**, **TensorRT-LLM**: both implement iteration-level scheduling as the default.

To reproduce the core behavior: set up vLLM, enable async engine mode, and benchmark with a mixed-length workload (e.g., using `ShareGPT` trace data). Compare against a naive request-level baseline by disabling continuous batching. The throughput gap is large and reproducible.

Key parameters to explore:
- `max_num_seqs`: maximum number of sequences in the in-flight batch. Increasing this improves throughput at the cost of memory and tail latency.
- `max_num_batched_tokens`: cap on tokens processed per iteration. Controls the prefill budget per step.
- Prefill/decode co-scheduling policy: vLLM's default prioritizes prefill; tuning this trades TTFT against TPOT.

## Commentary

Orca is important not because the idea is complicated — it isn't. The iteration-level scheduling idea is straightforward once stated. What made it impactful was that no one had bothered to implement it cleanly, measure it carefully, and publish the result in a top systems venue. The 36.9× number gave the community a shared reference point.

The deeper lesson is about **abstraction boundaries**. Request-level scheduling was inherited from classical batch inference (image classification, embedding generation) where all requests are the same length and there is no autoregressive decode phase. The abstraction was wrong for LLMs, and fixing the abstraction — without changing the model at all — recovered an order of magnitude of performance.

The remaining limitations of the Orca design trace directly to what it chose not to address:

- **Static KV allocation**: solved by PagedAttention (vLLM, 2023).
- **Prefill/decode interference**: the prefill of a long prompt blocks decode progress for all in-flight sequences. Sarathi (Agrawal et al., 2024) addresses this with chunked prefill — breaking large prefills into chunks that interleave with decode steps, smoothing TTFT and TPOT simultaneously.
- **Centralized scheduler at scale**: DistServe and Mooncake disaggregate the prefill and decode pools entirely, running them on separate hardware with separate schedulers. Orca's centralized design is the baseline they improve on.

For anyone learning inference infrastructure: understand Orca first. Every subsequent paper in LLM serving — vLLM, SGLang, Sarathi, DistServe, Mooncake — is either building on iteration-level scheduling or explicitly solving a problem that Orca left open.

## References

- [1] Yu et al. _Orca: A Distributed Serving System for Transformer-Based Generative Models._ OSDI '22. https://www.usenix.org/conference/osdi22/presentation/yu
- [2] Kwon et al. _Efficient Memory Management for Large Language Model Serving with PagedAttention._ SOSP '23 / arXiv:2309.06180.
- [3] Agrawal et al. _Sarathi: Efficient LLM Inference by Piggybacking Decodes with Chunked Prefills._ arXiv:2308.16369, 2023.
- [4] Zhong et al. _DistServe: Disaggregating Prefill and Decoding for Goodput-Optimized Large Language Model Serving._ OSDI '24.
- [5] Holmes et al. _Deepspeed-FastGen: High-throughput Text Generation for LLMs via MII and DeepSpeed-Inference._ arXiv:2401.08671, 2024.
- [6] vLLM project. https://github.com/vllm-project/vllm
