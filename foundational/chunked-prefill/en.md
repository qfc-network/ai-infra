# Chunked Prefill — Sarathi-Serve

- **Authors / Org**: Amey Agrawal, Nitin Kedia et al. (Microsoft Research India / IISc Bangalore)
- **Published**: 2024-03
- **Links**: [arXiv:2403.02310](https://arxiv.org/abs/2403.02310) · [code](https://github.com/microsoft/sarathi-serve) · [vLLM integration](https://docs.vllm.ai/en/latest/models/chunked_prefill.html)

## TL;DR

Chunked prefill splits long prompts into fixed-size chunks and interleaves them with decode steps within a single batch. This eliminates the prefill–decode interference problem that plagues standard continuous batching: a long prefill no longer monopolizes the GPU and stalls all concurrent decode tokens. Under mixed short-and-long prompt workloads, Sarathi-Serve cuts P99 time-to-first-token (TTFT) by up to 6.8× compared to vLLM's continuous batching, while maintaining decode throughput at 98% of baseline. The technique has since been merged into vLLM (v0.4.0) and SGLang, making it the de facto scheduling upgrade over plain Orca-style serving.

## Context & Motivation

### The Orca baseline and its remaining bottleneck

Orca (Yu et al., OSDI 2022) introduced continuous batching for LLM serving: instead of waiting for an entire batch to finish before starting the next, the server processes requests at the iteration level, dynamically inserting and evicting requests each forward pass. This eliminated the padding waste of static batching and substantially improved GPU utilization.

But Orca left one problem unsolved: **prefill–decode interference**. In LLM inference, every new request undergoes a prefill phase — processing the entire prompt in parallel to populate the KV cache — before entering the decode phase, where it generates tokens one at a time. These two phases have fundamentally different computational characters:

- **Prefill** is compute-bound: the entire prompt is processed in one (or a few) forward passes, resulting in a large matrix multiplication that fully saturates tensor cores. A single 8K-token prompt might take 50–200 ms depending on model size and hardware.
- **Decode** is memory-bandwidth-bound: each decode step processes a single token per request, producing a tiny matmul but requiring reading the entire KV cache for that request. GPU compute utilization is typically low; the bottleneck is memory bandwidth.

When a continuous-batching server mixes prefill and decode in the same forward pass, **prefill crowds out decode**. A long prefill request entered into a batch forces every other decode request in that batch to wait the full prefill duration before getting its next token. The decode requests are making zero forward progress during that time.

### Two failure modes

**High TTFT for short requests.** A user submitting a 50-token prompt queued behind a request with a 10K-token prompt waits hundreds of milliseconds for their first token, even though their prompt would individually take only a few milliseconds to prefill. Under bursty traffic, P99 TTFT balloons to seconds.

**Decode throughput degradation.** Prefill phases saturate the compute units that decode also needs. Even when batched together, the compute-intensive prefill phase leaves memory-bandwidth-bound decode underserved — the two operations are not cleanly pipelined on current hardware.

### Alternative approaches

**DistServe / P/D disaggregation** (Zhong et al., 2024) solves this by routing prefill requests to a dedicated prefill fleet and decode requests to a separate decode fleet, communicating KV caches over NVLink or InfiniBand. This cleanly separates the two phases but requires at least 2× the hardware and introduces network transfer latency for the KV cache. Mooncake (Moonshot AI, 2024) applies the same principle at datacenter scale.

**Sarathi-Serve** takes a different approach: solve the interference on a single GPU by never issuing a full prefill. This is cheaper, simpler to deploy, and covers the majority of serving scenarios where a dedicated prefill fleet is not economically justified.

## Core Method

### Chunked prefill scheduling

The central idea is straightforward:

1. **Chunk each prefill into segments of C tokens** (the chunk size, a tunable hyperparameter, typically 256–1024).
2. **Each forward pass is a hybrid batch**: one or more prefill chunks from pending prefill requests, plus all active decode tokens.
3. **Prefill requests advance C tokens per step** toward completing their prompt. Decode requests advance 1 token per step, as in standard continuous batching.
4. **Decode tokens are never displaced.** Every forward pass includes all active decode tokens. No decode request is ever stalled waiting for a prefill to complete.

This creates a cooperative schedule: prefill requests share GPU time with decode requests, spending C tokens of prefill budget per forward pass rather than monopolizing the entire pass. Long prefills are spread across many forward passes instead of occurring in a single large burst.

### Chunk size controls the tradeoff

The parameter C is the primary knob:

- **Small C (e.g., 128–256):** decode is almost never stalled; TTFT for queued short requests approaches the theoretical minimum; prefill throughput is reduced because each forward pass does less prefill work and more scheduling overhead per chunk.
- **Large C (e.g., 2048–4096):** prefill throughput approaches unchuked performance; the first chunk of a long prefill is still a large compute burst that can stall decode requests for longer; approaches vanilla continuous batching at C = full prompt length.

In practice, C = 512 is the Sarathi-Serve default and the value used in most vLLM deployments. It represents a near-optimal point on the TTFT–throughput Pareto curve for typical workload distributions.

### KV cache management and PagedAttention compatibility

A key engineering requirement is that chunked prefill must integrate cleanly with the KV cache management system. In PagedAttention (Kwon et al., 2023), the KV cache is divided into fixed-size pages (typically 16–32 tokens per page). A prefill request incrementally fills pages as its chunks are processed.

Sarathi-Serve aligns chunk boundaries with page boundaries to avoid partial-page writes: if the page size is P tokens, the chunk size C should be a multiple of P (or C should be chosen so chunks do not straddle page boundaries). This is a scheduling-level constraint — no changes to the PagedAttention kernel are required. The KV cache pages allocated for a prefill request are populated chunk-by-chunk and become available to the attention kernels as each chunk completes.

This cleanly separates concerns: **chunked prefill is a scheduling-layer technique**, not a kernel-layer technique. It is compatible with any attention kernel (FlashAttention-2, FlashAttention-3, FlashInfer) and any memory management scheme (PagedAttention, prefix caching, radix tree in SGLang).

### Hybrid batch composition

A single forward pass with chunked prefill processes tokens of two different types simultaneously:

- **Prefill tokens (from one or more chunks):** attend over the full prefix accumulated so far, using standard causal masking. All tokens in the current chunk attend to each other and to all previous KV cache entries for that request.
- **Decode tokens (one per active decode request):** attend over the full KV cache for their respective requests, as in standard decode.

The attention computation differs for the two token types: prefill tokens produce a sequence-length × sequence-length attention pattern; decode tokens produce a 1 × sequence-length pattern. FlashAttention handles this naturally via variable-length (varlen) batch processing — the query, key, and value tensors for the batch are concatenated across requests with a cumulative-sequence-length index array specifying boundaries.

**Attainable Utilization (AU)** is the key metric introduced by Sarathi-Serve: the fraction of theoretically achievable goodput (tokens per second at the hardware's peak rate) actually delivered. Standard continuous batching under mixed workloads exhibits AU well below 1.0 due to prefill bubbles — the GPU sits idle or underutilized during the compute-intensive prefill phases of large batches. Chunked prefill raises AU by keeping a consistent mix of decode and prefill work in every forward pass, avoiding the spikes and valleys of the unchuked schedule.

### Interaction with prefix caching and radix trees

SGLang's radix cache and vLLM's prefix caching both cache KV cache entries for shared prompt prefixes. Chunked prefill interacts gracefully: a chunk that falls entirely within a cached prefix can be skipped (the KV cache entries already exist), and the scheduler can jump to the first uncached chunk. This is a pure win — chunked prefill does not interfere with prefix caching.

## Engineering Tradeoffs

| Decision | Gained | Gave up |
|---|---|---|
| Chunked prefill vs full-length prefill | Decode requests never stall; P99 TTFT reduced up to 6.8× for mixed workloads; more predictable latency under bursty load | Slightly higher total prefill latency per request (multiple forward passes instead of one); small scheduling overhead per chunk transition |
| Small chunk size C (e.g., 256) | Near-zero decode stall; best P99 tail latency for short queued requests; most consistent inter-token latency | Lower effective prefill throughput; more scheduler invocations per long prompt; marginal increase in context-switching overhead |
| Large chunk size C (e.g., 2048) | Near-maximal prefill throughput per forward pass; fewer scheduler decisions for long prompts | Decode stall risk returns for the duration of a large chunk; tail latency approaches unchuked behavior for adversarial (very long) prompts |
| Chunked prefill vs P/D disaggregation (DistServe / Mooncake) | Single-GPU solution; no KV cache network transfer overhead; lower hardware cost; simpler operational deployment | Cannot independently scale prefill and decode resources; prefill and decode share the same GPU memory and compute budget; less effective under extreme prefill-heavy workloads |
| Chunk–page alignment (PagedAttention integration) | Clean integration with PagedAttention; no kernel modifications required; works with any attention implementation | Chunk size must be a multiple of page size; scheduler must track chunk-page offsets per request; adds state to the scheduler data structures |

## Experiments & Results

Sarathi-Serve is evaluated on LLaMA 2 70B served on two A100-80GB GPUs (tensor-parallel degree 2), under two workload distributions representing real-world request patterns:

**P99 TTFT reduction:** Under a mixed workload of short (512-token) and long (8192-token) prompts at high request rates, Sarathi-Serve achieves a **6.8× reduction in P99 TTFT** compared to vLLM continuous batching without chunking. The P50 TTFT improvement is more modest (~2–3×), reflecting that chunking primarily benefits requests queued behind long prefills (the tail).

**Decode throughput preservation:** Decode token generation throughput (tokens/second for already-in-progress requests) is maintained at **98% of vLLM baseline** with C = 512. The 2% overhead comes from the additional scheduling overhead of hybrid batches and the slight inefficiency of smaller prefill chunks.

**Attainable Utilization (AU):** AU improves from approximately 0.55 under unchuked continuous batching at peak load to approximately 0.85 under Sarathi-Serve — a 55% improvement in how efficiently the hardware is used. The remaining gap from 1.0 reflects irreducible scheduling overhead and memory-bandwidth-bound decode phases.

**Sensitivity to chunk size:** The TTFT–throughput Pareto curve shows a wide plateau between C = 256 and C = 1024, meaning the system is relatively insensitive to the exact chunk size choice in this range. The default C = 512 sits near the optimal point for most workloads.

**Comparison with P/D disaggregation:** DistServe achieves lower absolute TTFT at the cost of 2× hardware. Sarathi-Serve achieves comparable TTFT to DistServe at moderate request rates while using the same hardware as unchuked vLLM. At very high request rates with extreme prompt length variance, disaggregation retains an advantage.

## Reproducibility Notes

**Sarathi-Serve (original implementation):**
- Repository: `github.com/microsoft/sarathi-serve`
- Built on top of vLLM; requires CUDA 11.8+, Python 3.9+
- Chunk size C controlled via `--sarathi-chunk-size` flag
- Supports LLaMA, Mistral, Mixtral, and any vLLM-compatible architecture

**vLLM integration (v0.4.0+):**
```bash
python -m vllm.entrypoints.openai.api_server \
    --model meta-llama/Llama-2-70b-hf \
    --enable-chunked-prefill \
    --max-num-batched-tokens 2048
```
The `--max-num-batched-tokens` flag controls the effective chunk budget per forward pass. Setting it to 512–1024 approximates Sarathi-Serve's default behavior.

**SGLang:** Chunked prefill is supported and enabled by default for long-context models. No flag needed; chunk size is set automatically based on the model's context length.

**Key tuning knobs:**
- `chunk_size` (C): primary latency–throughput tradeoff. Start at 512; decrease toward 256 if P99 TTFT is the primary concern, increase toward 1024 if prefill throughput is the priority.
- `max_num_batched_tokens`: total token budget per forward pass (prefill chunks + decode tokens). Should be set to at least `chunk_size + max_concurrent_decode_tokens`.

**Workload considerations:** chunked prefill provides the most benefit under workloads with high prompt length variance. If all requests have similar prompt lengths (e.g., a pure summarization API with consistent 4K-token inputs), the interference problem is less severe and chunking provides smaller gains.

## Commentary

Chunked prefill is one of the cleanest system design wins in recent LLM serving research. The insight is simple: the interference between prefill and decode arises because they are scheduled at incompatible granularities — prefill monopolizes an entire forward pass, while decode wants to share every forward pass. Reducing prefill to chunk granularity restores parity.

The implementation cost is low. Chunked prefill is a scheduler-level change; it does not require new kernels, new hardware, or any changes to the model architecture. The chunk–page alignment constraint (for PagedAttention compatibility) adds modest scheduling complexity but nothing that a production inference system cannot handle.

The comparison with P/D disaggregation (DistServe, Mooncake) is instructive. Disaggregation is strictly more powerful — it allows independent scaling of prefill and decode fleets — but it requires network infrastructure, KV cache transfer protocols, and 2× hardware. For organizations that cannot afford dedicated prefill GPUs, Sarathi-Serve delivers most of the latency benefit at zero additional hardware cost. The right choice depends on scale: at small-to-medium serving loads, Sarathi-Serve is almost always the right answer; at very large scale with extreme SLA requirements on TTFT, disaggregation justifies its overhead.

One underappreciated aspect: chunked prefill also improves **fairness** across request priorities. Without chunking, a short request queued behind a long prefill experiences arbitrarily long waits. With chunked prefill, every request makes progress every forward pass, bounding the worst-case wait to roughly `(full_prompt_length / C)` forward passes — a predictable, controllable quantity.

The technique has been widely adopted: vLLM, SGLang, and TensorRT-LLM all support it. For any serving stack that uses continuous batching and handles variable-length prompts, chunked prefill is now the expected baseline.

## References

- [1] Agrawal, A. et al. (2024). "Sarathi-Serve: Efficient LLM Inference by Piggybacking Decodes with Chunked Prefills." *OSDI 2024*. arXiv:2403.02310.
- [2] Yu, G. et al. (2022). "Orca: A Distributed Serving System for Transformer-Based Generative Models." *OSDI 2022*.
- [3] Kwon, W. et al. (2023). "Efficient Memory Management for Large Language Model Serving with PagedAttention." *SOSP 2023*. arXiv:2309.06180.
- [4] Zhong, Y. et al. (2024). "DistServe: Disaggregating Prefill and Decoding for Goodput-Optimized Large Language Model Serving." *OSDI 2024*. arXiv:2401.09670.
- [5] Qin, H. et al. (2024). "Mooncake: A KVCache-centric Disaggregated Architecture for LLM Serving." arXiv:2407.00079.
- [6] Zheng, L. et al. (2023). "SGLang: Efficient Execution of Structured Language Model Programs." arXiv:2312.07104.
