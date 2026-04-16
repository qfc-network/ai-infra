# Prefix Caching

_A system design entry covering KV cache reuse across requests. The matching algorithms (RadixAttention) are described in the [SGLang entry](./sglang/); this entry focuses on the design space: matching strategies, eviction policies, hit rate economics, multi-tier caching, and disaggregated serving integration._

- **Key implementations**: vLLM Automatic Prefix Caching (APC, 2024), SGLang RadixAttention (2023), TensorRT-LLM KV cache reuse (2024)
- **Foundational work**: Zheng et al. (SGLang, arXiv:2312.07104); vLLM docs (APC); Mooncake (arXiv:2407.00079)

## TL;DR

Every LLM inference request that shares a token prefix with a prior request can skip recomputing the KV cache for that prefix — the cache from the prior request can be reused directly. In production LLM serving, prefixes are not random: system prompts repeat across every request, few-shot examples are shared across a dataset, retrieval context repeats across a session, and agent scaffolding appears in every call. Prefix caching converts this structure into measurable throughput and latency gains: a 1,000-token system prompt that hits the cache saves 1,000 tokens of prefill computation per request, potentially cutting TTFT by 50–80% for common deployments. The design space has three axes: **matching strategy** (how to identify reusable blocks), **eviction policy** (what to evict when GPU memory is exhausted), and **caching tier** (GPU HBM only, or spill to CPU DRAM and disk). Each axis involves non-obvious tradeoffs.

## Context & Motivation

### Where prefix repetition comes from

PagedAttention ([entry](./paged-attention/)) established that KV cache blocks are the right physical unit for memory management. Copy-on-write across blocks enabled beam search sharing. The next step: share blocks across *requests*, not just within one request's beam tree.

Real traffic has high prefix repetition at several levels:

| Repetition type | Example | Typical prefix length |
|---|---|---|
| System prompt | "You are a helpful assistant. Do not..." | 50–500 tokens |
| Few-shot examples | Classification with 5 labeled examples | 200–2,000 tokens |
| RAG context | Retrieved documents prepended to every query | 500–8,000 tokens |
| Agentic scaffolding | Tool descriptions, environment state | 500–5,000 tokens |
| Multi-turn history | Earlier turns of a conversation | Grows per turn |
| Dataset batch processing | Same instructions across thousands of items | 100–2,000 tokens |

In each case, the prefix is recomputed on every request — a pure waste of prefill compute and TTFT latency.

### The opportunity size

For a 70B model on H100, prefill throughput is approximately 10,000–20,000 tokens/second (compute-bound, batched). A 2,000-token shared system prompt + RAG context costs 0.1–0.2 seconds of prefill *per request* with no caching. With 100% prefix cache hits, that cost drops to near zero. At 100 requests/second, that is 10–20 GPU-seconds saved per second of serving — effectively a free 10–20× multiplier on prefill capacity for the shared portion.

## Core Method

### The matching unit: blocks

PagedAttention's block abstraction (fixed-size KV cache pages, typically 16–32 tokens per block) is the natural unit for prefix matching. A prefix is reusable if and only if the full block it belongs to has been previously computed and is still resident.

The matching granularity has a consequence: a prefix that does not end on a block boundary is only partially cacheable. The last partial block must be recomputed. For a block size of 16 tokens and a system prompt of 100 tokens:
- Blocks 0–5 (tokens 0–95): fully cacheable — 6 blocks reused
- Block 6 (tokens 96–100): partial — must be recomputed from its start

This is why **chunked prefill** ([entry](./chunked-prefill/)) and prefix caching compose well: chunked prefill lets the server fill partial blocks with new request tokens, completing the block so it becomes cacheable for the *next* request that shares this prefix.

### Strategy 1 — Hash-based block matching (vLLM APC)

vLLM's Automatic Prefix Caching computes a **hash of each block's content** (token IDs of the block's 16 tokens plus the hash of the preceding block, forming a chain hash). When a new request arrives:

1. Hash each block of its prompt sequentially.
2. Look up each hash in a global block hash table.
3. If found (and the block is still resident), skip recomputing that block — mark it as "reused" in the request's block table.
4. At the first cache miss, stop — all subsequent blocks are also misses (prefix property: a block is only reusable if all preceding blocks also hit).

The chain hash ensures that two identical blocks at different positions in the prompt are treated as different (a block "the cat" at position 3 and "the cat" at position 100 hash differently). This prevents accidental reuse of out-of-context blocks.

**Complexity**: O(prefix_length / block_size) hash lookups per request. With a hash table, each lookup is O(1). The overhead is negligible compared to prefill cost.

**Limitation**: exact match only. A prefix that differs in one token in the middle triggers a full cache miss from that point forward, even if the suffix is identical to a prior request. No fuzzy matching or semantic similarity — the unit is exact byte equality of token IDs.

### Strategy 2 — Radix trie matching (SGLang RadixAttention)

SGLang stores cached blocks in a **radix tree** (compressed trie) indexed by token ID sequences. See the [SGLang entry](./sglang/) for full details. The key difference from hash-based matching:

- The radix tree makes **longest prefix matching** explicit and efficient: a single traversal from the root finds the longest cached prefix that matches the new request.
- The tree structure makes eviction decisions cheaper: evicting a subtree removes a family of related prefixes atomically, maintaining trie invariants without scanning the entire cache.
- The tree directly exposes **sharing topology**: branches off a common node are known to share that node's prefix, making copy-on-write and cache accounting straightforward.

**Complexity**: O(prefix_length) for a lookup, but with better constant factors than hash chaining for long prefixes because the trie naturally stops at the first mismatch rather than computing hashes for all blocks.

### Eviction policies

When GPU HBM is full and a new block must be allocated, the cache manager must evict an existing block. The eviction policy determines which block to remove.

**LRU (Least Recently Used):**
Evict the block that was last accessed furthest in the past. Standard, well-studied, easy to implement with a doubly-linked list + hash map. Works well when access patterns are recency-dominated (recent requests are likely to repeat soon).

Failure mode: a single very large prefix (a 10,000-token RAG context used by 1% of traffic) that is accessed occasionally will survive in cache and crowd out many smaller, frequently-used prefixes.

**LFU-with-decay (Least Frequently Used, with exponential decay):**
Weight recent accesses more than distant ones. Evict blocks with the lowest decayed access count. Better for mixed workloads with both frequent short prefixes and infrequent long ones.

Failure mode: new prefixes start with low frequency and are immediately evicted, leading to a cold-start penalty on new system prompts.

**Size-aware eviction:**
Prefer evicting large blocks (or large subtrees) when the system is under memory pressure. Maximizes the memory recovered per eviction decision. The tradeoff: may evict large, high-frequency prefixes that happen to be the most expensive to recompute.

**Practical choice**: most production systems use LRU as the default. For workloads with a few dominant system prompts (the common case), LRU is near-optimal because system prompts are accessed on every request and will never be evicted under any recency policy.

### Hit rate economics

Prefix cache hit rate is not uniform across traffic. The hit rate depends on:

1. **Prefix length distribution**: longer shared prefixes increase the cacheable fraction of each request. A 2,000-token system prompt with 100-token user query gives 95% cacheable fraction; with a 10,000-token user query it's 17%.

2. **Request arrival pattern**: requests with the same prefix must arrive within the cache eviction window. High-QPS serving with a single dominant system prompt will have near-100% hit rate. Low-QPS serving with many distinct system prompts will have near-0% hit rate.

3. **Cache size relative to working set**: if the working set of active prefixes fits in GPU HBM, hit rate is high. If the working set exceeds HBM, hit rate collapses to the HBM fraction of working set.

The **effective hit rate** (fraction of prefill tokens that are served from cache rather than recomputed) is a better metric than request-level hit rate:

```
effective_hit_rate = (cached_tokens_served) / (total_prefill_tokens)
```

A workload where every request has a 2,000-token shared prefix and a 100-token unique suffix has an effective hit rate of 95% even if only 50% of requests have a cache-resident prefix (the other 50% recompute the prefix and populate the cache for the next request).

### Multi-tier caching: GPU → CPU → Disk

GPU HBM (80 GB on H100) limits the total prefix cache working set. For workloads with large or numerous distinct system prompts, the working set exceeds HBM. Two extensions:

**CPU DRAM offload:**
Evicted KV cache blocks are moved to CPU DRAM (hundreds of GB) rather than discarded. On a cache miss in GPU HBM, check CPU DRAM before recomputing. If found, transfer the block(s) via PCIe back to GPU HBM.

Transfer cost: PCIe 4.0 bandwidth is ~64 GB/s. A 1,000-token prefix (at 16 bytes per token per layer × 80 layers ≈ 1.28 MB) transfers in ~20 μs — much cheaper than recomputing (which takes ~5 ms of prefill at 70B scale). CPU DRAM is worth using when the recompute cost >> transfer cost, which holds for nearly any prefix longer than ~100 tokens.

**Disk (NVMe SSD):**
A second spill tier. NVMe read bandwidth is ~7 GB/s, making transfer ~18× slower than PCIe DRAM. Only worthwhile for very long prefixes (>10,000 tokens) that are infrequently accessed. 3FS ([entry](../../deepseek/open-source-week/3fs/)) is an example of a storage system built to serve large KV cache workloads efficiently; Mooncake's ([entry](../../moonshot/mooncake/)) KVCache pool extends this to a disaggregated cache store.

### Interaction with disaggregated prefill/decode

In disaggregated serving ([DistServe entry](./distserve/), Mooncake), prefill and decode run on separate machines. Prefix caching in this architecture has an additional dimension: **where is the cache stored, and how does it move?**

Mooncake's design: maintain a global KV cache pool distributed across all prefill instances. When a new request arrives, route it to the prefill instance that already holds the longest matching prefix in its local cache. This avoids both recomputing the prefix and transferring the KV cache — the request migrates to the cache, rather than the cache migrating to the request.

The routing problem (which prefill instance to send a request to, given a distributed prefix cache) becomes a key systems problem: consistent hashing over prefix hashes maps similar requests to the same prefill instance, improving local hit rates. This is equivalent to "cache-aware load balancing."

## Engineering Tradeoffs

| Decision | Gained | Gave up |
|---|---|---|
| Hash-based matching (vLLM APC) | Simple; O(1) lookups; no tree maintenance | Exact match only; no prefix-sharing topology exposed; eviction is unaware of shared ancestry |
| Radix trie matching (SGLang) | Longest-prefix matching; natural sharing topology; efficient subtree eviction | Tree maintenance overhead; slightly more complex implementation |
| LRU eviction | Simple; optimal for recency-dominated workloads | Crowded out by occasional large prefixes; cold-start penalty for new prompts |
| CPU DRAM spill tier | 4–8× more effective cache capacity | PCIe transfer latency (~20 μs per MB); CPU memory bus contention; prefetch logic needed to hide latency |
| Block size 16 tokens | Low partial-block waste; fine-grained sharing | 16-token alignment requirement; small blocks mean more hash table entries and more pointer chasing |
| Block size 64 tokens | Fewer hash entries; better DRAM locality | More waste on partial blocks at prefix boundaries; less sharing granularity |
| Disaggregated prefix cache (Mooncake) | Shared cache across instances; optimal hit rate across fleet | Network transfer cost for cache misses that hit a remote instance; routing complexity |

## Experiments & Results

**vLLM APC on shared system prompt workloads:**
- With a 512-token system prompt and short user queries (~50 tokens), APC reduces TTFT by ~60% at high QPS (prefill is the TTFT bottleneck; cache hits eliminate it).
- Effective hit rate of 95%+ on single-system-prompt deployments with warm cache.

**SGLang RadixAttention on agentic workloads:**
- On multi-call LLM programs (tree-of-thought, tool-use chains) with branching structure: 5× throughput vs vLLM era (2023) baseline without prefix sharing.
- Hit rates depend heavily on program structure; programs with wide shallow trees (many parallel branches off one root) achieve near-100% effective hit rates for the shared root.

**Mooncake (Moonshot production, 2024):**
- KVCache-centric disaggregated serving with a distributed prefix cache pool.
- Reports 75% KV cache hit rate in production across the prefill fleet, reducing prefill compute by ~3× compared to no caching.

**CPU DRAM tier (empirical from vLLM community):**
- For workloads with >10 distinct system prompts (each ~2,000 tokens), CPU DRAM tier raises effective hit rate from ~30% (HBM only) to ~85% (HBM + CPU DRAM) at comparable TTFT — PCIe transfer is fast enough to be imperceptible vs recompute.

## Reproducibility Notes

- **vLLM APC**: enabled by default in vLLM >= 0.4.0. No configuration needed; enable with `--enable-prefix-caching` in older versions. Block size is configurable (`--block-size`).
- **SGLang RadixAttention**: enabled by default. The radix tree eviction and LRU policy are visible in `sglang/srt/mem_cache/radix_cache.py`.
- **Measuring hit rate**: vLLM exposes `num_cached_tokens` in request metrics; SGLang logs cache hit statistics per request. Instrumenting these is the first step to understanding whether your workload benefits.
- **Optimizing for cache**: structure prompts so the shared prefix comes first, unique content last. Even a single token difference at the start of a prompt causes a full cache miss. Some frameworks offer explicit "prefix sealing" APIs to guarantee block alignment.
- **CPU offload**: vLLM >= 0.6 supports `--cpu-offload-gb` for CPU DRAM prefix cache. The prefetch heuristic (fetch on miss vs prefetch on access pattern) significantly affects latency — production deployments benefit from tuning prefetch aggressiveness to match QPS.

## Commentary

Prefix caching is the most straightforwardly profitable optimization in LLM serving: it reduces a real cost (prefill compute) by exploiting structure that already exists in real traffic (repeated system prompts, few-shot examples, RAG context). Unlike speculative decoding (requires draft model tuning) or quantization (requires accuracy validation), prefix caching is lossless and requires no model changes.

The practical ceiling is workload-dependent. For single-system-prompt chatbot deployments, a warm cache delivers near-100% hit rates and effectively free prefill for the system prompt. For diverse workloads (each request with a unique large context), prefix caching provides zero benefit. The ROI analysis is simple: measure effective hit rate on your traffic; if it's below 30%, the optimization is not worth tuning further.

Two underappreciated interactions:

**Prefix caching + chunked prefill = block completion.** Without chunked prefill, a request whose prompt doesn't align to block boundaries produces a partial last block that is neither cached nor cacheable for the next request. Chunked prefill completes the block by filling it with decode tokens or the next request's prefix tokens, making the block cacheable. This is why enabling both together consistently outperforms either alone.

**Prefix caching + disaggregation = routing problem.** In a disaggregated fleet (Mooncake, DistServe with multiple prefill nodes), a cache miss on one prefill node may be a hit on another. Naive round-robin load balancing destroys effective hit rate. Cache-aware routing (consistent hashing on prefix hashes, or a centralized cache directory) is required to achieve the theoretical hit rate. This is a well-defined distributed systems problem, but it's often an afterthought in deployments that inherit round-robin from their load balancer.

## References

- [1] Zheng et al. _SGLang: Efficient Execution of Structured Language Model Programs._ arXiv:2312.07104, 2023. (RadixAttention)
- [2] Kwon et al. _Efficient Memory Management for Large Language Model Serving with PagedAttention._ SOSP '23 / arXiv:2309.06180. (Block abstraction)
- [3] Qin et al. _Mooncake: A KVCache-centric Disaggregated Architecture for LLM Serving._ arXiv:2407.00079, 2024.
- [4] Agrawal et al. _Sarathi-Serve: Chunked Prefill for LLM Serving._ arXiv:2403.02310, 2024. (Block completion interaction)
- [5] vLLM. _Automatic Prefix Caching._ docs.vllm.ai, 2024.
- [6] DeepSeek. _3FS._ github.com/deepseek-ai/3FS, 2025. (Distributed KV cache storage)
