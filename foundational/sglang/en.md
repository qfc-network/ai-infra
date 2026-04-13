# SGLang — RadixAttention and a Structured Frontend for LLM Programs

- **Authors / Org**: Zheng et al., LMSYS Org + UC Berkeley + Stanford + CMU
- **Published**: 2023-12 (preprint), NeurIPS '24
- **Links**: [paper (arXiv:2312.07104)](https://arxiv.org/abs/2312.07104) · [code](https://github.com/sgl-project/sglang)

## TL;DR

SGLang pairs a **structured frontend language** for multi-call LLM programs (branching, looping, parallel sampling, tool use) with a **RadixAttention runtime** that reuses KV cache across calls automatically by storing prefixes in a **radix tree**. The key insight: LLM applications are no longer single forward passes — they're programs with many related calls, and the dominant inference cost is recomputing KV for shared prefixes. A radix tree over tokens gives **prefix-level cache reuse at O(prefix-length) cost** without manual annotation. Reported gains: up to **5× throughput** on agentic workloads vs vLLM-era baselines, with exact output equivalence. Today SGLang is one of the two or three production serving stacks (alongside vLLM and TensorRT-LLM) that most frontier-model deployments use.

## Context & Motivation

By late 2023, LLM workloads were shifting from "one prompt → one completion" to:

- **Multi-turn chat** with long shared history.
- **Few-shot prompting** with identical instruction prefixes across thousands of requests.
- **Agents / tool use** with common system prompts and branching exploration.
- **Tree-of-thought / self-consistency** with many sibling completions from one prefix.
- **RAG** with large repeated context chunks.

Serving systems of the era (vLLM, TGI) treated each request as independent. KV cache existed *within* a request but was thrown away at request end, and different requests with the same prefix each recomputed it. For some production traffic, this redundant prefill was **the majority of compute**.

vLLM had primitive prefix caching, but the matching logic was coarse (exact prefix, per-request opt-in). SGLang aimed to make prefix reuse **automatic, fine-grained, and program-aware**.

## Core Method

### RadixAttention

A **radix tree** (compressed trie) indexed by token IDs. Each node stores a KV cache block. When a new request arrives:

1. Walk the tree with the request's prompt tokens; the walk finds the **longest matching prefix** already cached.
2. That prefix's KV is reused directly — no recomputation.
3. The request's uncached suffix is prefilled, and the resulting KV extends the tree.

Crucially, the tree is **shared across all requests** in the serving process. A system-prompt-heavy workload sees all requests converge on a common ancestor path, and it's loaded exactly once. Branching workloads (multiple completions from the same prefix, beam search, tree-of-thought) share inner nodes.

Eviction: LRU over tree leaves, protected by a reference count for in-flight requests.

Unlike vLLM's per-block prefix matching, RadixAttention works at **arbitrary prefix boundaries** — a shared system prompt can end mid-block. Memory is paged (same idea as PagedAttention), but the index is a tree, not a hash.

### The frontend language

A Python-embedded DSL for describing multi-call LLM programs:

```python
@sgl.function
def multi_turn(s, question):
    s += sgl.system("You are a helpful assistant.")
    s += sgl.user(question)
    s += sgl.assistant(sgl.gen("answer", max_tokens=512))
    s += sgl.user("Explain it simpler.")
    s += sgl.assistant(sgl.gen("simpler", max_tokens=512))
```

The frontend:

- Makes prefix sharing **explicit** — each `s += ...` extends the program state, so shared prefixes across calls are structurally obvious.
- Enables **fork / merge**: explore several continuations in parallel from a state, re-join their outputs.
- Supports **constrained decoding** (regex, grammars) — with efficient compilation.
- Composes with tool calls, JSON mode, streaming.

The runtime uses the frontend's structure to prepare RadixAttention walks eagerly and to batch related requests together.

### Compressed FSM for constrained decoding

For regex/grammar-constrained generation (JSON output, structured responses), SGLang compiles the automaton and **collapses deterministic transitions into single steps** — skipping forward pass when the next token is uniquely determined by the current FSM state. Real programs with JSON schemas see substantial speedups this way.

### Other performance work

Subsequent SGLang releases added:

- **Flashinfer integration** for efficient CUDA attention kernels.
- **Tensor parallelism** and hybrid PD-disaggregation (converged with DistServe / Mooncake).
- **Speculative decoding** with MTP and EAGLE variants.
- **Hierarchical KV cache** extending the radix tree to CPU DRAM and SSD — a Mooncake-like store under the same RadixAttention abstraction.
- **Scheduler improvements** for chunked prefill, priority queues, fair sharing.

The frontend + RadixAttention pairing remains the distinguishing core.

## Engineering Tradeoffs

| Decision | Gained | Gave up |
|---|---|---|
| Radix tree over tokens | Automatic fine-grained prefix reuse | Tree lookup cost; eviction logic is non-trivial |
| Tree shared across all requests | Maximum reuse | Contention on the root node's neighborhood; needs lock-free or well-designed concurrency |
| Frontend language | Exposes program structure to runtime | Application must adopt the DSL to get full benefit (but REST-style APIs still work) |
| Compressed FSM constrained decoding | Fast JSON / grammar output | Only helps when the FSM is deterministic for stretches |
| Hierarchical KV cache | Offload cold prefixes to CPU/SSD | Complexity; latency on cache miss |

## Experiments & Results

- Original paper: **up to 5× throughput** on multi-turn / RAG / agentic benchmarks vs baseline serving; quality identical to naive baseline.
- On workloads *without* prefix reuse, SGLang performs comparably to vLLM — no penalty for adoption.
- Later SGLang versions (2024–2025) have incorporated PD-disaggregation and hierarchical cache, bringing production numbers close to Mooncake-class architectures.
- Adoption: powers major open deployments including several frontier-model inference stacks (Grok, Kimi, various Qwen deployments), and is a common choice for academic inference research.

## Reproducibility Notes

- Fully open source (`pip install sglang`).
- Supports OpenAI-compatible API out of the box, plus the native DSL.
- Integrates with the PyTorch / Triton / FlashInfer ecosystem.
- The reference implementation is production-used at LMSYS and partner orgs.

## Commentary

SGLang's staying power is unusual. Many inference papers get cited and then the ideas are absorbed into vLLM / TensorRT-LLM; the original code fades. SGLang instead **became a competing production stack**, because its two ideas were both strong and composed well:

1. **RadixAttention** is the right data structure for prefix caching. A hash-indexed block cache (vLLM early versions) finds exact matches. A radix tree finds the longest prefix regardless of block boundaries, and naturally supports branching. Two years on, everyone uses something radix-tree-shaped.
2. **The frontend language** is the right abstraction for **multi-call LLM programs**. The rise of agents — every call is implicitly a node in a program — validated the bet.

Where SGLang fits vs peers in 2026:

- **vLLM**: the most widely adopted. Has absorbed radix-style caching, PD-disaggregation. Biggest ecosystem.
- **SGLang**: the most aggressive on prefix caching and structured programs. Strong on agentic and RAG workloads.
- **TensorRT-LLM**: the most optimized for NVIDIA-specific peak perf. Tightest kernel integration.
- **NVIDIA Dynamo**: the youngest, bet on disaggregation and multi-model serving.

Pick by workload. For anything heavy on prefix reuse or structured multi-call programs, SGLang is still the default. For highest raw throughput on simple workloads with NVIDIA-only deployment, TensorRT-LLM. For broadest hardware and community support, vLLM.

## References

- [1] Zheng et al. _SGLang: Efficient Execution of Structured Language Model Programs._ NeurIPS '24 / arXiv:2312.07104.
- [2] SGLang repo: https://github.com/sgl-project/sglang
- [3] Kwon et al. _PagedAttention / vLLM._ SOSP '23. (Baseline + background)
- [4] Zhong et al. _DistServe._ OSDI '24. (Disaggregation, later adopted in SGLang)
- [5] Qin et al. _Mooncake._ FAST '25. (Hierarchical cache, later adopted in SGLang)
