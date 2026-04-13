# PagedAttention / vLLM

- **Authors / Org**: Kwon et al., UC Berkeley
- **Published**: 2023-09 (SOSP '23)
- **Links**: [paper (arXiv:2309.06180)](https://arxiv.org/abs/2309.06180) · [vLLM](https://github.com/vllm-project/vllm)

## TL;DR

LLM serving was bottlenecked not by compute but by **KV cache memory fragmentation**. Traditional systems pre-allocate contiguous KV cache per request at max sequence length, wasting 60–80% of GPU memory on padding and reservation. PagedAttention borrows the **virtual-memory paging** idea from operating systems: split each request's KV cache into fixed-size blocks, maintain a block table per request, and let the attention kernel gather K/V from non-contiguous blocks. This unlocks near-zero fragmentation, enables **continuous batching**, copy-on-write for shared prefixes / beam search, and delivers 2–4× throughput over prior serving systems. vLLM — the reference implementation — is now the de facto open-source inference stack.

## Context & Motivation

In 2023, serving systems like FasterTransformer / HF TGI allocated KV cache as one contiguous tensor per request, sized to the max possible sequence length. Two compound problems:

1. **Internal fragmentation** — most requests end before max length; the reserved space is wasted.
2. **External fragmentation** — once requests finish, the freed contiguous regions don't match the shapes new requests need.

The result: GPU memory was the serving bottleneck, throughput was low, and sharing KV cache across requests (prefix caching, beam search, parallel sampling) was impractical because cache was physically tied to one request.

## Core Method

### Blocks and block tables

KV cache is stored in **fixed-size blocks** (typically 16 tokens of K and V per block, per layer). Each request has a **block table** mapping its logical token positions to physical block IDs, exactly like a page table.

- New request → allocate blocks on demand as it generates tokens.
- Finished request → return blocks to the free pool; no fragmentation because all blocks are the same shape.
- Fragmentation drops from 60–80% to under 4%.

### PagedAttention kernel

The attention kernel is rewritten to **gather K, V from non-contiguous physical blocks** using the block table as an indirection. The gather is efficient because block size is tuned to align with warp / memory transaction shapes; the per-block overhead is small.

### Continuous batching

With paged storage, requests can be **added to and removed from the in-flight batch at any step**, not only at batch boundaries. Combined with iteration-level scheduling, this keeps the GPU saturated under variable-length workloads.

### Shared blocks: CoW and prefix caching

Multiple logical sequences can **share the same physical block** when their KV content is identical — useful for:

- **Beam search / parallel sampling**: sibling sequences share the prompt prefix.
- **Prefix caching**: repeated system prompts / few-shot examples are encoded once and reused across requests.
- **Copy-on-write**: when a shared block needs to diverge, copy at that moment only.

This is transparent to the model — all machinery lives in the block table and kernel.

## Engineering Tradeoffs

| Decision | Gained | Gave up |
|---|---|---|
| Fixed-size blocks | Zero external fragmentation, simple allocator | Small internal fragmentation within the last block of each sequence |
| Block table indirection | Flexible sharing, no physical contiguity | Attention kernel must do gather; slight per-op overhead |
| Continuous batching | High throughput on variable-length workloads | Scheduler complexity; harder to reason about latency tails |
| Prefix sharing via CoW | Huge wins on multi-sample and shared-system-prompt workloads | Reference counting + CoW bookkeeping in the memory manager |
| Python + custom CUDA hybrid | Fast iteration, broad model coverage | Python overhead visible at small-batch low-latency regime; addressed over time via C++ engine |

## Experiments & Results

- **2–4× throughput** over FasterTransformer and HuggingFace TGI on OPT and LLaMA at 13B–175B scales.
- Near-peak GPU memory utilization; fragmentation < 4%.
- vLLM adoption: default open-source inference for essentially every model family (Llama, Mistral, Qwen, DeepSeek, etc.), with support for tensor parallelism, quantization (AWQ / GPTQ / FP8), speculative decoding, and disaggregated prefill/decode.

## Reproducibility Notes

- Fully open source; easy to stand up locally (`pip install vllm`).
- Block size (default 16) is a tunable; larger blocks reduce overhead but increase internal fragmentation.
- Custom attention kernels now exist for most GPU / accelerator backends (CUDA, ROCm, TPU via JAX port, Inferentia). Some features (CoW, specific quant schemes) lag on non-CUDA paths.

## Commentary

PagedAttention is the clearest example in LLM infra of **OS ideas paying off in ML**. The paging analogy is not decorative — the paper literally maps `malloc/free` inefficiency and fragmentation onto KV cache, then applies the known fix. The deeper lesson is that **LLM serving is a memory-management problem, not a compute problem**, and that framing is still producing wins: radix trees over blocks (SGLang), disaggregated prefill (DistServe), hierarchical KV cache (host DRAM / SSD tiers) — all are direct descendants. For anyone running inference at scale, vLLM is the default starting point; understanding how it works is no longer optional.

## References

- [1] Kwon et al. _Efficient Memory Management for Large Language Model Serving with PagedAttention._ SOSP '23 / arXiv:2309.06180.
- [2] vLLM project. https://github.com/vllm-project/vllm
- [3] Zheng et al. _SGLang._ arXiv:2312.07104, 2023. (Radix-tree extension)
- [4] Zhong et al. _DistServe._ OSDI '24. (Disaggregated prefill/decode)
