# Streaming LLM and Attention Sinks

- **Authors / Org**: Xiao et al. (MIT, CMU, MIT-IBM Watson AI Lab)
- **Published**: 2023-09 (arXiv:2309.17453)
- **Links**: [Paper](https://arxiv.org/abs/2309.17453) · [Code](https://github.com/mit-han-lab/streaming-llm)

## TL;DR

Standard sliding-window attention keeps only the most recent W tokens in the KV cache, but fails catastrophically when the initial tokens of a sequence scroll out of the window — perplexity spikes even though recent context remains available. The culprit is the **attention sink phenomenon**: initial tokens (especially position 0) accumulate disproportionate attention weight regardless of their semantic content, acting as "dump bins" for excess softmax probability mass. StreamingLLM's fix is elegant: always retain the first K_sink tokens (typically 4) in the KV cache alongside the sliding window. This restores stable generation indefinitely with a constant-size KV cache of K_sink + W_recent tokens — bounded memory, no matter how long the sequence. The paper enables continuous multi-turn chat, long-horizon agents, and streaming inference on commodity hardware by reframing the infinite-sequence problem as a bounded-cache scheduling problem.

## Context & Motivation

### The serving gap: what LLMs need vs. what inference provides

LLMs are trained at a fixed maximum context length (e.g., 4,096 tokens for LLaMA, 8,192 for LLaMA 2). In production, two use cases push beyond this:

1. **Multi-turn chat**: a long conversation accumulates tokens rapidly. At 100 turns × 200 tokens/turn = 20,000 tokens — 5× LLaMA's training length.
2. **Streaming document processing**: real-time summarization, transcription, or analysis of a continuous data stream.

**Full attention** at these lengths is infeasible: KV cache memory grows linearly with sequence length; for a 7B model with 32 KV heads, head dim 128, bf16, 32 layers:

```
KV cache per token = 32 × 2 × 32 × 128 × 2 = 524,288 bytes ≈ 0.5 MB/token
```

At 20,000 tokens, the KV cache is 10 GB per sequence. This is the entire memory of a standard A10G GPU.

**Naive solutions and their failure modes**:
- *Re-encode from scratch*: infeasibly slow; the model must process all prior tokens at every generation step.
- *Truncate old tokens*: simple, but results in the model losing earlier context. For multi-turn chat, this means forgetting earlier turns.
- *Sliding window*: keep only the most recent W tokens. This seems correct — why would removing old tokens hurt if they're far from the current query?

### Why sliding window fails: the empirical observation

Xiao et al. tested sliding-window attention (window size W) on LLaMA-2-7B generating sequences of up to 100k tokens. The finding: perplexity is stable while token 0 is inside the window. The moment token 0 scrolls out of the window, perplexity spikes by 10–100× and coherence collapses — even though the recent W tokens remain available.

This is a non-obvious failure. The model does not "need" the initial tokens for content — it needs them for something else.

## The Attention Sink Phenomenon

### Observation

Visualizing attention patterns in autoregressive LLMs reveals a consistent pattern: initial tokens (especially token 0, and the first 3–4 tokens generally) receive disproportionately large attention weights across nearly all attention heads, at all layers, for most query positions. This occurs regardless of what the initial tokens semantically contain — even if the initial token is a semantically vacuous separator like `\n` or `<s>`.

This phenomenon is the **attention sink**: the initial tokens serve as "drain" positions where excess softmax probability mass accumulates.

### Why sinks form: the softmax constraint

Softmax must sum to 1 over all keys. In many-to-one attention scenarios (a query attending to N keys), not all keys are semantically relevant. But the model cannot output zero attention — softmax has no "null" option. When there is no relevant key to attend to (e.g., when the content of the sliding window doesn't contain an answer to the current query), the probability mass must go somewhere. It goes to the least-harmful positions: the initial tokens.

This is a **training artifact**. During training, the model is exposed to full sequences, and attention to initial tokens is penalized if those tokens contribute to wrong predictions. But when no available key is relevant, the model learns that parking probability mass on initial tokens — which are present in all training sequences — is safe. The sink tokens accumulate attention as a learned artifact of softmax normalization combined with attention over irrelevant context.

Empirically: token 0 (`<s>` or BOS) consistently receives 5–20% of all attention mass across heads and layers. Token 1–3 also receive elevated attention. The actual informative tokens at middle positions receive proportionally less.

### The consequence: why removing sinks breaks generation

When the sliding window evicts initial tokens, the model's softmax no longer has its usual "parking spot" for excess probability mass. The distribution collapses onto available tokens in unintended ways, corrupting the attention pattern. The effect cascades through layers: softmax distortions in early layers create distorted key/value representations in later layers. The model produces incoherent text.

## StreamingLLM: The Fix

### The algorithm

StreamingLLM maintains a KV cache of constant size by combining two components:

1. **Sink tokens**: always retain the KV cache entries for the first K_sink tokens (K_sink = 4 in the paper). These tokens are never evicted.
2. **Sliding window**: retain the most recent W_recent tokens (e.g., 2,048 or 4,096).

Total KV cache size = K_sink + W_recent, **independent of sequence length**.

**KV cache size equation:**

```
KV cache memory = (K_sink + W_recent) × n_layers × 2 × d_head × n_kv_heads × bytes_per_element
```

For a 7B model (32 layers, 32 KV heads, head dim 128, bf16) with K_sink=4, W_recent=2048:

```
= (4 + 2048) × 32 × 2 × 128 × 32 × 2 = 1.07 GB
```

This is constant. Whether you generate 10 tokens or 10 million tokens, the KV cache occupies the same 1.07 GB.

### Position encoding for retained sink tokens

A subtle implementation detail: when sink tokens are at original positions 0–3 and recent tokens are at positions 50,000–52,047, the position IDs used for RoPE encoding must be adjusted. The sink tokens retain their original position IDs (0–3). The recent tokens use their original position IDs (50,000–52,047). The cache ordering is non-contiguous but the position encoding is consistent.

This is critical: if position IDs were naively re-indexed from 0, the model would be confused by positions it has seen before. The sink tokens must maintain their identity as "position 0–3 tokens."

### Attention at each step

At each decoding step:
1. The new token attends to: [sink token KV entries] ++ [recent window KV entries].
2. Key/value for the new token is computed and appended to the recent window.
3. If the recent window is full (size > W_recent), evict the oldest entry from the recent window (FIFO).
4. Never evict sink entries.

The attention pattern is sparse: the new token sees K_sink + W_recent previous tokens, not all prior tokens.

### What information is lost

StreamingLLM does **not** provide true long-context memory. Information from tokens that scrolled out of the recent window and were not in the initial sink is lost. The model cannot answer questions about events that occurred 100,000 tokens ago (unless they were in the first 4 tokens, which is unlikely to be useful).

The claim is narrower: StreamingLLM enables **stable generation** over an infinite stream without KV cache explosion. The content outside the window is inaccessible, but the generation does not become incoherent. This makes it appropriate for:
- **Multi-turn chat** where earlier turns are often not needed verbatim, just their conclusions.
- **Streaming summarization** where the model summarizes the last W tokens continuously.
- **Long-horizon agents** where only recent state matters.

For tasks requiring **cross-window recall** (e.g., "earlier in our conversation you said X"), StreamingLLM is insufficient; the information is simply gone. Full context retrieval requires RAG, memory modules, or context compression techniques.

## Comparison to Related Approaches

| Method | KV cache size | Coherence at infinite length | Information retained | Notes |
|---|---|---|---|---|
| Full attention | O(L) — grows without bound | Degrades at L >> training length | All tokens | Needs PI/YaRN for L > L_train |
| Naive sliding window | O(W) — bounded | Collapses when token 0 scrolls out | Last W tokens | Fails without sink fix |
| StreamingLLM | O(K_sink + W) — bounded | Stable indefinitely | K_sink initial + last W | This paper |
| H2O / Scissorhands | O(budget) — bounded | Stable | Budget most-important tokens | Requires attention score tracking |
| SnapKV | O(selected) — bounded | Stable | Clustered important KV entries | Offline selection; different use case |
| RAG / Memory modules | O(W) for KV, O(all) for retrieval | Stable | All (via retrieval) | Retrieval latency; separate system |

StreamingLLM is the simplest approach with the best coherence guarantee. It pays for this with hard information loss beyond the window.

## Engineering Implications

### Multi-turn chat at scale

A deployment serving 1,000 concurrent multi-turn conversations with a 7B model:

- Full attention: 1,000 × ~10K tokens × 0.5 MB/token = 5 TB KV cache. Impossible.
- StreamingLLM (W=2048, K_sink=4): 1,000 × 1.07 GB = 1.07 TB. Still large, but feasible with KV offloading to host DRAM.
- StreamingLLM (W=512, K_sink=4): 1,000 × 275 MB = 275 GB. Fits on a standard 8×A100 node.

### Interaction with KV cache quantization

StreamingLLM's bounded cache interacts naturally with KV cache quantization (see [kv-cache-quantization entry](../kv-cache-quantization/en.md)). With 4-bit KV quantization, the 275 GB above becomes 140 GB. The sink tokens can be kept at higher precision (fp16) since they are accessed frequently; the recent window can use 4-bit quantization. This "mixed-precision streaming KV" is a practical production pattern.

### Interaction with paged attention

PagedAttention (see [paged-attention entry](../paged-attention/en.md)) manages KV cache as fixed-size blocks. StreamingLLM's K_sink + W_recent budget maps cleanly onto a fixed block allocation: allocate blocks for K_sink (small, permanent reservation) plus W_recent / block_size blocks for the sliding window. The FIFO eviction policy maps onto block recycling.

## Engineering Tradeoffs

| Decision | Gained | Gave up |
|---|---|---|
| Keep K_sink=4 initial tokens | Stable attention distribution; prevents perplexity explosion | 4 extra KV entries per sequence (negligible cost) |
| Fixed window size W | Bounded KV cache memory; predictable memory allocation | Information beyond W is permanently lost |
| Position IDs preserved (not re-indexed) | Correct RoPE behavior; no position confusion | Non-contiguous cache indices; slightly more complex kernel |
| Streaming without fine-tuning | Works on pretrained models; no additional training | Model was trained at L_train; quality degrades vs. fine-tuned long-context models for long sequences |
| FIFO eviction (oldest tokens go first) | Simple; no scoring overhead; predictable | Does not preserve semantically important mid-window tokens; H2O/Scissorhands-style importance scoring can do better |
| No memory modules / RAG | Simple system; no retrieval latency | True long-range recall requires external system |

## Experiments & Results

**Perplexity at long generation**: LLaMA-2-7B, LLaMA-2-13B, Falcon-7B, Pythia 6.9B. StreamingLLM with K_sink=4, W=2048 maintains constant perplexity for sequences up to 4 million tokens. Naive sliding window (same W, no sink) shows perplexity explosion the moment K_sink tokens leave the window. Dense attention shows gradual perplexity increase beyond the training length.

**Sink ablation**: reducing K_sink from 4 to 1 degrades stability. K_sink=1 (just the BOS token) works for most models. K_sink=0 (pure sliding window) fails. K_sink=4 is robust across all tested models. The paper also shows that a "learned sink" (a sentinel token prepended at training time explicitly designated as a sink) reduces K_sink needs but requires model retraining.

**Throughput**: StreamingLLM processes 4M-token streams at nearly the same throughput as fixed-context inference at W=2048. Since KV cache is bounded, there is no memory fragmentation or reallocation overhead as generation length grows.

## Commentary

Attention sinks are one of the cleaner insights in the LLM systems literature: an empirical observation (initial tokens get high attention) → a mechanistic explanation (softmax must sum to 1; initial tokens are the learned null-attention target) → a minimal engineering fix (keep K_sink=4 initial tokens). The fix costs virtually nothing in memory or compute, and it completely resolves the sliding-window failure mode.

The deeper implication is that LLM internals contain learned artifacts of the training setup that are invisible in short-context benchmarks but become load-bearing in production. The attention sink is one such artifact: it is never tested in standard evaluations, never mentioned in loss curves, but it is the load-bearing beam that holds up the entire generation process. Systems engineers need to know it exists.

StreamingLLM occupies a specific niche: infinite-length generation with bounded memory, zero additional training, and guaranteed coherence, at the cost of hard information loss beyond the window. This makes it the right tool for streaming inference use cases and an important mental model for anyone designing multi-turn serving infrastructure. For use cases where long-range recall is required, the [position interpolation techniques](../position-interpolation/en.md) combined with [paged attention](../paged-attention/en.md) and potentially [KV cache quantization](../kv-cache-quantization/en.md) are the appropriate stack. The two approaches are complementary: one extends the window you can hold, the other makes peace with a bounded window and handles it gracefully.

## References

- [1] Xiao et al. 2023, "Efficient Streaming Language Models with Attention Sinks," arXiv:2309.17453.
- [2] Zhang et al. 2023, "H2O: Heavy-Hitter Oracle for Efficient Generative Inference of Large Language Models," NeurIPS 2023. arXiv:2306.14048.
- [3] Liu et al. 2023, "ScissorHands: Exploiting the Persistence of Importance Hypothesis for LLM KV Cache Compression at Test Time," arXiv:2305.17118.
- [4] Li et al. 2024, "SnapKV: LLM Knows What You are Looking for Before Generation," arXiv:2404.14469.
- [5] Kwon et al. 2023, "Efficient Memory Management for Large Language Model Serving with PagedAttention," SOSP '23. See also: [PagedAttention entry](../paged-attention/en.md).
- [6] [KV cache quantization entry](../kv-cache-quantization/en.md).
- [7] [Position Interpolation entry](../position-interpolation/en.md) — the complementary approach: extending the window rather than bounding it.
- [8] [Prefix caching entry](../prefix-caching/en.md) — for efficient reuse of shared prefixes within a bounded window.
