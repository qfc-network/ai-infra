# Llama 4 — Meta's First Native MoE

- **Authors / Org**: Meta AI
- **Published**: 2025-04
- **Links**: [Llama 4 Scout (HuggingFace)](https://huggingface.co/meta-llama/Llama-4-Scout-17B-16E) · [Llama 4 Maverick (HuggingFace)](https://huggingface.co/meta-llama/Llama-4-Maverick-17B-128E) · [Meta blog](https://ai.meta.com/blog/llama-4-multimodal-intelligence/)

## TL;DR

Llama 4 is Meta's first MoE-based model family, released April 2025. Two open-weight variants: **Scout** (17B parameters per expert × 16 experts, 17B activated) and **Maverick** (17B × 128 experts, 17B activated, 400B+ total). Both are **natively multimodal**: a vision encoder is trained jointly with the language model from scratch, not added post-hoc. The attention design introduces **iRoPE** — interleaved local and global attention layers with RoPE applied only to local layers, enabling very long context (10M tokens for Scout) without the quadratic cost of global attention on every layer. This entry focuses on the infrastructure implications: MoE routing, multimodal token budgets, iRoPE serving, and how Llama 4 compares to the [Llama 3 dense architecture](../llama3/en.md) and [DeepSeek V3's MoE](../../deepseek/v3-tech-report/en.md).

## Context & Motivation

Llama 3 405B was dense. It proved that dense scaling at 16k H100s produces frontier-quality models, but the serving economics are punishing: 405B parameters × 2 bytes (BF16) = 810 GB just for weights. Activating the entire model for every token means every decode step reads 810 GB from HBM.

MoE addresses this directly: keep a large total parameter count for capacity but activate only a fraction per token. Llama 4's choices — 17B activated parameters regardless of total size — give a concrete serving target: the decode memory footprint is comparable to a 17B dense model, even for Maverick with 128 experts (400B+ total weights).

The multimodal decision reflects a shift in Meta's strategy: rather than releasing separate vision models, the Llama 4 family treats vision as a first-class modality co-trained from the beginning. This affects data pipelines, tokenization, and serving architecture in ways that bolted-on vision adapters do not.

## Architecture

### MoE Design

Both Scout and Maverick use a standard MoE transformer block structure:

- Attention layers: standard multi-head attention with GQA (unchanged from Llama 3).
- FFN layers: replaced by a mixture-of-experts FFN block.
  - `E` expert FFNs of standard size.
  - A top-K router selects `K = 1` expert per token (Scout/Maverick use top-1 routing, not top-2 as in Mixtral or DeepSeekMoE).
  - A shared expert (always active) is added alongside the routed experts — similar to [DeepSeekMoE's](../../deepseek/moe/en.md) shared expert concept.

Total parameters = `(attention + shared FFN + E × expert FFN) × layers`. The activated parameters per token = `(attention + shared FFN + 1 × expert FFN) × layers` ≈ 17B for both variants.

Top-1 routing simplifies load balancing relative to top-2 (one token, one expert, no partial weights to synchronize) but increases the routing variance — a single bad routing decision drops the token entirely rather than partially recovering with a second expert.

### iRoPE — Interleaved Attention

Standard RoPE + global attention across 10M tokens is quadratically infeasible. Llama 4's solution: **interleave** local and global attention layers.

- **Local attention layers** (majority): standard sliding-window attention with a fixed window size (e.g., 8192 tokens). RoPE position embeddings are applied here.
- **Global attention layers** (minority, e.g., every 4th layer): full attention across the entire context. **No RoPE** in global layers — positional information is already encoded by the local layers that precede them.

The intuition: local layers with RoPE handle position-relative information within windows; global layers aggregate across the full sequence without needing explicit position encoding because the representations they operate on already embed local positional structure.

This is distinct from [Gemma 2's](../../google/gemma2/en.md) alternating local/global attention, where both use their own independent position encodings. In iRoPE, the asymmetry is deliberate: RoPE only in local, bare attention in global.

**Serving implications**: KV cache requirements differ by layer type. Local layers have bounded KV cache = `window_size × layers_local × 2 × d_head × n_kv_heads`. Global layers have unbounded KV cache = `seq_len × layers_global × ...`. For Scout's 10M context, the global layer KV cache dominates; efficient serving requires either CPU offload or careful memory management. See [Prefix Caching](../../foundational/prefix-caching/en.md) for how radix-trie caching interacts with mixed attention types.

### Vision Encoder

Llama 4 includes a native vision encoder (based on a ViT-derived architecture) co-trained with the language model. Images are tokenized into a fixed number of visual tokens (patches), which are projected into the language model's embedding space and interleaved with text tokens.

Key infrastructure points:
- **Variable-resolution encoding**: images are resized to fit within a token budget (e.g., 4096 visual tokens); high-resolution images are tiled, with tiles processed independently and then concatenated.
- **No dedicated cross-attention**: visual tokens simply occupy positions in the sequence, attending to and from text tokens through the normal transformer attention. This simplifies the architecture but increases sequence length.
- **Prefill cost**: a single high-resolution image can contribute thousands of tokens to the prefill batch, significantly affecting TTFT. Chunked prefill (see [Sarathi-Serve](../../foundational/chunked-prefill/en.md)) becomes more important.

## Training Infrastructure

Meta's public blog and model cards describe a training cluster of tens of thousands of H100s. Key choices inherited from Llama 3 (see [Llama 3 entry](../llama3/en.md) for full details):

- 4D parallelism: TP + PP + CP + DP.
- BF16 mixed precision throughout.
- Gradient checkpointing for the vision encoder.

New considerations for MoE:
- **Expert parallelism (EP)**: the `E` expert FFNs are sharded across EP groups. Each GPU holds a subset of experts; all-to-all communication routes tokens to the correct expert device. At 128 experts, EP is essential — storing all experts on one node is infeasible.
- **Load balancing**: auxiliary loss term penalizes unequal expert utilization. Top-1 routing increases load imbalance risk; the shared expert absorbs tokens that the router is uncertain about.
- **Communication pattern**: EP all-to-all adds a network-intensive step per MoE layer, similar to [DeepEP's](../../deepseek/open-source-week/deep-ep/en.md) approach. NVLink handles intra-node; RDMA/InfiniBand handles inter-node.

## Model Sizes and Key Numbers

| Variant | Experts | Activated Params | Total Params | Context | Vision |
|---|---|---|---|---|---|
| Scout | 16 | ~17B | ~109B | 10M tokens | Yes |
| Maverick | 128 | ~17B | ~400B+ | 1M tokens | Yes |

Both variants use the same 17B expert size; the difference is how many experts (and thus how much total capacity) each has. Scout is optimized for long-context efficiency; Maverick for quality at standard context lengths.

## Engineering Tradeoffs

| Decision | Gained | Gave up |
|---|---|---|
| Top-1 routing (vs top-2) | Simpler load balancing; one all-to-all per layer | Higher routing variance; no partial recovery on bad routing |
| Shared expert alongside routed experts | Stable baseline representation; absorbs uncertain tokens | Additional compute per token (shared expert always active) |
| iRoPE (RoPE in local only) | 10M context without quadratic cost | Mixed KV cache sizes complicate memory management; non-standard attention |
| Native multimodal (co-trained) | Tighter vision-language alignment | Larger training dataset and pipeline complexity; visual tokens inflate prefill cost |
| Variable-resolution tiling | Handles high-resolution images within token budget | High-resolution images → large prefill batches → TTFT pressure |
| 17B activated params regardless of total size | Serving cost scales with activated, not total | 400B+ total weights for Maverick must still fit somewhere (cold storage or EP sharding) |

## Comparison: Llama 4 vs DeepSeek V3

Both are large MoE models released within months of each other, targeting similar quality levels. Key differences:

| Aspect | Llama 4 Maverick | DeepSeek V3 |
|---|---|---|
| Total params | ~400B+ | 671B |
| Activated params | ~17B | ~37B |
| Routing | Top-1 + shared expert | Top-2 + shared expert (DeepSeekMoE fine-grained) |
| Attention | iRoPE (local + global interleaved) | MLA (latent KV compression) |
| Vision | Native co-trained | Text-only |
| Training precision | BF16 | FP8 (first major model) |
| Context | 1M (Maverick) / 10M (Scout) | 128k |

DeepSeek V3 activates more parameters per token (37B vs 17B), which generally means higher quality per-token at the cost of higher serving compute. Llama 4's iRoPE gives a major context-length advantage; V3's MLA gives a major KV cache compression advantage.

## Commentary

Llama 4 represents Meta catching up with the MoE trend that DeepSeek and Mistral established. The technical contribution is less about novel mechanisms (iRoPE is a reasonable extension of established local/global attention ideas) and more about **integration**: native multimodal + MoE + very long context in a single open-weight model.

The infrastructure challenge is the product of three simultaneously tricky components. MoE serving requires expert parallelism and load balancing. Long-context serving (10M tokens) requires hierarchical KV cache management. Multimodal serving requires handling variable-length visual token sequences that can dominate prefill cost. Together, these make Llama 4 a significantly more complex serving target than Llama 3 — even though the activated parameter count is lower.

For practitioners: Scout's 10M context at 17B activated is the most interesting efficiency point — competitive quality, long context, and a serving footprint comparable to a standard 17B dense model. The catch is the global attention layers' unbounded KV cache.

## References

- [1] Meta AI. _Llama 4: Multimodal Intelligence at Scale._ Blog post, April 2025.
- [2] Meta AI. Model cards: `meta-llama/Llama-4-Scout-17B-16E`, `meta-llama/Llama-4-Maverick-17B-128E`. HuggingFace, 2025.
- [3] For MoE routing context: [DeepSeekMoE](../../deepseek/moe/en.md), [DeepSeek V3](../../deepseek/v3-tech-report/en.md), [Mixtral](../../mistral/mixtral/en.md).
- [4] For attention context: [GQA](../foundational/gqa/en.md), [RoPE](../foundational/rope/en.md), [Gemma 2 alternating attention](../../google/gemma2/en.md).
- [5] For serving context: [Prefix Caching](../../foundational/prefix-caching/en.md), [Chunked Prefill](../../foundational/chunked-prefill/en.md).
