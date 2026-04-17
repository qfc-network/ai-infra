# VLM Serving — Production Vision-Language Model Infrastructure

- **Authors / Org**: Synthesis entry — draws from vLLM multimodal support, LLaVA-NeXT, Llama 4, Qwen-VL, and production deployment experience
- **Published**: 2024 (ongoing)
- **Links**: [vLLM multimodal docs](https://docs.vllm.ai/en/latest/models/vlm.html) · [LLaVA-NeXT](https://llava-vl.github.io/blog/2024-01-30-llava-next/) · [Qwen-VL](https://arxiv.org/abs/2308.12966) · [Llama 4](https://ai.meta.com/blog/llama-4-multimodal-intelligence/)

## TL;DR

Serving vision-language models in production requires rethinking every assumption from text-only LLM serving. A single 1024×1024 image tiled at 336px produces 2304 visual tokens — 10–50× the text prompt length — inverting the prefill/decode balance and straining batch schedulers. The key infrastructure changes are: running the vision encoder as a separate preprocessing pass, implementing image-content-hash-based prefix caching for repeated images, using chunked prefill to prevent visual-heavy requests from monopolizing the GPU, and handling heterogeneous sequence composition (zero to many images per request) via sequence packing rather than padding.

## Context & Motivation

Text-only LLM serving is well-characterized: prefill processes the input prompt (hundreds of tokens, fast), decode generates the output token-by-token (slow, memory-bandwidth-bound). vLLM, PagedAttention, and continuous batching are optimized for this profile. VLMs break this model:

- **Prefill dominates**: a single LLaVA-NeXT image (4-tile, 336px) contributes 2304 visual tokens before any text. For a 50-word question, the visual tokens are 46× the text.
- **Heterogeneous requests**: some requests have no images, some have 1, some have 4. Batch composition varies wildly.
- **Vision encoder is not the LLM**: the vision encoder (typically CLIP ViT-L/14) runs as a separate model, potentially on different hardware, and must be scheduled separately.
- **Image KV cache**: visual tokens can be cached per-image-content across requests, unlike text which rarely repeats.

This entry synthesizes the production patterns that address these challenges, drawing from [LLaVA](../llava/en.md), [Llama 4](../../meta/llama4/en.md), [Qwen3](../../qwen/qwen3/en.md), and the vLLM + SGLang serving stacks.

## Visual Token Prefill Economics

### Token Count by Configuration

| Model / Config | Image size | Tiles | Tokens/tile | Visual tokens | Text tokens (typical) | Ratio |
|---|---|---|---|---|---|---|
| LLaVA-1.5 | 336×336 | 1 | 576 | 576 | 50 | 11.5× |
| LLaVA-NeXT 2×2 | 1024×1024 | 4+1 | 576 | 2880 | 50 | 57.6× |
| Qwen-VL 448px | 448×448 | 1 | 1024 | 1024 | 50 | 20.5× |
| Llama 4 Scout | 336px dynamic | up to 16 | 256 | ~4096 | 50 | 82× |
| Multi-image (4 images) | 336px each | 4 | 576 | 2304 | 100 | 23× |

At these ratios, standard prefill scheduling completely changes. A single 4-tile image request takes as long to prefill as generating 2880 output tokens — effectively blocking decode throughput for all other requests in the batch.

### Prefill Cost vs. Decode Cost

For a LLaVA-NeXT request with a 2880-token visual prefix and a 100-token text question:

- Prefill: 2980 tokens × LLM forward pass (one pass, fully parallel) → ~500ms on A100 for a 7B model.
- Decode: 200 output tokens × 200 LLM forward passes (sequential) → ~400ms.

The prefill dominates. In text-only serving, prefill rarely exceeds decode; in VLM serving, it routinely does.

## Variable-Resolution Tiling Strategies

### Fixed vs. Dynamic Tile Count

**Fixed tiling** (LLaVA-NeXT original): always use a 2×2 grid regardless of aspect ratio. Simple to implement; wastes tiles on non-square images (portrait photos pad one dimension).

**Dynamic tiling** (LLaVA-NeXT dynamic, Qwen-VL, Llama 4): choose the tile grid that best preserves aspect ratio with minimal padding:

```
For input image W×H, target tile size T×T:
- Enumerate grid options: 1×1, 1×2, 2×1, 1×3, 3×1, 2×2
- For each grid (n_w, n_h): scale image to fit n_w×T × n_h×T with padding
- Choose the grid with minimum padding
- Always append a downsampled global thumbnail
```

Dynamic tiling reduces average token count by 15–30% on real-world image distributions compared to fixed 2×2. Infrastructure implication: **token counts are now variable and unpredictable at batch construction time**.

### Aspect Ratio Preservation

Padding images to a fixed resolution wastes tokens on uniform gray borders. Aspect-ratio-preserving tiling ensures that, e.g., a 640×480 image uses a 2×2 grid (960×720 bounding box) rather than being stretched to 672×672 with distortion. The serving system must handle variable N per request — this directly complicates attention batching.

## KV Cache with Images — Image Prefix Caching

### The Opportunity

If the same image appears in multiple requests (FAQ bot answering questions about the same product image, batch processing a fixed document set), the visual tokens are identical across requests. If the visual tokens are always at the beginning of the sequence, they form a **cacheable prefix**.

[Prefix caching](../../foundational/prefix-caching/en.md) works by hashing the token sequence and reusing KV cache blocks when the prefix matches. For visual tokens, the hash is computed over the image content (or a deterministic hash of the pixel values + encoding configuration). Cache hit → skip the prefill for those tokens entirely, reusing stored K and V tensors.

### Cache Key Design

```
image_cache_key = sha256(
    image_bytes +
    encoder_config_hash +   # ViT model version, resolution
    projector_config_hash   # MLP weights version
)
```

This key is stable across requests with the same image and model version. On a cache hit, the 576 or 2880 K/V blocks are retrieved from the KV cache pool without rerunning the LLM prefill for those positions.

### Cache Hit Economics

For a 7B model, 576 cached visual tokens (FP16, 32 layers, 32 KV heads, d_k=128):
- KV size = 576 × 32 × 2 × 128 × 2 bytes = **9.4 MB per image**
- Prefill compute saved per hit ≈ 500ms × (576/2980) ≈ 96ms per hit

For a document processing pipeline running 10 queries against the same image: cache saves 9 × 96ms = 864ms of compute at the cost of 9.4 MB of KV cache storage. At GPU-hour prices, cache hit value is substantial for high-reuse workloads.

## Encoder-Decoder Serving Patterns

### Pattern 1: Separate Encoder Pass (Standard LLaVA)

```
Request arrives with image + text
  → Step 1: Route image to Vision Encoder (CLIP ViT-L/14)
            Output: 576 visual feature vectors [separate GPU or same GPU, earlier]
  → Step 2: Pass through projector MLP
            Output: 576 LLM-space token embeddings
  → Step 3: Concatenate [visual tokens] + [text tokens]
            Input to LLM prefill
  → Step 4: LLM prefill + decode (standard)
```

Steps 1-2 are embarrassingly parallel across images in a batch. The vision encoder can be batched separately from the LLM, running on a different CUDA stream or even a separate GPU. For high-throughput serving, a dedicated vision encoder worker preprocesses images and caches the projected embeddings.

### Pattern 2: Native Multimodal Attention (Llama 4 / MM-DiT)

Some architectures (Llama 4 Scout/Maverick, Flamingo, BLIP-2) integrate visual tokens differently:

- Cross-attention layers: visual features stay in a side buffer; language tokens attend to them via separate cross-attention heads. Visual encoder output stays in its native dimension (not projected to LLM dimension).
- Interleaved tokens: image tokens are inserted directly into the text token stream at the image position.

For [Llama 4](../../meta/llama4/en.md), native multimodal attention layers process visual and text tokens jointly in certain layers, enabling deeper vision-language fusion than a bottleneck projector allows. Serving is more complex: the vision encoder must be co-scheduled with the LLM and its output cannot be simply prepended.

## Batching with Heterogeneous Sequences

### The Padding Problem

A naive batching approach pads all sequences in a batch to the same length:

- Request A: 2880 visual + 50 text = 2930 tokens
- Request B: 0 visual + 100 text = 100 tokens

Padding B to 2930 tokens wastes 96.6% of compute for request B. At batch size 8 with mixed visual/text requests, average waste can reach 60–80%.

### Sequence Packing

Sequence packing (also known as "multi-pack" or "concatenated batch") concatenates multiple sequences with attention masking to prevent cross-sequence attention:

```
[seq_A tokens (2930)] [sep] [seq_B tokens (100)] [sep] [seq_C tokens (400)] ...
```

All sequences are packed into a single tensor with a block-diagonal attention mask. Each sequence attends only to its own tokens. This eliminates padding waste and is the approach used by vLLM's continuous batching for mixed requests.

Implementation requires careful handling of positional embeddings (reset per sequence) and the attention mask (block-diagonal). For visual tokens specifically, the positional embedding convention must be consistent: most systems reset position to 0 at the start of each sequence within a packed batch.

### [PagedAttention](../../foundational/paged-attention/en.md) with Visual Tokens

vLLM's PagedAttention allocates KV cache in fixed-size pages. Visual tokens allocate pages like any other prefill tokens. Key difference: visual token pages can be **shared across requests** if the same image is in the KV cache — the same physical pages are mapped into multiple virtual KV caches. This is identical to how PagedAttention shares system prompt KV cache across requests.

## Chunked Prefill for Visual Tokens

A 2880-token visual prefill in a single forward pass takes ~500ms on A100 — during which all other requests in the continuous batch are blocked. [Chunked prefill](../../foundational/chunked-prefill/en.md) breaks the 2880 tokens into chunks of, say, 512 tokens:

- Chunk 1: prefill tokens 0-511 (visual tokens 0-511)
- Chunk 2: prefill tokens 512-1023 (visual tokens 512-1023)
- ...
- Chunk 6: prefill tokens 2560-2879 (visual tokens 2560-2879) + text question
- Decode phase begins

Between chunks, the scheduler can inject decode steps from other requests, maintaining tail latency for text-only requests even when large VLM prefills are in progress. The tradeoff: more scheduler overhead, slightly more total compute due to KV cache not being reused across chunks in some implementations.

## Engineering Tradeoffs

| Decision | Gained | Gave up |
|---|---|---|
| Separate vision encoder pass | Encoder can be batched/pipelined independently; visual embeddings cacheable | Two-model scheduling complexity; encoder and LLM must be co-versioned |
| Image prefix caching | Eliminates redundant prefill for repeated images; large wins for FAQ/doc workloads | Cache memory overhead (~10MB per image); cache invalidation on model update |
| Dynamic tiling | Fewer tokens for non-square images; better quality per token | Variable token count complicates batch construction; scheduler must re-sort dynamically |
| Sequence packing | Near-zero padding waste; higher GPU utilization | Complex attention masking; positional embedding reset logic required |
| Chunked prefill | Prevents visual prefill monopolizing GPU; lower P99 latency for text requests | Slightly higher total compute; scheduler complexity; state management between chunks |
| Native cross-attention (Llama 4) | Deeper vision-language fusion; model quality gains | More complex serving graph; visual features must be co-scheduled with LLM; harder to cache |

## Experiments & Results

From vLLM VLM benchmarks and production measurements:

- **Throughput without packing**: mixed-modality batch (50% image requests) at batch size 16 achieves ~40% GPU utilization due to padding waste.
- **Throughput with sequence packing**: same workload → ~75% GPU utilization, ~1.9× throughput improvement.
- **Image prefix caching**: on a FAQ workload (10 questions per unique image), cache hits reduce mean TTFT (Time to First Token) by 45-60% after cache warmup.
- **Chunked prefill**: P99 TTFT for text-only requests in mixed batch drops from ~4s (blocked by image prefill) to ~0.8s with chunk size 512.

## Commentary

VLM serving exposes a fundamental tension in the LLM serving stack that text-only deployments never surface: the prefill and decode phases are so asymmetrically costly for visual inputs that the entire scheduling logic must be reconsidered. The continuous batching paradigm — process requests as they arrive, interleave prefills and decodes — was designed for text where prefills are small. When a single image injects 2880 prefill tokens, continuous batching without chunking degrades into essentially serial processing.

The image prefix caching opportunity is underexploited in current open-source stacks. For any application where users repeatedly ask about the same image (e-commerce product QA, document analysis, video frame analysis), caching visual KV tensors can reduce prefill compute by 80%+ on hot images. This requires treating images as first-class cache keys in the scheduler — a straightforward extension of [prefix caching](../../foundational/prefix-caching/en.md) but one that requires image content hashing to be wired into the request preprocessing pipeline.

The longer-term trajectory points toward learned visual token compression: instead of passing all 576 patch tokens to the LLM, a small resampler (Q-Former, Perceiver Resampler, or token merging) compresses them to 64-128 tokens. This is already present in BLIP-2, InstructBLIP, and [Llama 4](../../meta/llama4/en.md)'s cross-attention layers. At 64 tokens per image, the entire VLM serving problem simplifies dramatically: visual prefill becomes negligible, caching becomes less critical, and batch heterogeneity is manageable. The accuracy cost of this compression is currently 2–5% on standard benchmarks but is rapidly closing as resampler architectures mature.

## References

- [1] vLLM Team. _vLLM: Easy, Fast, and Cheap LLM Serving with PagedAttention._ arXiv:2309.06180, 2023.
- [2] Liu et al. _LLaVA-NeXT._ Blog post, 2024.
- [3] Wang et al. _Qwen-VL: A Versatile Vision-Language Model for Understanding._ arXiv:2308.12966, 2023.
- [4] Meta AI. _Llama 4: Natively Multimodal Intelligence._ Blog post, 2025.
- [5] Related: [PagedAttention](../../foundational/paged-attention/en.md), [Prefix Caching](../../foundational/prefix-caching/en.md), [Chunked Prefill](../../foundational/chunked-prefill/en.md), [LLaVA](../llava/en.md), [Llama 4](../../meta/llama4/en.md), [Qwen3](../../qwen/qwen3/en.md)
