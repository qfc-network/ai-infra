# LLaVA / Vision-Language Models

- **Authors / Org**: Haotian Liu, Chunyuan Li, Qingyang Wu, Yong Jae Lee (UW Madison / Microsoft Research)
- **Published**: LLaVA — 2023-04-17 · LLaVA-1.5 — 2023-10-05 · LLaVA-1.6/NeXT — 2024-01-30
- **Links**: [LLaVA arXiv:2304.08485](https://arxiv.org/abs/2304.08485) · [LLaVA-1.5 arXiv:2310.03744](https://arxiv.org/abs/2310.03744) · [code](https://github.com/haotian-liu/LLaVA)

## TL;DR

LLaVA (Large Language and Vision Assistant) is the dominant open-source recipe for building vision-language models: freeze a pretrained [CLIP](../clip/en.md) ViT-L/14 encoder, freeze a pretrained LLM (Vicuna/LLaMA), train only a lightweight MLP projector to map visual features into the LLM's embedding space, then instruction-tune the whole system. LLaVA-1.5 demonstrated that this minimal recipe — with the projector upgraded from linear to two-layer MLP — already achieves state-of-the-art on 11 benchmarks while requiring only ~1.2M training data points. The serving consequence is stark: a single 336px image contributes 576 visual tokens to the prompt, and high-resolution variants (LLaVA-1.6) push this to 2880 tokens per image.

## Context & Motivation

GPT-4V's release in late 2023 demonstrated that multimodal LLMs were not just research toys — they could read charts, describe complex scenes, and reason about images in ways that text-only models fundamentally could not. The open-source community needed a replicable recipe. LLaVA's insight was that the two hard parts (visual representation and language understanding) are already solved by CLIP and LLaMA respectively — the only missing piece is a trained bridge between them.

This data-efficient approach contrasted with concurrent work (Flamingo, InstructBLIP) that proposed more elaborate cross-attention architectures. LLaVA showed a two-layer MLP is sufficient, which has major infrastructure implications: a 2-layer MLP with 256k–1M parameters is negligible in both VRAM and FLOPs. The bottleneck is the number of visual tokens passed to the LLM, not the projector itself.

## Architecture

### Projector Design

The projector maps CLIP visual features to the LLM embedding space:

- **LLaVA (original)**: single linear layer. W_proj ∈ R^{D_CLIP × D_LLM}. For CLIP ViT-L/14 (D=1024) → LLaMA-13B (D=5120): a 1024×5120 weight matrix ≈ 20M parameters.
- **LLaVA-1.5**: two-layer MLP with GELU activation. Empirically +4-8 points on VQA benchmarks vs. linear. Still tiny relative to the 7B/13B LLM.

The projector is the only component that requires images and language to appear in the same training objective. Everything else can be pretrained independently, which is the key to LLaVA's data efficiency.

Which CLIP tokens? LLaVA uses **all spatial patch tokens** (N=576 for ViT-L/14@336px), not the [CLS] token. This preserves per-location visual information, enabling the LLM to answer spatial questions ("what is in the top-left corner?"). See [ViT](../vit/en.md) for why patch tokens carry more information than the pooled [CLS].

### Two-Stage Training

**Stage 1 — Feature alignment (pretraining the projector):**
- Freeze: CLIP encoder (all weights), LLM (all weights).
- Train: projector only.
- Data: ~595k image-text pairs (filtered CC3M). Simple captions, not instructions.
- Objective: next-token prediction on the caption given the visual tokens.
- Goal: teach the projector to produce visual embeddings that the frozen LLM can "read."
- Cost: ~4 H100 hours for a 7B model.

**Stage 2 — Visual instruction tuning:**
- Freeze: CLIP encoder.
- Train: projector + LLM (full fine-tune, or LoRA for efficiency).
- Data: ~665k instruction-following samples (LLaVA-Instruct-158K + VQA/caption/OCR data for 1.5).
- Objective: instruction-following responses given images + text questions.
- Cost: ~20 H100 hours for LLaVA-1.5-7B.

Using [LoRA](../../foundational/lora/en.md) for Stage 2 reduces VRAM from ~80GB to ~40GB for a 13B model by not computing full gradients for the LLM. The projector is always trained with full gradients (it is small enough).

### Visual Token Count — The Central Infrastructure Variable

| Configuration | Resolution | Patch tokens | Total visual tokens |
|---|---|---|---|
| LLaVA-1.0 ViT-L/14 | 224px | N=256 | 256 |
| LLaVA-1.5 ViT-L/14@336 | 336px | N=576 | 576 |
| LLaVA-1.6 NeXT (2×2 tiling) | 4 × 336px tiles | 4×576 | 2304 |
| LLaVA-1.6 NeXT (3×3 tiling) | 9 tiles | 9×576 | 5184 |

576 visual tokens is already substantial. A typical text prompt is 50-200 tokens; the image alone contributes 3–10× more tokens than the text. This inverts the compute profile: **prefill is dominated by visual tokens, not text**.

### LLaVA-1.6 / LLaVA-NeXT: High-Resolution Tiling

LLaVA-1.5's fixed 336px input loses fine-grained detail in high-resolution images (text in screenshots, small objects). LLaVA-NeXT addresses this via **dynamic tiling**:

1. Resize the input image to fit into a tile grid (up to 4 tiles: 1×1, 1×2, 2×1, 2×2).
2. Each tile is 336×336px and independently encoded by ViT-L/14 → 576 tokens per tile.
3. A downsampled global thumbnail (336px) is always appended → another 576 tokens.
4. Total: up to 4 tiles × 576 + 576 global = **2880 tokens** per image.

The tile count is dynamically selected based on aspect ratio, avoiding unnecessary padding. A tall portrait image uses a 1×2 grid; a wide panorama uses 2×1; a dense screenshot uses 2×2.

2880 tokens per image at typical LLM serving throughput means the prefill phase for one image-question pair takes as long as generating ~2880 text tokens. This is the motivating use case for [chunked prefill](../../foundational/chunked-prefill/en.md): break the 2880-token visual prefill into chunks so it can be interleaved with decode batches from other requests.

## Serving Implications

### Memory Breakdown for LLaVA-1.5-7B

| Component | VRAM |
|---|---|
| CLIP ViT-L/14@336 weights | ~1.2 GB |
| Projector MLP | ~40 MB |
| LLaMA-7B weights (FP16) | ~14 GB |
| KV cache (1024 ctx, bs=8) | ~4 GB |
| **Total** | **~20 GB** |

The KV cache grows with visual token count: 576 visual tokens per image in the KV cache consume the same space as 576 text tokens. For a batch of 8 requests each with one image at 2048 context: KV cache ≈ 8 × 2048 × (hidden_dim) × 2 (k+v) × num_layers × 2 bytes.

### Visual Encoder as Separate Pass

In production VLM serving, the CLIP encoder runs as a **separate preprocessing pass** before the LLM forward pass:

1. Image arrives → encode with CLIP → 576 embeddings (no autoregression, parallelizable).
2. Pass embeddings through projector → 576 LLM-space tokens.
3. Concatenate with text tokens → feed into LLM prefill.
4. Run LLM decode normally.

Step 1 is embarrassingly parallelizable across images in a batch and can be pipelined with step 3 for subsequent requests. See [VLM Serving](../vlm-serving/en.md) for the full production patterns.

### Prefix Caching for Images

If the same image appears in multiple requests (e.g., a product image queried with different questions), the visual tokens in the KV cache can be reused via [prefix caching](../../foundational/prefix-caching/en.md) keyed on a hash of the image content. This requires that visual tokens always appear at the beginning of the sequence (which LLaVA enforces by convention). Cache hit on a 576-token visual prefix saves ~576 KV cache lookups worth of prefill compute.

## Engineering Tradeoffs

| Decision | Gained | Gave up |
|---|---|---|
| Freeze CLIP + LLM, train only projector (Stage 1) | Tiny training cost; 4 GPU-hours; image-text alignment with minimal data | Projector must bridge two independently trained embedding spaces; limits alignment depth |
| Full LLM fine-tune in Stage 2 | LLM adapts to visual-grounded instructions; better benchmark performance | 2–4× VRAM; longer training; risk of catastrophic forgetting of text-only capabilities |
| LoRA for Stage 2 | VRAM halved; original LLM weights preserved (mergeable) | Slightly lower ceiling vs. full fine-tune on complex visual reasoning |
| All 576 patch tokens (not [CLS]) | Rich spatial features; enables grounded QA and OCR | 576 tokens dominate prefill; KV cache grows proportionally |
| Dynamic tiling (LLaVA-NeXT) | High-resolution detail; fine-grained OCR/chart reading | Up to 2880 tokens/image; 5× more prefill compute vs. LLaVA-1.5 |
| Single visual encoder (no re-sampling) | Simple architecture; no Q-Former complexity | Token count fixed to patch count; cannot compress visual info adaptively |

## Experiments & Results

- **LLaVA-1.5-7B**: 78.5% on VQAv2, 38.9% on MMBench, 66.8% POPE (hallucination benchmark) — state-of-the-art among 7B models at the time.
- **LLaVA-1.5-13B**: outperforms all prior 7B and 13B models including InstructBLIP-13B on 11/11 benchmarks with only 1.2M training samples vs. InstructBLIP's 129M.
- **LLaVA-1.6 / NeXT**: large gains on document understanding (DocVQA, ChartQA) due to high-resolution tiling; best open-source results on multiple benchmarks in early 2024.

## Commentary

LLaVA established a template that nearly every subsequent open-source VLM follows: CLIP backbone + lightweight projector + instruction-tuned LLM. The simplicity is the point. More complex architectures (Q-Former in InstructBLIP, cross-attention layers in Flamingo) offer marginal gains while dramatically increasing complexity and training cost. For practitioners, the LLaVA recipe means you can build a competitive VLM with a few hundred GPU-hours and 1M data points — the barrier is not the model recipe but the quality of instruction data.

The visual token inflation problem is the central serving challenge that LLaVA inadvertently created. When 576 tokens per image was introduced, it seemed manageable. LLaVA-NeXT's tiling pushed this to 2880. Modern multi-image VLMs (Qwen-VL, Llama 4) can serve 4–8 images per request, reaching 10k–20k visual tokens before any text. This requires architectural changes in the serving stack: [chunked prefill](../../foundational/chunked-prefill/en.md) to avoid GPU stalls, [prefix caching](../../foundational/prefix-caching/en.md) to reuse repeated images, and attention kernels that handle heterogeneous sequence lengths gracefully.

The trajectory points toward adaptive token reduction: methods like LLaVA-HR, TokenPacker, and learned visual resamplers compress 576 tokens down to 64–144 without significant accuracy loss. This compression happens in the projector or a dedicated resampler, and is likely to become standard in production deployments where the 5× prefill cost of high-resolution tiling is prohibitive. The [Llama 4](../../meta/llama4/en.md) architecture already incorporates a learned cross-attention approach that controls token count.

## References

- [1] Liu et al. _Visual Instruction Tuning (LLaVA)._ arXiv:2304.08485, 2023.
- [2] Liu et al. _Improved Baselines with Visual Instruction Tuning (LLaVA-1.5)._ arXiv:2310.03744, 2023.
- [3] Liu et al. _LLaVA-NeXT: Improved reasoning, OCR, and world knowledge._ Blog post, 2024.
- [4] Related: [CLIP](../clip/en.md), [ViT](../vit/en.md), [LoRA](../../foundational/lora/en.md), [Chunked Prefill](../../foundational/chunked-prefill/en.md), [Prefix Caching](../../foundational/prefix-caching/en.md), [Llama 4](../../meta/llama4/en.md)
