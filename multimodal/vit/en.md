# Vision Transformer (ViT)

- **Authors / Org**: Alexey Dosovitskiy, Lucas Beyer, Alexander Kolesnikov et al. (Google Brain)
- **Published**: 2020-10-22 (ICLR 2021)
- **Links**: [arXiv:2010.11929](https://arxiv.org/abs/2010.11929) · [code](https://github.com/google-research/vision_transformer)
- **Variants covered**: ViT-B/16, ViT-L/14, ViT-H/14 · DeiT (Facebook, arXiv:2012.12877)

## TL;DR

ViT applies a standard transformer encoder — unchanged from the NLP world — directly to sequences of fixed-size image patches. Each patch is linearly projected into an embedding vector, a [CLS] token prepended, positional embeddings added, and the resulting sequence fed through multi-head self-attention layers exactly as in BERT. With sufficient pretraining data (JFT-300M), ViT matches or beats CNNs at lower computational cost. At patch size 14 and resolution 224px, ViT-L/14 produces 256 patch tokens and became the canonical vision backbone for [CLIP](../clip/en.md) and all downstream VLMs.

## Context & Motivation

Convolutional networks had dominated vision for a decade by 2020. They embed strong inductive biases: locality (each filter only looks at a local region) and translation equivariance (the same filter applied everywhere). These biases are helpful in the low-data regime — you need fewer examples to learn that edges matter regardless of where they appear. But at web scale, inductive biases are a cage: they constrain the model from learning arbitrary long-range dependencies that the data would otherwise teach.

The core question ViT asked was: **can a pure transformer, with no convolutional priors, learn to see?** The answer was yes — but only if you pre-train on enough data. On ImageNet alone (~1.2M images), ViT underperforms ResNet because the transformer has no locality bias and needs to learn spatial structure from scratch. On JFT-300M (300M images), ViT-L/14 surpasses EfficientNet with fewer FLOPs per inference. This data-hungry behavior has infrastructure implications: ViT is the backbone of choice for large-scale systems that have compute budgets for pretraining, not for fine-tuning on small datasets.

The practical consequence: [CLIP](../clip/en.md) chose ViT because CLIP trains on 400M web pairs — plenty of data to overcome ViT's sample-inefficiency — and because ViT scales cleanly with both depth and width.

## Architecture

### Patch Embedding

An image of resolution H×W with C channels is divided into N non-overlapping patches of size P×P:

$$N = \frac{H \times W}{P^2}$$

Each patch is a tensor of shape P×P×C. It is flattened to a vector of length P²C and linearly projected to dimension D:

$$\mathbf{z}_i = \mathbf{E} \cdot \text{flatten}(\text{patch}_i) + \mathbf{p}_i$$

where E ∈ R^{D×P²C} is a learned projection matrix and p_i is the positional embedding. This linear projection is the only image-specific component in ViT. Everything else is standard transformer.

Common configurations:

| Model | Patch size | Resolution | Patch tokens N | Params |
|---|---|---|---|---|
| ViT-B/16 | 16×16 | 224px | 196 | 86M |
| ViT-L/14 | 14×14 | 224px | 256 | 307M |
| ViT-L/14@336 | 14×14 | 336px | 576 | 307M |
| ViT-H/14 | 14×14 | 224px | 256 | 632M |

ViT-L/14@336px is significant: it is the image encoder used in CLIP and subsequently in LLaVA, producing **576 visual tokens per image** — a number that dominates VLM prefill costs (see [LLaVA](../llava/en.md)).

### [CLS] Token and Sequence Structure

A learnable [CLS] token is prepended to the patch sequence before the transformer. The final [CLS] hidden state is used as the image-level representation for classification. The full input sequence has length N+1.

This mirrors BERT's architecture. For downstream VLMs, the per-patch tokens (not the [CLS] token) are what gets passed to the language model — the spatial detail in each patch token is more informative for grounded tasks than the globally pooled [CLS].

### Positional Embeddings

ViT uses **1D learned positional embeddings** indexed by patch position (0 to N). This is the simplest possible approach. Despite the 2D nature of images, 1D indexing in raster order works well in practice.

Alternatives explored: 2D sinusoidal embeddings (slightly worse or equal), relative position encodings. When fine-tuning at higher resolution than pretraining (e.g., CLIP fine-tunes ViT-L/14 at 336px after pretraining at 224px), positional embeddings are interpolated — the N increases from 256 to 576 and intermediate positions are 2D-bilinearly interpolated.

### Transformer Encoder

Standard transformer blocks: Multi-Head Self-Attention (MHSA) + MLP, with LayerNorm applied before each sub-layer (Pre-LN variant). [FlashAttention](../../foundational/flash-attention/en.md) applies directly to ViT's MHSA without modification — the QKV attention computation is identical to language model attention, just with shorter sequences.

For ViT-L (1024-dimensional hidden state, 16 heads): each head operates on d_k = 64 dimensions. The attention matrix per layer per head is (N+1)×(N+1) = 577×577 for ViT-L/14@336 — small enough that FlashAttention's tiling overhead is minimal but its memory savings (no O(N²) materialization) remain valuable when processing large batches.

### Key Scaling Equation

Self-attention FLOPs scale as:

$$\text{FLOPs}_{\text{attn}} \approx 4 \cdot N^2 \cdot D$$

For ViT-L/14@336 (N=576, D=1024): ~1.4 GFLOPs per layer just for attention. At 24 layers, ~34 GFLOPs total for attention. The MLP layers (D→4D→D) contribute more: ~48 GFLOPs. Total ViT-L/14@336 forward pass ≈ 190 GFLOPs per image — this is the cost paid before any LLM decode in a VLM pipeline.

## DeiT: Data-Efficient ViT

DeiT (Touvron et al., 2020) showed that ViT can train competitively on ImageNet-1k (1.2M images) using **knowledge distillation** from a CNN teacher. Key additions:

- A distillation token (analogous to [CLS]) is added, trained to match the teacher's hard prediction.
- Aggressive data augmentation (RandAugment, MixUp, CutMix) substitutes for the large-scale pretraining data.

DeiT demonstrated that ViT's sample inefficiency is not fundamental — it can be addressed by knowledge transfer and augmentation. For practitioners without JFT-scale data, DeiT is the starting point.

## FlashAttention Compatibility

ViT's attention is identical in form to LLM attention. FlashAttention applies without change. The practical benefits:

- **Memory**: standard attention materializes a (N+1)² attention matrix per head, per batch element. For a batch of 64 images through ViT-L/14@336: 64 × 577² × 16 heads × 2 bytes = ~687 MB just for attention maps. FlashAttention eliminates this.
- **Speed**: sequence lengths (≤577) are short enough that ViT is typically compute-bound, not memory-bound, but FA still reduces memory pressure enabling larger batches.

## Engineering Tradeoffs

| Decision | Gained | Gave up |
|---|---|---|
| No convolutional priors | Arbitrary long-range dependencies; scales cleanly with data | Sample-inefficient; needs 100M+ images to beat CNNs |
| Linear patch projection | Simple; image-specific code is minimal (one matrix multiply) | No multi-scale features; fine-grained details smaller than P×P are averaged away |
| 1D learned positional embeddings | Simple; works well in practice | No built-in 2D structure; interpolation needed for resolution changes |
| [CLS] token for classification | Clean pooling via attention; matches BERT pattern | Forces global info into one vector; spatial detail lost for dense tasks |
| Larger patch size (e.g. 16 vs 14) | Fewer tokens → faster attention; fewer FLOPs | Coarser spatial resolution; worse at fine-grained tasks |
| Per-patch tokens for VLMs | Rich spatial features; enables grounded QA | More tokens passed to LLM (576 per image); dominates prefill cost |

## Experiments & Results

- ViT-L/16 on JFT-300M achieves 87.8% top-1 on ImageNet, outperforming prior BiT (ResNet-based) with ~4× fewer FLOPs.
- ViT-B/16 on ImageNet-21k + fine-tune: 85.5% top-1 — competitive with EfficientNet-L2.
- DeiT-B (86M params, ImageNet-1k only): 83.1% top-1 — ViT-competitive with CNN baselines, no extra data.
- When used as CLIP's image encoder (ViT-L/14@336), produces 576 tokens that drive 76.2% zero-shot ImageNet top-1.

## Commentary

ViT's contribution to the infrastructure stack is its near-perfect composability with the transformer ecosystem. Because ViT is just "transformer applied to patches," every optimization developed for language transformers — FlashAttention, mixed-precision training, tensor parallelism, quantization — transfers directly without adaptation. This is architecturally underrated: CNNs required separate optimization paths (depthwise-separable kernels, specialized NHWC layouts) while ViT rides on the GEMMs that GPUs are already maximally optimized for.

The 576-token cost of ViT-L/14@336 is the dominant infrastructure variable in VLM serving. When a user sends a 1024×1024 image tiled into four 336px tiles, each tile produces 576 visual tokens, totaling 2304 tokens that must be prefilled before a single output token is generated. This motivates [chunked prefill](../../foundational/chunked-prefill/en.md) for VLMs — without it, image-heavy requests would monopolize the GPU and stall text-only requests.

ViT's weakness is multi-scale feature extraction. CNNs naturally build a feature pyramid (fine details in early layers, semantics in late layers). ViT lacks this hierarchy — patch size P defines a single fixed spatial scale. DINOv2, Swin Transformer, and other works address this by adding hierarchical pooling or overlapping windows, but the standard ViT-L/14 used in CLIP remains the dominant choice for VLM backbones, with the resolution tradeoff (336px@P14 vs the ideal higher resolution) a known limitation that high-resolution tiling strategies (LLaVA-NeXT) partially compensate for.

## References

- [1] Dosovitskiy et al. _An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale._ arXiv:2010.11929, 2020.
- [2] Touvron et al. _Training data-efficient image transformers & distillation through attention (DeiT)._ arXiv:2012.12877, 2020.
- [3] Zhai et al. _Scaling Vision Transformers (ViT-22B)._ arXiv:2302.05442, 2022.
- [4] Related: [FlashAttention](../../foundational/flash-attention/en.md), [CLIP](../clip/en.md), [LLaVA](../llava/en.md), [Llama 4](../../meta/llama4/en.md)
