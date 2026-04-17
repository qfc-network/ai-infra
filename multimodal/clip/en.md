# CLIP — Contrastive Language-Image Pretraining

- **Authors / Org**: Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh et al. (OpenAI)
- **Published**: 2021-02-26
- **Links**: [arXiv:2103.00020](https://arxiv.org/abs/2103.00020) · [blog](https://openai.com/research/clip) · [code](https://github.com/openai/CLIP)

## TL;DR

CLIP trains two encoders — one for images, one for text — jointly on 400 million image-text pairs scraped from the web using a contrastive objective. At inference, zero-shot classification is performed by encoding text descriptions of each class and selecting the class whose text embedding is most similar to the image embedding. CLIP demonstrated that web-scale weakly-supervised contrastive learning could match or beat supervised ImageNet baselines without ever seeing a single ImageNet training label. It became the de-facto vision backbone for vision-language models and RLHF reward modeling.

## Context & Motivation

Before CLIP, vision models were trained on carefully curated, hand-labeled datasets (ImageNet, COCO). This created a two-pronged bottleneck: label acquisition was expensive and the fixed label space made zero-shot transfer to new tasks impossible. NLP had solved an analogous problem with self-supervised pretraining on raw web text — the insight in CLIP was to apply the same philosophy to vision, using naturally occurring image-text pairs on the internet as a free source of supervision.

Concurrent weakly-supervised vision work (ALIGN, ConVIRT) explored similar territory. CLIP's contribution was scale: 400M pairs, a clean contrastive objective, and a thorough zero-shot evaluation across 30 datasets. The resulting embeddings generalize so well that CLIP features power essentially every subsequent VLM, including [LLaVA](../llava/en.md) and [Llama 4](../../meta/llama4/en.md).

## Architecture

### Dual-Encoder Design

CLIP uses two independent encoders:

**Image encoder**: either a ResNet (with attention pooling) or a Vision Transformer ([ViT](../vit/en.md)). The best-performing public model uses ViT-L/14 — a ViT with patch size 14px, trained at 336px resolution for fine-tuning. The encoder outputs a single fixed-dimensional vector per image (e.g., 768-d for ViT-L).

**Text encoder**: a transformer with causal masking, treating the BPE-tokenized text as a sequence. The representation of the `[EOS]` token is used as the text embedding, linear-projected to the shared embedding space.

Both encoder outputs are L2-normalized before computing similarities. The key design principle: the two towers share no weights and have no cross-attention — the only interaction between modalities happens through the contrastive loss.

### Contrastive InfoNCE Loss

For a batch of N image-text pairs, CLIP constructs an N×N similarity matrix:

$$S_{ij} = \frac{\text{img}_i \cdot \text{txt}_j}{\tau}$$

where τ is a learned temperature parameter. The loss treats the N correct pairings (diagonal entries) as positives and the N²−N off-diagonal entries as negatives:

$$\mathcal{L} = -\frac{1}{2N} \sum_{i=1}^{N} \left[ \log \frac{e^{S_{ii}}}{\sum_j e^{S_{ij}}} + \log \frac{e^{S_{ii}}}{\sum_j e^{S_{ji}}} \right]$$

The two terms are symmetric: image-to-text and text-to-image directions. The loss requires the model to identify, for each image in the batch, which of the N text descriptions matches it — and vice versa. This is InfoNCE (noise-contrastive estimation), also known as NT-Xent.

**Temperature τ**: CLIP learns τ as a log-scaled parameter, initialized to 0.07. Temperature controls the sharpness of the distribution: low τ makes the softmax peakier, effectively harder negatives. At large batch sizes (CLIP used up to 32,768) the in-batch negatives are numerous and diverse enough to drive learning without mined hard negatives.

### Zero-Shot Classification

At inference, zero-shot classification over K classes works as:

1. For each class label c, construct a text prompt: `"a photo of a {class}"` (prompt engineering matters here).
2. Encode all K prompts → K text embeddings.
3. Encode the query image → 1 image embedding.
4. Predict the class with highest cosine similarity.

No fine-tuning, no class-specific parameters. The model generalizes because it learned a shared embedding space where semantically related images and texts cluster together.

## Training at Web Scale

CLIP was trained on **WIT (WebImageText)** — 400M image-text pairs collected from the internet. No ImageNet labels were used. Key training details:

- Batch size: up to 32,768 — critical for having meaningful in-batch negatives.
- The N² negative pairs per batch are computed across all GPUs via distributed all-gather, making communication cost proportional to batch size.
- Mixed precision (FP16) training; [Hopper/H100-era](../../foundational/hopper-h100/en.md) tooling was not yet available but the same principles apply.
- Total compute: ~700 GPU-years equivalent for the largest model (ViT-L/14@336px).

The use of noisy web data rather than curated labels is philosophically similar to [Whisper's](../whisper/en.md) approach to speech recognition.

## Key Equation

The normalized cosine similarity used at inference:

$$\text{sim}(v, t) = \frac{v \cdot t}{\|v\| \cdot \|t\|}$$

where v is the image embedding and t is the text embedding, both output by their respective encoders and already L2-normalized (so this reduces to a dot product).

## Engineering Tradeoffs

| Decision | Gained | Gave up |
|---|---|---|
| Dual-encoder, no cross-attention | O(1) retrieval after encoding; image and text can be encoded independently and cached | Cannot model fine-grained image-text interactions; cross-encoder re-rankers needed for precision-critical tasks |
| In-batch negatives only | No offline mining pipeline; scales with batch size naturally | Need very large batches (32k+) for good coverage; communication overhead grows with batch |
| Learned temperature τ | Self-calibrating sharpness across training | One more hyperparameter to monitor; divergence risk if τ collapses to near 0 |
| Symmetric loss (both directions) | Richer gradient signal; better-aligned embeddings | ~2× loss computation vs. one-sided |
| Web-scraped noisy data | 400M pairs at near-zero label cost | Social biases and factual errors baked into the embedding space; requires careful bias evaluation |
| Shared embedding space | Clean zero-shot transfer; universal representations | Text and image modalities must agree on dimensionality; compression of rich visual content into single vector loses spatial detail |

## Experiments & Results

- **Zero-shot ImageNet**: ViT-L/14 CLIP matches ResNet-50 supervised on ImageNet without seeing any ImageNet labels (76.2% top-1 for ViT-L/14@336px).
- **Transfer breadth**: Evaluated across 30 diverse datasets (OCR, action recognition, geo-localization, texture). Competitive or superior to supervised baselines on most.
- **Linear probe**: Linear probe on frozen CLIP features outperforms full fine-tuning of many prior self-supervised methods.
- **Robustness**: CLIP embeddings are significantly more robust to distribution shift than standard ImageNet-trained models (ImageNet-A, ObjectNet benchmarks).

## Commentary

CLIP's lasting impact is not its ImageNet number — it is the realization that a shared image-text embedding space trained at scale becomes a universal interface between modalities. Every downstream VLM (LLaVA, InstructBLIP, Flamingo, LLaMA-Vision) plugs CLIP's ViT-L/14 as the frozen vision backbone, using a lightweight projector to bridge into the language model's embedding space. CLIP essentially solved "feature extraction for images" the same way BERT solved it for text — train once at scale, freeze, and reuse.

From an infrastructure perspective, CLIP's dual-encoder structure has strong serving advantages: image embeddings can be precomputed and cached offline. For a product serving millions of queries against a fixed image corpus (e.g., semantic image search), you encode the corpus once and then answer queries with a single matrix multiply. The same caching logic extends to [prefix caching](../../foundational/prefix-caching/en.md) of visual tokens in VLM serving.

The main limitation is inherent to the contrastive paradigm: CLIP produces a single vector per image and a single vector per text — all spatial and compositional structure is squeezed out. This is why LLaVA-style systems keep the full sequence of patch tokens from the ViT encoder rather than the pooled CLS token — the spatial tokens carry far more information for grounded QA tasks. CLIP as a zero-shot classifier is elegant; CLIP as a feature backbone is most powerful when you retain per-patch features, not the pooled output.

## References

- [1] Radford et al. _Learning Transferable Visual Models From Natural Language Supervision._ arXiv:2103.00020, 2021.
- [2] Jia et al. _Scaling Up Visual and Vision-Language Representation Learning With Noisy Text Supervision (ALIGN)._ arXiv:2102.05918, 2021.
- [3] Zhang et al. _ConVIRT: Contrastive Learning of Medical Visual Representations from Paired Images and Text._ arXiv:2010.00747, 2020.
- [4] Related: [ViT](../vit/en.md), [LLaVA](../llava/en.md), [RLHF](../../foundational/rlhf/en.md), [Llama 4](../../meta/llama4/en.md)
