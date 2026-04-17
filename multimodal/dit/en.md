# DiT — Scalable Diffusion Models with Transformers

- **Authors / Org**: William Peebles, Saining Xie (UC Berkeley / Meta AI)
- **Published**: 2022-12-19 (ICCV 2023)
- **Links**: [arXiv:2212.09748](https://arxiv.org/abs/2212.09748) · [code](https://github.com/facebookresearch/DiT)

## TL;DR

DiT replaces the U-Net backbone in latent diffusion models with a pure transformer, following the same patch-tokenization logic as [ViT](../vit/en.md). A VAE compresses images into latent space; the latent is patchified into tokens; a transformer denoises those tokens for T diffusion steps conditioned on timestep and class label via adaptive layer normalization (adaLN). DiT-XL/2 (675M parameters, patch size 2) achieves a new state-of-the-art FID of 2.27 on class-conditional ImageNet 256×256 while scaling better with compute than any prior U-Net variant. DiT became the backbone of Sora, Stable Diffusion 3, and most subsequent video generation systems.

## Context & Motivation

U-Net was the standard backbone for diffusion models (DDPM, LDM, Stable Diffusion 1.x). U-Net has strong inductive biases: skip connections between encoder and decoder stages provide multi-scale feature reuse, and convolutional layers exploit spatial locality. These biases were helpful at small scale but they also limited how cleanly U-Net could benefit from simply making it bigger.

The transformer scaling laws (Chinchilla, GPT-3) had established that transformers scale more predictably with compute than CNNs. DiT asked: if we replace U-Net with a transformer, do we get the same clean scaling in the generative setting? The answer was yes — DiT's Gflop-FID curve is monotonically improving and the transformer architecture allows the same optimizations (FlashAttention, tensor parallelism, FP8) that power LLMs.

DiT's practical impact was enormous. Sora (OpenAI, 2024) uses a DiT-based "spacetime patch" transformer for video generation. Stable Diffusion 3 and FLUX replace the U-Net with a multimodal DiT (MM-DiT) that jointly attends over image and text tokens. The generation side of multimodal AI is now as transformer-dominated as the understanding side.

## Latent Diffusion Background

DiT operates in **latent space** rather than pixel space, following Latent Diffusion Models (LDM / Stable Diffusion):

1. **Encoder**: a VAE encoder f compresses an image x ∈ R^{H×W×3} to a latent z ∈ R^{h×w×c}. Typical compression: 8× spatial (512px → 64×64 latent), c=4 channels. This reduces pixel-space computation by 64×.
2. **Diffusion in latent space**: add Gaussian noise to z over T timesteps following a noise schedule. The forward process: z_t = √ᾱ_t z + √(1−ᾱ_t) ε, ε ~ N(0,I).
3. **Denoising**: a neural network predicts the noise ε̂ or the denoised latent directly. DiT is this denoising network.
4. **Decoder**: the VAE decoder g maps the denoised latent back to pixel space.

The VAE is pretrained separately and frozen during DiT training/inference. The only component trained/run during generation is the DiT denoiser.

## Architecture

### Patch Tokenization of Latent

The latent z ∈ R^{h×w×c} is tokenized using the same strategy as ViT:

1. Divide z into non-overlapping patches of size p×p. For DiT with patch size 2 and a 32×32 latent (from 256px image, 8× VAE): N = (32/2)² = 256 patches.
2. Flatten each patch: p×p×c → vector of length p²c.
3. Linearly project to dimension D.
4. Add 2D sinusoidal positional embeddings.

This is the only image-specific code in DiT. The rest is transformer blocks.

For higher-resolution or video generation, the token count grows as (h×w)/p². Sora's "spacetime patches" tokenize (time, height, width) jointly: a 1-second 720p video at 24fps with patch size (t=1, h=8, w=8) gives N = 24 × (720/8) × (1280/8) = 24 × 90 × 160 = 345,600 tokens — why video generation is so much more expensive than image generation.

### DiT Transformer Blocks

Standard transformer blocks with two modifications:

**adaLN (adaptive layer normalization)** for conditioning on timestep t and class label c:

$$\text{adaLN}(x, t, c) = \gamma(t, c) \cdot \frac{x - \mu}{\sigma} + \beta(t, c)$$

The scale (γ) and shift (β) parameters are not fixed — they are functions of the diffusion timestep and class label, produced by a small MLP applied to the sum of the timestep embedding and class embedding. This allows the transformer to behave differently at different noise levels (early denoising steps vs. late refinement steps) without needing separate weights.

Four conditioning strategies were evaluated in the paper; adaLN-Zero performed best: γ and β are initialized to produce identity (γ=1, β=0) so early in training the residual blocks act as identity functions, making training stable.

**No causal masking**: unlike language models, DiT uses bidirectional attention — every patch token can attend to every other patch token. This is correct because denoising is not autoregressive; all latent patches are denoised simultaneously at each timestep.

### DiT Variants

| Model | Layers | D | Heads | Patch size | GFLOPs | Params |
|---|---|---|---|---|---|---|
| DiT-S/2 | 12 | 384 | 6 | 2 | 6.1 | 33M |
| DiT-B/4 | 12 | 768 | 12 | 4 | 5.6 | 130M |
| DiT-L/4 | 24 | 1024 | 16 | 4 | 19.7 | 458M |
| DiT-XL/2 | 28 | 1152 | 16 | 2 | 118.6 | 675M |

Patch size 2 yields 256 tokens (for 256px images) — 4× more than patch size 4, hence 16× more attention FLOPs (N² scaling). DiT-XL/2 is the best-performing model; its 119 GFLOPs per denoising step is the single-step cost, not the total inference cost.

## Inference: Sequential Denoising

DiT inference is fundamentally different from LLM inference:

- **No autoregressive loop** over tokens — all N latent tokens are denoised in parallel at each step.
- **Sequential loop** over T timesteps (T=250 for DDPM, T=50 for DDIM/DPM-Solver).
- Total FLOPs per image: 119 GFLOPs × 250 steps = **29.75 TFLOPs** for DiT-XL/2 at full DDPM sampling.

This is **compute-bound**, not memory-bound. Unlike LLM decode where the model weights are read repeatedly for each token and you are typically bandwidth-limited, DiT loads weights once per denoising step and processes all N tokens in a fully parallel forward pass. On H100:

- With DDIM (50 steps): 119 GFLOPs × 50 ≈ 5.95 TFLOPs. H100 FP16 peak ≈ 989 TFLOPs/s → ~6ms/image at 100% utilization. Practical throughput (accounting for overhead, VRAM bandwidth, etc.): ~50–200ms/image at batch size 1.

Increasing batch size improves GPU utilization until VRAM is full — there is no sequential dependency between images in a batch. This is the opposite of LLM decode, where batch size increases help throughput but each request must wait for its own sequential decode.

## Key Equation — Noise Schedule and Denoising Objective

The training objective is to minimize:

$$\mathcal{L} = \mathbb{E}_{z, \varepsilon, t} \left[ \| \varepsilon - \varepsilon_\theta(z_t, t, c) \|^2 \right]$$

where ε_θ is the DiT network predicting the noise, z_t is the noisy latent at timestep t, and c is the conditioning (class label or text embedding). This is the standard DDPM ε-prediction objective, unchanged from U-Net diffusion — DiT is purely a backbone replacement.

## Engineering Tradeoffs

| Decision | Gained | Gave up |
|---|---|---|
| Transformer instead of U-Net | Clean scaling laws; reuses LLM optimizations (FA, TP, FP8) | No multi-scale feature hierarchy; must learn global-to-local relationships from attention |
| Latent diffusion (VAE + denoise in latent) | 64× cheaper than pixel-space; quality maintained | VAE encoding/decoding adds latency; VAE reconstruction ceiling limits output quality |
| adaLN conditioning | Strong timestep + class conditioning; stable training via zero-init | adaLN MLP runs at every layer; minor FLOPs overhead |
| Bidirectional attention | All patches inform all others; more globally coherent generations | Cannot stream outputs; entire latent must be generated before decoding to pixels |
| Small patch size (p=2) | Higher spatial fidelity; more tokens capture fine detail | N² attention cost; 16× more attention FLOPs than p=4 |
| Sequential denoising (T steps) | Flexible quality-speed tradeoff via T; progressive refinement | Minimum T steps regardless of image complexity; few-step distillation (consistency models) needed for real-time use |

## Experiments & Results

- **DiT-XL/2**: FID-50K = 2.27 on class-conditional ImageNet 256×256 — new state-of-the-art, better than all prior diffusion and GAN models.
- **Scaling**: FID improves monotonically as GFLOPs increase (smaller patch size + larger model); the scaling is smooth, unlike U-Net variants that plateau.
- **Classifier-free guidance**: at guidance scale 4.0, DiT-XL/2 achieves FID=2.27, IS=278.24 — generation quality competitive with DALL-E 2.
- **Downstream**: DiT-XL/2 backbone was adopted in Sora, SD3, FLUX.1, and numerous open-source video generation models.

## Commentary

DiT's primary contribution is not a clever new attention mechanism or a better noise schedule — it is the **empirical confirmation that transformers scale better than U-Nets for diffusion**. This is infrastructure-relevant because it means the entire LLM optimization toolchain (FlashAttention, quantization, tensor parallelism, KV cache techniques adapted to cross-attention) transfers to generative models. The two major branches of foundation model deployment — understanding/generation — are now both transformer-shaped and can share infrastructure.

The compute profile of DiT inference (compute-bound, parallel over tokens, sequential over timesteps) requires a different batching strategy than LLM inference. LLMs are latency-constrained in the decode phase and benefit from continuous batching; DiT benefits from large image batches processed in parallel. A production image generation cluster running DiT does not need the complex request scheduler of a vLLM deployment — it looks more like a classical batch ML workload where you pack images to fill GPU memory and submit all denoising steps back-to-back.

The most interesting open problem in DiT serving is **few-step distillation**: consistency models and flow matching (as in SD3 and FLUX) can generate acceptable images in 4–8 steps instead of 50–250. This reduces DiT inference from ~100ms to ~10ms, pushing it into the same latency regime as VLM text generation. At that point, the serving infrastructure for combined understanding + generation multimodal systems starts to look like a single unified pipeline — and the separation between "generative" and "understanding" models at the infrastructure level begins to dissolve. See [Mamba-SSM](../../foundational/mamba-ssm/en.md) for hybrid DiT-Mamba architectures that also attempt to reduce the per-step compute.

## References

- [1] Peebles & Xie. _Scalable Diffusion Models with Transformers._ arXiv:2212.09748, 2022.
- [2] Rombach et al. _High-Resolution Image Synthesis with Latent Diffusion Models._ arXiv:2112.10752, 2021.
- [3] Ho et al. _Denoising Diffusion Probabilistic Models._ arXiv:2006.11239, 2020.
- [4] Song et al. _Consistency Models._ arXiv:2303.01469, 2023.
- [5] Related: [ViT](../vit/en.md), [FlashAttention](../../foundational/flash-attention/en.md), [Mamba-SSM](../../foundational/mamba-ssm/en.md)
