# DDPM / DDIM / Classifier-Free Guidance — Diffusion Model Fundamentals

- **Key Papers**: Ho et al. (2020), Song et al. (2020, 2022), Ho & Salimans (2022)
- **Published**: DDPM — 2020-06 · DDIM — 2020-10 · CFG — 2022-07 · Improved DDPM — 2021-02
- **Links**: [DDPM (arXiv:2006.11239)](https://arxiv.org/abs/2006.11239) · [DDIM (arXiv:2010.02502)](https://arxiv.org/abs/2010.02502) · [CFG (arXiv:2207.12598)](https://arxiv.org/abs/2207.12598) · [Improved DDPM (arXiv:2102.09672)](https://arxiv.org/abs/2102.09672)

## TL;DR

Diffusion models define a fixed forward process that gradually corrupts data with Gaussian noise, then train a neural network to reverse that process step by step. DDPM established the theoretical framework and the simple noise-prediction training objective. DDIM showed that the same trained model supports deterministic sampling with far fewer steps. Classifier-Free Guidance introduced a lightweight conditioning mechanism that trades sample diversity for quality without requiring a separate classifier. Together, these three ideas form the mathematical and engineering foundation for every modern image and video generation system — including DiT-based architectures (see [../multimodal/dit/](../../multimodal/dit/en.md)).

## Context & Motivation

Prior to diffusion models, generative image quality was dominated by GANs (mode collapse, training instability) and VAEs (blurry outputs, posterior collapse). Score-based generative models and diffusion probabilistic models emerged independently and were unified in 2021 under the score-matching / denoising objective. The key insight was that predicting the noise added to an image is equivalent to estimating the score (gradient of the log density) of the data distribution at each noise level, and this objective is stable and scalable in a way that GAN training is not.

The practical problem that motivated the DDIM and CFG improvements was cost: DDPM required 1000 sequential denoising steps, each a full neural network forward pass. For any network large enough to produce quality outputs, 1000 steps is prohibitively slow. DDIM halved the theoretical cost per image by a factor of ~20×; CFG made the outputs conditional on text or class without a separate classifier network, at the cost of doubling the forward passes per step.

## Forward Process

The forward process is a fixed (non-learned) Markov chain that progressively adds Gaussian noise to a data sample $x_0$ over $T$ timesteps:

$$q(x_t \mid x_{t-1}) = \mathcal{N}\!\left(x_t;\, \sqrt{1 - \beta_t}\, x_{t-1},\; \beta_t \mathbf{I}\right)$$

where $\beta_t \in (0, 1)$ is the noise schedule — a sequence of small positive values that controls how quickly the signal is destroyed. Because the process is Gaussian and Markovian, it admits a closed-form marginal:

$$q(x_t \mid x_0) = \mathcal{N}\!\left(x_t;\, \sqrt{\bar\alpha_t}\, x_0,\; (1 - \bar\alpha_t)\mathbf{I}\right)$$

where $\bar\alpha_t = \prod_{s=1}^t (1 - \beta_s)$. This means that at any arbitrary timestep $t$, the noisy sample can be computed in a single operation as $x_t = \sqrt{\bar\alpha_t}\, x_0 + \sqrt{1 - \bar\alpha_t}\, \varepsilon$, with $\varepsilon \sim \mathcal{N}(0, \mathbf{I})$. This closed-form marginal is critical for efficient training: there is no need to simulate the entire Markov chain to obtain a training example at step $t$.

As $t \to T$, $\bar\alpha_t \to 0$ and $x_T \approx \mathcal{N}(0, \mathbf{I})$ — pure isotropic noise. The forward process imposes no parameters to learn; it defines the target noisy distribution the reverse process must learn to denoise.

## Noise Schedules

The schedule $\{\beta_t\}$ determines how quickly information is destroyed.

**Linear schedule (DDPM original)**: $\beta_t$ increases linearly from $\beta_1 = 10^{-4}$ to $\beta_T = 0.02$ over $T = 1000$ steps. Simple, but at high resolutions it destroys too much signal in early steps, leaving a band of timesteps with poor SNR that don't contribute usefully to learning.

**Cosine schedule (Improved DDPM, 2021)**: defines $\bar\alpha_t$ directly as a cosine function: $\bar\alpha_t = \cos^2\!\left(\frac{t/T + s}{1 + s} \cdot \frac{\pi}{2}\right)$. This gives a smoother degradation curve, preserving more signal at moderate noise levels and avoiding the near-zero SNR plateau at the end of the linear schedule. The cosine schedule became standard for high-resolution image generation.

**Quadratic and sigmoid schedules**: variants that further tune the SNR trajectory for specific resolutions or latent spaces. When the denoising network operates on VAE latents rather than raw pixels (as in LDM / Stable Diffusion / DiT), the latent space statistics differ from pixels and the schedule often requires retuning.

Infrastructure consequence: the choice of schedule is not free at deployment. If you change the schedule after training, the learned denoiser is no longer calibrated — $\varepsilon_\theta(x_t, t)$ expects the noise statistics that correspond to the training schedule's $\bar\alpha_t$ at step $t$.

## Reverse Process and DDPM

The reverse process is a learned Markov chain that starts from $x_T \sim \mathcal{N}(0, \mathbf{I})$ and denoises back to $x_0$:

$$p_\theta(x_{t-1} \mid x_t) = \mathcal{N}\!\left(x_{t-1};\, \mu_\theta(x_t, t),\; \Sigma_\theta(x_t, t)\right)$$

DDPM parameterizes the mean via noise prediction: rather than directly learning $\mu_\theta$, the network $\varepsilon_\theta$ predicts the noise $\varepsilon$ that was added to $x_0$ to produce $x_t$. The mean is then recovered algebraically. The training objective simplifies to:

$$\mathcal{L}_{\text{simple}} = \mathbb{E}_{x_0,\, \varepsilon,\, t}\!\left[\, \|\varepsilon - \varepsilon_\theta(x_t, t)\|^2 \,\right]$$

This is a weighted evidence lower bound (ELBO) on the log-likelihood, with the weights set to a simple constant. The full VLB objective includes per-timestep KL terms that can improve log-likelihood at a small cost in sample quality; $\mathcal{L}_{\text{simple}}$ is preferred for visual generation tasks.

**Covariance parameterization**: DDPM fixes $\Sigma_\theta = \beta_t \mathbf{I}$ (the forward posterior variance). Improved DDPM learns a linear interpolation between the two valid variance choices ($\beta_t$ and $\tilde\beta_t = \frac{1-\bar\alpha_{t-1}}{1-\bar\alpha_t}\beta_t$), which improves log-likelihood scores. Fixed covariance is simpler and still sufficient for perceptual quality.

**Sampling (DDPM)**: starting from $x_T \sim \mathcal{N}(0, \mathbf{I})$, run the denoising chain for $T = 1000$ steps. Each step requires one forward pass through the denoising network. At an image resolution of 256×256 with a DiT-XL backbone, this is approximately 119 GFLOPs × 1000 steps = 119 TFLOPs per image.

## DDIM: Deterministic Sampling with Fewer Steps

DDIM (Denoising Diffusion Implicit Models) observes that DDPM's stochastic sampler is not the only valid reverse process. For any choice of auxiliary variable variance $\sigma_t \geq 0$, there exists a valid non-Markovian reverse process whose marginals are consistent with the same forward process and the same trained denoiser. The DDIM update rule is:

$$x_{t-1} = \sqrt{\bar\alpha_{t-1}}\underbrace{\left(\frac{x_t - \sqrt{1-\bar\alpha_t}\,\varepsilon_\theta}{\sqrt{\bar\alpha_t}}\right)}_{\text{predicted }x_0} + \sqrt{1 - \bar\alpha_{t-1} - \sigma_t^2}\,\varepsilon_\theta + \sigma_t\,\varepsilon_t$$

Setting $\sigma_t = 0$ for all $t$ gives a fully deterministic trajectory: the same $x_T$ always produces the same $x_0$. This determinism enables:

1. **Step skipping**: because the trajectory is consistent (not Markovian), you can evaluate the denoiser at a subset of $\{T, T-\Delta, T-2\Delta, \ldots, 0\}$ rather than every integer step. With 50 steps instead of 1000, DDIM produces comparable quality to DDPM's 1000 steps, reducing the number of function evaluations (NFE) by 20×.
2. **Latent space interpolation**: two images can be inverted to their $x_T$ representation (DDIM inversion) and interpolated in noise space, producing semantically smooth transitions.
3. **Controllable editing**: DDIM inversion followed by conditional re-sampling enables image editing without retraining.

**NFE is the primary latency driver**. For any diffusion model, wall-clock inference time is approximately:

$$\text{latency} \approx \text{NFE} \times t_{\text{step}}$$

where $t_{\text{step}}$ is the wall time of one forward pass through the denoising network. NFE is directly controllable at inference time; $t_{\text{step}}$ is fixed by model size, hardware, and batch size. A 50-step DDIM inference on DiT-XL/2 at batch size 1 is roughly 50 × 4–6ms ≈ 200–300ms on an H100. Batch inference multiplies $t_{\text{step}}$ by a factor that grows sublinearly until memory is saturated, so batch throughput is almost always preferable to latency optimization for non-interactive workloads.

The $\sigma_t > 0$ (stochastic) regime of DDIM interpolates between DDPM ($\sigma_t = \sqrt{\tilde\beta_t}$) and the deterministic case. At $\sigma_t = 0$ the diversity of outputs depends entirely on the diversity of the $x_T$ starting points; there is no stochastic injection during sampling.

## Classifier-Free Guidance (CFG)

Classifier Guidance (Dhariwal & Nichol, 2021) showed that shifting the denoising score by the gradient of a separately trained classifier's log-likelihood improved perceptual quality at the cost of diversity. This required training and running a separate classifier at every denoising step — an expensive proposition.

Classifier-Free Guidance collapses the classifier into the denoising network itself. During training, the conditioning signal $c$ (text embedding, class label) is dropped with probability $p_{\text{uncond}} \approx 0.1$–$0.2$, replaced by a null token $\varnothing$. The single network thereby learns both conditional and unconditional score functions. At inference, the guided score is:

$$\hat\varepsilon_\theta(x_t, t, c) = \varepsilon_\theta(x_t, t, \varnothing) + w \cdot \left(\varepsilon_\theta(x_t, t, c) - \varepsilon_\theta(x_t, t, \varnothing)\right)$$

where $w \geq 0$ is the guidance scale. At $w = 0$ you recover unconditional sampling; at $w = 1$ you recover standard conditional sampling; $w > 1$ exaggerates the conditioning direction, improving alignment to the condition at the cost of sample diversity and sometimes saturation artifacts.

**Typical values**: $w = 7.5$ in Stable Diffusion, $w = 4.0$–$6.0$ in DiT-XL/2 evaluations. The optimal $w$ is dataset and task dependent and must be tuned.

**Inference cost of CFG**: each denoising step now requires two forward passes — one conditional, one unconditional. This doubles $t_{\text{step}}$ compared to unconditional or standard conditional sampling. For a 50-step DDIM + CFG pipeline, the true NFE is 100. At $w = 0$ (uncond only) or if you batch the two passes, the cost can be recovered partially by packing both prompts into a single batch of size $2B$.

**Memory consequence of CFG**: the standard implementation doubles the batch size by concatenating conditional and unconditional inputs before the forward pass. For a batch of $B$ images, this means the denoiser runs on $2B$ samples per step. At high batch sizes, this can double peak activation memory, potentially halving the maximum feasible batch size before OOM.

## Engineering Consequences: NFE Is the Cost Axis

The core infrastructure insight of diffusion inference is that **NFE dominates latency**, and NFE is separately tunable from model quality via the sampling algorithm. This gives a different optimization surface than language model inference:

- LLM decode latency is determined by model size × output length × per-token compute, none of which are freely tunable at inference time without retraining.
- Diffusion latency is determined by NFE × per-step cost. NFE can be reduced from 1000 to 50 (DDIM), to 20 (DPM-Solver), to 4–8 (consistency models, flow matching distillation) without retraining the base model, at graduated quality cost.

**Per-step cost** is determined by the denoising backbone. For a UNet (Stable Diffusion 1.x/2.x), a single step involves an encoder-attention-decoder pass over the spatial activations. For a DiT, it is a standard transformer forward pass over $N$ patch tokens with bidirectional attention. DiT steps are compute-bound and parallelize over large batches cleanly; UNet steps are more bandwidth-bound at small resolutions due to their sequential encoder-decoder structure.

**Batch inference**: diffusion models have no sequential dependency between batch elements (unlike autoregressive LLMs), so batching images together amortizes weight loads perfectly. The GPU utilization curve is flat as a function of batch size until VRAM fills — at that point OOM rather than a soft throughput curve. With CFG doubling effective batch size, the practical maximum batch before OOM is approximately half the single-pass limit.

**Latency SLOs in production**: interactive use cases (consumer image generation) typically need end-to-end latency under 3–5 seconds. With a DiT-XL backbone (119 GFLOPs/step) on an H100:
- DDPM 1000 steps: ~100+ seconds at batch size 1 — unacceptable.
- DDIM 50 steps + CFG: ~0.5–1.5 seconds — borderline.
- Flow matching / consistency distillation 4–8 steps: ~100–400ms — acceptable.

This is why consistency models and flow matching (FLUX, SD3) are not optional optimizations but functional requirements for any production image generation system with latency constraints.

## Engineering Tradeoffs

| Approach | Pros | Cons | When to use |
|---|---|---|---|
| DDPM (1000 steps) | Highest log-likelihood; stochastic diversity; theoretically cleanest | 1000 NFE per image; prohibitive latency for most use cases | Offline research, FID benchmarking, when quality ceiling matters and latency is unconstrained |
| DDIM (50 steps, deterministic) | 20× fewer NFE; invertible (enables editing); same trained model | Slightly lower diversity than stochastic DDPM; quality degrades below ~20 steps | Production image generation; offline batch jobs; all use cases where DDPM is the baseline model |
| DDIM + CFG ($w > 1$) | Strong conditional alignment; high perceptual quality | 2× NFE vs. unconditional; batch-size doubling memory spike; diversity reduced | Text-to-image, class-conditional generation where prompt fidelity matters more than diversity |
| Flow matching / distillation (4–8 steps) | Latency competitive with VLM generation; interactive use viable | Requires additional training (distillation or flow-matching pretraining); some quality loss | Interactive applications, real-time generation, mobile deployment |

## Connection to DiT

The math above is backbone-agnostic: DDPM / DDIM / CFG operate identically whether the denoising network $\varepsilon_\theta$ is a UNet or a DiT. The DiT entry in this repository (see [../../multimodal/dit/](../../multimodal/dit/en.md)) uses exactly this training objective — Equation $\mathcal{L}_{\text{simple}}$ — with the DiT network serving as $\varepsilon_\theta$. The adaLN conditioning in DiT is the mechanism by which the timestep $t$ and class label $c$ are injected into the network; it implements the CFG null-token mechanism by dropping $c$ to a learned unconditional embedding during training.

The practical consequence of this separation is that you can swap denoising backbones without changing the diffusion algorithm, and you can swap sampling algorithms (DDPM, DDIM, DPM-Solver, consistency) without retraining the backbone. This modularity is a deliberate design property of the framework, and it is why the diffusion literature can iterate on sampling algorithms independently of architecture research.

## Reproducibility Notes

All three core algorithms are fully implemented in open-source libraries:

- HuggingFace `diffusers`: `DDPMScheduler`, `DDIMScheduler`, `UniPCMultistepScheduler`, all with CFG support.
- The reference implementations (Ho 2020, Song 2020) are small and reproducible on a single A100 in hours for class-conditional ImageNet.
- CFG guidance scale tuning requires grid search per model; $w = 7.5$ is a reasonable starting point for text-to-image but is not universal.
- Noise schedule changes require retraining or careful rescaling; do not mix schedule types between training and inference.

## References

- [1] Ho et al. _Denoising Diffusion Probabilistic Models._ arXiv:2006.11239, 2020.
- [2] Song et al. _Denoising Diffusion Implicit Models._ arXiv:2010.02502, 2020.
- [3] Nichol & Dhariwal. _Improved Denoising Diffusion Probabilistic Models._ arXiv:2102.09672, 2021.
- [4] Dhariwal & Nichol. _Diffusion Models Beat GANs on Image Synthesis._ arXiv:2105.05233, 2021.
- [5] Ho & Salimans. _Classifier-Free Diffusion Guidance._ arXiv:2207.12598, 2022.
- [6] Song et al. _Score-Based Generative Modeling through SDEs._ arXiv:2011.13456, 2020.
- [7] Lipman et al. _Flow Matching for Generative Modeling._ arXiv:2210.02747, 2022.
- [8] Related: [DiT](../../multimodal/dit/en.md)
