# Weight-Only Quantization — GPTQ and AWQ

- **Papers**:
  - **GPTQ** — Frantar et al., IST Austria. ICLR '23.
  - **AWQ** — Lin et al., MIT + NVIDIA + SJTU + UMass. MLSys '24.
- **Links**: [GPTQ (arXiv:2210.17323)](https://arxiv.org/abs/2210.17323) · [AWQ (arXiv:2306.00978)](https://arxiv.org/abs/2306.00978) · [GPTQ code](https://github.com/IST-DASLab/gptq) · [AWQ code](https://github.com/mit-han-lab/llm-awq)

## TL;DR

Two post-training quantization methods that reduce LLM weights from FP16/BF16 to **INT4 with minimal accuracy loss**, enabling inference on consumer GPUs and dramatic serving cost reductions. Both are **weight-only**: activations stay in FP16, only weights are compressed. They differ in how they decide what to quantize and how:

- **GPTQ** uses **second-order information (inverse Hessian via OBQ)** to greedily pick and update quantization grid points column-by-column, minimizing layer-wise reconstruction error.
- **AWQ** observes that **a small fraction of weight channels are "salient"** (large activations multiply them), and applies a **per-channel scaling trick** to protect those channels under uniform INT4 quantization — no gradient, no backprop, no second-order.

Both run **post-training** (hours on a calibration set, no retraining), preserve perplexity to within ~0.1–0.5 of FP16, and have become the default weight formats in llama.cpp, vLLM, TGI, and every local-LLM deployment. AWQ is generally preferred in modern stacks for its simplicity, hardware-friendliness, and per-channel activation awareness; GPTQ remains common for its broader shape support and robust open-source implementations.

## Context & Motivation

A 70B model in FP16 is 140 GB of weights — beyond a single consumer GPU, expensive on H100s. Full-training-time quantization (QAT) requires re-running the training, which is infeasible for released checkpoints. Post-training quantization (PTQ) to INT8 had long been trivial (minor accuracy loss at 8-bit); **INT4 without retraining had been the hard case**. Errors at 4-bit swamp the forward pass; naive round-to-nearest destroys perplexity.

The question: can a smart calibration step on a few hundred samples recover most of the lost accuracy at 4-bit, without touching training?

GPTQ answered "yes, with Hessian-informed column updates." AWQ answered "yes, and you don't need the Hessian — just protect the salient channels."

## GPTQ — Hessian-informed column-by-column quantization

Builds on **Optimal Brain Quantization (OBQ)**, which generalizes Optimal Brain Surgeon to discretization. The core idea: when you quantize a weight column, you can compute the *optimal update to the remaining (unquantized) columns* that compensates for the error, using the **inverse of the local Hessian** of the layer's reconstruction loss.

Naive OBQ is `O(d³)` per layer — infeasible for 70B models. GPTQ's contributions:

1. **Arbitrary order** — OBQ greedily picks the column with smallest expected error at each step. GPTQ shows a **fixed left-to-right order** works nearly as well, eliminating an `O(d²)` factor.
2. **Lazy batch updates** — instead of updating all remaining columns after each quantization, batch updates in groups. Memory-friendlier, numerically stable.
3. **Cholesky reformulation** — the inverse-Hessian math is rewritten in terms of a Cholesky factor, making the main loop a well-conditioned triangular solve.

Result: quantize a 70B model in about an hour on a single A100, with perplexity loss typically <0.3 at INT4.

The output is **uniform INT4 per group** (commonly group size 128 — every 128 consecutive weights along a row share one FP16 scale + zero-point). Group size trades off compression ratio vs accuracy: 128 is the common sweet spot.

## AWQ — activation-aware weight quantization

Starts from a different observation: if you look at the magnitudes of weights **scaled by their corresponding activations**, a small fraction (~1%) of weight channels contribute orders of magnitude more to the output than the rest. These are the **salient** channels — if you preserve them, the rest can be aggressively quantized; if you don't, accuracy collapses even at 8-bit.

Naive protection: keep salient channels in FP16 and quantize the rest to INT4. Works, but mixed-precision tensors are a pain for kernels — breaks simple matmul layouts, adds branches.

AWQ's trick: **scale salient channels up before quantization, scale the corresponding activations down during inference**. Mathematically, for an input `x` and weight column `w`:

```
y = x · w = (x / s) · (w · s)    for any s > 0
```

Choosing `s` so that salient columns' scaled weights fit comfortably in the INT4 range, and non-salient columns' scaled activations remain in range, preserves the product. **Every weight is still INT4** — the only change is a per-channel scale, which can be **absorbed into the preceding layer's weights or normalization**, free at inference.

Search: the optimal `s` per channel is found by grid search over a few values on a calibration set (~128 samples). No gradients, no Hessian. The whole process is a few minutes for typical models.

### Why AWQ often beats GPTQ in practice

- **Hardware-friendly**: per-channel scales are standard in every GEMM library. GPTQ's group-wise layout is less standard.
- **No per-layer calibration coupling**: AWQ's scaling is local to each layer; GPTQ's column updates chain.
- **Robust on outlier-heavy layers**: activations in LLMs have long tails in specific layers (first attention layer, FFN output) — AWQ's direct activation awareness handles these cleanly.
- **Simpler to implement correctly**: the AWQ paper's code is ~1000 lines; GPTQ's involves careful Hessian numerics.

GPTQ still wins on some specific layer shapes and on models with unusual weight distributions.

## Engineering Tradeoffs

| Decision | Gained | Gave up |
|---|---|---|
| Weight-only INT4 (either method) | ~4× model compression; fits on consumer GPUs | Activations still FP16 — compute savings limited to decode (memory-bound) |
| Group size 128 | Good accuracy / compression balance | Not optimal for every shape; 64 gives more accuracy, 256 more compression |
| GPTQ Hessian machinery | Theoretically grounded; good on some shapes | Complex; numerical care needed; memory overhead during quantization |
| AWQ per-channel scaling | Simple, fast, hardware-friendly | Harder to reason about formally; calibration sample selection matters |
| Post-training (both) | No retraining; quantize released checkpoints | Small accuracy loss; not as good as QAT for aggressive bit-widths |
| Per-group FP16 scales | Negligible overhead, good accuracy | Kernels must support the layout |

## Experiments & Results

- **GPTQ**: INT4 OPT-175B within 0.1 perplexity of FP16; 3.25× speedup over FP16 end-to-end; first paper to demonstrate PTQ INT4 at 100B+ scale.
- **AWQ**: INT4 Llama / Mistral / Qwen / Falcon within ~0.05–0.15 perplexity of FP16; runs faster than GPTQ at inference due to cleaner layout; particularly strong on instruction-tuned models where outliers are more pronounced.
- **Real-world impact**: llama.cpp's `Q4_K_M` and similar formats, vLLM's AWQ / GPTQ support, ExLlama, TGI quant kernels — essentially all local-LLM deployments use one of these two (or a variant like **GGUF**, which owes its design to both).

## Reproducibility Notes

- Both algorithms are fully open source, with reference implementations and well-tested wrappers.
- Calibration data matters: use ~128 in-domain samples; WikiText-2 is common but suboptimal for instruction-tuned models.
- **AutoAWQ** and **AutoGPTQ** provide Python wrappers; models on HuggingFace Hub are often pre-quantized.
- Inference kernels: AWQ kernels from MIT Han Lab, GPTQ via ExLlama / Marlin; Marlin in particular (hand-tuned INT4xFP16 GEMM) is a common shared backend.

## Commentary

The lasting lesson from GPTQ + AWQ is about **how much accuracy lives in a small number of channels**. The naive mental model of quantization ("every weight loses a bit of precision") is wrong: outlier channels do the work, and the entire accuracy question reduces to "did you preserve the outliers?" This framing is what makes AWQ simpler and often better than GPTQ — AWQ makes the outlier handling explicit; GPTQ implicitly captures it via the Hessian, which is more work for the same insight.

Looking at the 2026 landscape: **FP8/FP4 hardware support (Blackwell, MI300) is making weight-only quantization less important**, since the native format is already 8 or even 4 bits. But AWQ's **activation-aware scaling** survives — it's now used to decide where to keep higher precision in mixed-precision FP8 / FP4 training itself (V3's fine-grained scaling recipe is a close cousin). The algorithmic idea outlived the specific quantization target.

For practitioners: default to AWQ for new deployments; use GPTQ when a model has known issues with AWQ (rare). Don't invent your own quantization — the ecosystem is mature, the kernels are tuned, and re-implementing is not where you find performance in 2026.

## References

- [1] Frantar et al. _GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers._ ICLR '23 / arXiv:2210.17323.
- [2] Lin et al. _AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration._ MLSys '24 / arXiv:2306.00978.
- [3] Frantar & Alistarh. _Optimal Brain Compression (OBC/OBQ)._ NeurIPS '22. (GPTQ's basis)
- [4] Dettmers et al. _LLM.int8() and SmoothQuant_, contemporary W8 work.
- [5] Marlin INT4 kernels: https://github.com/IST-DASLab/marlin
