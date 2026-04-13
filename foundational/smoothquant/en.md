# SmoothQuant — Activation-Weight Outlier Migration for W8A8

- **Authors / Org**: Xiao et al., MIT + NVIDIA
- **Published**: ICML '23
- **Links**: [paper (arXiv:2211.10438)](https://arxiv.org/abs/2211.10438) · [code](https://github.com/mit-han-lab/smoothquant)

## TL;DR

Unlike weight-only methods (GPTQ, AWQ), SmoothQuant quantizes **both activations and weights to INT8** — enabling ~2× throughput on INT8 tensor cores, not just memory savings. The core obstacle: activations have extreme outliers in specific channels, and INT8 quantization on outliers destroys accuracy. SmoothQuant's insight: the outliers live in a **small, predictable set of activation channels**, and you can **mathematically migrate them from activations to weights** by applying a per-channel scale that makes activations easier to quantize while weights absorb the difficulty. Since weights have no outliers to begin with, they tolerate the extra dynamic range. Result: lossless W8A8 (INT8 weights, INT8 activations) on OPT-175B and Llama, with actual compute speedup (not just memory).

## Context & Motivation

Circa 2022, standard practice for INT8 LLM inference:

- **Weight-only INT8** (easy): weights have well-behaved distributions; round-to-nearest loses <0.1 perplexity. But activations stay FP16, so compute uses FP16 tensor cores.
- **W8A8** (hard): INT8 × INT8 tensor cores are 2× faster than FP16 on Ampere, 4× on Hopper. Big serving win *if* you can quantize activations without accuracy collapse.

LLM.int8() (Dettmers et al., 2022) showed activations have **persistent outlier channels** — specific channels where values are 10–100× larger than the rest. If you INT8-quantize the whole tensor per-tensor, outliers dominate the scale and non-outlier values lose all resolution. LLM.int8()'s workaround: **mixed-precision** — detect outliers at runtime, keep them in FP16, quantize the rest. Works but slow (runtime branching) and unwieldy.

SmoothQuant's goal: get to W8A8 with **pure INT8 math**, no mixed precision, no branching.

## Core Method

### The observation

Outliers in LLM activations are:

- **Channel-persistent** — specific activation channels have outliers across almost all tokens, not just rare tokens. This means the outlier structure is a property of the layer, not the input.
- **Layer-specific** — some layers (first few, FFN outputs) have strong outliers; others don't.
- **Weights are NOT outlier-heavy** — weight distributions are well-behaved. Weights can absorb extra dynamic range.

This creates an asymmetry: activations are hard to quantize, weights are easy. SmoothQuant asks: can we transfer quantization difficulty from activations to weights?

### The scaling transformation

For a linear layer `y = x · W`, introduce a per-channel scale `s`:

```
y = x · W = (x / s) · (s · W)    where s is per-(input-channel)
```

The equality holds for any positive `s`. Choose `s` so that:

- `x / s` has its outlier channels **damped** — easier for INT8 quantization.
- `s · W` has its corresponding columns **amplified** — weights pick up the slack.

Since activations have outliers and weights don't, moving the difficulty to weights is a win: weights go from "extremely easy" to "somewhat harder" to quantize, activations go from "impossible" to "feasible." Both end up at INT8.

### Choosing `s`

The paper uses a simple formula:

```
s_j = max(|x_j|)^α / max(|W_j|)^(1-α)
```

Per input-channel `j`, balance between how outlier-heavy the activation is and how much range the weight column can absorb. `α ∈ [0, 1]` is a migration strength hyperparameter; `α = 0.5` is the default, `α = 0` is no migration (standard quantization), `α = 1` pushes all difficulty to weights.

`α` per-model (or per-layer) is chosen on a calibration set (~128 samples). The scale `s` is then **fused into the preceding layer's weights** (absorbed into the FFN or LayerNorm that produced `x`). **Zero inference-time overhead** — after fusion, the model is just INT8 tensors and INT8 matmuls.

### Fusion details

The migration scale `s` needs to live somewhere. Two common fusion targets:

1. **LayerNorm**: LayerNorm has a per-channel scale (γ) anyway; multiply `γ` by `1/s`, and the output `x/s` comes out natively.
2. **Previous linear layer's weights**: multiply the previous layer's output-channel weights by `1/s`.

Either way, the transformed model produces `x/s` as the input to the quantized linear, so no runtime division is needed.

## Engineering Tradeoffs

| Decision | Gained | Gave up |
|---|---|---|
| W8A8 instead of weight-only | Full INT8 tensor core speedup (2–4× vs FP16) | Requires solving the activation outlier problem |
| Outlier migration via scaling | Pure INT8 inference, no mixed precision | Finding the right `α` per model; weights lose a bit of headroom |
| Scale fused into prior layer | Zero inference-time overhead | Requires controlling the upstream op; not always possible |
| Per-channel migration | Handles outliers at their natural granularity | More scales to track than per-tensor |
| Static (calibration-time) | Fast; deployable | Not adaptive to shifting activation distributions |

## Experiments & Results

- **OPT-175B**: W8A8 SmoothQuant matches FP16 within ~0.1 perplexity; 1.56× end-to-end latency improvement over FP16.
- **Llama-1/2**: near-identical accuracy; significant throughput gains.
- Compared to LLM.int8(): same or better accuracy with **no mixed-precision branching**; 2–3× faster in practice.
- Compared to naive W8A8 (no migration): accuracy collapses (perplexity increases by 10+) — confirming the outlier problem is the actual blocker.

## Reproducibility Notes

- Reference implementation open-sourced; integrates with PyTorch via hook-based model transformation.
- Calibration data: ~128 Pile samples work for base models; in-distribution data for fine-tuned models.
- Works out-of-the-box for OPT, Llama, GPT-2 family; transformers with unusual norm layouts may need minor adjustments.
- Now integrated into **NVIDIA TensorRT-LLM** and other production inference stacks as a standard INT8 preprocessing option.

## Commentary

SmoothQuant sits between the weight-only methods (GPTQ, AWQ) and the full-quant approaches (LLM.int8()'s mixed precision) and offers the cleanest engineering trade: **push the problem where it's solvable (to weights), accept a tiny accuracy cost, get actual compute speedup**.

The broader lesson is about **asymmetric quantization difficulty**: not every tensor is equally hard to quantize, and a clever re-parameterization can move difficulty between tensors without changing the result. The same idea shows up in:

- **AWQ** (protecting salient weight channels by scaling activations) — close mathematical cousin.
- **FP8 training** (V3's fine-grained scaling recipe: per-tile activation scales, per-block weight scales) — this is outlier migration at training time.
- **Log-linear quantization** and **rotation-based quantization** (QuaRot, SpinQuant) — later work that generalizes the "transform tensor to reduce outliers" frame.

By 2026, **SmoothQuant is less dominant** because hardware FP8 support (Hopper, Blackwell) made W8A8-level speedups available natively with even smaller accuracy cost. But the **activation-outlier migration pattern** is now a standard tool in the quantization toolbox, and the technique still wins in Ampere-era deployments without FP8 hardware.

For practitioners: if you're on H100/H200 with FP8 support, use FP8 natively; if you're on A100 or older, or need to push to INT8 for memory bandwidth reasons, SmoothQuant is the right starting point. For INT4 weight compression with FP16 activations, use AWQ. The three methods (GPTQ, AWQ, SmoothQuant) together define the quantization default toolbox for transformer inference.

## References

- [1] Xiao et al. _SmoothQuant: Accurate and Efficient Post-Training Quantization for Large Language Models._ ICML '23 / arXiv:2211.10438.
- [2] Dettmers et al. _LLM.int8(): 8-bit Matrix Multiplication for Transformers at Scale._ NeurIPS '22. (Outlier observation; mixed-precision baseline)
- [3] Ashkboos et al. _QuaRot: Outlier-Free 4-Bit Inference in Rotated LLMs._ NeurIPS '24.
- [4] Liu et al. _SpinQuant._ 2024. (Rotation-based outlier removal)
- [5] NVIDIA TensorRT-LLM SmoothQuant integration docs.
