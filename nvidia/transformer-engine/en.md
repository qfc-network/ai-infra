# NVIDIA TransformerEngine

- **Org**: NVIDIA
- **First release**: 2022-11 (open-sourced alongside H100 GA)
- **Links**: [GitHub](https://github.com/NVIDIA/TransformerEngine) · [Docs](https://docs.nvidia.com/deeplearning/transformer-engine/user-guide/)

## TL;DR

TransformerEngine (TE) is NVIDIA's open-source Python/C++ library that maps the FP8 Tensor Core instructions on Hopper (`wgmma`) into drop-in PyTorch and JAX modules. It sits between the hardware (`cuBLAS`/`cuDNN` FP8 GEMMs, Hopper `wgmma`) and the high-level framework layer (Megatron-Core, NeMo, FSDP). The key abstractions are fused modules — `te.Linear`, `te.LayerNormLinear`, `te.MultiheadAttention`, `te.TransformerLayer` — each of which performs FP8 cast, GEMM, and dequantization in a single kernel, along with a scaling recipe (`DelayedScaling`) that tracks per-tensor amplitude history to keep quantization safe without forcing synchronous scale updates on every step. End-to-end training throughput on H100 improves by 1.3–1.6× over BF16 baselines once attention, communications, and other non-GEMM costs are included.

## Context & Motivation

Hopper H100 ships with a new instruction class — `wgmma` (warpgroup matrix multiply-accumulate) — that supports FP8 (E4M3 and E5M2) inputs and accumulates in FP32. Peak FP8 GEMM throughput is 3958 TFLOPS (dense, sparsity off); BF16 GEMM is 1979 TFLOPS. The theoretical 2× ratio is the budget FP8 training is trying to capture.

The obstacle is numerical. FP8 E4M3 has a dynamic range of roughly [−448, 448] — roughly 11 bits narrower than BF16. Naively casting large weight or activation tensors to FP8 saturates or underflows immediately. Two things are required:

1. A per-tensor scale that maps the actual amplitude of each tensor into the FP8 representable range before the GEMM, and an inverse scale applied after.
2. A mechanism to maintain those scales with low overhead — computing the exact per-tensor maximum every step requires an extra reduction kernel and a device-to-host sync if you want the scale before the forward pass begins.

TransformerEngine solves both problems. It wraps every FP8-eligible operation in a fused cast+GEMM+dequant kernel, and it implements `DelayedScaling` to amortize the scale-update cost across multiple steps.

## Core Architecture

### Module Hierarchy

TransformerEngine exposes four main modules:

- `te.Linear` — replaces `nn.Linear`; input + weight cast to FP8, GEMM in FP8, output dequantized to BF16/FP32.
- `te.LayerNormLinear` — fuses LayerNorm into the cast path; saves one HBM round-trip for the normalized activations.
- `te.LayerNormMLP` — fuses LayerNorm + two Linears (up + down projection) with the GELU gate into a single launch.
- `te.MultiheadAttention` — wraps QKV projection, attention (optionally dispatching to FlashAttention-3 backend), and output projection; all linear layers use FP8.
- `te.TransformerLayer` — a full pre-norm transformer block (self-attention + MLP); Megatron-Core uses this directly as its layer implementation.

All modules share the same scaling infrastructure: each maintains `amax_history`, `scale`, and `scale_inv` buffers for inputs, weights, and gradient outputs.

### FP8 Formats: E4M3 and E5M2

TE uses two FP8 dtypes with different numeric properties:

| Format | Exponent bits | Mantissa bits | Max value | Dynamic range |
|--------|--------------|---------------|-----------|---------------|
| E4M3   | 4            | 3             | 448       | ~8 OOM        |
| E5M2   | 5            | 2             | 57344     | ~24 OOM       |

E4M3 has higher mantissa precision, making it suitable for forward pass tensors (weights, activations) where small numerical errors accumulate. E5M2 has a much larger dynamic range at the cost of mantissa resolution, making it appropriate for gradient tensors whose magnitudes can vary by orders of magnitude across training. TE's `DelayedScaling` recipe defaults to E4M3 for forward pass inputs and weights, and E5M2 for gradient outputs.

### The FP8 Cast + GEMM + Dequant Pattern

For a linear layer with weight matrix W and input X, the computation is:

```
# Scale selection (happens once per N steps, not per step)
scale_x   = fp8_max(E4M3) / amax_history_x.max()   = 448   / amax_x
scale_w   = fp8_max(E4M3) / amax_history_w.max()   = 448   / amax_w

# Forward (single fused kernel)
X_fp8     = cast_to_fp8(X * scale_x,  dtype=E4M3)
W_fp8     = cast_to_fp8(W * scale_w,  dtype=E4M3)
Y_fp32    = wgmma(X_fp8, W_fp8)                    # accumulates in FP32
Y_out     = Y_fp32 * (1 / scale_x) * (1 / scale_w) # dequant, back to BF16

# Backward (separate E5M2 recipe for dY)
scale_dY  = fp8_max(E5M2) / amax_history_dY.max()  = 57344 / amax_dY
dY_fp8    = cast_to_fp8(dY * scale_dY, dtype=E5M2)
dX        = wgmma(dY_fp8, W_fp8^T) * (1/scale_dY) * (1/scale_w)
dW        = wgmma(X_fp8^T, dY_fp8) * (1/scale_x)  * (1/scale_dY)
```

The key point is that the cast, GEMM, and dequant are not three separate kernels — they are a single fused kernel that receives the pre-computed scales as arguments. This avoids materializing the FP8 tensors as intermediate HBM buffers between the cast and the GEMM.

## DelayedScaling Recipe

`DelayedScaling` is the default and only production-ready scaling recipe in TE. It maintains a sliding window of per-tensor absolute-maximum values (amax history) and updates scales once every `amax_history_len` steps rather than every step.

```python
from transformer_engine.common.recipe import DelayedScaling, Format

recipe = DelayedScaling(
    margin=0,                    # log2 headroom below fp8_max; increase if overflows
    interval=1,                  # steps between scale updates (1 = every step)
    fp8_format=Format.HYBRID,    # E4M3 for fwd, E5M2 for bwd
    amax_history_len=16,         # sliding window of amax values
    amax_compute_algo="max",     # "max" or "most_recent"
)

with te.fp8_autocast(enabled=True, fp8_recipe=recipe):
    loss = model(batch)
```

`fp8_autocast` is a context manager that installs the FP8 recipe on all enclosed TE modules. Outside of this context the modules run in BF16, so the same model can be evaluated in full precision without code changes.

The `margin` parameter is a power-of-two headroom: the effective fp8_max used for scale computation is `fp8_max / 2^margin`. Setting `margin=1` halves the used range and reduces overflow risk at the cost of some representable precision. Production recipes typically leave `margin=0` and watch for loss spikes.

### Scale Update Mechanism

On each forward pass, TE computes the per-tensor amax from the actual tensor values (a reduction kernel on device) and stores it in the amax history buffer. The scale is updated using the sliding-window maximum:

```
scale = fp8_max / (amax_history[last N steps].max() * 2^margin)
```

Because the scale is derived from historical data, not the current step's tensor, it can be stale by up to `amax_history_len` steps. If an activation suddenly spikes — e.g., early in training or after a learning rate warmup jump — the stale scale may be too large, causing overflow. The `amax_history_len` parameter trades freshness for stability: shorter windows react faster but are noisier.

### Per-Tensor vs. Per-Channel Scaling

TE uses per-tensor scaling: one scale per tensor, regardless of shape. The alternative, per-channel (or per-row) scaling, computes one scale per output channel of a weight matrix, or one scale per token for activations. Per-channel scaling is strictly more accurate because it handles outlier channels without penalizing the whole tensor, but it requires significantly more metadata and hardware-level support for channel-wise dequantization — support that H100 `wgmma` does not provide natively for FP8. TE therefore accepts the accuracy gap in exchange for hardware compatibility and simplicity.

## Integration with Training Frameworks

### Megatron-Core

Megatron-Core's `TransformerLayer` can accept `te.TransformerLayer` as a drop-in replacement through its `transformer_layer_spec` argument. With TE enabled, every Linear in the transformer block is FP8. Activation checkpointing is handled transparently: TE modules implement custom `torch.autograd.Function` subclasses that save only the FP8-quantized activations (not the BF16 originals) during the forward pass, reducing checkpoint memory by up to 50%.

Pipeline parallelism (PP) boundaries in Megatron pass BF16 tensors between stages; TE does not attempt to keep inter-stage activations in FP8 because the receiver would need the sender's scaling metadata.

### FSDP

FSDP (FullyShardedDataParallel) requires `use_orig_params=True` when used with TE modules. This is the same requirement as `torch.compile`. Without it, FSDP reconstructs parameters as flat tensors during the all-gather, and TE's per-parameter FP8 scale/amax metadata (which lives as separate registered buffers on each `nn.Module`) becomes desynced from the gathered parameter tensor.

When `use_orig_params=True`, FSDP shards the FP8 scale and amax buffers alongside the weight parameters, keeping them aligned after all-gather. The scale buffers are small (one float per tensor) so the communication overhead is negligible.

### FlashAttention-3 Relationship

TE's `MultiheadAttention` module handles the QKV projection and output projection in FP8 (these are Linear layers). For the attention score computation itself — `softmax(QK^T / sqrt(d)) V` — TE dispatches to FlashAttention-3 on Hopper when available. FA3 natively uses `wgmma` with FP8 for the inner QK and PV matmuls and handles the attention IO-efficiency problem (tiled SRAM computation, no materialization of the full attention matrix). The two libraries are complementary: FA3 covers the attention score kernel; TE covers the linear projection layers. Using both together captures the maximum FP8 benefit.

## Performance Analysis

H100 SXM5 peak FLOPs for dense GEMMs:

| Dtype  | Peak TFLOPS | Notes                        |
|--------|-------------|------------------------------|
| FP32   | 67          | CUDA cores                   |
| TF32   | 989         | Tensor Cores                 |
| BF16   | 1979        | Tensor Cores (wgmma)         |
| FP8    | 3958        | Tensor Cores (wgmma, Hopper) |

The 2× ratio between BF16 and FP8 is a hardware ceiling, not an observed training speedup. End-to-end speedup is lower because:

1. Attention FLOPs do not double (FA3 FP8 is ~1.5–2× over FA3 FP16 for large sequences; attention is a fraction of total FLOPs).
2. Embedding layers, layer norms, and other non-GEMM ops are bandwidth-bound and unaffected.
3. Communication (AllReduce, AllGather, ReduceScatter) does not benefit.
4. Scale update and amax reduction kernels add overhead.

Observed end-to-end training speedup in production: **1.3–1.6× over BF16** for GPT-style models with large hidden dimensions. Models with longer attention fractions (small FFN ratio) see the lower end; models with large FFNs (e.g., MoE with high hidden dim) see the upper end.

## Engineering Tradeoffs

| Scaling Approach | Accuracy vs. BF16 | Implementation Complexity | Hardware Requirement | Training Stability |
|---|---|---|---|---|
| DelayedScaling (FP8 E4M3+E5M2) | ~0.1–0.3 ppl gap on LLMs; usually within noise | Moderate — requires amax history buffers, scale sync across FSDP shards | H100 / GH200 (Hopper wgmma required) | Good with tuned margin; risk of spike during rapid loss descent |
| Static per-tensor FP8 | Depends heavily on pre-calibrated scale accuracy | Low — no runtime state | H100 | Fragile — any distribution shift causes overflow/underflow |
| Per-channel (per-row) FP8 quantization | Better than per-tensor; similar to INT8 per-channel | High — dequant must be channel-wise, not supported natively by wgmma | Requires custom kernels or future hardware | Better stability for outlier-heavy models |
| BF16 (baseline) | Reference | Minimal — standard mixed precision | A100 and later | Very stable; well-understood loss landscape |
| INT8 (post-training, inference only) | Acceptable for inference, degrades training | Low (inference) / High (training) | Ampere and later | Not used for training |

## Failure Modes and Debugging

**Loss spike at training start**: amax history is cold (all zeros). The first scale update uses an incorrect reference value. Mitigation: pre-populate the amax history with a calibration forward pass before FP8 training begins, or set a large initial `margin` for the first few hundred steps.

**Overflow in backward pass**: E5M2 range (57344) can still be exceeded for gradient amplitudes in early training. Check with `NVTE_FP8_DFP_AMAX_REDUCE_DEBUG=1`. Increase `margin` or extend `amax_history_len`.

**Scale sync across data-parallel ranks**: each rank computes its own amax from its data shard. If different ranks see different input distributions, their scales diverge. TE provides `te.fp8_autocast` with `scale_factor_reduction` option to AllReduce amaxes across DP ranks before updating scales.

**FSDP + compile conflicts**: `torch.compile` with FSDP and TE requires `use_orig_params=True` and `dynamic=False`. The interaction between FSDP's deferred parameter gathering and TE's custom autograd functions is the main source of silent correctness bugs.

## Commentary

TransformerEngine is the most direct path from Hopper silicon to training throughput. The `wgmma` instruction with FP8 doubles the hardware throughput budget for GEMM; TE is what actually makes that budget accessible from Python without rewriting every training loop from scratch. The `DelayedScaling` recipe is not magic — it is a pragmatic approximation that works well when amplitude distributions are smooth, and breaks when they are not. Knowing when to widen the margin, when to extend the history window, and when the FP8 path is simply not appropriate (e.g., very early in training, or in very sensitive fine-tuning runs) is the practical engineering skill. As Blackwell introduces FP4 and wider wgmma variants, the same pattern will repeat: new low-precision formats, new hardware instructions, new fused kernels, new scaling recipes. TE's architecture is general enough to absorb those additions.

## Cross-References

- [`../hopper-h100/`](../hopper-h100/) — `wgmma` instruction, FP8 Tensor Core hardware, H100 memory hierarchy
- [`../../foundational/mixed-precision/`](../../foundational/mixed-precision/) — FP16/BF16/FP8 training progression; loss scaling history
- [`../../foundational/flash-attention/`](../../foundational/flash-attention/) — FA3 uses `wgmma` for attention; TE handles the linear layers around it
- [`../../foundational/zero-fsdp/`](../../foundational/zero-fsdp/) — FSDP integration; `use_orig_params=True` requirement
- [`../megatron-core/`](../megatron-core/) — Megatron-Core uses TE as the layer implementation via `transformer_layer_spec`

## References

- [1] NVIDIA TransformerEngine GitHub. https://github.com/NVIDIA/TransformerEngine
- [2] NVIDIA. _H100 Tensor Core GPU Architecture._ White paper, 2022.
- [3] Micikevicius et al. _FP8 Formats for Deep Learning._ arXiv:2209.05433, 2022.
- [4] Shah et al. _FlashAttention-3: Fast and Accurate Attention with Asynchrony and Low-precision._ arXiv:2407.08608, 2024.
- [5] Shoeybi et al. _Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism._ arXiv:1909.08053, 2019.
