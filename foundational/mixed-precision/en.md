# Mixed Precision Training

- **Authors / Org**: Paulius Micikevicius et al. (NVIDIA / Baidu) — original AMP; NVIDIA Research — FP8 on Hopper/Blackwell
- **Published**: 2018-02 (AMP paper, arXiv:1710.03740); extended through FP8 (2022–2024)
- **Links**: [AMP paper (arXiv:1710.03740)](https://arxiv.org/abs/1710.03740) · [NVIDIA H100 Whitepaper](https://resources.nvidia.com/en-us-tensor-core/gtc22-whitepaper-hopper) · [TransformerEngine docs](https://docs.nvidia.com/deeplearning/transformer-engine/user-guide/index.html)

## TL;DR

Mixed precision training is not a single paper — it is a decade-long progression across three hardware generations. The core idea: run forward and backward passes in low-precision numeric formats (FP16, then BF16, now FP8) to exploit narrow-format tensor cores and halve activation memory, while maintaining a full-precision (FP32) master copy of weights and optimizer state to keep gradient accumulation numerically stable.

- **FP16 (2018, V100)**: 2× memory over FP32; 2–8× tensor-core throughput. Problem: a narrow exponent range (max ≈ 65,504) causes gradient overflow. Requires loss scaling to survive.
- **BF16 (2019, A100+)**: same 8-bit exponent as FP32 — no overflow risk — with only 3 mantissa bits. Became the universal default. PyTorch `torch.autocast` defaults to BF16 on Ampere and later.
- **FP8 (2022–2024, H100/Blackwell)**: two sub-formats — E4M3 for weights and activations (higher precision), E5M2 for gradients (wider dynamic range). H100 delivers 3,958 TFLOPS peak in FP8 vs 1,979 TFLOPS in BF16. The obstacle is activation outliers, which cause large quantization errors if not handled with per-tensor or per-block scaling.

## Context & Motivation

Training a large model in FP32 is expensive in both memory and time. For a model with N parameters:

- **Weights**: 4N bytes (FP32)
- **Gradients**: 4N bytes (FP32)
- **Adam optimizer states** (m and v): 8N bytes (FP32)
- **Total**: 16N bytes minimum, not counting activations

A 7B-parameter model requires 112 GB of GPU memory before a single activation tensor appears. At that size, FP32 training demands eight A100-80GB GPUs just for the model state. Activation checkpointing reduces activation memory but not parameter memory.

The goal of mixed precision is to keep the arithmetic in a shorter format — fewer bits per value — while preserving numerical stability where it matters most.

**Why FP16 wasn't the final answer.** FP16's maximum representable value is 65,504. Gradients, especially in the early phases of training or in poorly conditioned layers, routinely exceed this. An overflowed gradient becomes infinity or NaN, corrupting the entire update. Loss scaling patches this by multiplying the loss by a large constant S before the backward pass and dividing gradients by S after, keeping values inside the FP16 range. Dynamic loss scaling — starting high, halving on detected overflow, slowly increasing otherwise — works but adds complexity and an extra CPU–GPU synchronization per step to check for NaNs.

**Why BF16 fixed the overflow problem.** BF16 uses the same exponent field as FP32 (8 bits), giving the same ±3.4 × 10³⁸ dynamic range. It sacrifices mantissa precision (3 bits vs FP32's 23 bits and FP16's 10 bits), but gradient accumulation tolerates this: you care that gradients are in the right ballpark, not that they are exact to many significant figures. BF16 eliminated the need for loss scaling on modern hardware and is now the unconditional default for every serious training run.

**Why FP8 is next.** Even with BF16, the activation tensors in a large transformer — the key/query/value projections, attention maps, FFN intermediates — consume many gigabytes per forward pass. Reducing these from 2 bytes to 1 byte per element directly cuts activation memory by half, which either shrinks GPU memory usage or allows a doubling of batch size. The arithmetic throughput benefit is equally large: tensor cores on H100 do twice as many FP8 operations per second as BF16 operations, so models that are compute-bound (large batch, long context) see near-2× speedups.

## Core Method

### Automatic Mixed Precision (AMP)

The AMP protocol introduced by Micikevicius et al. remains essentially unchanged today:

1. **Maintain an FP32 master copy** of all trainable parameters θ_fp32.
2. **Cast to low precision** before each forward pass: θ_low = cast(θ_fp32). This cast is cheap — one memory read, one write per element.
3. **Run forward and backward passes** entirely in the low-precision format. Intermediate activations, gradient tensors, and the output loss are all FP16 or BF16.
4. **Cast gradients back to FP32** and accumulate into the master copy: ∂L/∂θ_fp32 = cast_up(∂L/∂θ_low).
5. **Update the master copy** with the FP32 optimizer (Adam, AdamW, etc.).

With FP16, step 3 requires **loss scaling**: multiply the scalar loss by a scale factor S before calling `backward()`, then divide all gradients by S before the optimizer step. This keeps gradient values in the representable FP16 range. Dynamic loss scaling (PyTorch `GradScaler`) starts S high (e.g., 2¹⁶), detects inf/NaN in gradients each step, and adjusts S accordingly.

**Memory accounting.** A common misconception is that AMP halves total memory. The actual breakdown for a model with N parameters:

- FP32 baseline: 4N (weights) + 4N (gradients) + 8N (Adam m+v) = **16N bytes**
- BF16 AMP: 2N (BF16 weights) + 2N (BF16 gradients) + 4N (FP32 master) + 8N (FP32 Adam) = **16N bytes**

The parameter-related memory is identical. The real savings come from **activations**, which in a standard transformer are proportional to batch size × sequence length × hidden dimension. Storing these in BF16 rather than FP32 cuts activation memory by half — and activations dominate at large batch sizes.

### FP8 Training (TransformerEngine / DeepSeek V3)

FP8 introduces a new complication: with only 4 or 5 exponent bits and 3 or 2 mantissa bits, the representable range is very narrow. A single large-magnitude activation value can fill the entire range, leaving all smaller values clustered at the low end with severe quantization error.

The solution is **per-tensor scaling**: before casting a tensor to FP8, compute a scale factor

$$s = \frac{\max(|x|)}{x_{\max}^{\text{FP8}}}$$

where x_max^FP8 is the largest representable FP8 value for the chosen sub-format. Divide all elements by s before casting, then multiply by s after the FP8 matmul to recover the original scale. This keeps values spread across the full FP8 range.

**Delayed scaling** avoids the overhead of computing max(|x|) synchronously each step: cache the scale factor from the previous step and use it for the current step. This breaks the dependency chain that would otherwise require a GPU→CPU round-trip per tensor per step.

**Sub-format selection:**
- **E4M3** (4 exponent bits, 3 mantissa bits): used for forward-pass weights and activations. Max value: 448. Higher precision for the dominant computation path.
- **E5M2** (5 exponent bits, 2 mantissa bits): used for backward-pass gradients. Max value: 57,344. Wider dynamic range to handle the larger magnitude spread in gradient tensors.

**High-precision accumulation:** FP8 matmul units on H100 accumulate partial sums in FP32 before writing results back. This is mandatory — accumulating in FP8 would saturate immediately. The TransformerEngine API exposes this as the default behavior.

**DeepSeek V3's fine-grained tile scaling:** instead of one scale factor per tensor (which is dominated by the maximum element), DeepSeek V3 uses per-block scaling with 1×128 tiles for activations and 128×128 tiles for weights. This requires tracking more scale factors but dramatically reduces the impact of outlier activations on non-outlier values — effectively the same philosophy as SmoothQuant applied to training.

**Memory formula with FP8:**
- FP8 AMP (weights + activations in FP8, FP32 master + Adam): 1N (FP8 weights) + 1N (FP8 gradients) + 4N (FP32 master) + 8N (FP32 Adam) = **14N bytes** for parameters, plus roughly 2× reduction in activation memory vs BF16

The weight savings are modest; the activation memory and compute throughput gains are the primary benefit.

## Engineering Tradeoffs

| Decision | Gained | Gave up |
|---|---|---|
| FP16 vs FP32 training | 2× memory reduction on parameters (where used); 2–8× throughput on tensor cores | Overflow risk on gradients; requires loss scaling; dynamic loss scaling adds CPU–GPU sync overhead per step |
| BF16 vs FP16 | No gradient overflow (same exponent range as FP32); simpler training loop, no loss scaling needed | Slightly lower mantissa precision (3 bits vs 10); requires Ampere or newer hardware; slightly worse on some tasks sensitive to numeric precision |
| FP8 vs BF16 | ~2× compute throughput on H100; ~50% reduction in activation memory; larger effective batch size | Per-tensor scaling overhead; sensitivity to activation outliers; debugging NaN/inf is harder; requires Hopper or Blackwell hardware; kernel-level complexity in TransformerEngine |
| FP32 master weight copy (AMP) | Numerically stable gradient accumulation; full-precision weight updates; clean convergence on par with FP32 baseline | Doubles per-parameter memory cost vs keeping only low-precision weights; FP32 Adam states are the dominant memory consumer |
| Per-block FP8 scaling (DeepSeek V3 style) | Better outlier handling than per-tensor; improved model quality on outlier-heavy layers | More scale factors to store and synchronize; increased kernel complexity; custom CUDA/Triton kernels required; harder to reproduce with stock TransformerEngine |

## Experiments & Results

**AMP (2018):** Micikevicius et al. trained ResNet-50, VGG, and LSTM language models to match FP32 quality with no accuracy degradation. Reported 2–3× end-to-end training speedup on V100 GPUs. Loss scaling was required for FP16; the paper introduced dynamic loss scaling as the practical solution.

**BF16 (2019–present):** Adopted as the default at Google (TPU), then NVIDIA (A100). No accuracy regression observed across a wide range of transformer models. PyTorch `torch.autocast(device_type='cuda', dtype=torch.bfloat16)` is the recommended path. No GradScaler needed.

**FP8 on H100:** NVIDIA reports theoretical peak of 3,958 TFLOPS at FP8 vs 1,979 TFLOPS at BF16 (dense, non-sparse) — a clean 2× ratio. Real-world speedup is model-architecture-dependent; compute-bound workloads (large batch, long sequence) see the most benefit; memory-bandwidth-bound workloads (small batch decode) see less.

**DeepSeek V3 (2024):** Uses FP8 for the forward pass matmuls with BF16 gradients and BF16 optimizer states. Reports stable training of a 671B MoE model with no quality regression vs a BF16 baseline. Per-block scaling (1×128 for activations, 128×128 for weights) was essential for stability; naive per-tensor FP8 degraded quality noticeably on certain layers. Training throughput: approximately 180K tokens/second on a cluster of H800 GPUs, substantially above what would be achievable in BF16 at the same hardware budget.

**FlashAttention-3 (2024):** Leverages H100's asynchronous execution pipeline and FP8 support for the attention kernel specifically, achieving up to 1.5× speedup over FlashAttention-2 on long-context workloads.

## Reproducibility Notes

**BF16 AMP (PyTorch):**
```python
with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
    loss = model(input)
loss.backward()
optimizer.step()
```
Requires PyTorch >= 1.10 and Ampere or newer GPU. No GradScaler needed.

**FP16 AMP with GradScaler (PyTorch):**
```python
scaler = torch.cuda.amp.GradScaler()
with torch.autocast(device_type='cuda', dtype=torch.float16):
    loss = model(input)
scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
```

**FP8 via NVIDIA TransformerEngine:**
- Install `transformer-engine` package; replace `torch.nn.Linear` with `transformer_engine.pytorch.Linear`
- Set `fp8_autocast` context manager; TransformerEngine handles scaling, casting, and high-precision accumulation automatically
- Requires Hopper (H100) or Blackwell hardware

**DeepSeek V3 FP8 recipe:** described in the technical report (arXiv:2412.19437). Requires custom CUDA kernels for per-block scaling; reference implementation not yet fully open-sourced at time of writing.

**Key hardware requirements:**
- FP16/BF16 tensor cores: Volta (V100) and above for FP16; Ampere (A100) and above for BF16
- FP8 tensor cores: Hopper (H100, H200) and Blackwell (B100, B200) only

## Commentary

BF16 is table stakes. No serious training run uses FP32 today — the hardware efficiency gap is too large, and BF16 has no meaningful accuracy cost. The only scenario where FP32 remains relevant is small-scale debugging or gradient-sensitive applications where the extra precision matters (which is rare in large transformer training).

FP8 is the current frontier, and the engineering challenges are real. The activation outlier problem — the same phenomenon that motivates SmoothQuant for inference — appears acutely during training. A single outlier-heavy layer trained with per-tensor FP8 scaling can destabilize the entire run. DeepSeek V3's per-block scaling addresses this at the cost of kernel complexity, and their public results demonstrate that FP8 training at scale is achievable with the right implementation.

The broader lesson is that precision reduction is always a fight against outliers. Whether you are quantizing weights post-training (AWQ, GPTQ), serving in INT8 (SmoothQuant), or training in FP8 (TransformerEngine), the pattern is the same: identify the high-magnitude values that dominate the representable range, and migrate the quantization burden away from them. BF16 avoided this problem entirely by preserving the exponent width; FP8 reintroduces it and requires explicit solutions.

Looking forward: Blackwell's FP4 support (announced for GB200) would push the scaling curve further. At FP4, activation outliers and per-block scaling are even more critical — this is an area of active research heading into 2025.

## References

- [1] Micikevicius, P. et al. (2018). "Mixed Precision Training." *ICLR 2018*. arXiv:1710.03740.
- [2] NVIDIA (2022). "NVIDIA H100 Tensor Core GPU Architecture Whitepaper." NVIDIA Technical Report.
- [3] NVIDIA (2023). "Transformer Engine: FP8 Training for Large Language Models." NVIDIA Developer Documentation.
- [4] DeepSeek-AI (2024). "DeepSeek-V3 Technical Report." arXiv:2412.19437.
- [5] Shah, J. et al. (2024). "FlashAttention-3: Fast and Accurate Attention with Asynchrony and Low-precision." arXiv:2407.08608.
- [6] Xiao, G. et al. (2023). "SmoothQuant: Accurate and Efficient Post-Training Quantization for Large Language Models." *ICML 2023*. arXiv:2211.10438.
