# DeepSeek-V3 Technical Report

- **Authors / Org**: DeepSeek-AI
- **Published**: 2024-12
- **Links**: [paper (arXiv:2412.19437)](https://arxiv.org/abs/2412.19437) · [code](https://github.com/deepseek-ai/DeepSeek-V3) · [model](https://huggingface.co/deepseek-ai/DeepSeek-V3)

## TL;DR

A 671B-parameter MoE model (37B activated per token) trained on 14.8T tokens using only **2.788M H800 GPU-hours** (~$5.58M at $2/hr). Matches or beats Llama 3.1 405B and approaches GPT-4o / Claude 3.5 Sonnet on many benchmarks. The headline contribution is not the model — it is the **infrastructure stack** that made training this cheap: FP8 mixed-precision at scale, DualPipe pipeline parallelism, hand-tuned all-to-all kernels, and an auxiliary-loss-free MoE load-balancing scheme.

## Context & Motivation

Pre-V3, frontier dense / MoE training runs at this scale were assumed to require ≥10k H100-class GPUs and tens of millions of dollars. Two structural bottlenecks dominated:

1. **Communication**: MoE all-to-all across cross-node expert parallelism saturates inter-node bandwidth, leaving GPUs idle.
2. **Precision**: BF16 activations and gradients inflate memory and HBM traffic; FP8 had been demonstrated only at small scale.

H800s (the export-restricted variant available in China) cap NVLink bandwidth at roughly half of H100, making the communication problem worse. V3's infra choices are direct responses to this constraint.

## Core Method

### Architecture
- **MoE**: 256 routed experts + 1 shared expert per MoE layer; top-8 routing; 61 layers total.
- **MLA (Multi-head Latent Attention)**: KV cache compressed into a low-rank latent; at inference, KV cache per token is ~1/10 of standard MHA. Detailed in the V2 paper.
- **Multi-token prediction (MTP)**: auxiliary objective predicting the next 2 tokens; also enables speculative decoding at inference.

### Training infrastructure
- **FP8 mixed precision**: most GEMMs in FP8 E4M3; fine-grained per-tile (1×128 activations) and per-block (128×128 weights) scaling to contain outliers; master weights and optimizer state in BF16/FP32. First large-scale public validation of FP8 end-to-end training.
- **DualPipe**: a bidirectional pipeline schedule that overlaps forward and backward phases of adjacent micro-batches, reducing pipeline bubbles vs. 1F1B / ZeroBubble. Combined with customized PTX-level all-to-all kernels, compute and cross-node communication are near-fully overlapped.
- **Expert parallelism over 2048 H800s** with all-to-all routing tuned to H800's NVLink+IB topology (limit dispatch to at most 4 nodes per token).
- **Auxiliary-loss-free load balancing**: per-expert bias term added to routing logits, updated online based on observed load. Avoids the accuracy penalty of the standard auxiliary load-balance loss.

## Engineering Tradeoffs

| Decision | Gained | Gave up |
|---|---|---|
| FP8 GEMMs with fine-grained scaling | ~2× throughput, ½ memory vs BF16 | Complexity in scaling/accumulation; numerical risk required mitigation (high-precision accumulate, E4M3 for both fwd/bwd activations) |
| MLA | 10× smaller KV cache → long context + cheap inference | Extra projection compute; architectural complexity vs vanilla MHA |
| Aux-loss-free balancing | No accuracy tax from balancing loss | Extra bookkeeping; sensitivity to bias update rate |
| DualPipe | Near-zero pipeline bubble | 2× activation memory vs 1F1B (accepted, since FP8 + MLA freed memory) |
| Cap tokens to ≤4 nodes | Bounded comm cost | Routing constraint reduces expert utilization flexibility |

## Experiments & Results

- **Cost**: 2.664M H800 GPU-hours for pre-training + 0.124M for context extension and post-training = 2.788M total. At $2/GPU-hour assumption → ~$5.58M.
- **Throughput**: sustained model FLOPs utilization competitive with dense BF16 runs on H100 at similar scale, despite slower H800 interconnect.
- **Quality**: top open-source model at release; competitive with closed frontier on MMLU-Pro, MATH, coding benchmarks; strong long-context (128k).

Caveat: $5.58M counts only the final run. R&D, failed runs, data curation, and salary are excluded. Still, the relative improvement over contemporary reports is large.

## Reproducibility Notes

- Open weights and inference code are released. Training code is **not** released.
- PTX-level all-to-all kernels were later open-sourced as **DeepEP** (2025 Open Source Week). FP8 GEMM code as **DeepGEMM**. MLA inference kernels as **FlashMLA**.
- Reproducing the training run requires: cluster of ~2k H800/H100, FP8-capable kernels, multi-week stable operation. Infeasible outside well-resourced labs, but the component recipes are now reusable.

## Commentary

The genuinely novel contributions are (a) FP8 at this scale actually working with a published recipe, (b) aux-loss-free balancing as a clean replacement for a long-standing hack, and (c) the DualPipe + custom-kernel co-design showing that H800-class clusters can train frontier models if you are willing to do systems work. The model itself is strong but not architecturally revolutionary — MLA and fine-grained MoE were established in V2. The real lesson for the field: frontier-scale training is a systems problem, and there is still substantial headroom below the compute-optimal frontier if you invest in infra.

## References

- [1] DeepSeek-AI. _DeepSeek-V3 Technical Report._ arXiv:2412.19437, 2024.
- [2] DeepSeek-AI. _DeepSeek-V2._ arXiv:2405.04434, 2024. (MLA, DeepSeekMoE)
- [3] DeepSeek Open Source Week repos: FlashMLA, DeepEP, DeepGEMM, 3FS, DualPipe (2025).
