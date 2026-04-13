# Megatron-LM: Tensor, Pipeline, and Sequence Parallelism

- **Authors / Org**: Shoeybi et al., NVIDIA
- **Published**: v1 — 2019-09 · v2 — 2021-04 · v3 (sequence parallelism + selective recompute) — 2022-05
- **Links**: [v1 (arXiv:1909.08053)](https://arxiv.org/abs/1909.08053) · [v2 (arXiv:2104.04473)](https://arxiv.org/abs/2104.04473) · [v3 (arXiv:2205.05198)](https://arxiv.org/abs/2205.05198) · [code](https://github.com/NVIDIA/Megatron-LM)

## TL;DR

The trilogy of NVIDIA papers that defined **how transformers are parallelized at scale**. v1 introduced **tensor parallelism (TP)** — shard individual matmuls across GPUs with a carefully chosen communication pattern (two all-reduces per layer). v2 added **pipeline parallelism (PP)** with interleaved 1F1B scheduling and showed how TP + PP + DP compose into "3D parallelism" at 1T+ parameter scale. v3 added **sequence parallelism (SP)** to shard activation memory along the sequence axis and **selective activation recomputation** to cut memory without full recompute cost. Every large-scale training framework (DeepSpeed, Colossal-AI, MindSpore, even V3 and Llama 3) inherits Megatron's parallelism primitives.

## Context & Motivation

In 2019, ZeRO and data parallelism handled optimizer / gradient sharding, but a single transformer layer already didn't fit on one GPU at GPT-2-XL scale. Pipeline parallelism (GPipe) existed but had large bubbles. What was missing: a way to split **one layer** across GPUs without killing throughput.

v2's context: people were trying to train 500B+ models. Pure TP doesn't scale beyond a node (NVLink), pure PP has too many bubbles, pure DP hits memory. Combining them was the only path; the question was how.

v3's context: even with 3D parallelism, activation memory blew up on long sequences. Full recomputation was standard but wasted ~30% of compute. Needed: finer control over what to recompute.

## Core Method

### Tensor Parallelism (v1)

For a transformer block:

- **MLP**: `Y = GeLU(XA) · B`. Split `A` column-wise across GPUs → each GPU produces part of `GeLU(XA)`; then split `B` row-wise → each GPU computes its partial product of `B`. A single **all-reduce** combines the outputs. No communication in forward before the GeLU.
- **Self-attention**: split heads across GPUs (column-parallel on Q/K/V projections, row-parallel on output projection). Again, one all-reduce per block per direction.

Total: 2 all-reduces forward + 2 backward per transformer block. Efficient only when intra-node bandwidth (NVLink) is available — TP size is typically ≤ node size (4 or 8).

### Pipeline Parallelism (v2)

Layers sharded across PP stages. The bubble comes from pipeline fill/drain. Two ideas:

1. **1F1B schedule**: once a stage has processed a forward for a micro-batch, it immediately processes the backward for an earlier one. Keeps pipeline active with fewer in-flight activations.
2. **Interleaved 1F1B**: assign non-contiguous layer groups to each stage (e.g., stage 0 owns layers 0–3 and 16–19). More micro-batches in flight, smaller bubbles, more memory.

Bubble fraction roughly `(PP − 1) / (m + PP − 1)` where `m` is micro-batch count — so drive `m` up.

### 3D Parallelism

`total_gpus = TP × PP × DP`. Rule of thumb:
- TP inside a node (NVLink-rich), typically 8.
- PP across nodes, sized by layer count / memory.
- DP (+ ZeRO) on the outermost axis.

V3 scaled this to ~2k H800; Llama 3 to 16k H100 (with added CP).

### Sequence Parallelism (v3)

In a TP=8 setup, dropout / LayerNorm / residual are replicated across all TP ranks — they don't benefit from TP but still consume full activation memory per rank. SP shards **these non-matmul activations along the sequence dimension**. Extra all-gather / reduce-scatter required, but overall communication volume is similar to pure TP (one ring is replaced by two halves), and activation memory drops ~1/TP.

### Selective Activation Recomputation (v3)

Full activation checkpointing recomputes everything in backward → ~30% compute overhead. v3 profiles activation memory and finds: a small set of layers (especially attention matrices) dominate memory but are cheap to recompute. Recompute **only those**; keep expensive-to-recompute activations cached. Result: near-zero memory penalty, <5% compute overhead.

## Engineering Tradeoffs

| Decision | Gained | Gave up |
|---|---|---|
| Tensor parallelism | Single-layer fits across multiple GPUs | Communication-bound beyond a node; TP size capped by NVLink topology |
| Pipeline parallelism | Scales to any layer count | Pipeline bubbles; needs many micro-batches; cross-stage activation transfer |
| Interleaved 1F1B | Smaller bubbles | More concurrent micro-batches → more memory |
| Sequence parallelism | Frees ~1/TP of activation memory | Extra communication ops (offset by reduced ring length) |
| Selective recompute | ~5× less overhead than full recompute | Requires workload-specific profiling to choose which activations |
| 3D parallelism in general | Trains trillion-param models | Launch/config complexity; framework lock-in; hard to debug |

## Experiments & Results

- v1: 8.3B GPT-2, 15.1 PetaFLOPs sustained on 512 V100s, 76% scaling efficiency over DP.
- v2: 1T model on 3072 A100s at 52% of peak FLOPs; detailed scaling laws across TP/PP/DP choices.
- v3: sequence parallelism + selective recompute reduce activation memory by 5× on 22B and 175B configs; throughput improves ~29% vs full recompute.
- Megatron is the reference point — most subsequent framework papers compare against it.

## Reproducibility Notes

- Fully open source; NVIDIA maintains Megatron-Core as the long-term library.
- Most production training stacks either use Megatron directly or re-implement its primitives (DeepSpeed, Colossal-AI, FSDP-based Meta stack).
- Configuration (TP, PP, DP, micro-batch count, sequence length) is workload-specific and requires tuning; NVIDIA publishes recipes for common scales.

## Commentary

Megatron's three papers are the most important single body of work on distributed transformer training. The genuinely novel part is **how simple each step looks in retrospect**: TP is linear algebra + all-reduces; PP 1F1B is "schedule backward as soon as you can"; SP is "shard the axis that isn't already sharded." The engineering art is in (a) getting communication patterns to line up with network topology, and (b) realizing which activations matter. Everything built since — FSDP, ZeRO-3, ring attention, context parallelism — is a response to Megatron's limits, not a replacement for its abstractions. If you're designing a training system in 2026, you still start from the TP/PP/DP decomposition and justify deviations from there. Read v2 first (most material per page), then v3 for the modern tricks; v1 is mostly of historical interest now.

## References

- [1] Shoeybi et al. _Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism._ arXiv:1909.08053, 2019.
- [2] Narayanan et al. _Efficient Large-Scale Language Model Training on GPU Clusters Using Megatron-LM._ SC '21 / arXiv:2104.04473.
- [3] Korthikanti et al. _Reducing Activation Recomputation in Large Transformer Models._ arXiv:2205.05198, 2022.
- [4] Huang et al. _GPipe._ NeurIPS 2019.
- [5] Rajbhandari et al. _ZeRO._ SC '20.
