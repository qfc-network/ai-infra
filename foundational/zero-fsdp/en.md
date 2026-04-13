# ZeRO and FSDP

- **Authors / Org**: ZeRO — Rajbhandari et al., Microsoft (SC '20). FSDP — Meta (PyTorch).
- **Published**: ZeRO — 2019-10 / 2020-08 · FSDP paper — 2023-04
- **Links**: [ZeRO (arXiv:1910.02054)](https://arxiv.org/abs/1910.02054) · [ZeRO-Offload](https://arxiv.org/abs/2101.06840) · [ZeRO-Infinity](https://arxiv.org/abs/2104.07857) · [FSDP (arXiv:2304.11277)](https://arxiv.org/abs/2304.11277) · [DeepSpeed](https://github.com/microsoft/DeepSpeed) · [PyTorch FSDP](https://pytorch.org/docs/stable/fsdp.html)

## TL;DR

Data parallelism replicates the entire model on every GPU — fine for small models, disastrous at scale. **ZeRO** (Microsoft) removes that replication by partitioning **optimizer state (ZeRO-1)**, **gradients (ZeRO-2)**, and **parameters (ZeRO-3)** across data-parallel ranks, gathering on demand during forward / backward. **FSDP** (Meta) is the PyTorch-native re-implementation of ZeRO-3 with better overlap, flattening, and tooling — now the default distributed training primitive in the PyTorch ecosystem. The pair defines the "memory axis" of modern distributed training, and composes orthogonally with Megatron's TP/PP.

## Context & Motivation

Classic DDP: every rank holds a full copy of model parameters, gradients, optimizer states. For an Adam-trained 1B model in mixed precision, that's roughly 16 GB per rank (params 2B + grads 2B + optimizer 12B). At 100B params, 1.6 TB per rank — impossible.

Megatron's TP/PP shards **within a model instance** (intra-model parallelism). ZeRO/FSDP shards **across data-parallel copies** (no replication needed). These are orthogonal. The combined axis decomposition is the basis for all modern large-scale training.

## Core Method

### ZeRO-1: partition optimizer state

Each rank stores only `1/N` of optimizer state (Adam `m`, `v`, FP32 master weights). At step time, each rank updates its slice, then all-gather to get full updated params. Gradients and params still fully replicated.

Memory savings: typically 4× for Adam (optimizer is the fat part).

### ZeRO-2: also partition gradients

Each rank receives only the gradients for its optimizer-state slice via **reduce-scatter** (replaces the DDP all-reduce at the same communication volume). Further ~2× savings.

### ZeRO-3: also partition parameters

Each rank owns only `1/N` of parameters. Forward / backward require an **all-gather** before each layer (to materialize the full parameter tensor), compute, then free. Backward is analogous: all-gather params, compute grad, reduce-scatter grad, free.

Communication rises — an additional all-gather per forward and per backward vs DDP — but memory becomes proportional to `1/N`, enabling training with params that don't fit on any single GPU.

### ZeRO-Offload and ZeRO-Infinity

- **Offload**: move optimizer state + master weights to CPU RAM; CPU does the Adam step. Trades PCIe bandwidth for memory — viable at small-to-mid scale.
- **Infinity**: extends offload to NVMe; supports models much larger than aggregate GPU memory, at the cost of significant bandwidth pressure on storage.

### FSDP: the PyTorch-native ZeRO-3

Conceptually identical to ZeRO-3, but with key engineering differences:

- **FlatParameter**: per-unit parameters flattened into one 1-D tensor per shard, so all-gather is one op rather than many.
- **Unit granularity**: the user wraps the model in "FSDP units" (commonly per transformer block). Each unit is gathered independently, allowing **overlap between the next unit's all-gather and the current unit's compute**.
- **Mixed precision, CPU offload, activation checkpointing** integrated natively.
- **Full shard vs hybrid shard**: hybrid shards within a node and replicates across, trading memory for fewer cross-node gathers.

FSDP-2 (2024) moves further: per-parameter sharding (no flattening), better composability with TP/PP, cleaner state-dict semantics.

## Engineering Tradeoffs

| Decision | Gained | Gave up |
|---|---|---|
| ZeRO-1 → 2 → 3 (incremental) | Monotonic memory savings, adopt as needed | Each stage adds communication |
| ZeRO-3 / FSDP full shard | Memory scales as `1/N`, trains huge dense models on pure DP | Extra all-gather per layer forward and backward |
| FSDP FlatParameter | One comm op per unit, good for overlap | Complicates partial loading, state-dict handling, mixed dtypes |
| Unit granularity | Tunable memory / comm tradeoff | Requires wrapping choices; wrong choice silently hurts throughput |
| Hybrid shard | Fewer cross-node all-gathers | More per-node memory; replication factor cost |
| Offload to CPU / NVMe | Train beyond aggregate GPU memory | PCIe / storage bandwidth becomes the new bottleneck |

## Experiments & Results

- **ZeRO paper**: 170B model training on 400 V100s, super-linear speedup in some configs. First demonstrated 100B+ dense training without model parallelism.
- **FSDP paper**: near-linear scaling to 512 A100s on Llama-style 175B models; >55% MFU with activation checkpointing.
- Adoption: FSDP is the default in torchtitan, llama-recipes, and most PyTorch-native training stacks. DeepSpeed ZeRO remains popular for offload-heavy, HuggingFace-centric pipelines.

## Reproducibility Notes

- Both are fully open source and well-documented.
- Correctness is usually trivial; the challenge is **performance tuning** — unit size, overlap, precision, and interaction with TP/PP all matter.
- FSDP-2 is the recommended PyTorch path going forward; FSDP-1 still widely in production.

## Commentary

ZeRO and FSDP are the "quiet backbone" of large-scale training. Megatron gets credit for enabling trillion-param models, but without ZeRO-style sharding, even 7B dense models wouldn't fit in DDP on commodity hardware. The design is conceptually simple — **don't replicate what you can shard and gather on demand** — and the engineering is all about overlap and sizing. The most underappreciated part: ZeRO-3 / FSDP is **orthogonal to Megatron**, and the combined `TP × PP × FSDP` cube is what makes modern frontier training composable. Looking forward: FSDP-2's per-parameter sharding removes the last few rough edges, and FSDP + TP overlap (activation sharding, etc.) is an active research area. If you're training anything above ~1B params in PyTorch and not using FSDP, you are probably leaving memory or throughput on the table.

## References

- [1] Rajbhandari et al. _ZeRO: Memory Optimizations Toward Training Trillion Parameter Models._ SC '20 / arXiv:1910.02054.
- [2] Ren et al. _ZeRO-Offload: Democratizing Billion-Scale Model Training._ USENIX ATC '21 / arXiv:2101.06840.
- [3] Rajbhandari et al. _ZeRO-Infinity: Breaking the GPU Memory Wall for Extreme Scale Deep Learning._ SC '21 / arXiv:2104.07857.
- [4] Zhao et al. _PyTorch FSDP: Experiences on Scaling Fully Sharded Data Parallel._ arXiv:2304.11277, 2023.
