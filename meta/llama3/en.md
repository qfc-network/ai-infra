# The Llama 3 Herd of Models

- **Authors / Org**: Meta AI
- **Published**: 2024-07
- **Links**: [paper (arXiv:2407.21783)](https://arxiv.org/abs/2407.21783) · [models](https://huggingface.co/meta-llama)

## TL;DR

A 90-page engineering report on training a **405B dense transformer** on **15.6T tokens** using up to **16,000 H100s**. Architecturally conservative (dense, GQA, RoPE, standard transformer) — the value is in the systems work: 4D parallelism (TP + PP + CP + DP), BF16 throughout, a detailed account of hardware failures at scale, and a staged approach to context extension (8k → 128k). The clearest public picture of what frontier-scale dense pretraining actually looks like as of 2024.

## Context & Motivation

By mid-2024, frontier training split into two camps: sparse MoE (DeepSeek, Mistral, rumored GPT-4) and dense (Anthropic, Meta). Meta's bet: dense is simpler to train, serve, and fine-tune, and at 405B the capability gap vs. MoE at similar activated parameters is acceptable. The paper's job is to prove that bet at a scale no one else has publicly documented.

## Core Method

### Architecture (deliberately boring)
- Dense transformer, 405B params, 126 layers, hidden 16384, 128 heads.
- GQA with 8 KV heads.
- RoPE with base frequency scaled for long context.
- Tokenizer: 128k vocab.

No MoE, no MLA, no exotic attention. The thesis is that **the architecture does not matter much at frontier scale if the data and system are right**.

### Data
- 15.6T tokens after aggressive filtering. Multi-stage classifier-based filtering, de-duplication at scale, quality heuristics.
- Mix tuned by running thousands of **scaling-law experiments on small models** — data-mix is treated as a learnable hyperparameter.
- Annealed LR and data-mix changes in the final phase improve math and code.

### Training infrastructure
- **16k H100s** in a single cluster; network: RoCE Ethernet for the biggest job, also InfiniBand variants.
- **4D parallelism**: tensor (TP=8, within node) + pipeline (PP=16) + context (CP=2, for long-context stage) + data (DP=128). The CP axis is an explicit answer to long-context activation memory.
- **BF16 throughout** — no FP8. Meta's position: stability and simpler tooling outweigh the compute savings at this scale.
- **Reliability engineering**: 419 unexpected interruptions over 54 days; 78% hardware-related (GPU/HBM/NVLink failures). Detailed MTBF analysis; automated checkpoint restart; burn-in procedures.

### Context extension
Staged: pretrain at 8k → extend to 128k via continued pretraining on long documents with adjusted RoPE base frequency. CP parallelism introduced at this stage.

### Post-training
- SFT + DPO rather than full RLHF (claimed simpler, stable, competitive).
- Iterated rounds with reward-model filtering on synthetic data.

## Engineering Tradeoffs

| Decision | Gained | Gave up |
|---|---|---|
| Dense over MoE | Simpler training, serving, fine-tuning; predictable scaling | Higher active FLOPs per token; worse quality per activated param |
| BF16 over FP8 | Stability, mature tooling | ~2× throughput left on the table vs FP8 |
| RoCE Ethernet at 16k | Avoided IB supply constraints at scale | More network-level engineering; custom congestion control |
| DPO over PPO-RLHF | Stability, less infra | Some quality headroom in alignment |
| 4D parallelism with CP | Long-context trainable at this scale | Scheduler complexity; CP is not free in communication |
| Scaling-law-driven data mix | Principled data decisions | Thousands of small runs before main training — meaningful prep cost |

## Experiments & Results

- **405B** competitive with GPT-4 class models on many benchmarks at release.
- **8B / 70B** derivatives are the workhorses — most of the field's open-weight fine-tuning runs on these.
- Detailed MFU, failure, and scaling numbers make it the best public dataset for anyone planning a large run.

## Reproducibility Notes

- Weights open (8B, 70B, 405B); training code is not open-sourced, but enough detail in the paper to reproduce architecture and data pipeline.
- The specific cluster-level engineering (network topology, scheduling, fault handling) is Meta-internal and not reproducible outside a hyperscaler.
- Excellent companion read to DeepSeek-V3: same era, opposite architectural bets, similar level of infra rigor.

## Commentary

Read this paper to understand **what frontier training cost and complexity look like** when you refuse to take architectural shortcuts. The contrast with V3 is instructive: V3 spends on infra (FP8, DualPipe, hand-tuned kernels) to buy capability at low compute; Llama 3 spends on compute (BF16, dense, 16k GPUs) to buy simplicity. Both work; neither generalizes to every lab. The detailed failure statistics are the most under-appreciated part of the paper — if you are planning ≥1k-GPU training, **treat reliability engineering as a first-class design axis**, not an afterthought. Expect the next Llama to move toward MoE (the economic pressure is strong) but likely retain Meta's stability-first culture: BF16, proven parallelism, paranoid checkpointing.

## References

- [1] Meta AI. _The Llama 3 Herd of Models._ arXiv:2407.21783, 2024.
- [2] Dubey et al. (Meta). Companion blog posts on Llama 3 training infra.
