# Mixtral of Experts

- **Authors / Org**: Mistral AI
- **Published**: 2024-01
- **Links**: [paper (arXiv:2401.04088)](https://arxiv.org/abs/2401.04088) · [weights](https://huggingface.co/mistralai/Mixtral-8x7B-v0.1)

## TL;DR

Mixtral 8×7B is a **sparse mixture-of-experts** built on Mistral 7B's architecture: 8 feed-forward experts per layer, top-2 routing, **46.7B total parameters but only 12.9B activated per token**. It matches or beats Llama 2 70B on most benchmarks while running at ~13B-dense inference cost. Conservative MoE design — **few, large, uniform experts, no shared expert, no fine-grained segmentation** — but the first open-weight MoE at meaningful scale. Useful read alongside DeepSeekMoE as the "coarse MoE" end of the design spectrum.

## Context & Motivation

When Mixtral dropped in late 2023 / early 2024, the open-weight world was dense. MoE at scale was rumored in GPT-4 but not publicly validated in open source. Mistral's goal was pragmatic: show that MoE works at the ~50B parameter band, keeps inference cheap, and fits on two consumer GPUs with quantization. The paper is short (~15 pages) and deliberately light on novelty — it is a **release paper**, not an architecture paper.

## Core Method

### Architecture
- Base: Mistral 7B (GQA, RoPE, SwiGLU, sliding-window attention).
- **Replace each FFN with a sparse MoE layer**: 8 experts, top-2 routing.
- Router: linear layer over token hidden state, softmax → top-2 picks.
- Final output: weighted sum of the two chosen experts' outputs.
- 32 MoE layers, 32k vocab, 32k context.

### Routing and load balancing
- Standard top-2 router with softmax normalization.
- Load balancing via an **auxiliary loss** (Switch Transformer style) plus **router z-loss** for stability.
- No shared expert, no fine-grained segmentation, no capacity factor tricks worth noting — the point is to keep it simple.

### Training
- Paper gives almost no training detail — no total tokens, no cluster size, no throughput numbers. This is a non-trivial contrast with V3 and Llama 3.
- Fine-tuned variant (Mixtral 8×7B Instruct) released alongside.

## Engineering Tradeoffs

| Decision | Gained | Gave up |
|---|---|---|
| Coarse 8 experts, top-2 | Simple routing, low dispatch overhead | Less specialization than DeepSeekMoE's 256 top-8 |
| No shared expert | Simpler architecture | Routed experts must re-learn generic features |
| Drop-in replacement of FFN | Easy to build, tool chains work out of the box | Cannot rebalance compute between attention and MoE |
| Standard aux balancing loss | Stable, well-understood | Mild quality tax vs V3's aux-loss-free scheme |
| 46.7B total / 12.9B active | Runs on 2× 24GB with 4-bit quant | Memory footprint still awkward — falls between 13B and 70B serving classes |

## Experiments & Results

- Matches Llama 2 70B on most benchmarks, beats it on math/code.
- Inference throughput and latency close to a 13B dense model.
- Instruct variant outperforms GPT-3.5 on MT-Bench.
- **No training cost disclosed.** Significant limitation for any infra takeaway.

## Reproducibility Notes

- Weights open (Apache 2.0); inference code in `transformers`, vLLM, llama.cpp, etc.
- No training code, data, or infra detail.
- For MoE research, this is a **clean, well-understood baseline** — most open MoE tooling was stabilized against Mixtral's shape.

## Commentary

Mixtral's contribution is **ecosystem, not architecture**. By shipping a permissively-licensed MoE at useful quality, Mistral forced every inference framework, quantization library, and serving stack to support sparse routing. DeepSeekMoE is the more ambitious architecture; Mixtral is the one that made MoE mainstream in open source. Read the two papers together: Mixtral shows the **minimum viable MoE** (simple, works, ships), DeepSeekMoE shows what you gain by **pushing the architecture harder** (more experts, finer grain, shared isolation). The field is converging on the DeepSeek end of the spectrum — V3's recipe is likely what the next Mixtral will look like.

## References

- [1] Mistral AI. _Mixtral of Experts._ arXiv:2401.04088, 2024.
- [2] Jiang et al. _Mistral 7B._ arXiv:2310.06825, 2023.
- [3] Fedus et al. _Switch Transformers._ JMLR, 2022. (Aux loss baseline)
- [4] Dai et al. _DeepSeekMoE._ arXiv:2401.06066, 2024. (Contrast point)
