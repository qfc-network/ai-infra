# MLA — Multi-head Latent Attention

- **Authors / Org**: DeepSeek-AI (introduced in DeepSeek-V2)
- **Published**: 2024-05
- **Links**: [DeepSeek-V2 paper (arXiv:2405.04434)](https://arxiv.org/abs/2405.04434) · [FlashMLA](https://github.com/deepseek-ai/FlashMLA)

## TL;DR

MLA replaces the per-head KV cache of Multi-Head Attention with a **single low-rank latent vector per token**, from which per-head keys and values are reconstructed on the fly. The KV cache shrinks to roughly 1/10 of standard MHA while matching — not underperforming — MHA in quality. Unlike MQA/GQA, which trade quality for cache size, MLA is presented as dominating MHA on both axes once you absorb the up-projection matrices into the attention computation.

## Context & Motivation

KV cache dominates long-context inference memory and bandwidth:

- **MHA**: cache size = `2 · n_h · d_h · L · n_layers` per sequence. Expensive.
- **MQA**: all heads share one K,V. ~`n_h`× smaller cache, measurable quality loss.
- **GQA** (Llama 2 onward): compromise — group heads, quality closer to MHA.

MQA/GQA are strictly quality-for-memory trades. DeepSeek asked: can we keep full MHA expressivity with near-MQA cache?

## Core Method

### Low-rank joint KV compression

For each token, project the hidden state `h_t` once into a **compressed latent** `c_t^{KV} ∈ R^{d_c}` with `d_c ≪ n_h · d_h`:

```
c_t^{KV} = W_DKV · h_t
```

At attention time, per-head keys and values are decompressed:

```
k_t^{(i)} = W_UK^{(i)} · c_t^{KV}
v_t^{(i)} = W_UV^{(i)} · c_t^{KV}
```

**The cache stores only `c_t^{KV}`** (plus a small RoPE component, below). For a typical config `d_c ≈ 4·d_h`, cache per token is comparable to MQA while heads remain independently parameterized.

### Absorption trick

Naively, reconstructing K, V per head at every step seems to cost compute. But `W_UK` can be **absorbed into W_Q** (via `Q · K^T = (h · W_Q) · (W_UK · c)^T = h · (W_Q · W_UK^T) · c^T`) and `W_UV` into the output projection. At inference, no explicit up-projection of K, V is needed — attention runs directly over the compressed latent.

### Decoupled RoPE

RoPE is position-dependent, so it cannot be applied to an absorbed matrix product. MLA resolves this by splitting each key into:

- A **compressed part** (no RoPE) reconstructed from `c_t^{KV}` via the absorbed path.
- A **decoupled part** `k_t^R` of small dimension `d_h^R` that *does* receive RoPE and is cached separately.

Queries are split analogously. Concatenation of both parts gives the final per-head K, and the RoPE contribution flows correctly.

### Query compression (optional)

The same low-rank trick is applied to queries to reduce training activation memory. This does not affect cache size.

## Engineering Tradeoffs

| Decision | Gained | Gave up |
|---|---|---|
| Single latent instead of per-head K,V | ~10× smaller KV cache | Two extra projections (absorbed at inference, still present at training) |
| Decoupled RoPE dimension | Correct position encoding with absorption | Cache is latent + small RoPE slice (minor overhead) |
| Absorption at inference | Eliminates up-projection compute | Kernels must fuse the composite weight; naive implementations leave perf on the floor |
| Full per-head expressivity | Matches MHA quality (no GQA-style tax) | More parameters than MQA's shared K, V |

## Experiments & Results

Reported in V2:

- KV cache per token: MLA ≈ 1/7 to 1/14 of MHA depending on config.
- On standard benchmarks, MLA matches or slightly beats MHA at the same parameter budget; clearly outperforms GQA variants at comparable cache size.
- Training throughput improves because activation memory for attention drops.

FlashMLA (2025, open-sourced) gives tuned kernels achieving near-peak HBM bandwidth for decode on H800.

## Reproducibility Notes

- Architecture described in enough detail in V2 to re-implement; several open re-implementations exist.
- FlashMLA provides production-grade decode kernels; prefill kernels are still evolving in the ecosystem.
- Training recipe (init, stability) less documented than architecture.

## Commentary

MLA is the rare architectural change that dominates its predecessor on both quality and cost, rather than trading. The insight — that K and V projections can be algebraically absorbed, so compression need not cost compute at inference — is the kind of systems-aware modeling that defines DeepSeek's style. Expect variants of this idea to propagate: any cache that can be refactored as a low-rank product of a stored latent and a fixed weight is a candidate for the same trick. The open question is whether MLA composes cleanly with future attention variants (sliding window, linear attention hybrids); the decoupled-RoPE machinery suggests yes, at the cost of some implementation complexity.

## References

- [1] DeepSeek-AI. _DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts Language Model._ arXiv:2405.04434, 2024.
- [2] DeepSeek. _FlashMLA._ https://github.com/deepseek-ai/FlashMLA, 2025.
- [3] Shazeer. _Fast Transformer Decoding: One Write-Head is All You Need._ arXiv:1911.02150, 2019. (MQA)
- [4] Ainslie et al. _GQA: Training Generalized Multi-Query Transformer Models._ arXiv:2305.13245, 2023.
