# Mamba and State Space Models

- **Key papers**:
  - **S4** — Gu et al., Stanford. ICLR '22.
  - **Mamba** — Gu & Dao, CMU + Princeton. 2023-12.
  - **Mamba-2** — Dao & Gu. ICML '24.
- **Links**: [S4 (arXiv:2111.00396)](https://arxiv.org/abs/2111.00396) · [Mamba (arXiv:2312.00752)](https://arxiv.org/abs/2312.00752) · [Mamba-2 (arXiv:2405.21060)](https://arxiv.org/abs/2405.21060) · [code](https://github.com/state-spaces/mamba)

## TL;DR

**State Space Models (SSMs)** are a non-transformer sequence architecture with **linear-time prefill** and **constant-memory decode** — no KV cache. They re-emerged from classical control theory (S4, 2022) as a deep-learning-friendly form, then **Mamba** (2023) made them competitive with transformers up to ~7B scale by adding **input-dependent (selective) state transitions** while preserving the linear scaling. **Mamba-2** (2024) then showed SSMs and attention are closely related (the SSD framework), enabling fast hardware-friendly kernels. The infrastructure story is the main point: SSMs have **fundamentally different inference economics** — long context is cheap, decode doesn't grow with context — but **lose attention's in-context recall** on some tasks. Pure SSMs didn't dethrone transformers, but **hybrid architectures** (Jamba, Zamba, Samba, Nemotron-H) that mix SSM and attention layers are now a real deployment option, especially for long-context workloads where transformer KV cache is painful.

## Context & Motivation

Transformers have a specific infra pain point: **attention's quadratic cost in sequence length** (compute and memory). FlashAttention made it exact and cache-friendly, Ring Attention made it parallel, but neither changed the complexity. For 1M+ token contexts, transformer prefill is expensive, decode KV cache is huge.

Linear-attention variants (Performer, Linformer, Linear Attention) tried to change the complexity but all lost quality. The question: is there a **sub-quadratic architecture that preserves transformer-level quality**?

State space models offered a promising direction. The math: an SSM evolves hidden state `h_t` via

```
h_t = A · h_{t-1} + B · x_t
y_t = C · h_t
```

for matrices `A, B, C`. Pick `A` structured enough for efficient computation and you get:

- **Prefill**: a convolution along the sequence (FFT possible), linear time.
- **Decode**: a recurrent step, `O(hidden^2)` per token, **no KV cache**.

The blocker was quality. Classical SSMs had `A, B, C` fixed per layer (linear time-invariant, LTI). They could compress *any* sequence but couldn't **selectively attend to the right parts** — a skill attention has trivially.

## S4 — efficient SSMs at scale

Gu et al. 2022 made three contributions:

1. **HiPPO initialization** — a principled way to choose `A` so the state optimally memorizes history.
2. **Diagonal-plus-low-rank structure** — `A` decomposable into parts that admit fast computation.
3. **Parallel scan via FFT** for prefill.

S4 showed SSMs could match transformers on Long Range Arena. But it was LTI — every token used the same `A, B, C`. That meant no selective recall; for language, quality lagged transformers.

## Mamba — selective SSMs

Mamba's main move: **make `B` and `C` (and the time-step `Δ`) input-dependent**. Given input `x_t`, compute:

```
B_t = W_B · x_t
C_t = W_C · x_t
Δ_t = softplus(W_Δ · x_t)
A_t = discretize(A, Δ_t)
```

The SSM is no longer LTI. The model can now **dynamically decide what to store, what to ignore, and how fast state decays** based on input content. This is conceptually similar to attention's Q/K/V routing — each token gets to choose what it reads from state.

### The efficiency catch

Input-dependent SSMs break FFT-based parallel scan. Naively, you'd fall back to per-token recurrence (slow). Mamba's solution: **a custom CUDA kernel for "parallel selective scan"** — computes the scan in `O(N log N)` or `O(N)` depending on hardware, using work-efficient parallel prefix algorithms.

The kernel is hand-tuned (Tri Dao again — the same person who wrote FlashAttention). Keeps state in SRAM, minimizes HBM traffic. This is what made Mamba usable in practice: the **kernel is as important as the math**.

### Results

- Mamba-2.8B matches Llama-1-3B on common LM benchmarks with 2–4× faster inference throughput.
- Near-constant decode latency regardless of context length.
- Prefill scales linearly with sequence length (vs quadratic for transformers).
- Particularly strong on DNA, audio, and long-range synthetic tasks.

## Mamba-2 — State Space Duality

2024. Key insight: **the selective SSM is mathematically equivalent to a specific form of linear attention (scalar-valued mask)**. The authors call this the **SSD framework**.

Implications:

- Many tricks from efficient attention (FlashAttention-style kernels, tensor-core-friendly layouts) transfer directly to SSMs.
- Mamba-2 kernels are simpler and 2–8× faster than Mamba-1 at the same state size.
- Opens the door to **hybrid SSM/attention kernels** — same math substrate.

Mamba-2 is the version of SSMs that's practically deployable in mixed architectures.

## Hybrid architectures

Pure SSMs hit ceilings on tasks requiring precise in-context recall ("find this token in the prompt and use it"). Attention is essentially free for this; SSMs have to encode the prompt into fixed state, which compresses.

**Hybrid architectures interleave SSM and attention layers** — SSMs do most layers (cheap, linear), attention does a few (precise recall). Examples:

- **Jamba** (AI21, 2024) — production-deployed hybrid at 52B params with 256K context.
- **Zamba** (Zyphra, 2024) — hybrid SSM + shared attention.
- **Samba** — another mixed stack with sliding-window attention + SSM.
- **Nemotron-H** (NVIDIA, 2025) — hybrid at scale.

Ratio matters: typically 1 attention layer per 6–8 SSM layers. Enough to handle recall-dependent tasks, cheap enough to keep the infra advantages.

## Infrastructure implications

### Inference

- **Constant-memory decode** — no KV cache, just a fixed-size hidden state (~MB per layer per sequence). Massive memory savings at long context.
- **Long-context prefill** linear, not quadratic.
- **No prefix caching across requests** — state is compressed irreversibly; you can cache prefix *states*, but they're less compositional than KV cache (concatenating prefixes isn't exactly trivial).
- **PagedAttention irrelevant** — no KV blocks to page. Different memory management entirely.

For serving: SSM inference is a different regime. Mooncake-style KV cache pools don't apply; prefix caching requires new thinking; PD-disaggregation is different because there's no KV to ship.

### Training

- Selective scan kernel replaces attention kernel — the primary compute + memory shape is different.
- Sequence parallelism (Ring Attention) doesn't directly transfer; SSMs need their own sequence-axis parallelism story.
- Model parallelism otherwise similar (TP along hidden dim works).

### Hybrid serving

A Jamba-style model still has attention layers, so KV cache still exists for those — just smaller. Hybrid serving stacks need to manage **both** KV cache blocks and SSM states. Implementations (vLLM Jamba support, for example) split memory management across both.

## Engineering Tradeoffs

| Decision | Gained | Gave up |
|---|---|---|
| Pure SSM | Linear time, constant-memory decode | In-context recall quality |
| Input-dependent selectivity (Mamba) | Closes the quality gap for many tasks | Breaks FFT; requires custom scan kernel |
| SSD framework (Mamba-2) | Unified with attention; faster kernels | Less architecturally pure |
| Hybrid SSM + attention | Best of both: cheap most layers, precise where needed | Serving stack must handle two cache types; hybrid ratios require tuning |
| No KV cache | Dramatic inference memory savings | Existing inference infrastructure (paged attention, prefix caching) doesn't apply unchanged |
| Hand-tuned scan kernel | 2-4× over naive | Kernel complexity; specific to arch generation |

## Experiments & Results

- Pure Mamba scales cleanly to 2.8B; gets harder to match transformers above 7B (in the pure form).
- Hybrid models match or exceed transformers of similar size on general benchmarks; significantly better on long-context throughput benchmarks.
- Jamba-1.5-Large (398B MoE + SSM hybrid) is the largest public SSM-containing model as of 2025.

## Reproducibility Notes

- Reference implementations open-sourced.
- Mamba-2 is the version to use; Mamba-1 is superseded.
- Hugging Face Transformers has Mamba support; vLLM supports Jamba for serving.
- Training at scale has real quirks (state initialization, normalization placement) — the papers have details worth reading carefully.

## Commentary

The SSM story is an infra-driven one. The mathematics has been around since control theory; what changed was (a) HiPPO gave a sensible initialization, (b) Mamba made selectivity work, and (c) Tri Dao wrote the kernel. Without the kernel, Mamba wouldn't have been practical; with it, the architecture could compete.

The honest assessment: **SSMs didn't replace transformers, and they probably won't**. Attention's in-context recall is too useful, and transformers have too much ecosystem momentum. But SSMs **don't need to win** — they need to occupy a useful niche. That niche is **long context, cheap decode, and hybrid architectures where attention is rare**. Jamba, Nemotron-H, and the next generation of Claude / Gemini / others quietly include SSM layers now, not out of architectural purism but because **10M-token context is painful with pure attention, easy with SSM + sparse attention**.

The broader lesson: **inference economics drive architecture choices at scale**. Transformers won the training era because attention is great for training efficiency. SSMs and hybrids are winning the long-context-inference niche because KV cache at 1M+ tokens is painful. The next architectural shift will likely be driven by whichever inference shape becomes most expensive — probably agentic workloads with deep, diverse state.

For anyone building long-context products in 2026: **check hybrid SSM+attention models before defaulting to pure transformer**. The serving cost profile is different enough that for long-context-heavy use cases, the hybrid wins on cost even at slight quality cost.

## References

- [1] Gu et al. _Efficiently Modeling Long Sequences with Structured State Spaces._ ICLR '22 / arXiv:2111.00396.
- [2] Gu & Dao. _Mamba: Linear-Time Sequence Modeling with Selective State Spaces._ arXiv:2312.00752, 2023.
- [3] Dao & Gu. _Transformers are SSMs: Generalized Models and Efficient Algorithms Through Structured State Space Duality._ ICML '24 / arXiv:2405.21060.
- [4] Lieber et al. _Jamba: A Hybrid Transformer-Mamba Language Model._ arXiv:2403.19887, 2024.
- [5] state-spaces/mamba: https://github.com/state-spaces/mamba
