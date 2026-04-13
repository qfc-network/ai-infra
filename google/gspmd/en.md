# GSPMD — Compiler-Driven Automatic Parallelization

- **Authors / Org**: Xu et al., Google
- **Published**: 2021-05 (arXiv)
- **Links**: [paper (arXiv:2105.04663)](https://arxiv.org/abs/2105.04663) · [XLA SPMD partitioner](https://github.com/openxla/xla)

## TL;DR

GSPMD (Generalized SPMD) is the **XLA compiler pass** that takes a JAX / TensorFlow program plus a **sharding annotation on a few key tensors** and produces a fully parallelized SPMD program for thousands of devices — automatically inserting collective ops, re-sharding, and optimizing communication. It's the reason you can write `w = jax.lax.with_sharding_constraint(w, P('data', 'model'))` and have the compiler figure out TP, FSDP, and mixed strategies from there. GSPMD powers **PaLM, Gemini, and all modern Google-scale JAX training**, as well as every non-Google user of JAX with TPU / multi-GPU today. Alongside Pathways (the runtime), GSPMD is the other half of Google's answer to "how do you program a pod."

## Context & Motivation

Manual parallelism in frameworks like Megatron or DeepSpeed requires the programmer to:

- Split a weight matrix across devices along the correct axis.
- Insert `all-reduce` or `all-gather` at the correct points.
- Ensure collectives align across devices.
- Re-derive everything when changing TP size, PP layout, or adding ZeRO.

This is error-prone at small scale and impossible at frontier scale without a small army of specialists. TF 1.x's `tf.distribute` handled data parallelism but not model parallelism. GPipe and Mesh-TensorFlow had answers but required heavy rewriting.

GSPMD's goal: **annotate a few critical tensors with their sharding; the compiler figures out the rest**. Programmer specifies intent, compiler handles mechanics.

## Core Method

### Sharding as a type

Each tensor has a **sharding spec** — which mesh axes shard which tensor axes. A mesh is a named multi-dim grid of devices (e.g., `{'data': 8, 'model': 4}` for 32 devices arranged as 8×4).

A sharding spec for a 2D tensor might be `P('data', 'model')`: first axis sharded along the `data` mesh dim, second along `model`. A `P(None, 'model')` tensor is replicated along `data`, sharded along `model`. `P()` fully replicates.

### Propagation

Given sharding on inputs and a few explicit constraints, GSPMD **propagates** sharding through the computation graph:

- If `a` is sharded `P('data', None)` and `b` is `P(None, 'model')`, then `a @ b` naturally becomes `P('data', 'model')` — the compiler figures this out.
- Reductions, reshapes, broadcasts each have their propagation rules.
- Where multiple propagations conflict, the compiler **picks one and inserts re-sharding** (an `all-to-all`, `all-gather`, or similar collective) to reconcile.

The propagation is done by iterated rewriting over XLA HLO. A **cost model** over collectives determines which re-sharding placements are cheapest.

### Partitioner pass

Once every HLO op has a consistent sharding, the **partitioner** rewrites each op into its sharded form:

- `matmul` with sharded inputs becomes a sharded matmul + possibly `all-reduce` over the contracted dim.
- `conv` is partitioned similarly with spatial reductions handled.
- Element-wise ops are trivially partitioned.
- Shape-changing ops (reshape, transpose) may require re-sharding; the compiler emits the needed collectives.

The result: an SPMD program where every device runs the same code, reading/writing its own shard. Identical to what a human would write in Megatron — but generated from a few dozen annotations.

### Pipeline parallelism via explicit rewrites

GSPMD doesn't do pipeline parallelism automatically; PP is added via a **different rewrite** (`pipeline_parallelism_pass`) that splits the graph into stages and inserts stage boundaries. PP stays outside GSPMD's core sharding logic because it requires different cost reasoning (bubbles, schedules).

### Nested and hybrid strategies

Because sharding is compositional, GSPMD supports **hybrid strategies natively**:

- `P('data', 'model')` on weights = tensor parallelism.
- `P('fsdp', None)` on weights with `all-gather` at forward = FSDP.
- Combine both on different tensors = ZeRO-3 + TP.
- Add `shard_map` annotations around a block = local SPMD within a global SPMD.

Writing all of this by hand in Megatron takes thousands of lines and careful state management. GSPMD accepts it as annotations.

## Engineering Tradeoffs

| Decision | Gained | Gave up |
|---|---|---|
| Annotation-driven auto-parallel | Trivially adjust TP/FSDP/hybrid | Annotations still needed in key spots; not zero-config |
| Compiler cost model | Picks near-optimal re-sharding | Cost model can be wrong; opaque to user |
| SPMD-only for core | Simple code model; same code per device | PP needs a separate pass |
| XLA-based | Mature compiler, good peak perf | XLA compile times; debugging is a skill |
| Propagation + partitioner | Minimal user annotation | Partitioner bugs show up as subtle perf issues |
| Sharding as a type | Cleanly compositional | Requires the framework's cooperation (JAX, TF only) |

## Experiments & Results

- Trained **PaLM (540B)** via GSPMD + Pathways, achieving 57.8% MFU.
- GSPMD itself handles sharding for **every production JAX training at Google** from 2021 onward.
- Public analog: JAX's `jit`+`pjit`+`shard_map` ecosystem makes GSPMD's behavior available to all JAX users. Production frameworks (MaxText, T5X, Praxis) are GSPMD-based.

## Reproducibility Notes

- **The compiler is open source** — `openxla/xla` includes the SPMD partitioner. Used by every JAX TPU / multi-GPU workload.
- JAX documentation covers the user-facing annotation model; the compiler internals are in XLA HLO passes.
- Outside JAX, OpenXLA is being adopted by PyTorch via PyTorch/XLA — meaning GSPMD-style parallelism is trickling into PyTorch too.

## Commentary

GSPMD is the **quietest load-bearing tool in frontier ML**. Most people have never heard of it; most JAX users are using it every day without realizing. It's one of the cleanest examples of a compiler pass whose value is "eliminated a class of manual work." Before GSPMD, every major parallel training framework reinvented its own mechanism for splitting tensors and inserting collectives; after GSPMD, you annotate and move on.

The more interesting comparison is to PyTorch's trajectory. PyTorch historically bet on **the eager mental model** — define everything explicitly. Parallelism was user-owned (DDP, FSDP, Megatron-for-PyTorch). This is great for research flexibility and terrible for composing strategies. PyTorch is now adding compiler paths (`torch.compile`, `torch.distributed._tensor`, `DTensor`) that look increasingly GSPMD-ish. The convergence is real but slow — PyTorch's installed base of eager-style code is huge and not trivially rewriteable.

The philosophical point: **if your compiler understands sharding, parallelism becomes a hyperparameter; if it doesn't, parallelism becomes a rewrite**. The former is the long-run winning position, and GSPMD was first to execute on it at frontier scale. Expect the PyTorch ecosystem to close this gap over 2025–2026 via DTensor + `torch.compile`, but the design language is copying from GSPMD whether it's acknowledged or not.

## References

- [1] Xu et al. _GSPMD: General and Scalable Parallelization for ML Computation Graphs._ arXiv:2105.04663, 2021.
- [2] Barham et al. _Pathways._ MLSys '22. (Runtime counterpart)
- [3] Lepikhin et al. _GShard._ 2020. (Predecessor work on sharded MoE — "GSPMD" generalizes)
- [4] OpenXLA: https://github.com/openxla/xla
- [5] JAX `shard_map` and `pjit` docs: https://jax.readthedocs.io/
