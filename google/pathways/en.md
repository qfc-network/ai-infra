# Pathways — Asynchronous Distributed Dataflow for ML

- **Authors / Org**: Barham et al., Google
- **Published**: MLSys '22
- **Links**: [paper (arXiv:2203.12533)](https://arxiv.org/abs/2203.12533) · [Google blog](https://blog.google/technology/ai/introducing-pathways-next-generation-ai-architecture/)

## TL;DR

Pathways is Google's distributed ML runtime, purpose-built for **thousand-TPU, multi-pod** training with a **single-controller** architecture. Unlike Megatron-style multi-controller systems (MPI-ish, every worker runs the same Python), Pathways has **one scheduler driving a dataflow graph across many pods**, communicating via a sharded dataflow abstraction. Its central claim: you can get **near-peak hardware utilization at massive scale while keeping the programming model single-client**, and you can run **heterogeneous, sparse, multi-task computations** that multi-controller systems can't easily express. Pathways is the runtime behind **PaLM (540B)** and every Google frontier training run since (Gemini included, though details aren't public). It's also the architectural counterpoint to JAX's `pmap`-era multi-controller style — JAX now targets Pathways as a backend via `jit` / `pjit` / `shard_map`.

## Context & Motivation

Pre-Pathways, frontier-scale ML training fell into two camps:

1. **Multi-controller (SPMD)** — Megatron, DeepSpeed, Horovod: every worker runs the same program, coordinates via collectives. Simple, scales well for dense uniform computations, but every worker must be online for every step, and heterogeneous / sparse / async patterns are awkward. Sharding is implicit (everyone knows "my rank").
2. **Single-controller (MPMD or MPP)** — TensorFlow 1.x, `tf.distribute`: one client builds a graph, workers execute pieces. Elegant for complex graphs, but historically slow: the controller becomes a bottleneck, and per-op RPC latency kills small-op workloads.

Google faced two forcing functions beyond the single-controller perf problem:

- **TPU pods are physically heterogeneous** — a training run wants a specific pod shape, not "512 arbitrary chips." The system must understand topology.
- **Future workloads are sparse, multi-task, heterogeneous** — mixture-of-experts, multi-task, speculative compute, RL with learner/actor asymmetry. Multi-controller struggles with these.

Pathways' bet: **fix single-controller performance**, so you can keep its expressivity at frontier scale.

## Core Method

### Sharded dataflow computation

A Pathways program is a **dataflow graph** whose nodes are **sharded computations** — each node takes in shards, produces shards, runs on a specific device subset. The scheduler turns this graph into device-level operations.

Crucially, the graph is **per-program**, not global: you can have multiple programs sharing the same pod, the scheduler knows how to co-schedule them without requiring identical shapes.

### Asynchronous gang-scheduled dispatch

For each computation, Pathways:

1. **Gang-schedules** it — all participating devices commit to running it at the same logical time. Ensures collectives align.
2. **Dispatches asynchronously** from the controller to a **resource manager** per pod; the pod then issues device-level instructions.
3. **Pipelines dispatch with execution** — the next step is being sent while the current step runs. Amortizes controller→pod latency over many steps.

This is the key perf trick: single-controller dispatch is slow per-step, but **pipelined with compute**, the latency is hidden as long as per-step compute > dispatch latency. At frontier scale, per-step compute is always big.

### Parallel asynchronous dispatch (PAD)

For programs with many independent branches (multi-task, speculative evaluation), Pathways dispatches them in parallel without waiting for earlier branches. Controller isn't serialized on dependencies that don't exist.

### Compilation and shape specialization

The controller JIT-compiles shards per shape — like XLA but across pods. Compiled artifacts are cached. Most training steps hit the cache.

### Deployment: pod resource managers

Each TPU pod runs a resource manager that owns local scheduling, memory, and topology. The Pathways controller talks to resource managers, not individual chips. This **hides the pod's internal topology** from the controller — the controller just requests "run this sharded computation" and the pod RM figures out which exact chips and how.

## Engineering Tradeoffs

| Decision | Gained | Gave up |
|---|---|---|
| Single-controller | Expressive for heterogeneous / multi-task / sparse workloads | Controller throughput must scale; historically a bottleneck |
| Pipelined async dispatch | Hides per-step latency | Only works when compute ≫ dispatch; small-op workloads still expensive |
| Gang scheduling | Collective correctness | Stragglers stall the gang; no flexibility on step-level timing |
| Per-pod resource manager | Topology abstraction | Extra software layer; pod-level scheduling complexity |
| Purpose-built for TPU pods | Deep integration with TPU topology / networking | Not portable to GPU fabrics without major rework |
| XLA as the compiler | Mature, good peak perf | Compilation time; debugging XLA is its own skill |

## Experiments & Results

- PaLM (540B) was trained on two TPU v4 pods (6144 chips total) using Pathways — the first public confirmation of Pathways' frontier capability.
- **57.8% hardware FLOP utilization** on PaLM training — extraordinarily high for the scale, aided by the single-controller design's ability to schedule heterogeneous comm/compute patterns.
- Pathways demonstrated **two-pod training with 50% efficiency** — spanning data-center-scale networks, something multi-controller systems would handle much worse.

## Reproducibility Notes

- Pathways itself is **not open source**. No public implementation, no code.
- JAX's `pjit` / `shard_map` is the public-facing programming model that targets Pathways behind the scenes on Google Cloud TPU. JAX users on TPU are implicitly using Pathways.
- **MaxText** (Google's open JAX LLM training framework) is the closest public analog to how Pathways is used; runs on TPU Pathways or GPU-JAX.
- **Ray** (UC Berkeley → Anyscale) is philosophically similar at a higher level — single-controller distributed actor graph — but targets a broader domain and makes different perf trade-offs.

## Commentary

Pathways is the **counter-thesis to the Megatron / PyTorch FSDP school** of frontier training. The PyTorch/Megatron camp says: "everything is SPMD, stay out of the controller's way, scale via collectives and careful parallelism config." The Pathways camp says: "a smart controller can schedule workloads the SPMD camp can't, and if we make dispatch fast enough, the expressivity wins."

Which is right? Both, for different workloads:

- **Dense, uniform training (Llama 3, V3 pretraining)** — SPMD wins. There's no heterogeneity to exploit; single-controller overhead isn't needed.
- **Sparse / mixed / heterogeneous (huge MoE with disparate expert sizes, multi-task, async RL with learner/actor, speculative branching)** — Pathways' architecture fits naturally.

In 2026, the two camps are converging. PyTorch's `torch.distributed` has added features that feel Pathways-like (collective groups, heterogeneous process groups). JAX's `shard_map` brings explicit sharding to the multi-controller style. The intellectual distance is shrinking.

The deeper lesson is about **when to centralize**: Pathways shows that if your schedule is interesting enough (heterogeneous, async, sparse), the controller has value; if your schedule is boring (dense uniform), decentralize. Most LLM pretraining today is in the "boring" regime. Agentic / RL / multi-task workloads are moving into the "interesting" one, and this is where Pathways-style systems will matter more.

For anyone reading this without Google TPU access: understand the paper for the architecture ideas, not for immediate use. The Ray ecosystem, MaxText, and JAX on TPU are the practical takeaways.

## References

- [1] Barham et al. _Pathways: Asynchronous Distributed Dataflow for ML._ MLSys '22 / arXiv:2203.12533.
- [2] Chowdhery et al. _PaLM: Scaling Language Modeling with Pathways._ JMLR '23.
- [3] Xu et al. _GSPMD._ arXiv:2105.04663, 2021. (Companion parallelism compiler)
- [4] MaxText: https://github.com/AI-Hypercomputer/maxtext
- [5] JAX: https://github.com/jax-ml/jax
