# Megatron-Core (mcore): The Modular Library Behind Modern Large-Scale Training

- **Org**: NVIDIA
- **Initial extraction from Megatron-LM**: ~2023 (first appeared as `megatron/core/` in the Megatron-LM repository)
- **Links**: [GitHub — NVIDIA/Megatron-LM (core/)](https://github.com/NVIDIA/Megatron-LM/tree/main/megatron/core) · [NeMo](https://github.com/NVIDIA/NeMo) · [NeMo-Aligner](https://github.com/NVIDIA/NeMo-Aligner)

## TL;DR

Megatron-LM (the 2019–2022 paper series) is a monolithic GPT training script. Megatron-Core — `mcore` — is the refactored library extracted from it: reusable, independently importable Python modules that implement tensor parallelism, pipeline parallelism, context parallelism, FP8 routing, and pipeline schedules. NeMo, NeMo-Aligner, Nemotron, xAI's Grok training stack, and external research groups all consume mcore as a library dependency rather than forking the training script. If a 2024–2025 paper or system report references "Megatron-style TP" or "4D parallelism," the implementation it actually ran on is almost certainly mcore, not the original script.

## Context & Motivation

The original Megatron-LM repository trained GPT models end-to-end. Its parallelism logic was correct and well-tested, but it was deeply entangled with model definitions, data loading, and optimizer loops in a single codebase. Reusing just the tensor-parallel linear layers in a new architecture — say, a mixture-of-experts model or an RLHF training loop — required forking the entire repository, which diverged from upstream and created maintenance burden.

Megatron-Core solves this by extracting the parallelism primitives into a structured Python package (`megatron/core/`) with clean APIs. The key design decision: process group management, model construction, and schedule execution are separate concerns, each handled by its own module. External frameworks import only what they need.

The practical consequence for the ecosystem: NeMo no longer ships its own tensor-parallel linear layers; it wraps mcore's. NeMo-Aligner's GRPO and DPO training loops sit on top of mcore's DDP and schedule modules. When NVIDIA publishes Nemotron model weights, the training configuration is expressed as a `TransformerConfig` object from mcore. The paper entry in `../../foundational/megatron-lm/` describes the *algorithms*; this entry describes the *library* that executes them at production scale.

## Core Abstractions

### ParallelState (`megatron/core/parallel_state.py`)

`ParallelState` is the single source of truth for which GPU occupies which rank in each parallelism dimension. At job launch, `initialize_model_parallel(tensor_model_parallel_size, pipeline_model_parallel_size, ...)` creates NCCL process groups for TP, PP, DP, and CP by computing the combinatorial layout of `world_size` ranks. Every subsequent communication call — all-reduce, reduce-scatter, all-gather — queries a function like `get_tensor_model_parallel_group()` or `get_data_parallel_group()` rather than passing group handles manually.

The importance of this design: because the process group is a global singleton, any module in the call stack — a linear layer deep inside a transformer block, a custom MoE router, an optimizer — can retrieve its correct group without that group being passed through every function signature. It is effectively a process-group registry. The downside is that it introduces global mutable state, which complicates testing and makes multi-job-in-process setups awkward.

### TransformerConfig

`TransformerConfig` is the central configuration dataclass for the entire mcore transformer stack. Key fields:

```python
TransformerConfig(
    num_layers=96,
    hidden_size=12288,
    num_attention_heads=96,
    num_query_groups=8,               # GQA: 8 KV heads
    tensor_model_parallel_size=8,
    pipeline_model_parallel_size=16,
    context_parallel_size=2,          # CP for long context
    sequence_parallel=True,
    use_flash_attention=True,
    fp8=True,                         # route to TransformerEngine
    fp8_recipe=te.recipe.DelayedScaling(...),
    normalization="RMSNorm",
    activation_func=F.silu,           # SwiGLU
)
```

This single object is passed through `GPTModel`, `TransformerBlock`, `TransformerLayer`, and down to individual attention and MLP modules. Every component reads the config it needs from it. The practical effect: changing parallelism degree, precision policy, or normalization type requires editing one object, not threading new arguments through dozens of call sites. For `fp8=True`, the layer construction path routes to TransformerEngine's `te.TransformerLayer` instead of mcore's native implementation — mcore becomes the orchestration layer and TE provides the actual FP8 kernels.

### TransformerLayer and Attention Backends (`megatron/core/transformer/`)

`TransformerLayer` is configurable on multiple axes:

- **Attention backend**: flash attention (via `FlashAttentionCore`), fused attention (NVIDIA's custom fused kernel), dot-product attention (reference). Selected by `config.use_flash_attention` and a runtime capability check.
- **Normalization**: `LayerNorm` or `RMSNorm`, selected by `config.normalization`.
- **MLP variant**: standard dense MLP, SwiGLU, or mixture-of-experts (MoE) MLP (`config.num_moe_experts`).
- **Bias terms**: controllable per projection.

The pluggability matters for experimentation: swapping from LayerNorm to RMSNorm, or from standard attention to GQA, does not require touching any parallelism code. The parallelism wiring is handled by `MegatronModule` and the column/row-parallel linear layers one level below.

### MegatronModule and Parallel Linears

`MegatronModule` is the base class for all sharded modules. Its two principal subclasses are `ColumnParallelLinear` and `RowParallelLinear`, which implement the core TP communication pattern from the original Megatron-LM v1 paper:

```
# TP forward pass for a single linear layer pair (simplified)
# Column-parallel: input X is broadcast, output Y is split across TP ranks
Y_local = X @ W_col_local      # no comm
# Row-parallel: input is already split, output requires all-reduce
Z = Y_local @ W_row_local
Z_full = all_reduce(Z)          # across tensor_model_parallel_group
```

`ColumnParallelLinear` accepts `gather_output=False` to skip the all-gather when its output feeds directly into a `RowParallelLinear`, reducing communication volume by one operation per matmul pair. The TP communication in mcore is thus zero-overhead for the MLP interior and one all-reduce per attention output projection and one per MLP output projection — exactly the count from the original paper.

### DistributedDataParallel (mcore DDP)

mcore ships its own DDP implementation (`megatron/core/distributed/`), separate from `torch.nn.parallel.DistributedDataParallel`. The reason: PyTorch DDP buckets gradients and overlaps all-reduce with backward, but it does not understand tensor-parallel sharding. A parameter that is column-parallel across TP ranks should *not* have its gradient all-reduced across TP ranks (those ranks already hold complementary shards); it should only be reduced across DP ranks. mcore's DDP uses `ParallelState` to look up the correct reduction group per parameter, and it implements gradient bucketing with explicit overlap against the backward compute stream.

## Context Parallelism in mcore

Context parallelism (CP) is the fourth dimension in mcore's parallelism decomposition, added to support sequences too long for a single TP group's activation memory. The total GPU count becomes:

```
world_size = TP × PP × CP × DP
```

mcore's CP implementation uses **all-gather + causal load balancing**. The sequence is distributed across CP ranks using an even-odd interleaving: rank 0 gets tokens [0, 2, 4, ...], rank 1 gets [1, 3, 5, ...], etc. This interleaving ensures that each CP rank processes roughly equal numbers of causally-relevant tokens (tokens near the end of the sequence have longer causal context, which is the expensive part of attention; by interleaving, this work is spread evenly). During the attention forward pass, each rank all-gathers the K and V tensors from all CP peers, computes its portion of attention, and reduces.

This is the Megatron-CP variant described in detail in `../../foundational/sequence-parallelism/`. Llama 3 405B used CP=2 with sequence length 128k, giving each GPU an effective 64k token window while retaining full causal attention correctness.

## Pipeline Schedule Module (`megatron/core/pipeline_parallel/schedules.py`)

The schedule module implements the actual pipeline execution logic — the code that decides, for each GPU at each time step, whether to run a forward pass, a backward pass, or wait:

- **Non-interleaved 1F1B** (`forward_backward_pipelining_without_interleaving`): the baseline schedule from Megatron-LM v2. Bubble fraction `(PP-1)/(m+PP-1)`.
- **Interleaved 1F1B** (`forward_backward_pipelining_with_interleaving`): virtual pipeline stages; each GPU owns multiple non-contiguous layer groups. Bubble fraction `(PP-1)/(m·v+PP-1)` where `v` is the number of virtual stages per GPU. Requires `m % PP == 0`.
- **Dual-pipe style schedules**: newer additions that implement the ideas behind the DualPipe paper — overlapping computation and communication across PP stages using paired forward/backward passes, reducing bubble ratio below the 1F1B floor.

The schedule module is where mcore's pipeline implementation actually lives — the DualPipe paper describes the algorithm, but `schedules.py` is the running code that NeMo and other consumers invoke. This distinction matters when debugging pipeline bubble inefficiency: the problem is almost certainly in a schedule variant selection or micro-batch count setting, and the fix happens in this module.

## FP8 Integration via TransformerEngine

When `TransformerConfig(fp8=True)` is set, mcore's `TransformerLayer` constructor checks for TransformerEngine availability and, if present, constructs `te.TransformerLayer` instead of its own implementation. The TransformerConfig fields `fp8_recipe` (a `te.recipe.DelayedScaling` or `te.recipe.MXFP8BlockScaling` object) and `fp8_wgrad` are forwarded to TE.

The consequence: mcore does not implement FP8 kernels. It provides the *configuration surface* and the *context management* (setting up `te.fp8_autocast` contexts around forward passes), while TE provides cuBLAS FP8 GEMM, fused attention with FP8 I/O, and amax history tracking. A user who wants FP8 training sets one flag in `TransformerConfig`; the kernel selection and scaling-factor management are handled below the mcore API surface. See `../transformer-engine/` for TE internals.

## Engineering Tradeoffs

| Dimension | Megatron-Core | PyTorch FSDP (native) | DeepSpeed | Nanotron |
|---|---|---|---|---|
| Tensor parallelism | Native, production-grade (TP up to 8 via NVLink) | Not natively supported; requires custom sharding | Supported via DS-Tensor, less tested | Supported, matches mcore design |
| Pipeline parallelism | Native; 1F1B + interleaved + dual-pipe | Not supported | GPipe-style; coarser granularity | Supported for basic 1F1B |
| Context parallelism | Native (CP=2–8 in production use) | Not supported | Not supported | Not supported |
| Modularity | High: importable as a library; NeMo, Aligner consume it | Framework-native; tightly coupled to PyTorch DDP | Plugin-based; heavy runtime overhead | Lightweight; single-file design intentional |
| External adoption | NeMo, NeMo-Aligner, xAI Grok, Nemotron | torchtitan, llama-recipes, most PyTorch-native stacks | HuggingFace Transformers, many academic setups | Minimal; primarily for research/education |
| Maintenance overhead | High: NVIDIA-maintained, active development, frequent API changes | PyTorch core; stable but slower feature velocity | Large codebase; ZeRO + TP + MoE in one framework | Low: small codebase, minimal dependencies |

## The Relation to the Original Megatron-LM Entry

The entry at `../../foundational/megatron-lm/` covers the algorithms: tensor parallelism (v1), 1F1B pipeline scheduling (v2), and sequence parallelism plus selective activation recomputation (v3). Those algorithms are correct, well-described, and worth understanding in isolation.

What that entry does not cover: *how those algorithms are actually called by NeMo when training Llama 3 on 16,000 H100s*. The answer is: through mcore's `ParallelState`, `TransformerConfig`, `TransformerLayer`, and `schedules.py`. The paper entry is the algorithm specification; this entry is the implementation. When 2024–2025 technical reports reference "Megatron-style 4D parallelism," they mean the mcore API surface described here.

## Cross-References

- `../../foundational/megatron-lm/` — the paper trilogy whose algorithms mcore implements
- `../transformer-engine/` — plugged in via `TransformerConfig(fp8=True)`; mcore orchestrates, TE executes
- `../../foundational/sequence-parallelism/` — mcore's CP is the Megatron-CP variant described there
- `../../meta/llama3/` — Llama 3 405B training used mcore's 4D parallelism with TP=8, PP=16, CP=2, DP=512
- `../../foundational/zero-fsdp/` — the alternative data-parallel approach; composes with mcore TP/PP orthogonally

## References

- [1] Shoeybi et al. _Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism._ arXiv:1909.08053, 2019.
- [2] Narayanan et al. _Efficient Large-Scale Language Model Training on GPU Clusters Using Megatron-LM._ arXiv:2104.04473, 2021.
- [3] Korthikanti et al. _Reducing Activation Recomputation in Large Transformer Models._ arXiv:2205.05198, 2022.
- [4] Dubey et al. _The Llama 3 Herd of Models._ arXiv:2407.21783, 2024. (Section 3.3: training infrastructure using mcore 4D parallelism.)
- [5] NVIDIA Megatron-Core source: `github.com/NVIDIA/Megatron-LM/tree/main/megatron/core`
