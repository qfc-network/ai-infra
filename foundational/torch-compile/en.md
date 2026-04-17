# torch.compile and the Inductor Backend

- **Org**: PyTorch / Meta
- **Released**: PyTorch 2.0 (March 2023); Inductor shipped stable in 2.1
- **Links**: [PyTorch 2.0 paper (arXiv:2311.13608)](https://arxiv.org/abs/2311.13608) · [Inductor blog](https://dev-discuss.pytorch.org/t/torchinductor-a-pytorch-native-compiler-with-define-by-run-ir-and-symbolic-shapes/747) · [torch.compile docs](https://pytorch.org/docs/stable/torch.compiler.html)

## TL;DR

`torch.compile` is PyTorch's production compilation path, layered from Python bytecode down to Triton kernels or C++/OpenMP. Three components do the work: **TorchDynamo** captures the computational graph from live Python execution without breaking user code; **AOTAutograd** lifts the captured graph into a joint forward-backward unit so the backward pass is compiled along with the forward; **Inductor** lowers the resulting graph to Triton (GPU) or C++/OpenMP (CPU), applying operator fusion, memory layout optimization, and reduction/pointwise fusion. The net effect is that memory-bandwidth-bound operations — LayerNorm, GELU, residual adds, RMSNorm — are fused into far fewer CUDA kernel launches, and the gap between "eager PyTorch" and "hand-tuned Triton" collapses to roughly 15–30% end-to-end for training, 20–50% for inference.

## Context & Motivation

Eager PyTorch pays a per-operator overhead: each `torch.add`, `torch.nn.functional.layer_norm`, or attention projection dispatches a separate CUDA kernel. For compute-heavy workloads (large matmuls), this doesn't matter — the kernel runs for milliseconds. But modern transformers are saturated with memory-bound ops: every LayerNorm, every GELU activation, every residual addition reads and writes tensors that fit in L2 cache but must round-trip through HBM. A LayerNorm for a 4096-wide hidden state reads the input once and writes the output once per kernel launch. If LayerNorm, GELU, and residual add are three separate kernels, HBM traffic is 3× what it needs to be.

The answer known since the CUDA era is **operator fusion**: combine these into a single kernel that reads the input once, applies all three operations in registers, and writes the output once. Pre-`torch.compile`, this required hand-writing Triton or CUDA kernels, or relying on nvFuser. `torch.compile` automates it.

## The Three-Layer Stack

### Layer 1: TorchDynamo (Frontend)

TorchDynamo intercepts Python bytecode at the `CPython` level. When `torch.compile` wraps a function, Dynamo installs a bytecode evaluation hook: instead of executing instructions, it **traces** them, building an FX (functional transform) graph of the PyTorch operations encountered.

Critically, Dynamo does this **without requiring the user to write static models**. Control flow, loops, and Python-level side effects are handled through **guards**: assumptions about tensor shapes, dtypes, and Python values that were true at trace time. The compiled artifact is cached and reused as long as guards hold. If a guard fails — for instance, the batch size changes — Dynamo falls back to eager mode for that call and optionally recompiles with the new configuration.

The guard mechanism is what makes `torch.compile` composable with arbitrary research code. It does introduce a cost: guard checking on every call, and a recompilation penalty on guard failure. `torch._dynamo.config.cache_size_limit` (default 8) controls how many compiled variants are stored before Dynamo stops recompiling and falls back permanently to eager.

**Graph breaks** are the failure mode to watch for. Whenever Dynamo encounters something it cannot trace — a `print` statement, data-dependent control flow, a call into a non-`torch` library — it emits a **graph break**: the trace splits into a compiled segment followed by an eager segment. Each break is a compilation boundary and a performance cliff. Diagnose breaks with `torch._dynamo.explain(fn, *args)`, which prints every break point and its cause.

### Layer 2: AOTAutograd

The FX graph from Dynamo represents only the forward pass. AOTAutograd lifts this into a **joint forward-backward graph**: it symbolically traces both the forward computation and the corresponding backward (gradient) computation into a single combined graph.

The key win here is that backward ops become first-class citizens in the compilation unit. Inductor can then fuse backward pointwise ops, hoist redundant computations, and optimize memory layout for the backward pass just as aggressively as the forward. Without AOTAutograd, a compiler would see the forward graph and produce a fused forward kernel, then fall back to eager for the backward — missing half the optimization opportunity in training workloads.

AOTAutograd also handles saved-tensor logic: it tracks which intermediate activations must be saved for the backward pass and which can be recomputed (interacting with `torch.utils.checkpoint`), threading that information into the joint graph.

### Layer 3: Inductor (Backend)

Inductor is the code-generation backend. It receives the joint FX graph and:

1. **Fuses pointwise and reduction ops** — adjacent elementwise operations (bias adds, activations, residuals, normalizations) are merged into single loops (CPU) or single Triton kernels (GPU).
2. **Optimizes memory layout** — Inductor may choose to use channels-last or custom strides to improve coalescing and TensorCore alignment.
3. **Generates Triton kernel source** — for GPU targets, Inductor emits Triton kernel code automatically (see `../triton/`). The generated code is then JIT-compiled by the Triton compiler, which handles thread mapping, shared memory allocation, and software pipelining. `torch.compile` is the production orchestration layer above the Triton DSL.
4. **For CPU**, Inductor generates C++ with OpenMP parallelism.

The fusion result is most visible on memory-bandwidth-bound ops. A fused LayerNorm + GELU + residual add runs in one kernel launch: read input once, compute mean and variance, normalize, apply GELU, add residual, write output once. Three separate kernels would incur three reads and three writes to HBM. On an A100 where HBM bandwidth is 2 TB/s and kernel launch latency is ~5 µs, the saving is both bandwidth (2–4× for these ops) and launch overhead.

## Compilation Modes and CUDA Graphs

`torch.compile` exposes three compilation modes via the `mode` argument:

**`default`**: Standard Inductor optimization — fusion, layout optimization, Triton kernel generation. Balances compile time (10–30s for a typical transformer block on first call) with runtime performance. Appropriate for most training and inference workloads.

**`reduce-overhead`**: Adds CUDA Graphs on top of Inductor. A CUDA Graph records the sequence of GPU commands (kernel launches, memory ops) as a single graph object. On replay, the GPU executes the recorded sequence with no CPU involvement — CPU-side launch overhead, which is ~5–10 µs per kernel, is eliminated entirely.

The additional speedup from CUDA Graphs is **~10–30% for small-batch inference** where launch overhead is a significant fraction of runtime. For large batches where kernels run for milliseconds, the benefit is smaller. The critical constraint is **static shapes**: CUDA Graphs capture pointers and launch parameters at record time. Shape changes (different batch sizes, different sequence lengths) invalidate the graph and force re-recording. In production, this means either bucketing inputs to a small set of shapes or accepting a re-record penalty.

**`max-autotune`**: Runs autotuning over Triton kernel configurations — tile sizes, number of warps, number of pipeline stages — to find the optimal setting for each kernel on the specific hardware. Compile time increases to 2–10 minutes for a large model; runtime is the best achievable. Appropriate for inference deployments where the model is compiled once and served repeatedly.

## Warm-Up and Compilation Overhead

The first forward pass through a `torch.compiled` model incurs the full JIT compilation cost. For a 7B-parameter transformer, this is typically **30–120 seconds** depending on the number of unique FX graph segments, the number of distinct shapes encountered, and the compilation mode. Training loops with dynamic sequence lengths (e.g., variable-length batches from a dataloader) can trigger multiple recompilations in the first few steps.

Practical mitigations:

- **Warm up explicitly** before timing or evaluation: run several batches through the model before measuring throughput.
- **Use static shapes where possible**: fixed batch size, padded sequences, `drop_last=True` in DataLoader.
- **Serialize compiled artifacts**: `torch._dynamo.config.cache_size_limit` and `torch.compiler.reset()` manage the cache; persistent disk cache is available via `TORCHINDUCTOR_CACHE_DIR`.

The compilation cost is a one-time expense per shape signature. In training, it amortizes over hundreds or thousands of steps. In inference serving, the model is compiled once at startup.

## Integration with FSDP

`torch.compile` + FSDP (Fully Sharded Data Parallel) is the standard training configuration for models that don't fit in a single GPU's memory. The critical requirement is `use_orig_params=True` in the FSDP wrapper:

```python
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
model = FSDP(model, use_orig_params=True)
model = torch.compile(model)
```

Without `use_orig_params=True`, FSDP flattens parameters into a single `FlatParameter` per shard, which Dynamo cannot trace correctly — it will emit graph breaks at every FSDP unit boundary. With `use_orig_params=True`, parameters retain their original shapes from Dynamo's perspective, and the FX graph captures the full forward pass cleanly. Compilation then happens **per-rank after sharding**: each rank compiles its own shard of the model, and the collective communication ops (all-gather, reduce-scatter) are treated as opaque ops within the graph. See `../zero-fsdp/` for FSDP fundamentals.

## Integration with FlashAttention

`torch.nn.functional.scaled_dot_product_attention` (SDPA) is registered in PyTorch's dispatcher and auto-dispatches to the FlashAttention backend when:
- the head dimension is supported (typically ≤128)
- the input dtype is FP16 or BF16
- the sequence is not masked in a way that requires the non-flash path

`torch.compile` treats SDPA as a single fused op — it does not attempt to lower it to Triton pointwise ops, because the existing FlashAttention kernel already achieves near-optimal HBM utilization. The compile + FlashAttention combination is thus: Inductor fuses everything around attention (QKV projections, residuals, LayerNorm), and the SDPA op dispatches to FlashAttention inside, giving optimal fusion on both sides of the attention boundary. See `../flash-attention/` for FlashAttention mechanics.

## Operator Fusion: The Core Performance Driver

The dominant win from `torch.compile` in LLM workloads is pointwise operator fusion. Consider the forward pass through a single transformer block, post-attention:

1. Add residual: `x = x + attn_output`
2. LayerNorm: `x = layer_norm(x)`
3. Linear projection: `x = F.linear(x, W1)`
4. GELU activation: `x = F.gelu(x)`
5. Linear projection: `x = F.linear(x, W2)`
6. Add residual: `x = x + ff_output`
7. LayerNorm: `x = layer_norm(x)`

Steps 1–2 and 6–7 are pure pointwise/reduction ops. In eager, each is a separate kernel with separate HBM reads/writes. Inductor fuses steps 1–2 into one kernel and 6–7 into another (and can fuse further with surrounding ops depending on shapes). The memory access pattern is:

$$\text{Bandwidth savings} \approx \frac{N_\text{ops} \cdot T_\text{read/write}}{T_\text{fused read/write} + T_\text{fused compute}}$$

For memory-bound ops where `T_compute` is negligible, fusing N ops reduces HBM traffic by close to N×. Empirically, LayerNorm + residual fusion achieves **2–4× speedup** on the fused ops, and 15–30% end-to-end training speedup on typical LLM training configurations.

## Engineering Tradeoffs

| Approach | Pros | Cons | When to use |
|---|---|---|---|
| `default` mode, static shapes | Moderate compile time (30–60s); reliable fusion; works with FSDP | No CUDA Graph benefit; some ops unfused if shapes dynamic | General training; fine-tuning; development |
| `reduce-overhead` + CUDA Graphs | 10–30% extra speedup from eliminating launch overhead | Requires strictly static shapes; shape change triggers re-record | Inference serving with fixed batch size; benchmark runs |
| `max-autotune` | Best achievable runtime; autotuned Triton configs | 2–10 min compile time; only worth it if model runs for hours | Production inference deployment compiled once; serving throughput critical |
| `torch.compile` + FSDP (`use_orig_params=True`) | Full fusion with sharded training; per-rank compilation | Requires correct FSDP wrapping; debug complexity increases | Training large models (7B+) on multi-GPU setups |
| Eager (no compile) | Zero warm-up; easiest to debug; supports dynamic shapes freely | 15–50% slower; every op is a separate kernel launch | Debugging, shape-unstable research code, very short jobs |
| `torch.compile` + gradient checkpointing | Fused forward and selective recompute; good memory/compute balance | Interaction between AOTAutograd and checkpointing requires care | Memory-constrained training with long sequences |

## Practical Numbers

These figures apply to BF16/FP16 LLM training and inference on A100/H100 hardware:

- **Training speedup (forward + backward)**: 15–30% over eager for dense transformer layers; higher at the low end for models saturated with LayerNorm/activation ops.
- **Inference speedup**: 20–50% over eager depending on batch size; small-batch inference benefits most because launch overhead dominates kernel time.
- **CUDA Graphs additional speedup**: 10–30% on top of fused kernels for small-batch inference (batch 1–8).
- **Memory-bound op fusion**: 2–4× speedup on LayerNorm, GELU, residual add in isolation; the contribution to end-to-end speedup is diluted by matmuls.
- **Compilation warm-up**: 30–120s for a 7B-scale model on first call in `default` mode; 2–10 min in `max-autotune`.

## Limitations and Failure Modes

**Graph breaks** are the primary source of lost performance. A break occurs when Dynamo encounters untraceable Python: `print` statements, `if tensor.item() > threshold` branches, calls into non-PyTorch libraries, `torch.Tensor.numpy()`. Each break produces a compilation boundary: the segments on either side of the break are compiled independently, but the break point itself is always eager. A model with 5 graph breaks is getting far less fusion than a clean graph. Diagnosis: `torch._dynamo.explain(fn, *args)` or `TORCH_COMPILE_DEBUG=1`.

**Data-dependent shapes** (shapes that depend on tensor values, e.g., after `torch.nonzero` or `torch.where` with boolean masks) force either a guard-and-recompile approach or a graph break. Padding inputs to static shapes is usually the right engineering response for serving.

**Compile time** can be prohibitive for exploratory work. A research loop that runs 10 different model configurations is hard to develop with `torch.compile` if each compile takes 2 minutes. The recommended workflow is to develop in eager mode and add `torch.compile` as the last step before benchmarking or production deployment.

## Cross-References

- `../triton/` — Inductor's GPU code generation backend; `torch.compile` is the production orchestration layer above the Triton DSL; every fused kernel Inductor emits is a Triton kernel.
- `../flash-attention/` — SDPA dispatch to FlashAttention; `torch.compile` + FlashAttention is the standard training and inference combination; compile fuses ops around SDPA, FA handles SDPA itself.
- `../zero-fsdp/` — FSDP integration requires `use_orig_params=True`; the compile + FSDP combination is the default for multi-GPU PyTorch training in 2024+.

## References

- [1] Ansel et al. _PyTorch 2: Faster Machine Learning Through Dynamic Python Bytecode Transformation and Graph Compilation._ ASPLOS '24 / arXiv:2311.13608.
- [2] TorchInductor design doc: https://dev-discuss.pytorch.org/t/torchinductor-a-pytorch-native-compiler-with-define-by-run-ir-and-symbolic-shapes/747
- [3] torch.compile tutorial: https://pytorch.org/tutorials/intermediate/torch_compile_tutorial.html
- [4] torch.compile documentation: https://pytorch.org/docs/stable/torch.compiler.html
- [5] CUDA Graphs in PyTorch: https://pytorch.org/blog/accelerating-pytorch-with-cuda-graphs/
