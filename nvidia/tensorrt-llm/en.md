# TensorRT-LLM — NVIDIA's Production LLM Inference Engine

- **Authors / Org**: NVIDIA TensorRT Team
- **Published**: 2023-09 (open-source release)
- **Links**: [code](https://github.com/NVIDIA/TensorRT-LLM) · [docs](https://nvidia.github.io/TensorRT-LLM/) · [blog](https://developer.nvidia.com/blog/accelerating-inference-with-tensorrt-llm/)

## TL;DR

TensorRT-LLM is NVIDIA's open-source library for optimizing and serving LLMs on NVIDIA GPUs. It compiles model definitions into TensorRT engines with fused kernels, in-flight batching (continuous batching), paged KV cache, multi-GPU tensor/pipeline parallelism, and INT4/INT8/FP8 quantization — all exposed through a Python API that wraps CUTLASS and custom CUDA kernels. It is the inference backend for NVIDIA Triton Inference Server and the reference implementation for H100 LLM throughput benchmarks.

## Context & Motivation

By mid-2023 the LLM serving landscape had fragmented into a set of tools with overlapping but incomplete coverage:

- **vLLM**: excellent throughput via PagedAttention, but Python-heavy scheduler, limited quantization support, and no engine compilation.
- **FasterTransformer** (NVIDIA's own predecessor): fast for its time but research-grade, difficult to extend, and tied tightly to specific model architectures.
- **HuggingFace Transformers + Accelerate**: maximally flexible but 3–5× slower than hand-optimized engines due to lack of kernel fusion and eager PyTorch execution.

Labs running at scale — cloud providers, NVIDIA partners, frontier model deployments — needed four things simultaneously: (1) peak hardware utilization on A100/H100, (2) production-grade reliability with predictable tail latency, (3) comprehensive quantization support spanning INT4, INT8, FP8, and SmoothQuant, and (4) straightforward multi-GPU scaling from a single host to multi-node deployments. None of the existing tools delivered all four at once.

FasterTransformer was already inside NVIDIA, but it had grown organically around specific model architectures rather than being designed as a composable inference platform. Rather than continuing to patch it, NVIDIA built TensorRT-LLM from scratch — open-sourcing it in September 2023 to serve as the reference production inference stack for NVIDIA hardware. It is also the basis for the serving tier in NVIDIA NIM (formerly "AI Foundations") and the recommended backend for Triton Inference Server's LLM use cases.

## Core Method

### Compilation Pipeline

TensorRT-LLM operates in three distinct phases:

**1. Model definition.** The user describes the model in Python using `tensorrt_llm.Module` — a TensorRT-specific module API that resembles PyTorch's `nn.Module` but builds a TensorRT functional graph of operations rather than executing them eagerly. Weight loading from HuggingFace checkpoints is handled by per-model conversion scripts; from the graph's perspective, weights are constants that get embedded in the engine.

**2. Engine build.** The `trtllm-build` command ingests the model graph, applies TensorRT's optimizer (layer fusion, precision selection, CUTLASS tile-config search), and serializes the result to a `.engine` file. This is where all the expensive optimization happens: TensorRT profiles every eligible kernel at every relevant input shape, selects the fastest tile configuration for each GEMM, fuses adjacent elementwise ops into single CUDA kernels, and inlines quantization/dequantization into the surrounding matmul. Build time ranges from a few minutes for a 7B model to 30–60 minutes for a 70B model across four GPUs.

**3. Runtime.** A C++ runtime loads the serialized engine and drives inference. It owns the scheduler, the KV cache allocator, and the token streaming logic. Python bindings via `tensorrt_llm.runtime` expose a `GenerationSession` interface. For production deployments the C++ runtime is typically accessed through `tritonserver` with the `tensorrtllm_backend` plugin.

### In-Flight Batching

TensorRT-LLM implements continuous batching through a custom C++ scheduler rather than adopting Orca's Python-based iteration-level scheduler. At each decode step the scheduler inspects the completion status of every active sequence, evicts finished sequences, and promotes waiting prefill requests into the active batch — all within a single synchronous call inside the C++ runtime. The batch seen by the TensorRT engine therefore changes shape between steps, which TensorRT handles via dynamic shapes configured at engine build time (`--max_batch_size`, `--max_input_len`, `--max_output_len`).

### Paged KV Cache

The runtime maintains a block pool for KV cache. Each block holds a fixed number of tokens (configurable, typically 16–64) worth of key and value tensors for all layers and heads. When a new request arrives, blocks are allocated from the pool on demand; when a request completes, its blocks are immediately returned. The attention kernel receives a block-pointer table rather than contiguous KV tensors, enabling the same PagedAttention memory efficiency as vLLM. Cross-request KV sharing (prefix caching) is also supported: if two requests share a common system prompt, their prefix blocks can be aliased.

### Fused Attention Kernel

The core attention implementation is a custom CUDA kernel built on CUTLASS's MMA (Matrix Multiply-Accumulate) primitives, or optionally FMHA (Fused Multi-Head Attention from APEX). A single kernel call handles a full variable-length batch using `cuSeqLens` to describe per-sequence lengths — no padding wasted on short sequences. On Hopper (H100), TensorRT-LLM switches to a Hopper-native kernel that uses WGMMA and TMA to reach near-FlashAttention-3 throughput.

The effective attention computation for a single head follows the standard scaled dot-product form, but the kernel fuses softmax scaling, masking, and the dropout stochastic mask (if enabled) into the same pass over the KV tiles, never writing the full attention weight matrix to HBM:

```
# Pseudocode: per Q-tile, sweep K/V tiles
m_i = -inf; l_i = 0; O_i = 0
for each K/V tile j:
    S_ij = Q_i @ K_j^T / sqrt(d)    # on-chip, tile-sized
    m_ij = rowmax(S_ij)
    P_ij = exp(S_ij - m_ij)
    # online softmax correction
    m_new = max(m_i, m_ij)
    l_i = exp(m_i - m_new) * l_i + exp(m_ij - m_new) * rowsum(P_ij)
    O_i = exp(m_i - m_new) * O_i + P_ij @ V_j
    m_i = m_new
O_i = O_i / l_i                     # final rescale
```

This is identical in structure to FlashAttention; the CUTLASS implementation achieves the same IO-awareness but through NVIDIA's own kernel infrastructure rather than Dao-AILab's.

### Quantization

Quantization is configured at engine build time via `trtllm-build` flags and a separate calibration step:

- **INT4 weight-only** (GPTQ-style or AWQ-style): weights are stored in INT4, dequantized to FP16/BF16 inline inside the matmul kernel using CUTLASS's mixed-precision GEMM. Group quantization (group size 128) is standard for accuracy. No activation quantization; the compute is still FP16/BF16.
- **INT8 weight + activation (SmoothQuant)**: both weights and activations are quantized to INT8; the matmul runs on INT8 tensor cores (IMMA). Smooth-quantization migration factors absorb the per-channel activation scale into the weight, enabling per-tensor or per-channel activation quantization with minimal accuracy loss. Available on Ampere and later.
- **FP8 (Hopper only)**: weights and activations stored and computed in E4M3 FP8. TensorRT-LLM is the reference implementation for Hopper FP8 LLM inference. Scaling factors are per-tensor or per-channel; they are absorbed into the engine at build time.

Dequantization and scaling are fused into the surrounding matmul kernels — there is no separate "dequant" pass that reads and writes weights back to HBM.

### Tensor and Pipeline Parallelism

Tensor parallelism shards QKV projections and FFN weight matrices across GPUs along the head or hidden dimension. After each matmul, an NCCL AllReduce synchronizes partial sums. The tensor-parallel degree is specified at build time; the engine is built once per TP configuration. Pipeline parallelism assigns contiguous layer ranges to separate GPUs (or nodes), with micro-batch pipelining to hide inter-stage activation transfer latency. TP and PP can be composed: a 4-node × 8-GPU cluster can run TP=8 within a node and PP=4 across nodes, giving a 32-GPU logical device.

### Speculative Decoding

TensorRT-LLM has built-in support for draft+verify speculative decoding. The draft model (e.g., a 7B model) runs as a separate TensorRT-LLM engine and produces `k` draft tokens per step. The target model verifies all `k` tokens in a single forward pass (a batch operation over the draft proposals). The rejection-sampling step runs on CPU or as a small CUDA kernel. The net effect is that wall-clock latency scales with the number of accepted tokens per step rather than with `k`, giving 2–3× speedup for tasks with high draft-acceptance rates (code generation, structured output).

### Plugin System

Custom CUDA operations — attention, layer norm, quantized matmul, rotary embedding — are registered as TensorRT plugins: C++ classes implementing `IPluginV2DynamicExt`. This interface lets TensorRT's graph optimizer treat custom ops as first-class graph nodes, fusing them with adjacent standard ops when profitable. The plugin registration pattern is:

```cpp
class FMHAPlugin : public nvinfer1::IPluginV2DynamicExt {
    nvinfer1::DimsExprs getOutputDimensions(...) override;
    void enqueue(const PluginTensorDesc* inputDesc,
                 const PluginTensorDesc* outputDesc,
                 const void* const* inputs,
                 void* const* outputs,
                 void* workspace,
                 cudaStream_t stream) override;
    // ... serialize / deserialize / clone
};
```

The plugin model is TensorRT's extensibility mechanism but comes with a cost: plugin API versions track TensorRT releases and break backward compatibility across major versions.

## Engineering Tradeoffs

| Decision | Gained | Gave up |
|---|---|---|
| Ahead-of-time engine compilation | Peak kernel selection via exhaustive profiling; zero JIT overhead at serving time; reproducible latency | Build times of 10–60 min for large models; engine is not portable across GPU generations (A100 engine won't run on H100); any model change requires a full rebuild |
| C++ runtime vs. Python (vLLM) | Lower per-token latency; better tail latency; production-grade scheduling without GIL contention | Harder to customize; Python API is a thin wrapper; debugging scheduling logic or memory management requires C++ expertise |
| TensorRT plugin system | Custom ops fused as first-class graph nodes; full fusion with surrounding elementwise layers | Plugin API is verbose and has breaking changes between TensorRT major versions; porting kernels across TRT versions is manual work |
| Compile-time quantization (INT4/INT8/FP8) | Single framework covers full stack; quantization is zero-overhead at runtime; consistent throughput numbers | Calibration must be done upfront (offline); switching quantization scheme requires a new engine build; less flexible than vLLM's runtime quantization APIs |
| NVIDIA-only hardware support | Direct access to WGMMA, TMA, FP8 tensor cores, NCCL; H100 reference performance; tight driver/compiler co-design | No AMD/CPU/Intel GPU support; complete vendor lock-in; customers on non-NVIDIA hardware must use a different stack entirely |
| Dynamic shape engine (vs. static) | Single engine handles variable batch/sequence lengths within configured max bounds | Shape profiling at build time must cover the expected range; shapes far from profiled optima can run suboptimally |

## Experiments & Results

**LLaMA-2-70B, 4× A100-80GB, FP16 (batch 64):** TensorRT-LLM achieves approximately 2,100 tokens/sec generation throughput versus approximately 1,400 tokens/sec for vLLM — a 1.5× advantage attributable primarily to tighter kernel fusion, ahead-of-time GEMM tuning, and lower Python overhead in the scheduler.

**LLaMA-2-70B, 4× H100-80GB, FP8 (batch 64):** Approximately 4,500 tokens/sec — near-theoretical throughput given H100's FP8 tensor core peak. This is the clearest performance advantage of TensorRT-LLM: it is the only open-source stack that fully exploits Hopper FP8 end-to-end.

**INT4 weight-only quantization (AWQ):** 1.9× memory reduction with less than 1% quality loss on standard benchmarks (MMLU, HellaSwag), enabling LLaMA-2-70B on 2× A100 instead of 4×. This is the most common production configuration for cost-sensitive deployments.

**Speculative decoding (7B draft + 70B target):** 2.2× latency reduction on HumanEval code generation tasks, where draft-acceptance rates are high (60–75%) because code is locally predictable. Latency improvement degrades gracefully to 1.0× on tasks with low acceptance rate (open-ended creative text).

**Latency at batch=1:** TensorRT-LLM's compiled engine achieves approximately 30–35 ms time-to-first-token for LLaMA-2-7B on A100, versus approximately 45–50 ms for vLLM in the same configuration — relevant for interactive applications where single-request latency matters more than aggregate throughput.

## Reproducibility Notes

**Code**: github.com/NVIDIA/TensorRT-LLM (Apache 2.0).

**Docker**: `nvcr.io/nvidia/tensorrt-llm` from NVIDIA NGC. This is the recommended starting point; building from source requires careful CUDA/TensorRT version matching.

**Build workflow**:
```bash
# Convert HuggingFace checkpoint
python convert_checkpoint.py --model_dir <hf_dir> --output_dir <ckpt_dir> \
    --tp_size 4 --dtype float16

# Build TensorRT engine
trtllm-build --checkpoint_dir <ckpt_dir> \
    --output_dir <engine_dir> \
    --max_batch_size 64 --max_input_len 2048 --max_output_len 512 \
    --gemm_plugin float16 --gpt_attention_plugin float16

# Serve via Python runtime
python run.py --engine_dir <engine_dir> --tokenizer_dir <hf_dir> \
    --input_text "Hello, world"
```

**Triton deployment**: integrates with `tritonserver` using the `tensorrtllm_backend` repository. The backend handles gRPC protocol, dynamic batching at the server level, and health checking.

**Requirements**: TensorRT 9.x or 10.x, CUDA 12.x, Python 3.10+. INT8 SmoothQuant requires Ampere or newer. FP8 requires Hopper (H100, H200) or Ada Lovelace (RTX 40xx for consumer use).

**Model support** (as of 2024): LLaMA 1/2/3, Mistral, Mixtral (MoE), Falcon, GPT-NeoX, GPT-J, Gemma, Phi-1/2/3, Qwen 1/2, Baichuan, ChatGLM, BLOOM, OPT, Whisper.

**Known friction points**: engine build is sensitive to TensorRT version; engines built with TRT 9.x are not loadable by TRT 10.x runtimes. The model conversion scripts in `examples/` are the canonical reference and evolve quickly — pin to a specific commit for production builds.

## Commentary

TensorRT-LLM represents NVIDIA's deliberate answer to the fragmented inference ecosystem: a single, optimized, open-source stack that unifies compilation, quantization, batching, and multi-GPU serving under one framework. The central design bet is that ahead-of-time compilation — accepting long build times in exchange for peak runtime performance — is the right tradeoff for production serving, where the engine runs for weeks and the build cost is amortized across millions of requests.

This contrasts sharply with vLLM's approach: keep everything in Python, accept modest overhead, and maximize iteration speed and ecosystem breadth. In practice both systems coexist in production: hyperscalers use TensorRT-LLM for high-QPS serving tiers and vLLM for research clusters and lower-traffic endpoints. The choice mirrors the long-standing CUTLASS vs. Triton split — CUTLASS/TRT-LLM for production peak performance; Triton/vLLM for research flexibility.

The FP8 path on H100 is where TensorRT-LLM's advantage is clearest and most durable. FP8 requires Hopper-specific hardware (the FP8 GEMM units are distinct from the INT8 tensor cores on Ampere), and TensorRT-LLM is the reference implementation with the most mature calibration tooling, the most complete model coverage, and the tightest driver co-design. Any team deploying LLMs on H100 at scale should treat TensorRT-LLM as the performance baseline that alternative stacks must beat.

The main long-term question is whether ahead-of-time compilation remains viable as model diversity and update cadence increase. Fine-tuned model variants that differ only in weights can reuse the same engine (weights are loaded separately from the compiled graph), which helps significantly. But architecture changes — new attention variants, MoE routing, SSM layers — each require new plugin implementations and engine rebuilds. The NVIDIA team moves quickly, but the plugin-per-architecture tax is real and will grow as the model zoo expands.

## References

- [1] NVIDIA 2023. _TensorRT-LLM: A TensorRT Toolbox for Optimized Large Language Model Inference._ github.com/NVIDIA/TensorRT-LLM.
- [2] NVIDIA 2022. _FasterTransformer._ github.com/NVIDIA/FasterTransformer (predecessor, now archived).
- [3] Kwon et al. 2023. _Efficient Memory Management for Large Language Model Serving with PagedAttention._ arXiv:2309.06180.
- [4] Yu et al. 2022. _Orca: A Distributed Serving System for Transformer-Based Generative Models._ OSDI 2022.
- [5] Lin et al. 2023. _AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration._ arXiv:2306.00978.
- [6] Xiao et al. 2023. _SmoothQuant: Accurate and Efficient Post-Training Quantization for Large Language Models._ ICML 2023.
- [7] Dao et al. 2022. _FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness._ arXiv:2205.14135.
- [8] NVIDIA 2024. _TensorRT-LLM Documentation._ nvidia.github.io/TensorRT-LLM.
