# llama.cpp & GGUF

- **Author / Org**: Georgi Gerganov (ggerganov) and community contributors
- **Repository**: [ggml-org/llama.cpp](https://github.com/ggerganov/llama.cpp)
- **First commit**: March 2023 (Llama 1 release weekend)
- **Related**: [GGUF spec](https://github.com/ggerganov/ggml/blob/master/docs/gguf.md)

## TL;DR

llama.cpp is a pure C/C++ inference engine for transformer LLMs with zero runtime dependencies. It runs on CPU alone, GPU alone (CUDA, Metal, Vulkan, OpenCL), or in a CPU+GPU hybrid mode where layers are split between VRAM and system RAM. It is the substrate under Ollama, LM Studio, Jan, and dozens of local-inference tools. The companion file format, GGUF, encodes model weights, tokenizer, and all configuration in a single self-describing binary — no separate `config.json`, no Python import chain at load time. Together, llama.cpp and GGUF define the dominant stack for running LLMs on edge hardware.

## Context & Motivation

When Meta released Llama 1 in early 2023, the smallest usable model was 7B parameters — roughly 14 GB in FP16. Running that on a laptop required either a large GPU or a working implementation of INT4 quantization for CPUs. No such implementation existed as a standalone binary. Existing inference code (PyTorch, Transformers) dragged in gigabytes of Python dependencies and was not designed for CPU execution.

Gerganov had already written GGML — a C tensor library targeting CPU compute. Within days of the Llama 1 weight leak, he produced a self-contained C implementation that loaded quantized weights and ran inference entirely on CPU. Two engineering choices made it work:

1. **No allocator overhead at inference time**: all weight tensors are memory-mapped from disk; only the activation buffers are malloc'd at startup.
2. **Q4_0 quantization from the start**: 4-bit integer weights reduced a 7B model from 14 GB to ~4 GB, fitting in the RAM of a 2019 MacBook Pro.

This shipped as a single `make && ./main` build. The community adopted it immediately. Over the following 18 months, it grew backends for CUDA, Metal (Apple GPU), Vulkan, and OpenCL; support for every major open model; and a production-grade HTTP server. By 2025, llama.cpp was the inference engine behind more local LLM installations than any other codebase.

## GGUF Format

GGUF (GPT-Generated Unified Format) replaced the earlier GGML format in August 2023 to address backward compatibility problems and extend metadata capacity. The binary layout is:

```
[magic: 4 bytes "GGUF"] [version: u32] [tensor_count: u64] [metadata_kv_count: u64]
[metadata: key-value pairs, variable length]
[tensor info array: name, dimensions, dtype, file offset per tensor]
[padding to alignment boundary]
[tensor data: raw bytes, mmap-able]
```

The metadata key-value section is the defining feature. Standard fields include:

- `general.architecture` — string, e.g., `"llama"`, `"mistral"`, `"qwen2"`
- `general.name` — human-readable model name
- `llama.context_length` — maximum sequence length the model was trained with
- `llama.embedding_length` — hidden dimension
- `llama.attention.head_count`, `llama.attention.head_count_kv` — attention head configuration
- `llama.rope.freq_base` — RoPE base frequency
- `tokenizer.ggml.model` — tokenizer type (`"llama"`, `"gpt2"`, `"bpe"`, etc.)
- `tokenizer.ggml.tokens` — full vocabulary as an array of strings
- `tokenizer.ggml.token_type` — token type flags (normal, BOS, EOS, unknown, control)
- `tokenizer.ggml.scores` — SentencePiece scores if applicable

Because architecture, tokenizer, and all hyperparameters live in the file, a GGUF consumer needs no external configuration. `llama_load_model_from_file()` reads one file and returns a fully configured model object.

**Memory mapping at load time**: the tensor data section is mapped with `mmap(MAP_SHARED)` rather than copied into heap memory. The operating system pages in weight data on first access and can evict cold pages under memory pressure. For a Q4_K_M 70B model (~40 GB on disk), only the layers currently being computed need to be resident. On macOS, this integrates with the unified memory architecture: the Metal backend can import the mmap'd region directly into GPU address space, avoiding a second copy.

**File naming convention**: the community convention encodes quantization type in the filename, e.g., `Meta-Llama-3-70B-Instruct-Q4_K_M.gguf`. The quantization type is also recorded in each tensor's metadata, so the runtime does not rely on the filename.

## Quantization Types

llama.cpp defines a zoo of quantization types, all stored in GGUF. They differ in block structure, number of bits, and scale precision.

**Q4_0**: 4-bit uniform quantization. Weights are partitioned into 32-element blocks; each block has one FP16 scale and 32 INT4 values. Storage: `(16 + 32×0.5) = 32 bytes` per 32 weights = 8 bpw (bits per weight). Fast on all backends; lowest quality in the modern lineup.

**Q4_K_M**: 4-bit k-quant. The "K" family uses super-blocks of 256 elements subdivided into sub-blocks of 32. Two sets of scales: one set per 128 elements at FP16 precision ("M" = medium scale precision), one quantized scale per 32-element sub-block. The extra scale metadata reduces quantization error by adapting to weight distribution within the super-block. At the same 4-bit average, Q4_K_M is consistently 0.1–0.3 perplexity points better than Q4_0 on Llama family models. This is the default recommendation for general use.

```
Q4_K_M super-block (256 weights):
  2x FP16 scales (for two 128-element halves)   =   4 bytes
  8x quantized sub-block scales (4-bit each)    =   4 bytes
  256x 4-bit weights                            = 128 bytes
  Total: 136 bytes / 256 weights ≈ 4.25 bpw
```

**Q5_K_M, Q6_K**: 5-bit and 6-bit variants using the same k-quant super-block structure. Q6_K reaches near-Q8_0 accuracy with ~30% better compression; preferred for quality-sensitive tasks.

**Q8_0**: 8-bit uniform quantization. One FP16 scale per 32-element block. Storage 8.5 bpw. Near-lossless — perplexity gap to FP16 is typically < 0.05. Doubles model file size versus Q4_K_M; used when accuracy is non-negotiable and VRAM/RAM is available.

**IQ2_XXS, IQ3_S (importance-matrix quantization)**: A separate quantization pipeline using a calibration dataset (the "importance matrix" or imatrix). For each weight, the calibration pass computes its contribution to layer activations on sample data. Weights that contribute more receive higher precision; weights that matter less receive lower precision. IQ2_XXS at ~2.06 bpw beats naive Q2 by a large margin and is competitive with Q3_K_M on some benchmarks. This trades calibration compute (an offline pass of several hours) for dramatically better accuracy at extreme compression.

**Rule of thumb**: Q4_K_M for everyday use (good balance, universal support); Q6_K for quality-sensitive applications; Q8_0 when near-lossless is required; IQ3_S or IQ4_XS for constrained devices where Q4 is too large.

## CPU+GPU Hybrid Offload

The `--n-gpu-layers N` flag (short form `-ngl N`) controls how many transformer layers are offloaded to the GPU. The remaining layers run on CPU using system RAM.

**How it works**: llama.cpp builds a static compute graph for the model. During initialization, it assigns the last `N` layers (by convention, top layers in the transformer stack) to the GPU backend and the first `L - N` layers to the CPU backend. At inference time, the forward pass runs CPU layers first, then transfers the activation tensor to GPU memory, runs the GPU layers, and transfers the result back.

**Memory arithmetic** for a 70B Q4_K_M model (~40 GB quantized):
- Each transformer layer is approximately 40 GB / 80 layers = 500 MB in Q4_K_M
- A system with 8 GB VRAM can hold ~14 GPU layers (accounting for KV cache and activation buffers: ~6–7 GB of those 8 GB usable for weights)
- Remaining 66 layers run in system RAM

**Throughput consequence**: tokens generated per second is bounded by the slower of the two paths. CPU decode throughput on a modern laptop (Apple M2 Pro, 12 cores) is typically 2–6 tokens/s for 70B layers in DDR5; GPU throughput for offloaded Metal/CUDA layers is 20–80 tokens/s for the same layers on GPU. The bottleneck is the CPU portion. In practice, a 70B hybrid on an M2 MacBook Pro with 64 GB unified memory achieves 5–12 tokens/s — usable for interactive use, not competitive with server-grade GPU deployment.

The Apple Silicon case is special: unified memory means CPU and Metal GPU share the same physical memory. There is no PCIe copy cost for the layer transfer. A MacBook Pro M3 Max with 128 GB memory can hold an entire 70B Q4_K_M model in unified memory and run all 80 layers on Metal, achieving 20–40 tokens/s.

## Metal Backend (Apple Silicon)

The Metal backend maps GGUF quantized tensors onto Metal compute shaders via `MPSCommandBuffer` and custom Metal kernels for quantized mat-vec multiplication. The critical operation during decode (batch size 1) is a matrix-vector product: a single token's activation vector multiplied against a large weight matrix. At Q4_K_M, each weight requires dequantization before the multiply-accumulate.

Performance on Apple Silicon is governed by memory bandwidth rather than FLOPs. The M3 Pro has ~150 GB/s unified memory bandwidth. A 7B Q4_K_M model is ~4.3 GB; at ~150 GB/s, one forward pass reads the model in ~29 ms, yielding a theoretical ceiling around 34 tokens/s. Observed performance on M3 Pro is typically 28–50 tokens/s for 7B, depending on context length and KV cache size.

The Metal backend avoids explicit memory management for the mmap region: the Metal framework can reference the same physical pages as the CPU mmap, so model weights are never duplicated.

## CUDA Backend

The CUDA backend uses cuBLAS for GEMM operations during prefill (batch size > 1) and custom CUDA kernels for quantized mat-vec during decode. Key kernels:

- `mul_mat_vec_q4_0_q8_1_cuda`: dequantizes Q4_0 weights on-the-fly and performs dot product with Q8_1 activations in a single kernel. Thread block processes one row; warp-level reductions accumulate partial sums.
- `mul_mat_q4_K_cuda`: k-quant variant; reads super-block scale metadata from a separate buffer, dequantizes, then accumulates.

Prefill uses cuBLAS `cublasGemmEx` with FP16 or BF16 weights (loaded from a dequantized copy in VRAM if using CUDA-only mode) or with quantized kernels if using flash-style implementations.

Approximate throughput on RTX 4090 (24 GB VRAM):
- 7B Q4_K_M (fully in VRAM, ~4.3 GB): 130–180 tokens/s decode, 2,500–4,000 tokens/s prefill
- 13B Q4_K_M (~7.6 GB): 80–120 tokens/s decode
- 70B requires multi-GPU or hybrid with system RAM

## KV Cache Memory and Quantization

The KV cache stores key and value tensors for all past tokens and all layers. In FP16, a 70B model with context length 8192 requires approximately:

```
KV cache size = 2 (K+V) × context_len × n_layers × n_kv_heads × head_dim × 2 bytes
= 2 × 8192 × 80 × 8 × 128 × 2 bytes ≈ 26.8 GB
```

This competes directly with weight memory. For long contexts, the KV cache can exceed the model weights. The `--cache-type-k q8_0 --cache-type-v q8_0` flags quantize the KV cache to INT8, halving KV memory to ~13.4 GB for the same context. Accuracy impact is minimal for most tasks; some long-context reasoning tasks show small degradation. `--cache-type-k q4_0` is also available for 4-bit KV cache, reducing further to ~7 GB at greater accuracy risk.

## Serving with llama-server

`llama-server` (formerly `server`) exposes an OpenAI-compatible HTTP API at `/v1/chat/completions`, `/v1/completions`, and `/v1/embeddings`. It is the backend that Ollama calls when processing requests.

Continuous batching was added in mid-2024: the server maintains a slot pool; new requests enter available slots without waiting for in-flight requests to complete. Under this model, a single llama-server instance can handle multiple simultaneous users at the cost of increased memory (each slot maintains its own KV cache fragment).

`llama-server` does not implement dynamic block-level KV sharing (as vLLM does with PagedAttention). Each slot gets a statically allocated KV buffer. This simplifies the implementation but means multi-user throughput does not scale as efficiently as vLLM at high concurrency.

## Engineering Tradeoffs

| Dimension | llama.cpp | vLLM | Ollama | TGI |
|---|---|---|---|---|
| Hardware requirement | CPU, any GPU, or hybrid; no CUDA required | CUDA primary (ROCm experimental) | Wraps llama.cpp or llama backends; Mac/Linux/Win | CUDA primary |
| Quantization support | GGUF native: Q4_0, Q4_K_M, Q6_K, Q8_0, IQ2_XXS, FP16 | GPTQ, AWQ, FP8, BF16; no GGUF | Inherits llama.cpp quants | GPTQ, AWQ, BNB; no GGUF |
| Single-user decode throughput | High for local hardware; memory-bandwidth limited | Lower than llama.cpp at single stream | Same as llama.cpp | Comparable to vLLM |
| Multi-user / high-concurrency | Continuous batching added 2024; no paged KV allocation | PagedAttention; best-in-class multi-user | Queues requests to llama-server | PagedAttention (v2); good multi-user |
| Deployment complexity | Single binary + one file; `./llama-server -m model.gguf` | `pip install vllm`; CUDA driver required | Docker or installer; simplest UI | Docker; heavier stack |
| Model coverage | All major open models; fastest to support new architectures | Good; lags llama.cpp on newest models | Same as llama.cpp via backend | Good; lags on newest models |

## Cross-References

- `../../guides/on-prem-llm-deployment/` — Ollama wraps llama.cpp and is the recommended entry point for on-premise deployment; that guide covers operational concerns.
- `../weight-quantization/` — GPTQ and AWQ are the GPU-server quantization methods; GGUF k-quants are the CPU/hybrid counterpart. The same AWQ insight (protecting salient channels) influenced IQ-family quants.
- `../../apple/afm/` — MLX is Apple's alternative inference framework for Apple Silicon; llama.cpp Metal also targets the same hardware. MLX prioritizes Python ergonomics; llama.cpp prioritizes zero-dependency portability.
- `../paged-attention/` — vLLM implements PagedAttention for server-grade multi-user inference. llama.cpp targets local/edge single-user or low-concurrency scenarios; for >10 concurrent users on GPU, vLLM is the better choice.

## Commentary

llama.cpp succeeded because it solved the right problem at the right time with the right constraints: self-contained, minimal dependencies, ran on hardware people actually had. The GGUF format took that insight further — a single file that needs no ecosystem to open. This design philosophy is the reason llama.cpp is how most people experience LLMs privately.

The technical ceiling is now visible: without paged KV allocation, llama.cpp does not scale to high-concurrency serving. The Metal backend approaches the theoretical memory-bandwidth ceiling on Apple Silicon, leaving limited headroom for algorithmic improvements. The CPU path saturates DDR bandwidth at large model sizes. These are not bugs — they are the natural consequence of optimizing for local execution rather than server throughput.

What llama.cpp demonstrates for the field is that **the right abstraction boundary** (one binary, one file, hardware-adaptive quants) can outcompete technically superior systems in adoption. For engineers building local or edge inference, the lesson is: quantization format and deployment friction matter more than peak throughput numbers from data center benchmarks.

## References

- [1] Gerganov et al. llama.cpp repository. https://github.com/ggerganov/llama.cpp
- [2] GGUF format specification. https://github.com/ggerganov/ggml/blob/master/docs/gguf.md
- [3] Gerganov. GGML tensor library. https://github.com/ggerganov/ggml
- [4] Frantar et al. _GPTQ._ ICLR '23 / arXiv:2210.17323. (Quantization context)
- [5] Lin et al. _AWQ._ MLSys '24 / arXiv:2306.00978. (Activation-aware scaling; influenced IQ quants)
- [6] llama.cpp k-quants PR. https://github.com/ggerganov/llama.cpp/pull/1684 (Original k-quant introduction by ikawrakow)
- [7] Ollama. https://github.com/ollama/ollama (llama.cpp-based local serving)
