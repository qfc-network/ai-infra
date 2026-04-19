# LMDeploy / TurboMind

- **Author / Org**: Shanghai AI Lab (OpenMMLab)
- **Repository**: [InternLM/lmdeploy](https://github.com/InternLM/lmdeploy)
- **First public release**: 2023-07
- **Backend options**: TurboMind (C++/CUDA, default), PyTorch (Python)

## TL;DR

LMDeploy is Shanghai AI Lab's production LLM serving engine, built alongside InternLM and Qwen. Its default TurboMind backend is a custom C++/CUDA engine with hand-tuned kernels for GQA attention, W4A16 AWQ quantization, paged KV cache, and — uniquely among open-source serving engines at its time of release — efficient support for DeepSeek's Multi-head Latent Attention (MLA) with the compressed KV cache layout. W4A16 AWQ on TurboMind delivers approximately 2× single-stream decode throughput versus BF16 on the same GPU, and benchmarks on H100 show ~5,500 tokens/s for Llama 3 70B at batch 32 AWQ versus ~3,500 tokens/s BF16 — outperforming vLLM on the same hardware. For Chinese-lab models (InternLM, Qwen, DeepSeek), LMDeploy is the first-class serving solution.

## Context & Motivation

By mid-2023 the open-source serving landscape had TGI (HuggingFace), nascent vLLM (Berkeley), and FasterTransformer (NVIDIA). Each had gaps: TGI lacked quantization-optimized kernels; vLLM had PagedAttention but no weight quantization at the kernel level; FasterTransformer required model conversion and was NVIDIA-internal in origin.

Shanghai AI Lab had specific constraints. InternLM models were being trained and deployed internally; Qwen (Alibaba, same city and research overlap) was emerging as a major model family. The lab needed a serving engine that:
1. Had first-class W4A16 AWQ kernel support — the primary quantization method for their deployment targets.
2. Ran well on both 80 GB H100 clusters (internal serving) and smaller cards (external deployment by users).
3. Supported diverse architectures quickly — InternLM, Qwen, later DeepSeek — without months of per-model engineering.
4. Eventually supported MLA, which is structurally incompatible with standard multi-head attention KV cache layouts.

TurboMind was designed to satisfy these constraints. The PyTorch backend was added for model families where custom CUDA kernel development hadn't yet landed.

## TurboMind Backend Architecture

TurboMind is a C++ inference runtime with custom CUDA kernels, a paged KV cache manager, and a continuous batching scheduler. The execution path for a typical request:

1. **Request enters the scheduler**: the scheduler assigns available KV blocks from the physical pool and creates a sequence slot.
2. **Prefill pass**: token IDs are embedded; the forward pass runs through all layers. For W4A16 models, weight tensors are stored in INT4 and dequantized inside the GEMM kernel. For BF16 models, cuBLAS handles the dense matmul.
3. **Decode loop**: at each step, a single-token activation runs through the model. The key operation is a quantized matrix-vector multiply (GEMV). TurboMind's W4A16 GEMV kernel fuses dequantization and dot product; activations are kept in FP16/BF16 throughout.
4. **KV cache update**: after attention, new K and V tensors are written to the assigned physical blocks.
5. **Output**: sampled token is returned; the slot's block table is extended for the next step.

The scheduler is iteration-level (continuous batching): it checks for completed sequences and new arrivals at every decode step, not at batch boundaries.

## W4A16 AWQ Throughput Advantage

Weight-only INT4 quantization (W4A16) makes single-stream decode materially faster because decode is memory-bandwidth-bound, not compute-bound.

At batch size 1, each decode step loads the entire model's weights once to compute one output token. The performance equation is:

```
tokens/s ≈ HBM_bandwidth / model_size_bytes

BF16 70B: 140 GB / (3.35 TB/s on H100 SXM) ≈ 24 ms/token → ~42 tokens/s
W4A16 70B: ~37 GB / (3.35 TB/s) ≈ 11 ms/token → ~90 tokens/s
```

In practice, TurboMind achieves closer to 80–100 tokens/s for 70B W4A16 decode on H100 at batch size 1, versus 35–45 tokens/s for BF16 — roughly a 2× ratio, consistent with the theoretical prediction. The reason W4A16 beats W4A16 naive implementations is the custom GEMV kernel: standard CUDA INT4 matmul is designed for batched GEMM (prefill), not for the streaming GEMV pattern of decode. TurboMind's kernel reads 128-bit aligned INT4 tiles, unpacks 8 values at a time using bit manipulation, and performs fused multiply-accumulate with the FP16 activations. Warp-level reductions accumulate partial sums without global memory round-trips.

At larger batches, the advantage narrows. At batch size 32, both BF16 and W4A16 become increasingly compute-bound (the matmul is no longer memory-bound), and the ratio shrinks to approximately 1.3–1.5×. TurboMind's H100 benchmark numbers for Llama 3 70B:

- **BF16, batch 32**: ~3,500 tokens/s total throughput
- **W4A16 AWQ, batch 32**: ~5,500 tokens/s total throughput
- **vLLM BF16, batch 32, same H100**: ~3,200 tokens/s

The vLLM comparison is not exactly apples-to-apples (different scheduling and overhead profiles), but the direction is consistent across multiple third-party benchmarks as of early 2025.

## MLA Support

DeepSeek V2 introduced Multi-head Latent Attention (MLA), which compresses the KV cache by projecting keys and values into a lower-dimensional latent space. Standard multi-head attention stores `n_heads × head_dim` K and V tensors per token per layer; MLA stores a single compressed latent vector of dimension `kv_lora_rank` (e.g., 512 for DeepSeek V2) plus a small rope-component key.

The consequence for serving engines: the physical KV block layout must change. A system that stores KV as `[n_heads, head_dim]` per token cannot directly store MLA's compressed `[kv_lora_rank]` latent. vLLM initially handled MLA by expanding it back to full K/V at the attention kernel boundary, preserving the large KV cache size. TurboMind implemented native MLA KV layout from the start, storing the latent vectors directly in KV blocks.

The memory saving is substantial. For DeepSeek V2 (236B MoE), with `kv_lora_rank = 512` and `n_kv_heads × head_dim = 128 × 128 = 16384`:

```
Standard KV per token per layer: 16384 × 2 × 2 bytes = 65,536 bytes
MLA KV per token per layer:        512 × 2 × 2 bytes =  2,048 bytes
Compression ratio: ~32×
```

At context length 8192 and 61 attention layers (DeepSeek V2), the difference is 32 GB versus 1 GB for the KV cache alone on a single stream. This is what makes serving MoE models with MLA viable on a reasonable GPU budget.

LMDeploy was one of the first production-grade serving engines to implement this MLA-native path, which gave it an early advantage for DeepSeek deployment. The implementation fuses the up-projection of the latent into key-head and value-head space inside the attention kernel, avoiding a separate projection step in Python.

## Quantization Pipeline

LMDeploy provides an integrated quantization CLI:

```bash
# W4A16 AWQ calibration and quantization
lmdeploy lite auto_awq \
    model_path \
    --calib-dataset ptb \
    --calib-samples 128 \
    --calib-seqlen 2048 \
    --work-dir quantized_model_path

# W8A8 SmoothQuant
lmdeploy lite smooth_quant \
    model_path \
    --calib-dataset ptb \
    --calib-samples 128 \
    --work-dir quantized_model_path
```

The AWQ path uses the same channel-scaling calibration as AutoAWQ (protecting salient weight channels by scaling them before quantization, absorbing the inverse scale into preceding layers), but the resulting quantized weights are stored in LMDeploy's own format and consumed by TurboMind's W4A16 kernels rather than AutoAWQ's kernels. The kernel implementations differ; TurboMind's GEMV kernel is optimized for the serving GEMV pattern while AutoAWQ targets GEMM.

FP8 KV cache is also available: `--quant-policy 8` enables FP8 KV quantization, halving KV memory compared to BF16 with minimal accuracy impact on standard benchmarks. Combined with W4A16 weights, a 70B model can run at a KV cache footprint roughly equivalent to a 35B BF16 model, enabling longer effective context lengths within the same GPU budget.

## Model Coverage

TurboMind provides optimized kernel paths for:

- **Llama family** (Llama 2/3, Mistral, Mixtral MoE): standard GQA attention, dense FFN or MoE routing
- **InternLM 1/2/3**: first-class support as the origin model family; architecture co-evolved with TurboMind
- **Qwen 1.5/2/2.5/3**: first-class support; Qwen and Shanghai AI Lab share research overlap; LMDeploy is the reference serving engine in Qwen's official documentation
- **DeepSeek V1/V2/V3/R1**: MLA-native path; MoE routing support; among the first open-source engines to support DeepSeek V2 efficiently
- **Vision-language models**: InternVL (InternLM + ViT), LLaVA (CLIP + Llama); vision encoder runs in PyTorch, text decoder in TurboMind

The PyTorch backend covers models not yet ported to TurboMind kernels, at lower throughput.

## Deployment

Launching an OpenAI-compatible server:

```bash
lmdeploy serve api_server \
    model_path \
    --server-port 23333 \
    --tp 2 \
    --cache-max-entry-count 0.8 \
    --quant-policy 4
```

`--tp N` enables tensor parallelism across N GPUs. `--cache-max-entry-count` sets the fraction of GPU memory reserved for KV cache pages. `--quant-policy 4` selects W4A16 AWQ (if the model has been AWQ-quantized).

Docker image: `openmmlab/lmdeploy:latest` provides CUDA 12.1 + TurboMind binaries. Kubernetes Helm charts are available in the repository's `kubernetes/` directory. LMDeploy integrates with OpenCompass for post-quantization accuracy evaluation — running the eval harness against the quantized serving endpoint rather than a local model file.

Approximate server-grade throughput (H100 80GB SXM, as of early 2025):
- Llama 3 8B BF16: ~12,000 tokens/s at batch 32
- Llama 3 8B W4A16: ~18,000 tokens/s at batch 32
- Llama 3 70B BF16: ~3,500 tokens/s at batch 32
- Llama 3 70B W4A16: ~5,500 tokens/s at batch 32
- DeepSeek V2 236B (2× H100, MLA native): ~2,800 tokens/s at batch 32

## Engineering Tradeoffs

| Dimension | LMDeploy / TurboMind | vLLM | TGI | SGLang |
|---|---|---|---|---|
| W4A16 AWQ kernel quality | Custom GEMV kernel; best single-stream W4A16 throughput | AWQ via Marlin / custom kernels; competitive at large batches | AWQ supported; kernel quality varies by version | AWQ via Marlin; competitive |
| MLA support | Native MLA KV layout; full compression benefit | Initially expanded K/V; native MLA added later | No native MLA as of 2024 | MLA support added for DeepSeek |
| Paged KV cache | Yes; block-based physical pool | Yes; PagedAttention; most mature implementation | Yes (v2); comparable | Yes; radix tree on top for prefix sharing |
| Throughput (BF16, large batch) | ~3,500 tokens/s 70B H100 | ~3,200 tokens/s 70B H100 | Comparable to vLLM | Comparable or slightly higher with RadixAttention |
| Chinese-lab model support | First-class: InternLM, Qwen, DeepSeek | Good; lags on newest Chinese models | Partial | Growing; good DeepSeek support |
| Community size | Medium; primarily Chinese-speaking contributors | Largest open-source LLM serving community | Large; HuggingFace backing | Medium; UC Berkeley origin |
| Deployment complexity | Docker + single CLI command; good docs in English and Chinese | Good Python tooling; extensive documentation | Good; HuggingFace Hub integration | Python-first; slightly more setup |

## Cross-References

- `../paged-attention/` — LMDeploy implements the same physical-block KV cache concept as vLLM. The block table, free-block pool, and copy-on-write mechanics are conceptually identical; the implementation is independent.
- `../weight-quantization/` — AWQ is the primary quantization method in LMDeploy's production pipeline. The calibration algorithm is the same as AutoAWQ; the serving kernels are LMDeploy-specific.
- `../smoothquant/` — W8A8 SmoothQuant is also available via `lmdeploy lite smooth_quant`, suited for scenarios where INT4 accuracy loss is unacceptable and memory budget allows 8-bit weights.
- `../../deepseek/mla/` — MLA-native KV layout is the key differentiator for DeepSeek deployment; the compression mechanism and its serving implications are detailed there.
- `../tgi/` — TGI is the peer European-lab serving engine; comparison is covered there.
- `../../qwen/qwen3/` — Qwen models are first-class in LMDeploy; Qwen's official deployment documentation references LMDeploy as the primary serving solution.

## Commentary

LMDeploy's technical identity is defined by two choices: W4A16 AWQ kernel quality and MLA-native support. Both are consequences of the same organizational fact — the engine was built by and for the teams shipping the models. When your users are the engineers who wrote the model architecture, you get MLA support on day zero and AWQ kernels that are tuned for the access patterns of your specific model shapes.

The broader lesson is about the value of **vertical integration between training and serving**. vLLM is a better general-purpose serving engine in some dimensions (community, ecosystem, multi-modal coverage, tooling). LMDeploy is better on the specific problem of running Chinese-lab models efficiently, especially under weight quantization. Neither engine is universally dominant — the tradeoff is real and depends on your model family.

For practitioners: if you are deploying InternLM, Qwen, or DeepSeek at scale, LMDeploy is the correct starting point. The W4A16 throughput advantage alone (~2× at single stream, ~1.5× at batch 32) justifies the slightly smaller community relative to vLLM. If you are running a mixed fleet of model families with complex scheduling requirements and high concurrency, vLLM's more mature paged allocator and broader tooling may be the better choice. The decision is not ideological — benchmark both on your specific traffic pattern.

## References

- [1] InternLM team. LMDeploy repository. https://github.com/InternLM/lmdeploy
- [2] Lin et al. _AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration._ MLSys '24 / arXiv:2306.00978.
- [3] Xiao et al. _SmoothQuant: Accurate and Efficient Post-Training Quantization for Large Language Models._ ICML '23 / arXiv:2211.10438.
- [4] Kwon et al. _Efficient Memory Management for Large Language Model Serving with PagedAttention._ SOSP '23 / arXiv:2309.06180. (PagedAttention concept)
- [5] DeepSeek-AI. _DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts Language Model._ arXiv:2405.04434. (MLA specification)
- [6] Zheng et al. _SGLang: Efficient Execution of Structured Language Model Programs._ arXiv:2312.07104.
- [7] OpenCompass. https://github.com/open-compass/opencompass (Evaluation integration)
