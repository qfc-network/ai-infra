# Apple Foundation Models (AFM) + Private Cloud Compute

- **Authors / Org**: Apple
- **Published**: 2024-06 (Apple Intelligence announced at WWDC 2024); PCC security paper: 2024-10
- **Links**: [Apple Intelligence overview](https://www.apple.com/apple-intelligence/) · [Private Cloud Compute security paper](https://security.apple.com/blog/private-cloud-compute/) · [MLX (GitHub)](https://github.com/ml-explore/mlx) · [AFM technical report](https://machinelearning.apple.com/research/apple-foundation-models)

## TL;DR

Apple Foundation Models (AFM) is the inference stack underpinning Apple Intelligence, announced at WWDC 2024. Unlike every other frontier AI deployment, the primary compute target is an end-user device — the Neural Engine on A17 Pro or M-series silicon — not a GPU data center. A two-tier architecture routes simple requests to a ~3B on-device model and offloads complex requests to Private Cloud Compute (PCC), a custom Apple Silicon cloud built explicitly around the constraint that Apple's own engineers cannot inspect user request content. Every systems decision — quantization scheme, routing logic, hardware choice for PCC, the MLX framework — flows from two non-negotiable requirements: latency under 100 ms on-device and cryptographically verifiable privacy at the cloud tier. This document traces those decisions and their engineering consequences.

## Context & Motivation

Apple's position in 2024 was structurally different from OpenAI, Google, or Anthropic. It had no public language model product, a reputation for privacy as a differentiator, and 2 billion active devices with Neural Engines already in customers' hands. Running inference at the edge — rather than centralizing it in a GPU cluster — was both a business and engineering imperative. The constraints shaped the stack from the beginning:

- On-device memory budgets are tight. iPhone 16 has 8 GB DRAM shared by OS, apps, and the Neural Engine. The model must fit alongside everything else.
- Network latency is unacceptable for inline features. Writing suggestions in Mail or notification summaries must appear fast enough that the user does not perceive a round trip. Sub-100 ms total latency with a cloud roundtrip is physically impossible for most users; it requires local inference.
- Apple's privacy promise is a product claim, not just a policy claim. For it to be credible, the infrastructure must be auditable. A policy saying "we do not log requests" is weaker than hardware and software that structurally prevents logging.

The result is AFM: a family of models with two deployment targets and an inference framework (MLX) purpose-built for Apple Silicon.

## Core Architecture: Two-Tier Serving

AFM follows a strict two-tier serving model. The device — not a server-side router — decides which tier handles each request. This matters because client-side routing means no server-side component ever sees plaintext request content to make a routing decision; the routing signal is derived from local inference outputs (estimated request complexity, token length of expected response, task type).

**Tier 1 — On-device (AFM-on-device):** approximately 3B parameters, quantized to ~4-bit palettized weights, fits in roughly 2 GB of device DRAM. Handles the majority of requests: writing rewrites, notification summaries, quick completions, Siri short-context queries.

**Tier 2 — Private Cloud Compute (PCC):** larger models (Apple has not disclosed exact sizes; capability positioning suggests 8B–70B range) running on Apple Silicon server nodes. Handles requests that exceed the on-device model's capability: longer context summarization, complex Siri requests, cross-app reasoning.

There is no Tier 3 using third-party GPU cloud for Apple-branded features. PCC is wholly Apple-owned and Apple-operated, specifically to allow the hardware attestation model described below.

## On-Device Models: Quantization and the Neural Engine

### Palettized 4-bit Quantization

Standard INT4 quantization maps each weight independently to a 4-bit integer. Apple's published approach uses **palettization**: groups of weights share a codebook entry, and the codebook is learned during or after training via a vector quantization step. The operational form is:

For a weight tensor W partitioned into blocks of size G, each block is represented as an index into a codebook C of size 2^b (b = 4 for 4-bit):

    W_block ≈ C[k],   k ∈ {0, ..., 2^b − 1}

The codebook C is trained jointly with or after the full-precision model, typically via a K-means or learned-codebook objective that minimizes reconstruction loss. Because the codebook is learned rather than linearly spaced, it captures the actual distribution of each weight block more accurately than naive INT4, particularly for outliers. The Apple Core ML Tools `coremltools.optimize.torch.palettization` API exposes this publicly.

Memory impact: a 3B parameter model at BF16 requires ~6 GB. At 4-bit palettized, it requires approximately 1.5–2 GB depending on codebook storage overhead. This is the budget that makes on-device deployment viable on iPhone 16.

### Neural Engine Throughput

The A17 Pro Neural Engine delivers 35 TOPS at INT8; the M4 Neural Engine delivers 38 TOPS. For practical inference:

| Chip | Neural Engine | Memory BW | Target latency (3B, 4-bit) |
|------|--------------|-----------|---------------------------|
| A17 Pro (iPhone 15 Pro) | 35 TOPS (INT8) | ~68 GB/s | ~80–120 ms / request |
| A18 (iPhone 16) | 35+ TOPS | ~68 GB/s | ~80–100 ms / request |
| M4 (Mac, iPad) | 38 TOPS | 120 GB/s | ~40–60 ms / request |
| M4 Max (Mac Studio) | 38 TOPS per cluster | 546 GB/s | <30 ms / request |

The bottleneck for autoregressive decode is memory bandwidth, not compute — weights must be read from DRAM for every token. Higher TOPS ratings matter mainly for the prefill phase; the bandwidth figures above dominate throughput in practice.

### Unified Memory Architecture Advantage

Apple Silicon's defining infra advantage for inference is the unified memory model: CPU, GPU, and Neural Engine share a single DRAM pool over a high-bandwidth fabric. There is no PCIe bus between CPU and accelerator. Consequences:

- Model weights loaded once into DRAM are directly readable by the Neural Engine without an explicit copy or DMA transfer. On a discrete GPU system, weights reside in VRAM and must be managed separately; any CPU-side operation that needs weight data incurs a PCIe copy.
- Effective model capacity for a given DRAM capacity is higher. An M2 Ultra with 192 GB DRAM can fit a 70B BF16 model in full; a system with 192 GB CPU RAM and an H100 (80 GB VRAM) cannot fit the model in GPU memory alone and must offload, incurring NVMe or PCIe round-trips.
- KV cache management differs. On-device, KV cache lives in the same DRAM pool as weights. Cache pressure is felt directly as competition with the model itself, not as a separate VRAM budget problem.

### Speculative Decoding On-Device

Apple uses speculative decoding on-device to improve token throughput on the Neural Engine. The draft model is a smaller, shallower network run first to propose K tokens; the target model (AFM-on-device at full size) verifies all K in one batched forward pass. Both draft and target models run on-device. The constraint is that both must fit simultaneously in device DRAM, which tightens the budget further. For more on the algorithm, see [`../../foundational/speculative-decoding/`](../../foundational/speculative-decoding/).

## MLX: The Inference Framework

MLX was open-sourced by Apple in December 2023. It is the inference (and training) framework for Apple Silicon, analogous to JAX or PyTorch but designed around unified memory from the ground up.

### Design Choices

**Lazy evaluation with graph compilation.** Operations do not execute when called; they accumulate in a computation graph. Execution triggers at `.eval()` or when a Python primitive forces materialization. This enables automatic kernel fusion: adjacent elementwise operations, layernorm + linear sequences, and similar patterns are fused into single Metal kernel dispatches. The result is analogous to `torch.compile` with the Inductor backend on CUDA. See [`../../foundational/torch-compile/`](../../foundational/torch-compile/) for the CUDA-side analogy.

**No explicit `.to(device)` calls.** Arrays exist in a single address space. Moving computation from CPU-side Python to GPU-side Metal is a scheduling decision, not a data movement decision. This removes an entire class of bugs and simplifies model porting substantially.

**First-class quantization.** `mlx.core.quantize` implements 4-bit quantization including group quantization. The palettized scheme used for AFM-on-device is a production evolution of this — the open-source MLX API exposes the building blocks.

**MLX-LM.** The higher-level library for LLM inference built on MLX supports Llama, Mistral, Qwen, Phi, and other open-weight model families. It handles KV cache, speculative decoding, and quantized weight loading. Developers deploying Llama 3 70B on a Mac Studio use MLX-LM.

### Mac as Local Inference Hardware

The unified memory architecture makes high-end Mac hardware viable for serious local inference:

| Hardware | Unified Memory | Bandwidth | Llama 3 70B INT4 throughput |
|----------|---------------|-----------|----------------------------|
| M2 Ultra (Mac Studio) | 192 GB | 800 GB/s | ~20–35 tokens/s |
| M4 Max (MacBook Pro, Mac Studio) | 128 GB | 546 GB/s | ~25–40 tokens/s |
| M4 Ultra (Mac Pro) | 192 GB | 800 GB/s | ~35–50 tokens/s |
| H100 SXM (single GPU, comparison) | 80 GB HBM3 | 3350 GB/s | ~80–120 tokens/s (batch=1) |

The H100 wins on throughput but requires a $30,000+ server, CUDA toolchain, and does not fit 70B INT4 without offloading (model is ~35 GB at INT4; H100 has 80 GB VRAM so it fits, but the comparison clarifies the tradeoff). For single-user local deployment without the data center overhead, M2/M4 Ultra is technically competitive. See [`../../guides/on-prem-llm-deployment/`](../../guides/on-prem-llm-deployment/) for the operational guide.

## Private Cloud Compute: Privacy Architecture

PCC is the most technically differentiated part of the AFM stack. The design goal is not just strong privacy; it is *verifiable* privacy — a structural guarantee that survives even a compromised or malicious Apple infrastructure team.

### Hardware Foundation

PCC nodes run Apple Silicon (M2 Ultra or M4 Ultra based on timing and capability claims). The choice of Apple Silicon is not incidental: it enables the same Secure Enclave, hardware attestation, and Secure Boot chain that Apple uses for iPhone. An NVIDIA GPU node cannot provide the same attestation guarantees because NVIDIA's firmware and hardware root of trust are outside Apple's control.

### Stateless Processing

Each request handled by PCC is processed independently. No session state is persisted beyond the lifetime of the request. The OS running on PCC nodes is stripped of all persistent storage mechanisms that could accumulate user data: there is no write path to non-volatile storage for request content. Logs of request content are structurally absent, not just administratively prohibited.

### No Privileged Access

Apple engineers cannot SSH into PCC nodes and inspect traffic. This is enforced at the hardware level: the OS image running on PCC nodes is signed, and the Secure Enclave rejects boot if the image does not match a published, auditable hash. Mechanisms that would allow ad-hoc code execution — debug interfaces, privileged shell access — are removed from the production image. The constraint is not a policy ("engineers are not allowed"); it is a capability restriction ("the system does not allow it").

### Cryptographic Attestation and Verifiable Transparency

Before a user's device sends any data to PCC, it performs a cryptographic attestation check:

1. The PCC node presents an attestation certificate rooted in its Secure Enclave hardware key.
2. The certificate binds the node's hardware identity to the specific OS image hash running on it.
3. The user's device verifies the OS image hash against a transparency log maintained by Apple and auditable by independent security researchers.
4. Only if the attestation is valid and the OS image is in the published transparency log does the device encrypt and send the request payload.

This means a man-in-the-middle attack by Apple's own networking infrastructure is structurally blocked: intercepting traffic provides ciphertext that cannot be decrypted without the Secure Enclave key on the specific attested node. Replacing the PCC node with a modified OS image breaks attestation and the device refuses to send.

Apple publishes the PCC software image in a transparency log so that external security researchers can audit what code runs. This is the "trust but verify" model made operational: the privacy claim is not "Apple says so" but "here is the code, verify it yourself."

### Routing Logic Revisited

Because device-side routing ensures the server never sees unencrypted request content for routing purposes, PCC cannot implement quality-of-service tiering based on request semantics. The routing decision (on-device vs. PCC) is made locally by the device based on request length, task type, and estimated model capability requirements. This is a meaningful constraint: it prevents server-side A/B testing on request content and prevents Apple from building behavioral profiles based on which requests route to PCC.

## AFM Server: Inference at PCC

The server-side models running on PCC use the same Apple Silicon inference stack but with different operational parameters:

- **Flash Attention variant for M-chip GPU**: standard Flash Attention (FA2/FA3) assumes HBM-SRAM tiling on NVIDIA hardware. On Apple Silicon, the memory hierarchy is different: unified DRAM with a large L2 cache and GPU tile memory (comparable to SRAM in function). Apple has adapted Flash Attention for this hierarchy.
- **KV cache management**: on PCC nodes, the KV cache competes with model weights in the same DRAM pool. Unlike vLLM on H100 (where KV cache occupies a separate VRAM pool managed by PagedAttention), Apple's KV management must account for unified memory pressure. The absence of HBM means lower peak bandwidth for KV cache reads at large batch sizes; this is a throughput ceiling for PCC relative to H100-based deployments.
- **LoRA adapters for task-specific fine-tuning**: AFM uses adapter-based specialization (LoRA-style) for writing tools, summarization, notification triage, and other Apple Intelligence features. The base model weights are shared; adapters are small and can be swapped per-request. See [`../../foundational/lora/`](../../foundational/lora/).

## Engineering Tradeoffs

| Dimension | On-Device (AFM ~3B) | Private Cloud Compute | Cloud GPU (vLLM / TGI on H100) |
|-----------|--------------------|-----------------------|-------------------------------|
| Privacy | Maximum — data never leaves device; no network exposure | Strong — stateless, cryptographic attestation, no engineer access | Standard — provider trust required; logs possible; GDPR/contractual controls only |
| Latency | <100 ms typical (no network RTT) | ~200–500 ms (LAN-equivalent inside Apple DC) | 500 ms–5 s (WAN RTT + queue + prefill) |
| Model size | ~3B params, 4-bit palettized, ~2 GB | ~8B–70B estimated | Unlimited with scale-out |
| Hardware | Neural Engine (35–38 TOPS), ~68–546 GB/s BW | Apple Silicon M-chip (same ISA as device) | H100/A100, 3.35 TB/s HBM3 BW |
| Throughput | Single user; serial requests | ~10–100 concurrent requests per node (estimated) | Thousands concurrent with continuous batching |
| Cost model | Zero marginal (device already owned) | Bundled with Apple device cost; no per-token billing | Pay-per-token; OPEX-heavy at scale |

## Reproducibility Notes

Apple has not published the full AFM training recipe, model architecture details, or PCC cluster size. The published artifacts are:

- The Apple Intelligence feature set (publicly observable via iPhone 16 / macOS Sequoia).
- The PCC security research paper (Apple Security Research blog, October 2024), which describes the privacy architecture in detail.
- MLX (fully open-source, Apache 2.0); the production AFM inference path uses an internal version with additional optimizations.
- Core ML Tools and the palettization API (open-source).
- AFM technical report on Apple Machine Learning Research (June 2024): describes model capabilities and some architecture choices, but not training data or cluster specs.

What is verifiable by external researchers: the PCC transparency log (OS image hashes), the attestation protocol, and the absence of privileged-access mechanisms in published PCC images.

## Commentary

AFM's engineering significance is not the model itself — a 3B on-device model is not frontier capability. The significance is the systems architecture that deploys AI features to 2 billion devices under a privacy constraint that no other frontier lab has attempted to make structurally enforceable.

Three points worth marking:

**The PCC attestation model is genuinely novel at production scale.** The combination of hardware-rooted attestation, published software transparency, and no-privileged-access enforcement has no direct precedent in cloud AI. Google and Microsoft make privacy policy commitments; Apple has engineered a system where breaking those commitments requires breaking hardware attestation.

**Unified memory is a real architectural advantage for on-device inference, not marketing.** The ability to run a 3B model at 4-bit quantization with sub-100 ms latency on a phone is a consequence of bandwidth and memory architecture that discrete GPU systems at similar TDP simply cannot match. MLX makes this programmable.

**The throughput ceiling is real.** PCC's Apple Silicon nodes have lower peak memory bandwidth than H100 clusters. For batch inference at scale, Apple Silicon is not competitive with NVIDIA. Apple's bet is that the privacy and latency requirements of personal AI features make batch throughput the wrong metric — and for consumer device features, that bet is probably correct. Whether it holds for more compute-intensive workloads (long-context reasoning, multi-modal generation) is the open question for the next hardware generation.

## References

- [1] Apple, "Private Cloud Compute: A new frontier for AI privacy in the cloud," Apple Security Research blog, October 2024.
- [2] Apple, "Introducing Apple's On-Device and Server Foundation Models," Apple Machine Learning Research, June 2024.
- [3] Apple, MLX: An array framework for machine learning on Apple Silicon, GitHub (open-sourced December 2023). https://github.com/ml-explore/mlx
- [4] Leviathan et al., "Fast Inference from Transformers via Speculative Decoding," ICML 2023. arXiv:2211.17192.
- [5] Hu et al., "LoRA: Low-Rank Adaptation of Large Language Models," ICLR 2022. arXiv:2106.09685.
- [6] Apple, Core ML Tools — Palettization API. https://coremltools.readme.io/docs/palettization-overview
- [7] Dao et al., "FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning," ICLR 2024. arXiv:2307.08691.
