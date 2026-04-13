# ai-infra

> Deep dives into AI infrastructure papers and open-source systems from frontier labs.
> 前沿 AI 实验室基础设施论文与开源系统的深度解析。

[中文版 README](./README.zh.md)

## Scope

This repo collects engineering-focused analyses of papers and open-source releases covering:

- **Training systems** — parallelism strategies, mixed precision, communication primitives
- **Inference systems** — KV cache, speculative decoding, serving architectures
- **Model architectures with infra implications** — MoE routing, attention variants, long context
- **Open-source infra components** — kernels, schedulers, file systems

Every paper gets both an English (`en.md`) and Chinese (`zh.md`) write-up using the same template. See [`_template/`](./_template).

## Index

### Foundational
- [FlashAttention 1 / 2 / 3](./foundational/flash-attention/) — IO-aware exact attention, Hopper async + FP8
- [Triton](./foundational/triton/) — block-level GPU kernel DSL + MLIR compiler; the productivity multiplier
- [PagedAttention / vLLM](./foundational/paged-attention/) — OS-style paging for KV cache, continuous batching
- [Megatron-LM (TP / PP / SP)](./foundational/megatron-lm/) — tensor, pipeline, sequence parallelism + selective recompute
- [ZeRO / FSDP](./foundational/zero-fsdp/) — sharded data parallelism; orthogonal to Megatron
- [Speculative Decoding](./foundational/speculative-decoding/) — draft + verify; lossless 2–4× decode speedup
- [Ring Attention / Context Parallelism](./foundational/ring-attention/) — exact attention at 1M+ context via sequence sharding
- [DistServe](./foundational/distserve/) — prefill/decode disaggregation; goodput as the right metric
- [SGLang](./foundational/sglang/) — RadixAttention prefix caching; frontend DSL for multi-call LLM programs
- [Weight Quantization — GPTQ & AWQ](./foundational/weight-quantization/) — INT4 post-training; Hessian vs activation-aware
- [SmoothQuant](./foundational/smoothquant/) — W8A8; activation-to-weight outlier migration

### DeepSeek
- [V3 Technical Report](./deepseek/v3-tech-report/) — FP8 training, DualPipe, MoE at 671B
- [MLA — Multi-head Latent Attention](./deepseek/mla/) — KV cache compression
- [DeepSeekMoE](./deepseek/moe/) — fine-grained + shared experts
- [R1](./deepseek/r1/) — reasoning via rule-based RL; GRPO; R1-Zero emergence
- [Open Source Week — FlashMLA walkthrough](./deepseek/open-source-week/flash-mla/) — seesaw schedule, FP8 sparse decode
- [Open Source Week — DeepEP walkthrough](./deepseek/open-source-week/deep-ep/) — expert-parallel all-to-all, IBGDA low-latency
- [Open Source Week — DeepGEMM walkthrough](./deepseek/open-source-week/deep-gemm/) — JIT FP8/BF16 GEMM, MoE layouts, V3.2 indexer
- [Open Source Week — DualPipe walkthrough](./deepseek/open-source-week/dualpipe/) — bidirectional pipeline schedule; halves bubbles
- [Open Source Week — 3FS walkthrough](./deepseek/open-source-week/3fs/) — RDMA-native distributed FS; CRAQ + FDB + USRBIO

### Meta
- [Llama 3 Herd of Models](./meta/llama3/) — 405B dense, 16k H100s, 4D parallelism

### Mistral
- [Mixtral of Experts](./mistral/mixtral/) — 8×7B, top-2 routing, open-weight MoE baseline

### Moonshot
- [Mooncake](./moonshot/mooncake/) — KVCache-centric disaggregated inference; PD-disaggregation; cache pool

### Google
- _Gemini / Pathways / TPU systems_ — planned

### Anthropic
- _Public infra writings_ — planned

## Contributing

New papers: copy [`_template/`](./_template) into the appropriate vendor directory, fill in both `zh.md` and `en.md`, update this index.

## License

Documentation: [CC BY 4.0](./LICENSE).
