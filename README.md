# ai-infra

> Deep dives into AI infrastructure papers and open-source systems from frontier labs.
> 前沿 AI 实验室基础设施论文与开源系统的深度解析。

[中文版 README](./README.zh.md)

> **Companion repo:** [qfc-network/ai-infra-cheatsheet](https://github.com/qfc-network/ai-infra-cheatsheet) —
> side-by-side hardware spec tables (NVIDIA, AMD, Apple) plus VRAM and KV cache
> sizing math. This repo explains the mechanisms; that one holds the numbers.

## Scope

This repo collects engineering-focused analyses of papers and open-source releases covering:

- **Training systems** — parallelism strategies, mixed precision, communication primitives
- **Inference systems** — KV cache, speculative decoding, serving architectures
- **Model architectures with infra implications** — MoE routing, attention variants, long context
- **Open-source infra components** — kernels, schedulers, file systems

Every paper gets both an English (`en.md`) and Chinese (`zh.md`) write-up using the same template. See [`_template/`](./_template).

## Index

### Foundational
- [Scaling Laws — Kaplan 2020 + Chinchilla 2022](./foundational/scaling-laws/) — power-law compute/data/param tradeoffs; the 20-tokens-per-param rule
- [FlashAttention 1 / 2 / 3](./foundational/flash-attention/) — IO-aware exact attention, Hopper async + FP8
- [Triton](./foundational/triton/) — block-level GPU kernel DSL + MLIR compiler; the productivity multiplier
- [PagedAttention / vLLM](./foundational/paged-attention/) — OS-style paging for KV cache, continuous batching
- [Orca — Continuous Batching](./foundational/orca/) — iteration-level scheduling; goodput as the right metric; foundation for vLLM/SGLang
- [Megatron-LM (TP / PP / SP)](./foundational/megatron-lm/) — tensor, pipeline, sequence parallelism + selective recompute
- [ZeRO / FSDP](./foundational/zero-fsdp/) — sharded data parallelism; orthogonal to Megatron
- [Speculative Decoding](./foundational/speculative-decoding/) — draft + verify; lossless 2–4× decode speedup
- [Ring Attention / Context Parallelism](./foundational/ring-attention/) — exact attention at 1M+ context via sequence sharding
- [Grouped Query Attention (GQA)](./foundational/gqa/) — H/G KV cache reduction; the default attention variant in Llama 3, Mistral, Gemma
- [Rotary Position Embeddings (RoPE)](./foundational/rope/) — position-by-rotation; relative, parameter-free, flash-friendly; long-context extensions
- [DistServe](./foundational/distserve/) — prefill/decode disaggregation; goodput as the right metric
- [SGLang](./foundational/sglang/) — RadixAttention prefix caching; frontend DSL for multi-call LLM programs
- [Prefix Caching](./foundational/prefix-caching/) — hash-based vs radix-trie matching; eviction policies; hit rate economics; multi-tier GPU→CPU→disk; disaggregated routing
- [Weight Quantization — GPTQ & AWQ](./foundational/weight-quantization/) — INT4 post-training; Hessian vs activation-aware
- [SmoothQuant](./foundational/smoothquant/) — W8A8; activation-to-weight outlier migration
- [RLHF / InstructGPT](./foundational/rlhf/) — three-stage SFT+RM+PPO; the post-training foundation
- [DPO](./foundational/dpo/) — collapse RLHF into one supervised step; the simpler default
- [GRPO — Group Relative Policy Optimization](./foundational/grpo/) — critic-free RL via group advantage; the algorithm behind DeepSeek-R1
- [SimPO — Simple Preference Optimization](./foundational/simpo/) — reference-free DPO; length-normalized reward + target margin
- [LoRA / QLoRA](./foundational/lora/) — low-rank weight adaptation; 4-bit fine-tuning on a single GPU
- [Mixed Precision Training](./foundational/mixed-precision/) — FP16 → BF16 → FP8; AMP, loss scaling, per-tensor/block FP8 scaling
- [Chunked Prefill — Sarathi-Serve](./foundational/chunked-prefill/) — interleave prefill chunks with decode; eliminate TTFT stalls on a single GPU
- [KV Cache Quantization — KIVI & KVQuant](./foundational/kv-cache-quantization/) — INT2 keys + INT4 values; completes the quantization arc
- [verl — HybridFlow](./foundational/verl/) — per-model parallelism + CPU offload for PPO/GRPO at 70B+ scale; the infra behind R1-style training
- [GPU Interconnect primer](./foundational/gpu-interconnect/) — NVLink, NVSwitch, RDMA, IBGDA; the fabric assumed by everything else
- [Hopper / H100 Architecture Primer](./foundational/hopper-h100/) — wgmma, TMA, FP8 Tensor Cores, Thread Block Clusters; the compute primitives behind FlashAttention-3, DeepGEMM, DualPipe
- [Mamba and State Space Models](./foundational/mamba-ssm/) — linear-time, constant-memory-decode; hybrids with attention
- [Inference-Time Scaling — Test-Time Compute](./foundational/inference-time-scaling/) — parallel search (best-of-N, PRM beam search) vs sequential refinement (thinking tokens); compute-optimal strategy by difficulty
- [Process Reward Models (PRMs)](./foundational/process-reward-models/) — step-level verification; PRM800K human labels vs MC rollout auto-labeling; beam search and MCTS integration
- [Speculative Decoding Variants — Medusa / EAGLE](./foundational/speculative-decoding-variants/) — tree-structured self-draft; Medusa parallel heads; EAGLE feature-level autoregressive draft; EAGLE-2 adaptive trees
- [Blackwell / B200 Architecture Primer](./foundational/blackwell-b200/) — FP4 Tensor Cores, HBM3e 192 GB, NVLink 5 1.8 TB/s, GB200 NVL72 rack-scale fabric
- [Tokenization — BPE, SentencePiece, Tiktoken](./foundational/tokenization/) — subword algorithms; multilingual token budget economics; fertility by language
- [Data Pipelines — FineWeb / MinHash / Quality Filtering](./foundational/data-pipeline/) — Common Crawl → pretraining corpus; MinHash dedup; quality classifiers at trillion-token scale
- [Position Interpolation — YaRN / LongRoPE / NTK-aware](./foundational/position-interpolation/) — extending pretrained RoPE models to 128k–2M context; NTK-aware base change; YaRN non-uniform scaling
- [Streaming LLM & Attention Sinks](./foundational/streaming-llm/) — bounded KV cache for infinite-length generation; attention sink phenomenon; K_sink + sliding window
- [Switch Transformer & GShard](./foundational/switch-gshard/) — top-1 routing; auxiliary load-balancing loss; expert capacity factor; the foundational MoE papers
- [TGI — Text Generation Inference](./foundational/tgi/) — HuggingFace's Rust + Python serving stack; continuous batching; quantization formats; vLLM comparison
- [Multi-Tenant LoRA Serving — SLoRA / Punica](./foundational/multi-tenant-lora/) — paged adapter pool (SLoRA); SGMV batched kernel (Punica); serving thousands of adapters from one base model
- [Diffusion Fundamentals — DDPM / DDIM / CFG](./foundational/diffusion-fundamentals/) — forward/reverse diffusion process; DDIM deterministic sampling; classifier-free guidance; NFE as the primary latency driver
- [Sequence Parallelism Variants — Ulysses & Megatron-CP](./foundational/sequence-parallelism/) — all-to-all on head dim (Ulysses); causal even-odd interleaving (Megatron-CP); completes the sequence-axis parallelism story with Ring Attention
- [torch.compile / Inductor](./foundational/torch-compile/) — TorchDynamo bytecode tracing; AOTAutograd joint graph; Inductor Triton codegen; operator fusion; CUDA Graphs
- [Knowledge Distillation at Scale](./foundational/knowledge-distillation/) — soft label loss with temperature; white-box / black-box / sequence-level regimes; R1 distillation pipeline; Gemma 2 logit soft-capping
- [Tool Use / Function Calling Infrastructure](./foundational/tool-use-infra/) — JSON schema tool definitions; parallel tool call dispatch; streaming delta parsing; safety layers; context window growth arithmetic
- [Agent Framework Landscape](./foundational/agent-frameworks/) — LangGraph stateful graphs; AutoGen multi-agent conversations; OpenAI Swarm handoffs; DSPy prompt compilation; state persistence and p99 latency tradeoffs
- [Agent Evaluation Infrastructure](./foundational/agent-evaluation/) — SWE-bench, TAU-bench, WebArena; trajectory metrics beyond binary pass/fail; sandboxed eval environments; LLM-as-judge calibration
- [NCCL Internals](./foundational/nccl/) — ring vs tree AllReduce; LL/LL128 protocols; NVLS in-fabric reduction; SHARP IB offload; IBGDA; topology detection and key debug env vars
- [Async Checkpointing & PyTorch DCP](./foundational/distributed-checkpointing/) — sharded save/reshardable load; async in-memory copy; ZeRO sharded format; checkpoint frequency optimization; recovery bandwidth arithmetic
- [MoE Routing Improvements — Expert Choice & Loss-Free Balancing](./foundational/moe-routing/) — Expert Choice inverted assignment; Loss-Free Balancing bias update rule; fine-grained+shared pattern; closes the MoE routing arc
- [Jamba / Hybrid SSM-Transformer](./foundational/hybrid-ssm/) — interleaved Mamba+Attention blocks; KV cache reduction at 256k context; decode economics; why pure SSMs plateaued on recall tasks
- [llama.cpp & GGUF](./foundational/llama-cpp/) — GGUF binary format; Q4_K_M super-block quantization; CPU+GPU hybrid offload; Metal/CUDA backends; the substrate under Ollama
- [LMDeploy / TurboMind](./foundational/lmdeploy/) — W4A16 AWQ custom CUDA kernels; MLA-aware KV cache; FP8 KV; H100 throughput vs vLLM/TGI; first-class Qwen/DeepSeek support
- [LLM Evaluation Harness](./foundational/eval-harness/) — lm-eval-harness; log-likelihood vs generation scoring; MMLU/GSM8K/HumanEval plumbing; pass@k formula; Open LLM Leaderboard pipeline
- [Chatbot Arena & Pairwise Evaluation](./foundational/chatbot-arena/) — Bradley-Terry model; Elo rating; 1M+ human preference votes; MT-Bench LLM-as-judge; Arena Hard; why static benchmarks saturated
- [Confidential LLM Inference](./foundational/confidential-inference/) — H100 CC mode attestation; AWS Nitro Enclaves; Azure SEV-SNP; TEE-gated serving pattern; 5–10% latency overhead; 20–30% cost premium

### Multimodal
- [CLIP — Contrastive Language-Image Pretraining](./multimodal/clip/) — dual-encoder contrastive objective; InfoNCE loss over N² pairs; zero-shot classification; web-scale training
- [Vision Transformer (ViT)](./multimodal/vit/) — patch embedding; [CLS] token; ViT-L/14 → 576 tokens; FlashAttention-compatible; DeiT distillation
- [LLaVA / Vision-Language Models](./multimodal/llava/) — MLP projector; two-stage training; visual token counts (256→576→2880); image prefix caching
- [Whisper — Speech Recognition](./multimodal/whisper/) — log-mel spectrogram; 30s chunking; multitask via special tokens; weakly-supervised at 680k hours
- [DiT — Diffusion Transformers](./multimodal/dit/) — latent diffusion + transformer backbone; adaLN conditioning; Sora / SD3 / FLUX lineage; compute-bound inference
- [VLM Serving](./multimodal/vlm-serving/) — visual token prefill economics; variable-resolution tiling; image prefix caching; heterogeneous batching

### DeepSeek
- [V2 — Economical MoE at 236B](./deepseek/v2/) — MLA + DeepSeekMoE as a system; 21B activated / 236B total; 5.76× throughput over dense predecessor
- [V3 Technical Report](./deepseek/v3-tech-report/) — FP8 training, DualPipe, MoE at 671B
- [MLA — Multi-head Latent Attention](./deepseek/mla/) — KV cache compression
- [DeepSeekMoE](./deepseek/moe/) — fine-grained + shared experts
- [R1](./deepseek/r1/) — reasoning via rule-based RL; GRPO; R1-Zero emergence
- [Open Source Week — FlashMLA walkthrough](./deepseek/open-source-week/flash-mla/) — seesaw schedule, FP8 sparse decode
- [Open Source Week — DeepEP walkthrough](./deepseek/open-source-week/deep-ep/) — expert-parallel all-to-all, IBGDA low-latency
- [Open Source Week — DeepGEMM walkthrough](./deepseek/open-source-week/deep-gemm/) — JIT FP8/BF16 GEMM, MoE layouts, V3.2 indexer
- [Open Source Week — DualPipe walkthrough](./deepseek/open-source-week/dualpipe/) — bidirectional pipeline schedule; halves bubbles
- [Open Source Week — 3FS walkthrough](./deepseek/open-source-week/3fs/) — RDMA-native distributed FS; CRAQ + FDB + USRBIO
- [DeepSeek-Prover](./deepseek/prover/) — Lean 4 theorem proving via RL + RMaxTS; compiler as perfect PRM; auto-formalization pipeline
- [V3.2 / Native Sparse Attention (NSA)](./deepseek/v3-2-nsa/) — compressed + selected + window three-path sparse attention; V3.2 indexer kernel; native 128k context without RoPE interpolation

### Meta
- [Llama 3 Herd of Models](./meta/llama3/) — 405B dense, 16k H100s, 4D parallelism
- [Llama 4](./meta/llama4/) — first MoE family; Scout 17B×16E / Maverick 17B×128E; iRoPE interleaved attention; native multimodal

### Qwen
- [Qwen3](./qwen/qwen3/) — thinking/non-thinking toggle; MoE 235B/22A + dense 0.6B–32B; RL post-training; top open-weight family

### Mistral
- [Mixtral of Experts](./mistral/mixtral/) — 8×7B, top-2 routing, open-weight MoE baseline

### Moonshot
- [Mooncake](./moonshot/mooncake/) — KVCache-centric disaggregated inference; PD-disaggregation; cache pool
- [Kimi K3 Infrastructure Stack](./moonshot/kimi-k3-infra/) — MoonEP balanced expert parallelism; FlashKDA CUTLASS kernels; AgentENV Firecracker sandboxes for million-token agentic RL

### Google
- [Pathways](./google/pathways/) — async distributed dataflow runtime; single-controller at TPU pod scale
- [GSPMD](./google/gspmd/) — XLA compiler pass for auto-parallelization; sharding as a type
- [Gemma 2](./google/gemma2/) — distillation from 27B teacher; logit soft-capping; alternating local/global attention

### Microsoft
- [DeepSpeed — MoE, Chat, and Inference Engine](./microsoft/deepspeed/) — expert parallelism, hybrid RLHF engine, fused INT8 inference

### NVIDIA
- [CUTLASS](./nvidia/cutlass/) — C++ GEMM template hierarchy; the kernel library under FlashAttention, DeepGEMM, and cuBLAS
- [TensorRT-LLM](./nvidia/tensorrt-llm/) — AOT-compiled LLM inference; paged KV cache, FP8, continuous batching at H100 peak
- [Dynamo](./nvidia/dynamo/) — disaggregated inference orchestration; KV-aware routing; prefill/decode pool management; TensorRT-LLM integration
- [TransformerEngine](./nvidia/transformer-engine/) — FP8 drop-in modules (te.Linear, te.TransformerLayer); E4M3/E5M2 format split; DelayedScaling amax history; 1.3–1.6× end-to-end training speedup
- [Megatron-Core](./nvidia/megatron-core/) — modular library superseding the 2021 paper; ParallelState, TransformerConfig, mcore DDP; CP integration; TE routing; used by Nemotron, NeMo, Grok

### xAI
- [Grok + Colossus](./xai/grok-colossus/) — 314B MoE Grok-1; 100k H100 single-site Memphis cluster; 4D parallelism at frontier scale; single-site AllReduce latency advantage

### ByteDance
- [Seed / Doubao](./bytedance/seed/) — MegaScale fault tolerance at 12k GPUs; verl/HybridFlow origin lab; H800 export-control constraints; PD-disaggregated inference at 100M+ QPS

### Apple
- [Foundation Models (AFM)](./apple/afm/) — on-device 3B (4-bit palettized) + Private Cloud Compute; Apple Silicon attestation-based privacy; MLX unified memory; two-tier routing

### Anthropic
- [Building Effective Agents](./anthropic/building-effective-agents/) — workflow vs agent, five workflow patterns
- [Model Context Protocol (MCP)](./anthropic/mcp/) — LSP for LLMs; tools / resources / prompts via JSON-RPC
- [Constitutional AI](./anthropic/constitutional-ai/) — critique-revision loop + RLAIF; harmlessness from AI feedback
- [Computer Use & Browser Automation](./anthropic/computer-use/) — screenshot-based pixel-level action space; visual grounding; per-step latency arithmetic; container sandboxing

## Guides

Career and onboarding guides — engineering-first, minimal ML algorithm prerequisites.

- [From DevOps to AI Infrastructure](./guides/devops-to-ai-infra/) — skills that transfer, gaps to fill, three vertical paths (cluster ops / inference platform / training infra), 6-month milestones
- [On-Premise LLM Deployment](./guides/on-prem-llm-deployment/) — from Mac Studio + Ollama to multi-node GPU clusters; hardware sizing, software stack, cost reference, decision flowchart
- [Secure On-Prem Agent Deployment](./guides/secure-agent-deployment/) — tool call sandboxing (gVisor, NetworkPolicy, seccomp); MCP permission boundaries; secret management in agent loops; minimal secure stack reference

## Contributing

New papers: copy [`_template/`](./_template) into the appropriate vendor directory, fill in both `zh.md` and `en.md`, update this index.

## License

Documentation: [CC BY 4.0](./LICENSE).
