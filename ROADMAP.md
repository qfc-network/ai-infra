# ai-infra Deep Dive Roadmap

## Context

The repo currently has 96 topics plus 3 guides across training, inference, architectures, open-source infra, and multimodal. The goal is to systematically fill the gaps — topics that are either directly referenced by existing write-ups or are widely used in practice but not yet covered. Each new entry follows the established template: ~2,000–2,500 words, 1–2 equations, one Engineering Tradeoffs table (4–6 rows), both `en.md` and `zh.md`, and an update to the root `README.md` / `README.zh.md` index.

---

## Completed Phases

### Phase 1 — Foundational Gaps ✓
All six entries complete: Scaling Laws, GQA, RoPE, Orca, LoRA/QLoRA, GRPO.

### Phase 2 — Inference & Training Depth ✓
All four entries complete: Mixed Precision Training, Chunked Prefill, KV Cache Quantization, verl.

### Phase 3 — Alignment & Post-Training ✓
Both entries complete: Constitutional AI, SimPO.

### Phase 4 — New Vendor Sections ✓
All three entries complete: DeepSpeed (Microsoft), Cutlass (NVIDIA), TensorRT-LLM (NVIDIA).

### Phase 5 — Model Family Gaps ✓

| # | Topic | Directory | Status |
|---|-------|-----------|--------|
| 16 | Gemma 2 | `google/gemma2/` | ✓ done |

### Phase 6 — Inference-Time Scaling & DeepSeek Arc ✓ (added April 2026)

| # | Topic | Directory | Status |
|---|-------|-----------|--------|
| 17 | Inference-Time Scaling — Test-Time Compute | `foundational/inference-time-scaling/` | ✓ done |
| 18 | DeepSeek-V2 | `deepseek/v2/` | ✓ done |

---

## Active Gap Analysis (post-Phase 15)

Phases 1–15 shipped 54 entries across foundational / multimodal / vendor sections plus three guides. The repo now covers the full training + inference + agent stack end-to-end. Remaining gaps are in **the production-systems layer under the paper-level abstractions** (libraries like TransformerEngine, NCCL, DCP that appear by name in shipped entries but have no standalone treatment), **architectural breadth** (MoE routing beyond the foundational papers, hybrid SSM-Transformer), and **evaluation + deployment surfaces** (base LLM eval plumbing vs the agent-eval entry in Phase 14, confidential/TEE inference).

### Training-stack primitive gaps (high priority — cited, not covered)

- **NVIDIA TransformerEngine** — FP8 primitives library referenced by mixed-precision, FSDP, Megatron; the missing layer between Hopper/H100 and the training papers.
- **NCCL internals** — ring vs tree algorithms, SHARP offload, PXN; referenced by every training entry and the DevOps guide but never explained standalone.
- **Megatron-Core vs the Megatron-LM paper** — the productionized modular library (used by Grok, Nemotron, NeMo) diverged from the 2021 paper; the paper entry is stale as a reference to current practice.
- **Async checkpointing & PyTorch DCP** — mentioned in DualPipe and Llama 3 entries; no standalone treatment of distributed checkpoint, async writes, or recovery bandwidth.

### Architecture / deployment breadth gaps (medium priority)

- **MoE routing improvements** — DeepSeekMoE + Switch/GShard cover foundations; Expert Choice routing and Loss-Free Balancing (DeepSeek V3's actual routing) are referenced but not covered. Closes the MoE arc.
- **Hybrid SSM-Transformer (Jamba)** — Mamba entry exists; the pragmatic hybrid approach (AI21 Jamba, Nemotron-H) is where SSMs actually shipped.
- **llama.cpp / GGUF** — the on-prem guide mentions Ollama (which wraps it); GGUF format and CPU+GPU hybrid inference deserve standalone coverage.
- **LMDeploy / TurboMind** — major Chinese serving stack (Shanghai AI Lab); W4A16 kernels; missing peer to TGI / vLLM / SGLang.

### Evaluation & secure-inference gaps (medium priority)

- **Base LLM evaluation harness** — agent eval is covered (Phase 14); lm-eval-harness, MMLU/GSM8K/HumanEval plumbing, and Chatbot Arena / LMSYS Elo infrastructure have no entry. Cited implicitly by every model-family entry.
- **Confidential LLM inference** — H100 Confidential Computing, Nitro Enclaves, TEE-gated serving; natural pair for the secure-agent-deployment guide (Phase 14b).

### Lower-priority leftovers (unchanged)
- **RLHF reward model training pipeline** — could extend the existing RLHF entry rather than a new write-up.
- **SLO-aware scheduling depth** — covered in pieces across Orca / Sarathi / DistServe; dedicated synthesis possible but not urgent.

---

## Phased Roadmap — Next Phases

### Phase 7 — Hardware & Systems Depth ✓

| # | Topic | Directory | Status |
|---|-------|-----------|--------|
| 19 | Hopper / H100 Architecture Primer | `foundational/hopper-h100/` | ✓ done |
| 20 | Speculative Decoding — Medusa / EAGLE | `foundational/speculative-decoding-variants/` | ✓ done |
| 21 | Prefix Caching | `foundational/prefix-caching/` | ✓ done |

### Phase 8 — Post-Training Depth ✓

| # | Topic | Directory | Status |
|---|-------|-----------|--------|
| 22 | Process Reward Models (PRMs) | `foundational/process-reward-models/` | ✓ done |
| 23 | Gemma 2 | `google/gemma2/` | ✓ done |
| 24 | DeepSeek Prover | `deepseek/prover/` | ✓ done |

### Phase 9 — Production Inference & New Architectures ✓

| # | Topic | Directory | Status |
|---|-------|-----------|--------|
| 25 | Speculative Decoding — Medusa / EAGLE | `foundational/speculative-decoding-variants/` | ✓ done |
| 26 | Llama 4 | `meta/llama4/` | ✓ done |
| 27 | Qwen3 | `qwen/qwen3/` | ✓ done |
| 28 | Blackwell / B200 Architecture Primer | `foundational/blackwell-b200/` | ✓ done |

---

### Phase 10 — Multimodal Infrastructure ✓

| # | Topic | Directory | Status |
|---|-------|-----------|--------|
| 29 | CLIP — Contrastive Language-Image Pretraining | `multimodal/clip/` | ✓ done |
| 30 | Vision Transformer (ViT) | `multimodal/vit/` | ✓ done |
| 31 | LLaVA / Vision-Language Model architecture | `multimodal/llava/` | ✓ done |
| 32 | Whisper — speech recognition | `multimodal/whisper/` | ✓ done |
| 33 | DiT — Diffusion Transformers | `multimodal/dit/` | ✓ done |
| 34 | VLM Serving | `multimodal/vlm-serving/` | ✓ done |

### Phase 11 — Data, Tokenization & Long Context ✓

| # | Topic | Directory | Status |
|---|-------|-----------|--------|
| 35 | Tokenization — BPE, SentencePiece, Tiktoken | `foundational/tokenization/` | ✓ done |
| 36 | Data Pipelines — FineWeb / MinHash / Quality Filtering | `foundational/data-pipeline/` | ✓ done |
| 37 | Position Interpolation — YaRN / LongRoPE / NTK-aware | `foundational/position-interpolation/` | ✓ done |
| 38 | Streaming LLM & Attention Sinks | `foundational/streaming-llm/` | ✓ done |

### Phase 12 — MoE Foundations & Production Inference 2.0 ✓

| # | Topic | Directory | Status |
|---|-------|-----------|--------|
| 39 | Switch Transformer & GShard | `foundational/switch-gshard/` | ✓ done |
| 40 | Hugging Face TGI — Text Generation Inference | `foundational/tgi/` | ✓ done |
| 41 | NVIDIA Dynamo | `nvidia/dynamo/` | ✓ done |
| 42 | Multi-tenant LoRA Serving — SLoRA / Punica | `foundational/multi-tenant-lora/` | ✓ done |

---

## Phase 13 — Completing the Training/Inference Stack ✓

| # | Topic | Directory | Status |
|---|-------|-----------|--------|
| 43 | DDPM / DDIM / Classifier-Free Guidance | `foundational/diffusion-fundamentals/` | ✓ done |
| 44 | Sequence Parallelism Variants — Ulysses & Megatron-CP | `foundational/sequence-parallelism/` | ✓ done |
| 45 | `torch.compile` / Inductor | `foundational/torch-compile/` | ✓ done |
| 46 | Knowledge Distillation at Scale | `foundational/knowledge-distillation/` | ✓ done |
| 47 | DeepSeek V3.2 / Native Sparse Attention | `deepseek/v3-2-nsa/` | ✓ done |

---

## Phase 14 — Agent Infrastructure ✓

| # | Topic | Directory | Status |
|---|-------|-----------|--------|
| 48 | Tool Use / Function Calling Infrastructure | `foundational/tool-use-infra/` | ✓ done |
| 49 | Agent Framework Landscape | `foundational/agent-frameworks/` | ✓ done |
| 50 | Computer Use & Browser Automation | `anthropic/computer-use/` | ✓ done |
| 51 | Agent Evaluation Infrastructure | `foundational/agent-evaluation/` | ✓ done |

## Phase 14b — Secure Agent Deployment ✓

| # | Topic | Directory | Status |
|---|-------|-----------|--------|
| — | Secure On-Prem Agent Deployment (guide) | `guides/secure-agent-deployment/` | ✓ done |

---

## Phase 15 — Frontier Lab Completeness ✓

| # | Topic | Directory | Status |
|---|-------|-----------|--------|
| 52 | xAI Grok + Colossus | `xai/grok-colossus/` | ✓ done |
| 53 | ByteDance Seed / Doubao | `bytedance/seed/` | ✓ done |
| 54 | Apple Foundation Models (AFM) | `apple/afm/` | ✓ done |

---

## Phase 16 — Training Infrastructure Primitives ✓

| # | Topic | Directory | Status |
|---|-------|-----------|--------|
| 55 | NVIDIA TransformerEngine | `nvidia/transformer-engine/` | ✓ done |
| 56 | NCCL Internals | `foundational/nccl/` | ✓ done |
| 57 | Megatron-Core | `nvidia/megatron-core/` | ✓ done |
| 58 | Async Checkpointing & PyTorch DCP | `foundational/distributed-checkpointing/` | ✓ done |

---

## Phase 17 — Architectural & Serving Breadth ✓

| # | Topic | Directory | Status |
|---|-------|-----------|--------|
| 59 | MoE Routing Improvements — Expert Choice & Loss-Free Balancing | `foundational/moe-routing/` | ✓ done |
| 60 | Jamba / Hybrid SSM-Transformer | `foundational/hybrid-ssm/` | ✓ done |
| 61 | llama.cpp & GGUF | `foundational/llama-cpp/` | ✓ done |
| 62 | LMDeploy / TurboMind | `foundational/lmdeploy/` | ✓ done |

---

## Phase 18 — Evaluation & Confidential Inference ✓

| # | Topic | Directory | Status |
|---|-------|-----------|--------|
| 63 | LLM Evaluation Harness | `foundational/eval-harness/` | ✓ done |
| 64 | Chatbot Arena & Pairwise Eval | `foundational/chatbot-arena/` | ✓ done |
| 65 | Confidential LLM Inference | `foundational/confidential-inference/` | ✓ done |

---

## Execution Notes

- **Priority order (Phases 16–18)**: Phase 16 first — fills the library-layer gap under paper-level entries; highest internal-coherence leverage. Phase 17 second — architectural breadth and serving-engine completeness. Phase 18 last — orthogonal axes (evaluation infra, confidential inference); valuable for framing but less referenced by existing entries.
- **Historical priority (Phases 13–15)**: Phase 13 (stack completeness) first — each entry fills a gap already implicitly referenced by existing entries, so returns compound across the repo. Phase 14 (agent infra) second — highest strategic value but introduces a new axis, worth doing after internal coherence is improved. Phase 15 (lab completeness) last — valuable but narrative-heavy and lower leverage than the technical gaps in 13.
- **Each topic**: `en.md` + `zh.md`, update `README.md` + `README.zh.md` index. Phase 10 also needs new `multimodal/` section headers in both READMEs.
- **Format**: ~2,000–2,500 words, 1–2 key equations, 1 tradeoffs table (4–6 rows)
- **Tone**: engineering-first — what are the memory/compute/latency consequences?
- **Cross-references**: link to related entries; do not repeat content already in a standalone entry. See the "Cross-references to seed" note at the end of each phase for suggested links.
- **New vendor/section templates**: for `multimodal/`, follow the same pattern as `foundational/` — flat directory of topic folders, no vendor README required unless more than ~5 entries accumulate.

## Files to modify per entry
- `{vendor}/{topic}/en.md` (new)
- `{vendor}/{topic}/zh.md` (new)
- `README.md` (index entry)
- `README.zh.md` (index entry)
- `{vendor}/README.md` (if vendor-specific index exists)
