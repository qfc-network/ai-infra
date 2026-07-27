# ai-infra Deep Dive Roadmap

## Context

The repo currently has 95 topics plus 3 guides across training, inference, architectures, open-source infra, and multimodal. The count excludes the bilingual `_template/` pair. The goal is to systematically fill the gaps — topics that are either directly referenced by existing write-ups or are widely used in practice but not yet covered. Each new entry follows the established template: ~2,000–2,500 words, 1–2 equations, one Engineering Tradeoffs table (4–6 rows), both `en.md` and `zh.md`, and an update to the root `README.md` / `README.zh.md` index.

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

### Out-of-Cycle Release Ingestion — Kimi K3 Infrastructure ✓ (added July 2026)

| Topic | Directory | Status |
|---|---|---|
| Kimi K3 infrastructure stack — MoonEP, FlashKDA, AgentENV | `moonshot/kimi-k3-infra/` | ✓ done |

This synthesis connects model architecture to three released infrastructure layers: balanced expert-parallel communication, architecture-specific GPU kernels, and stateful microVM sandboxes for agentic RL. It was added out of roadmap order because all three codebases and the K3 technical report were released together.

---

## Active Gap Analysis (post-Phase 18)

Phases 1–18 shipped 65 topic entries + 3 guides. The repo now covers training systems, inference systems, architecture, agents, evaluation, secure/confidential deployment, and library-layer primitives end-to-end. Remaining gaps are **framework/toolchain references** (HF Transformers, Ray, FlexAttention — implementations cited everywhere but no standalone), **training recipe primitives** (stability, LR schedules, activation/norm variants, synthetic data — the "recipe" under every training paper), and **frontier model family gaps** (Gemini 1.5/2.0, NVIDIA Nemotron, reasoning-model landscape).

### Framework / toolchain gaps (high priority — referenced everywhere)

- **Hugging Face stack** — Transformers, PEFT, TRL, Accelerate; the reference implementation under most entries. The on-prem guide mentions PEFT/TRL; no standalone.
- **FlexAttention** — PyTorch 2.5+ composable attention primitive with score-mod API; score_mod replaces custom Triton kernels for many attention variants; referenced by torch.compile entry but not standalone.
- **Ray & Ray Serve for LLM** — orchestration layer under verl, ByteDance Seed, Anyscale; no standalone. Cited in verl entry.
- **Fine-tuning toolchain (Axolotl / Llama-Factory / Unsloth)** — production SFT+LoRA stacks; on-prem guide references Axolotl; no standalone.

### Training recipe gaps (medium priority)

- **Training stability** — loss spikes, Z-loss, muP, weight init; referenced implicitly by Megatron / Llama 3 / Grok. Recipe-level content that sits under every training entry.
- **LR schedules & optimizers** — WSD (warmup-stable-decay, now the default), AdamW tuning, Lion / Sophia optimizers; no coverage.
- **Activation & norm primitives** — SwiGLU, GLU variants, RMSNorm, QK-norm; universal in modern LLMs, no standalone entry.
- **Synthetic data generation for training** — Phi series, Nemotron-4 340B for reward modeling, R1-distill data pipelines; pairs with Knowledge Distillation (which covers training side) but not the data-generation economics.

### Frontier model family gaps (medium priority)

- **Google Gemini 1.5 / 2.0** — native multimodal, 2M context, MoE; notable infra (TPU-first); no entry.
- **NVIDIA Nemotron** — 340B synthetic-data + reward-modeling pipeline; NeMo stack; TransformerEngine/Megatron-Core reference consumer.
- **Reasoning-model landscape** — meta-analysis linking o1/o3, Claude 3.7/4 thinking, R1 (covered), Qwen3 thinking mode (covered); infrastructure implications of long-chain-of-thought (KV cache growth, thinking-token economics).

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

## Phase 19 — Open-Source Framework Stack (recommended next)

**Rationale**: Phases 16–18 filled the library layer underneath paper-level entries. Phase 19 covers the **framework layer above** — the reference implementations (HF Transformers, Ray, FlexAttention, Axolotl) that most entries assume as the baseline. Highest internal-coherence leverage of the remaining phases: every model-family and fine-tuning entry implicitly depends on these.

| # | Topic | Directory | Why |
|---|-------|-----------|-----|
| 66 | Hugging Face Stack — Transformers / PEFT / TRL / Accelerate | `foundational/huggingface-stack/` | The reference implementation family: `Transformers` (model zoo + generate loop), `PEFT` (LoRA/adapters surface), `TRL` (RLHF/DPO/GRPO trainers, used by OpenRLHF/verl peers), `Accelerate` (device mesh + launcher); nearly every entry links here implicitly |
| 67 | FlexAttention | `foundational/flex-attention/` | PyTorch 2.5+ composable attention with `score_mod` API; replaces custom Triton kernels for many variants (causal, sliding, ALiBi, document masks); compiled via Inductor; the productivity successor to FlashAttention for common cases |
| 68 | Ray & Ray Serve for LLM | `foundational/ray-for-llm/` | Distributed orchestration substrate: Ray actors for training (verl, OpenRLHF), Ray Serve for inference, Ray Data for preprocessing; the glue under multi-stage RL + inference pipelines |
| 69 | Fine-tuning Toolchain — Axolotl / Llama-Factory / Unsloth | `foundational/finetuning-toolchain/` | Config-driven SFT + LoRA pipelines (Axolotl YAML, Llama-Factory CLI); Unsloth's Triton kernel optimizations for 2–5× speedup; where LoRA/QLoRA theory meets production fine-tuning |

**Cross-references to seed**: HF Stack → LoRA/QLoRA, RLHF, DPO, GRPO, all model-family entries. FlexAttention → FlashAttention, torch.compile, Triton, Ring Attention, streaming-llm. Ray → verl (Ray-based), ByteDance Seed, agent-frameworks. Fine-tuning Toolchain → LoRA/QLoRA, Multi-tenant LoRA, on-prem-llm-deployment guide.

---

## Phase 20 — Training Recipe Primitives

**Rationale**: Every training entry assumes the "recipe" — optimizer schedule, stability tricks, activation choices, data mix. These are short but universal, and the cross-references compound across every model-family entry.

| # | Topic | Directory | Why |
|---|-------|-----------|-----|
| 70 | Training Stability — Loss Spikes, Z-loss, muP | `foundational/training-stability/` | Loss-spike detection + skip/restart, Z-loss regularization (PaLM, Gemini), muP (Maximal Update Parametrization) for HP transfer across scale, weight init schemes; every frontier paper mentions these in passing |
| 71 | LR Schedules & Optimizers | `foundational/lr-schedules-optimizers/` | WSD (warmup-stable-decay) as the new default, cosine vs WSD, Adafactor (memory-efficient), Lion (sign-momentum), Sophia (Hessian-aware); how batch-size / LR scaling laws interact |
| 72 | Activation & Norm Primitives | `foundational/activation-norm-primitives/` | SwiGLU (Llama/Mistral default), GeGLU, RMSNorm vs LayerNorm economics, QK-norm (stability), post-LN vs pre-LN; the architecture-level building blocks every modern transformer uses |
| 73 | Synthetic Data for Training | `foundational/synthetic-data/` | Phi series (filtered web + synthetic textbooks), Nemotron-4 340B as a reward-model+data-gen model, R1-distill data pipelines, Self-Instruct / Evol-Instruct; the data-side partner to Knowledge Distillation |

**Cross-references to seed**: Training Stability → Megatron-LM, Llama 3, Grok/Colossus, Mixed Precision. LR Schedules → Scaling Laws, all model-family entries. Activation/Norm → GQA, RoPE, Mamba, Llama 3, Qwen3, DeepSeek V3. Synthetic Data → Knowledge Distillation, Data Pipeline, DeepSeek R1 (distill models), Apple AFM.

---

## Phase 21 — Frontier Model Family & Reasoning Landscape

**Rationale**: Three remaining high-profile frontier-model gaps. Gemini and Nemotron are concrete infra stories; the reasoning-model landscape entry synthesizes across existing R1/Qwen3/inference-time-scaling entries to cover publicly-known-or-speculated o-series and Claude thinking mode. More narrative than Phases 19–20, lowest leverage but closes the frontier-family coverage.

| # | Topic | Directory | Why |
|---|-------|-----------|-----|
| 74 | Google Gemini 1.5 / 2.0 | `google/gemini/` | Native multimodal from pretraining (audio + video + image + text), 2M-token context, MoE family (Pro/Flash/Nano); TPU-first training stack; notable for long-context infra |
| 75 | NVIDIA Nemotron | `nvidia/nemotron/` | 340B base + reward + instruct family; synthetic-data pipeline (86% synthetic in Nemotron-4-340B-Instruct SFT); NeMo training stack; TransformerEngine + Megatron-Core reference consumer |
| 76 | Reasoning Models Landscape | `foundational/reasoning-models/` | Meta-analysis across o1/o3 (publicly known), Claude 3.7/4 thinking, DeepSeek R1, Qwen3 thinking mode; infra implications of long-CoT: KV cache growth, thinking-token economics, budget-forcing, hidden-reasoning pricing models |

**Cross-references to seed**: Gemini → Pathways, GSPMD, Gemma 2 (smaller sibling), Ring Attention (2M context peer), DiT/multimodal. Nemotron → TransformerEngine, Megatron-Core, Knowledge Distillation, Synthetic Data (Phase 20), CUTLASS. Reasoning Models → Inference-Time Scaling, Process Reward Models, R1, Qwen3, GRPO.

---

## Execution Notes

- **Priority order (Phases 19–21)**: Phase 19 first — reference-framework entries referenced by nearly every existing topic; returns compound. Phase 20 second — recipe primitives that universally sit under every training entry. Phase 21 last — narrative frontier-model entries; valuable for completeness but lower leverage.
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
