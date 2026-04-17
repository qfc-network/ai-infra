# ai-infra Deep Dive Roadmap

## Context

The repo currently has 78 topics plus 2 guides across training, inference, architectures, open-source infra, and multimodal. The goal is to systematically fill the gaps — topics that are either directly referenced by existing write-ups or are widely used in practice but not yet covered. Each new entry follows the established template: ~2,000–2,500 words, 1–2 equations, one Engineering Tradeoffs table (4–6 rows), both `en.md` and `zh.md`, and an update to the root `README.md` / `README.zh.md` index.

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

## Active Gap Analysis (post-Phase 12)

### Stack-completeness gaps (high priority)

- **Diffusion foundations** — the DiT entry (Phase 10) assumes DDPM / DDIM / classifier-free guidance without explaining them. Retroactive prerequisite.
- **Sequence Parallelism variants** — Ring Attention is covered; Ulysses (DeepSpeed) and Megatron-CP (the two other mainstream sequence-axis parallelism approaches used at frontier scale) are not.
- **`torch.compile` / Inductor** — Triton is covered at the DSL level; the PyTorch compilation layer that actually orchestrates Triton in production training is missing. Referenced implicitly by FSDP, FlashAttention, and nearly every modern training entry.
- **Knowledge Distillation at Scale** — referenced by Gemma 2, R1's distilled models, and the InstructGPT entry; no standalone treatment of white-box vs black-box distillation, step-by-step, or MiniLLM-style.
- **DeepSeek V3.2 / Native Sparse Attention** — the DeepGEMM walkthrough mentions the V3.2 sparse indexer; DeepSeek's 2025 sparse attention work deserves standalone coverage.

### New-axis gap (high strategic value)

- **Agent infrastructure** — the Anthropic section covers theory (MCP, Building Effective Agents) but has zero coverage of how agent systems are actually built in production: tool calling schemas, parallel tool calls, agent framework landscape (LangGraph / AutoGen / Swarm / DSPy), Computer Use, or agent evaluation infrastructure (SWE-bench, TAU-bench). This is the direction frontier-lab infra work is heading in 2026.

### Lab-completeness gaps (medium priority)

- **xAI** — no entry; Colossus cluster (100k+ H100) is the largest single-site training cluster currently operating.
- **ByteDance / Seed** — major Chinese frontier lab with Doubao series; no coverage despite being a significant infra contributor (verl originated here).
- **Apple** — no entry on Apple Foundation Models; on-device LLM inference is a distinct infra regime worth documenting.

### Lower-priority leftovers
- **RLHF reward model training pipeline** — could extend the existing RLHF entry rather than a new write-up.
- **SLO-aware scheduling depth** — covered in pieces across Orca / Sarathi / DistServe; dedicated synthesis possible but not urgent.
- **MoE routing improvements** — Expert Choice, Loss-Free Balancing (used in DeepSeek V3) — candidate for a future MoE-deep-dive entry.

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

## Phase 14 — Agent Infrastructure (new strategic axis)

**Rationale**: The Anthropic section has theory (MCP, Building Effective Agents); zero coverage of how agent systems are actually built and evaluated in production. This is where frontier-lab infra investment is going in 2026.

| # | Topic | Directory | Why |
|---|-------|-----------|-----|
| 48 | Tool Use / Function Calling Infrastructure | `foundational/tool-use-infra/` | Parallel tool calls, schema validation, tool-call trajectories, safety layers; how OpenAI / Anthropic / Gemini function-calling protocols actually work in serving |
| 49 | Agent Framework Landscape | `foundational/agent-frameworks/` | LangGraph (stateful graphs), AutoGen (multi-agent), OpenAI Swarm (handoffs), DSPy (prompt compilation); when to use which, tradeoffs, infra implications |
| 50 | Computer Use & Browser Automation | `anthropic/computer-use/` | Anthropic's Computer Use model + API; screenshot-based visual grounding, action space, latency economics; pairs naturally with VLM Serving entry |
| 51 | Agent Evaluation Infrastructure | `foundational/agent-evaluation/` | SWE-bench (code agents), TAU-bench (customer service), WebArena (browser agents); how agent workloads are actually graded; why this is harder than LM benchmarks |

**Cross-references to seed**: Tool Use → MCP, Building Effective Agents. Agent Frameworks → Building Effective Agents, SGLang. Computer Use → VLM Serving, LLaVA. Agent Evaluation → Inference-Time Scaling, Process Reward Models.

---

## Phase 15 — Frontier Lab Completeness

**Rationale**: Three major lab-level infra stories currently absent from the repo. Lower priority than Phases 13–14 because entries are more narrative than technical-reference, but important for "frontier labs infra" framing.

| # | Topic | Directory | Why |
|---|-------|-----------|-----|
| 52 | xAI Grok + Colossus | `xai/grok-colossus/` | Largest single-site training cluster currently operating (100k+ H100, 200k target); Memphis deployment; story of building at that scale |
| 53 | ByteDance Seed / Doubao | `bytedance/seed/` | Major Chinese frontier lab; Doubao model series; origin lab for verl and HybridFlow; infra choices reflective of Chinese scaling constraints |
| 54 | Apple Foundation Models (AFM) | `apple/afm/` | On-device LLM inference as a distinct infra regime: server + device models, Private Cloud Compute, MLX at lab scale; different tradeoffs than cloud-only labs |

**Cross-references to seed**: Grok/Colossus → Llama 3 (comparative cluster-scale infra), GPU Interconnect. ByteDance Seed → verl (originated here), DeepSeek (peer Chinese lab). Apple AFM → the on-prem-llm-deployment guide (MLX section), Mamba-SSM (on-device architecture tradeoffs).

---

## Execution Notes

- **Priority order**: Phase 13 (stack completeness) first — each entry fills a gap already implicitly referenced by existing entries, so returns compound across the repo. Phase 14 (agent infra) second — highest strategic value but introduces a new axis, worth doing after internal coherence is improved. Phase 15 (lab completeness) last — valuable but narrative-heavy and lower leverage than the technical gaps in 13.
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
