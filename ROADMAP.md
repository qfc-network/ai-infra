# ai-infra Deep Dive Roadmap

## Context

The repo currently has 59 topics plus 2 guides across training, inference, architectures, and open-source infra. The goal is to systematically fill the gaps — topics that are either directly referenced by existing write-ups or are widely used in practice but not yet covered. Each new entry follows the established template: ~2,000–2,500 words, 1–2 equations, one Engineering Tradeoffs table (4–6 rows), both `en.md` and `zh.md`, and an update to the root `README.md` / `README.zh.md` index.

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

## Active Gap Analysis (post-Phase 9)

### Structural gaps (high priority)

- **Multimodal infrastructure — zero coverage** — Llama 4 and Qwen3 are natively multimodal but the repo has no entries on CLIP, ViT, VLM architecture, speech, or diffusion-transformer video. The single biggest structural gap. No `multimodal/` section exists yet.
- **Data & tokenization — zero coverage** — every training entry assumes a tokenized corpus; nothing explains how the corpus is built. BPE / SentencePiece / tokenizer training, data pipelines (FineWeb, MinHash dedup, quality filtering) are all missing.
- **Position-interpolation long context** — Ring Attention covers the compute story; YaRN / LongRoPE / NTK-aware scaling (what every production long-context model actually uses to extend pretrained context) are missing.

### Completeness gaps (medium priority)

- **MoE foundations** — DeepSeekMoE and Mixtral are covered, but Switch Transformer and GShard (the papers that made sparse MoE viable at scale) are missing. Historical-foundations gap.
- **Production inference 2.0** — vLLM, SGLang, TensorRT-LLM are covered; TGI (Hugging Face), NVIDIA Dynamo (2025), and multi-tenant LoRA serving (SLoRA / Punica) are not. Recency gap for anyone operating a serving platform today.

### Lower-priority leftovers
- **RLHF reward model training pipeline** — data collection, preference labeling, RM architecture, Goodharting mitigations — could extend the existing RLHF entry rather than a new write-up.
- **SLO-aware scheduling depth** — chunked prefill + decode interleaving policies are covered across Orca / Sarathi / DistServe entries; a dedicated synthesis entry is possible but not urgent.

---

## Phased Roadmap — Next Phases

### Phase 7 — Hardware & Systems Depth (recommended next)

| # | Topic | Directory | Why |
|---|-------|-----------|-----|
| 19 | Hopper / H100 Architecture Primer | `foundational/hopper-h100/` | ✓ done |
| 20 | Speculative Decoding — Medusa / EAGLE | `foundational/speculative-decoding-variants/` | Extends existing speculative decoding entry; tree-based verification, self-draft methods in production |
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

## Phase 10 — Multimodal Infrastructure (recommended next)

**Rationale**: The biggest remaining structural gap. Llama 4 and Qwen3 ship native vision; multimodal serving has its own prefill economics (a single 1024×1024 image becomes ~2000+ tokens), its own tokenizers (ViT patches, log-mel frames), and its own fusion architectures. This phase creates the new `multimodal/` section and fills the core.

| # | Topic | Directory | Why |
|---|-------|-----------|-----|
| 29 | CLIP — Contrastive Language-Image Pretraining | `multimodal/clip/` | The foundation of all modern VLMs; dual-encoder contrastive objective; cited by every downstream multimodal paper |
| 30 | Vision Transformer (ViT) | `multimodal/vit/` | How images become token sequences; patch embedding; directly assumed by LLaVA and every native-vision LLM |
| 31 | LLaVA / Vision-Language Model architecture | `multimodal/llava/` | How vision features fuse with a pretrained LLM decoder; projector design; two-stage training; the dominant VLM recipe |
| 32 | Whisper — speech recognition | `multimodal/whisper/` | Speech infra: log-mel spectrograms, encoder-decoder, weakly-supervised pretraining; the analog of CLIP for audio |
| 33 | DiT — Diffusion Transformers | `multimodal/dit/` | The backbone of Sora / Stable Diffusion 3 / video generation; transformer replaces UNet in diffusion; infra implications for image + video gen |
| 34 | VLM Serving | `multimodal/vlm-serving/` | Production specifics: variable-resolution tiling, visual-token prefill cost, KV cache sizing with images, cross-modal attention patterns — synthesis entry |

**Cross-references to seed**: CLIP → RLHF (reward modeling uses CLIP-style encoders). ViT → FlashAttention (same attention kernel). LLaVA → LoRA (vision projector is often LoRA-tuned). DiT → FlashAttention, Mamba-SSM (hybrid DiT variants). VLM Serving → PagedAttention, Prefix Caching, Chunked Prefill.

---

## Phase 11 — Data, Tokenization & Long Context

**Rationale**: Closes the "how was this data prepared" gap and the position-interpolation gap. Prerequisite for anyone pretraining or fine-tuning from scratch.

| # | Topic | Directory | Why |
|---|-------|-----------|-----|
| 35 | Tokenization — BPE, SentencePiece, Tiktoken | `foundational/tokenization/` | Every model entry assumes a tokenizer; this explains how one is trained; multilingual coverage, byte-level fallback, token budget economics |
| 36 | Data Pipelines — FineWeb / MinHash / Quality Filtering | `foundational/data-pipeline/` | The unglamorous infrastructure behind pretraining: Common Crawl processing, deduplication at trillion-token scale, quality classifiers |
| 37 | Position Interpolation — YaRN / LongRoPE / NTK-aware | `foundational/position-interpolation/` | How RoPE models extend from 4k → 128k+ without retraining from scratch; used by Llama 3, Qwen3, DeepSeek; extends the existing RoPE entry |
| 38 | Streaming LLM & Attention Sinks | `foundational/streaming-llm/` | Sliding-window attention with attention sinks; infinite-length inference with bounded KV cache; used in practice for very long conversations |

**Cross-references**: Tokenization → Scaling Laws (20 tokens/param assumes *which* tokens). Data Pipeline → Scaling Laws, Llama 3. Position Interpolation → RoPE, Ring Attention, Llama 3. Streaming LLM → PagedAttention, KV Cache Quantization.

---

## Phase 12 — MoE Foundations & Production Inference 2.0

**Rationale**: Fills the historical MoE gap (DeepSeekMoE / Mixtral both assume Switch / GShard) and covers the production inference stacks launched in 2024–2025.

| # | Topic | Directory | Why |
|---|-------|-----------|-----|
| 39 | Switch Transformer & GShard | `foundational/switch-gshard/` | The foundational MoE papers; top-1 routing, auxiliary load balancing, expert capacity; prerequisite context for DeepSeekMoE entry |
| 40 | Hugging Face TGI — Text Generation Inference | `foundational/tgi/` | Rust-based production serving used widely outside the vLLM ecosystem; different design choices, different tradeoffs |
| 41 | NVIDIA Dynamo | `nvidia/dynamo/` | NVIDIA's 2025 unified inference platform for disaggregated serving at scale; pairs with TensorRT-LLM entry |
| 42 | Multi-tenant LoRA Serving — SLoRA / Punica | `foundational/multi-tenant-lora/` | Serving thousands of LoRA adapters from one base model; memory management, batching across adapters; fills a growing production niche |

**Cross-references**: Switch/GShard → DeepSeekMoE, Mixtral (backward cross-link). TGI → vLLM, SGLang. Dynamo → TensorRT-LLM, DistServe, Mooncake. Multi-tenant LoRA → LoRA, PagedAttention.

---

## Execution Notes

- **Priority order**: Phase 10 (multimodal) first — it's the largest structural gap and unblocks future multimodal entries. Phase 11 and 12 can proceed in either order; data/tokenization (Phase 11) is more foundational, production inference (Phase 12) is more recency-driven.
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
