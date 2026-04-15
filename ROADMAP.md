# ai-infra Deep Dive Roadmap

## Context

The repo currently has 35 topics across training, inference, architectures, and open-source infra. The goal is to systematically fill the gaps — topics that are either directly referenced by existing write-ups or are widely used in practice but not yet covered. Each new entry follows the established template: ~2,000–2,500 words, 1–2 equations, one Engineering Tradeoffs table (4–6 rows), both `en.md` and `zh.md`, and an update to the root `README.md` / `README.zh.md` index.

---

## Gap Analysis

### What existing write-ups assume but don't explain
- **Scaling Laws** — DeepSeek V3, Llama 3, and Chinchilla results are cited constantly; no standalone entry
- **GQA** — used in Llama 3, Mistral, DeepSeek V3; MLA write-up compares against it
- **RoPE** — used in virtually every modern LLM; referenced in MLA, Llama 3, R1
- **Orca / Continuous Batching** — vLLM, SGLang, DistServe all build on this; no entry
- **LoRA / QLoRA** — the default fine-tuning method; RLHF and DPO entries assume it

### Inference gaps
- **Chunked Prefill (Sarathi-Serve)** — addresses prefill/decode interference that DistServe motivates
- **KV Cache Quantization (KIVI / KVQuant)** — natural companion to Weight Quantization and SmoothQuant

### Training gaps
- **Mixed Precision Training (AMP / BF16)** — the prerequisite for FP8 work in DeepSeek V3 and FlashAttention-3
- **GRPO** — described inside R1 but deserves a standalone treatment as a general RL algorithm

### Alignment / post-training gaps
- **Constitutional AI** — Anthropic's method; pairs with existing RLHF and DPO entries
- **SimPO** — simpler DPO variant gaining adoption; natural successor to DPO entry

### New vendor sections
- **NVIDIA**: Cutlass (the kernel template library), TensorRT-LLM (production inference engine)
- **Microsoft**: DeepSpeed (the other major distributed training framework alongside Megatron/ZeRO)
- **OpenAI**: Scaling Laws (Kaplan 2020) — the original, plus Chinchilla (Hoffmann 2022)

---

## Phased Roadmap

### Phase 1 — Foundational Gaps (highest leverage, unblock other write-ups)

| # | Topic | Directory | Why now |
|---|-------|-----------|---------|
| 1 | Scaling Laws (Kaplan + Chinchilla) | `foundational/scaling-laws/` | Every compute-budget decision references this |
| 2 | Grouped Query Attention (GQA) | `foundational/gqa/` | Used in Llama 3, Mistral, DeepSeek; MLA entry compares against it |
| 3 | Rotary Position Embeddings (RoPE) | `foundational/rope/` | Assumed by MLA, Llama 3, R1; enables long-context work |
| 4 | Orca — Continuous Batching | `foundational/orca/` | Foundation for vLLM, SGLang, DistServe, Mooncake |
| 5 | LoRA / QLoRA | `foundational/lora/` | Default fine-tuning; assumed by RLHF, DPO, R1 distillation |

### Phase 2 — Inference & Training Depth

| # | Topic | Directory | Why |
|---|-------|-----------|-----|
| 6 | Mixed Precision Training (AMP / BF16 / FP8) | `foundational/mixed-precision/` | Prerequisite for FlashAttention-3 FP8, DeepSeek FP8 training |
| 7 | Chunked Prefill / Sarathi-Serve | `foundational/chunked-prefill/` | Closes prefill-decode interference gap after DistServe |
| 8 | KV Cache Quantization (KIVI / KVQuant) | `foundational/kv-cache-quantization/` | Completes the quantization arc (weights → activations → KV) |
| 9 | GRPO | `foundational/grpo/` | Standalone treatment; R1 entry only sketches it |

### Phase 3 — Alignment & Post-Training

| # | Topic | Directory | Why |
|---|-------|-----------|-----|
| 10 | Constitutional AI | `anthropic/constitutional-ai/` | Anthropic's core alignment method; pairs with RLHF + DPO |
| 11 | SimPO | `foundational/simpo/` | Reference-free DPO successor; growing adoption |

### Phase 4 — New Vendor Sections

| # | Topic | Directory | Why |
|---|-------|-----------|-----|
| 12 | Scaling Laws paper (OpenAI) | `openai/scaling-laws/` | Author org; new `openai/` section |
| 13 | DeepSpeed (ZeRO-Infinity, ZeRO-Offload) | `microsoft/deepspeed/` | Major training framework; new `microsoft/` section |
| 14 | Cutlass | `nvidia/cutlass/` | The kernel template library Triton competes with; new `nvidia/` section |
| 15 | TensorRT-LLM | `nvidia/tensorrt-llm/` | Production inference engine; pairs with Triton and vLLM |

### Phase 5 — Model Family Gaps

| # | Topic | Directory | Why |
|---|-------|-----------|-----|
| 16 | Gemma 2 | `google/gemma2/` | Google open model; distillation, logit-softcapping, alt-attention |
| 17 | Llama 2 | `meta/llama2/` | Bridges the gap before Llama 3 entry |

---

## Execution Notes

- **Each topic**: `en.md` + `zh.md`, update `README.md` + `README.zh.md` index
- **Format**: ~2,000–2,500 words, 1–2 key equations, 1 tradeoffs table (4–6 rows)
- **Tone**: engineering-first — what are the memory/compute/latency consequences?
- **Start with Phase 1** — each entry in Phase 1 is a prerequisite for understanding 3+ existing write-ups

## Files to modify per entry
- `{vendor}/{topic}/en.md` (new)
- `{vendor}/{topic}/zh.md` (new)
- `README.md` (index entry)
- `README.zh.md` (index entry)
