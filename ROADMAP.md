# ai-infra Deep Dive Roadmap

## Context

The repo currently has 51 topics across training, inference, architectures, and open-source infra. The goal is to systematically fill the gaps — topics that are either directly referenced by existing write-ups or are widely used in practice but not yet covered. Each new entry follows the established template: ~2,000–2,500 words, 1–2 equations, one Engineering Tradeoffs table (4–6 rows), both `en.md` and `zh.md`, and an update to the root `README.md` / `README.zh.md` index.

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

## Active Gap Analysis

### Hardware gaps
- **Hopper / H100 Architecture Primer** — FlashAttention-3, DeepGEMM, DualPipe, DeepEP all target Hopper-specific features (wgmma, TMA, async proxy, IBGDA, NVLink 4) without explaining them. A primer here makes ~6 existing entries more self-contained. Cross-reference leverage: highest in the repo.

### DeepSeek arc gaps
- **DeepSeek Prover / Math Reasoning Infra** — applies RLHF-style training to formal theorem proving (Lean 4); uses MCTS over proof states as the search backend. Bridges the inference-time scaling entry with the formal verification world.

### Model family gaps
- **Gemma 2** (Phase 5, still pending) — distillation at scale, logit-softcapping, alternating local/global attention.

### Inference systems
- **Speculative Decoding variants — Medusa / EAGLE** — vanilla speculative decoding is covered; the production-grade tree-based variants (Medusa heads, EAGLE draft model) are what vLLM/SGLang actually ship. Natural extension of the existing speculative decoding entry.
- **Prefix Caching** — SGLang's RadixAttention is mentioned but prefix caching as a standalone system design (eviction policy, hash-based matching, hit rate economics) has no dedicated entry.

### Post-training / alignment
- **Process Reward Models (PRMs)** — referenced in the inference-time scaling entry and GRPO commentary; no standalone treatment of how PRMs are trained, labeled (PRM800K), and integrated into search.
- **RLHF at scale — reward model training** — the RLHF entry covers PPO; the reward model training pipeline (data collection, preference labeling, RM architecture, Goodharting mitigations) has no dedicated entry.

---

## Phased Roadmap — Next Phases

### Phase 7 — Hardware & Systems Depth (recommended next)

| # | Topic | Directory | Why |
|---|-------|-----------|-----|
| 19 | Hopper / H100 Architecture Primer | `foundational/hopper-h100/` | ✓ done |
| 20 | Speculative Decoding — Medusa / EAGLE | `foundational/speculative-decoding-variants/` | Extends existing speculative decoding entry; tree-based verification, self-draft methods in production |
| 21 | Prefix Caching | `foundational/prefix-caching/` | ✓ done |

### Phase 8 — Post-Training Depth

| # | Topic | Directory | Why |
|---|-------|-----------|-----|
| 22 | Process Reward Models (PRMs) | `foundational/process-reward-models/` | ✓ done |
| 23 | Gemma 2 | `google/gemma2/` | Phase 5 carry-over; distillation, softcapping, alternating attention |
| 24 | DeepSeek Prover | `deepseek/prover/` | Lean 4 theorem proving via MCTS; connects formal verification to the reasoning model arc |

---

## Execution Notes

- **Each topic**: `en.md` + `zh.md`, update `README.md` + `README.zh.md` index
- **Format**: ~2,000–2,500 words, 1–2 key equations, 1 tradeoffs table (4–6 rows)
- **Tone**: engineering-first — what are the memory/compute/latency consequences?
- **Cross-references**: link to related entries; do not repeat content already in a standalone entry

## Files to modify per entry
- `{vendor}/{topic}/en.md` (new)
- `{vendor}/{topic}/zh.md` (new)
- `README.md` (index entry)
- `README.zh.md` (index entry)
- `{vendor}/README.md` (if vendor-specific index exists)
