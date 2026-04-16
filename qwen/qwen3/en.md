# Qwen3

- **Authors / Org**: Qwen Team, Alibaba Cloud
- **Published**: 2025-04
- **Links**: [Qwen3 blog](https://qwenlm.github.io/blog/qwen3/) · [HuggingFace collection](https://huggingface.co/collections/Qwen/qwen3-67dd247413c7f01c2f5ed641)

## TL;DR

Qwen3 is Alibaba's most capable open-weight model family, released April 2025. The lineup spans from 0.6B to 235B parameters and introduces a key serving design decision: a **thinking/non-thinking toggle** that switches between a full chain-of-thought reasoning mode and a direct-answer mode at inference time, without requiring two separate models. The flagship is **Qwen3-235B-A22B**, a MoE model with 235B total parameters and 22B activated per token — outperforming GPT-4o and Claude 3.7 Sonnet on major benchmarks at roughly half the serving compute of a comparably-capable dense model. Dense variants (Qwen3-32B down to 0.6B) cover deployment scenarios where MoE is impractical. This entry covers the infrastructure implications: the thinking toggle mechanism, MoE architecture choices, and how the model family is structured for practical deployment.

## Context & Motivation

Qwen2.5 (late 2024) established Alibaba as a serious open-weight competitor: strong multilingual coverage, competitive coding and math, and a wide range of sizes. Qwen3 extends this with two structural changes:

1. **Thinking mode**: inspired by DeepSeek-R1's success, Qwen3 incorporates reasoning-via-chain-of-thought as a native capability. Rather than training separate "thinking" and "chat" models, Qwen3 trains a single model that can operate in both modes via a soft switch.

2. **MoE at the top**: Qwen3-235B-A22B follows the pattern established by DeepSeek V3 and Llama 4 — a large sparse model where serving cost is determined by activated parameters (22B), not total parameters (235B).

The combined result: a model that can match frontier quality on reasoning tasks while being deployable on hardware that a 235B dense model would require three times as much of.

## Thinking / Non-Thinking Toggle

### The Mechanism

Qwen3 is trained with both chain-of-thought reasoning (thinking) and direct answer (non-thinking) data. The switch is controlled at inference time by a **system prompt instruction** or API flag rather than separate model weights.

- **Thinking mode** (`enable_thinking=True`): the model generates an internal `<think>...</think>` block before the final answer. The thinking tokens are not shown to the user (or are shown as a collapsible block, depending on the application). This is equivalent to the reasoning trace pattern in [DeepSeek-R1](../../deepseek/r1/en.md) and the test-time compute scaling described in [Inference-Time Scaling](../../foundational/inference-time-scaling/en.md).

- **Non-thinking mode** (`enable_thinking=False`): the model produces a direct answer, suppressing the `<think>` block. Latency and cost drop significantly — no reasoning tokens are generated.

### Training Implications

To support both modes from the same weights, the model is trained in multiple stages:

1. **Long-CoT cold start**: fine-tune on curated chain-of-thought traces to establish reasoning ability.
2. **Reasoning RL**: GRPO-style reinforcement learning with rule-based rewards (verifiable answers: math, code) — same approach as [R1](../../deepseek/r1/en.md).
3. **Mode merging**: further SFT and RL that teaches the model to switch cleanly between thinking and non-thinking, so that disabling thinking does not degrade non-reasoning task performance.

The key challenge in stage 3: thinking mode requires long outputs (reasoning traces can be thousands of tokens); non-thinking mode requires short, direct outputs. Training one model to do both well requires careful data mixing and reward design to prevent mode collapse.

### Serving Implications

The toggle has direct infrastructure consequences:

- **KV cache sizing**: in thinking mode, the KV cache must accommodate the reasoning trace (potentially thousands of tokens) before the answer begins. Non-thinking mode has standard KV cache requirements.
- **TTFT vs throughput tradeoff**: thinking mode has high TTFT (must generate the full reasoning trace before the answer can begin). Non-thinking mode has low TTFT. Serving infrastructure must accommodate both patterns in the same deployment.
- **Cost accounting**: some providers charge for thinking tokens; others hide them. The actual compute cost of a thinking-mode request is 3–10× higher than non-thinking for the same final answer length.

## MoE Architecture: Qwen3-235B-A22B

### Design Choices

Qwen3's MoE follows the pattern established in [DeepSeekMoE](../../deepseek/moe/en.md) and refined in V3:

- **Fine-grained experts**: each expert is smaller than in coarser designs (e.g., Mixtral's 8 large experts), with more total experts. 235B / 22B activated means roughly 10× expansion ratio.
- **Top-K routing**: top-K with K > 1 provides redundancy — a token that is poorly served by one expert is partially compensated by a second. Unlike Llama 4's top-1, Qwen3 uses top-2+ routing.
- **No shared expert** (unlike DeepSeekMoE): routing is entirely dynamic; there is no always-active FFN.

The tradeoffs relative to dense models are the same as any MoE: lower serving compute (22B activated), higher total memory requirement (235B parameters), additional all-to-all communication per MoE layer for expert parallelism.

### Dense Variants

For scenarios where MoE is impractical (single-GPU deployment, absence of expert parallelism infrastructure), Qwen3 ships dense variants: 32B, 14B, 8B, 4B, 1.7B, 0.6B. All support the thinking/non-thinking toggle.

The 32B dense variant is a notable point: it outperforms Qwen2.5-72B (the previous generation flagship) at less than half the parameter count, reflecting the gains from RL-based post-training.

## Architecture Details

Across the Qwen3 family:

- **Attention**: GQA throughout (all sizes).
- **Position encoding**: RoPE with YaRN-based long-context extension.
- **Context length**: 32k base, extended to 128k via YaRN scaling for most variants; 1M context for the MoE variant with additional position interpolation.
- **Vocabulary**: 151,936 tokens (inherited from Qwen2.5, covering ~30 languages).
- **Normalization**: RMSNorm pre-norm, same as Llama 3 / Qwen2.5.

No architectural novelties beyond the MoE structure and thinking toggle — the value is in the training recipe.

## Training Recipe Highlights

### Data

Qwen3 is trained on ~36T tokens (Qwen3-235B). Data quality is heavily filtered:

- Multi-stage quality scoring (language model perplexity filters, rule-based heuristics, classifier-based filtering).
- Synthetic data generated by stronger models for math, code, and reasoning.
- Strong multilingual coverage: Chinese and English dominate; ~30 other languages represented.

### Post-Training

The post-training pipeline is the key differentiator from Qwen2.5:

1. **SFT on diverse tasks** (instruction following, coding, math, multilingual).
2. **Long-CoT SFT** for reasoning capability cold-start.
3. **RL with verifiable rewards** (GRPO; same pattern as DeepSeek-R1).
4. **Mode-merging SFT + RL** to stabilize thinking/non-thinking switching.
5. **Final DPO pass** for response style and safety.

The RL stage is responsible for the largest quality jump — aligning with the [Inference-Time Scaling](../../foundational/inference-time-scaling/en.md) thesis that reasoning capability is more efficiently gained by RL-driven chain-of-thought than by pure pretraining scale.

## Engineering Tradeoffs

| Decision | Gained | Gave up |
|---|---|---|
| Single-model thinking toggle | One set of weights for both modes; simpler deployment | Training complexity; mode collapse risk during post-training |
| MoE at 235B/22A | Serving cost of a 22B model; quality of a 235B model | Expert parallelism infrastructure required; total weight storage |
| Fine-grained top-2 routing | Redundancy; smoother loss surface | 2× all-to-all per layer vs top-1 |
| Dense variants (32B–0.6B) | Wide deployment range; no EP infrastructure needed | Higher serving compute than MoE at equivalent quality |
| 128k–1M context via YaRN | Long-context capable out of the box | YaRN interpolation degrades slightly on mid-range context; KV cache grows linearly |
| RL-heavy post-training | Large reasoning quality gains; MATH/code benchmarks | RL instability risk; reward hacking on verifiable benchmarks |

## Benchmark Position (April 2025)

Qwen3-235B-A22B on major reasoning benchmarks:

- **AIME 2024**: ~85% pass@1 (competitive with o1-level models in thinking mode).
- **LiveCodeBench**: top-3 open-weight.
- **MMLU-Pro, GPQA**: competitive with GPT-4o.

Qwen3-32B (dense): competitive with Qwen2.5-72B and Llama 3 405B on many benchmarks, at 32B parameters.

These numbers are most meaningful when read alongside the serving compute: Qwen3-235B-A22B achieves frontier results at 22B activated parameters per token, significantly below comparably-capable dense models.

## Commentary

Qwen3's thinking toggle is the most operationally interesting aspect for inference infrastructure. It acknowledges that reasoning capability and direct-answer capability are best served by different inference strategies — but operational simplicity argues strongly for a single set of weights. The technical challenge of training one model to do both well without mode collapse is non-trivial; the public Qwen3 release suggests it is achievable with careful RL-based post-training.

For teams choosing between Qwen3-235B-A22B and Llama 4 Maverick: the MoE cost structure is similar (both ~17–22B activated), but Qwen3 has stronger multilingual coverage and a more established RL post-training pipeline; Llama 4 has native multimodal capability and better long-context architecture (iRoPE vs YaRN). The choice depends heavily on use case.

The 32B dense variant is the practical default for most deployments without expert parallelism infrastructure — it fits in 2 × 80 GB GPUs in BF16, supports thinking mode, and matches previous-generation 70B+ models in quality.

## References

- [1] Qwen Team. _Qwen3 Technical Report._ Alibaba Cloud, 2025.
- [2] Qwen Team. Blog post: "Qwen3: Think Deeper, Act Faster." qwenlm.github.io, April 2025.
- [3] For thinking mode context: [DeepSeek-R1](../../deepseek/r1/en.md), [GRPO](../../foundational/grpo/en.md), [Inference-Time Scaling](../../foundational/inference-time-scaling/en.md).
- [4] For MoE context: [DeepSeekMoE](../../deepseek/moe/en.md), [Mixtral](../../mistral/mixtral/en.md), [Llama 4](../llama4/en.md).
- [5] For long-context context: [RoPE and long-context extensions](../../foundational/rope/en.md), [Ring Attention / Context Parallelism](../../foundational/ring-attention/en.md).
