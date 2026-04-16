# DeepSeek-V2 — Economical MoE at 236B

- **Authors / Org**: DeepSeek-AI
- **Published**: 2024-05
- **Links**: [paper (arXiv:2405.04434)](https://arxiv.org/abs/2405.04434) · [model (HuggingFace)](https://huggingface.co/deepseek-ai/DeepSeek-V2) · [code](https://github.com/deepseek-ai/DeepSeek-V2)

## TL;DR

DeepSeek-V2 is the first public demonstration that two architectural innovations — MLA (Multi-head Latent Attention) and DeepSeekMoE — can be deployed together at frontier scale to produce a model that is simultaneously more capable, cheaper to train, and cheaper to serve than a comparable dense model. At 236B total parameters with only 21B activated per token, V2 achieves KV cache reduction of ~93% versus a comparable MHA model, and sustains generation throughput 5.76× higher than DeepSeek 67B while matching or outperforming it on most benchmarks. V2 is the architectural proof-of-concept that V3 then scaled to 671B; the two papers form a continuous engineering narrative.

## Context & Motivation

### The cost problem in 2024

By mid-2024, the economics of frontier LLM inference were under scrutiny. A GPT-4-class dense model requires:

1. **KV cache proportional to sequence length and head count**: at long context, the KV cache alone can exceed the model weight memory for a single request.
2. **All parameters active per token**: a 70B dense model reads and computes through all 70B parameters for every token generated — no sparsity, no reuse.
3. **Inference cost that scales with model size**: serving a 70B model costs roughly 10× more per token than serving a 7B model.

The MoE approach addresses (2) by activating only a fraction of parameters per token. The KV cache problem (1) is orthogonal and not solved by standard MoE. DeepSeek-V2 is the first model to attack both simultaneously at scale.

### The V2 hypothesis

Given MLA (compresses KV cache by ~10× with no quality loss — see [MLA entry](../mla/)) and DeepSeekMoE (activates ~9% of parameters per token — see [MoE entry](../moe/)), the question is: do these two innovations compose well at scale? V2 is the affirmative answer.

The target: a model competitive with GPT-4-class systems, deployable at a cost closer to a 21B dense model than a 236B dense model.

## Core Method

### Architecture overview

| Dimension | Value |
|---|---|
| Total parameters | 236B |
| Activated per token | 21B (~9%) |
| Transformer layers | 60 |
| Context window | 128k tokens |
| Attention | MLA throughout |
| FFN layers | DeepSeekMoE (fine-grained experts + shared experts) |

The first layer uses a standard dense FFN; all subsequent FFN layers use MoE. This pattern (dense first layer + sparse remainder) stabilizes training by ensuring the initial representation is not immediately filtered through routing.

### MLA: eliminating 93% of KV cache

Full details in the [MLA entry](../mla/). The key system-level consequence: at 128k context, the KV cache per sequence falls from the MHA equivalent (~12 GB for a 7B-activated model) to ~800 MB. This changes the memory arithmetic of batched inference fundamentally — you can fit far more concurrent sequences before HBM is exhausted. The result is higher throughput at the same hardware, not just lower memory.

The decoupled RoPE design (small dimension that receives position encoding, stored separately from the compressed latent) means MLA is compatible with any positional embedding extension technique. V2's 128k window is achieved with YaRN-style scaling of the RoPE frequency base — no architectural change required.

### DeepSeekMoE: fine-grained specialists with shared isolation

Full details in the [MoE entry](../moe/). At V2 scale, the fine-grained configuration means a large number of smaller routed experts with top-K routing, plus a small number of shared experts that every token passes through. The shared experts absorb high-frequency, cross-domain features; the routed experts specialize.

The system-level implication: the all-to-all communication pattern for expert dispatch involves more experts than a standard 8-expert MoE, but each expert is smaller. The per-expert activation time is shorter, giving the network stack more opportunity to overlap communication with compute — a property V3 would exploit explicitly with DualPipe.

### Expert parallelism at V2 scale

Training V2 requires partitioning experts across multiple nodes. Each node holds a shard of the expert pool; the all-to-all dispatch sends token representations to the node hosting the selected expert, and the results return. At V2's expert count, each token's dispatch touches expert shards across multiple nodes per MoE layer, and this happens for every token in every micro-batch.

DeepSeek mitigates this with:
1. **Expert load balancing**: auxiliary loss on routing logits to prevent collapse to a small subset of experts (V3 later replaced this with the cleaner bias-based approach).
2. **Topology-aware routing constraints**: limiting the number of nodes a token's experts can span to bound the all-to-all volume. This trades some routing flexibility for predictable communication cost.

### Training data and curriculum

V2 was trained on 8.1T tokens of predominantly English and Chinese text, with code and math data. The curriculum follows two stages:

1. **Pre-training on 8.1T tokens at 4k context**: standard next-token prediction loss; cosine learning rate decay.
2. **Long-context extension to 128k**: continued pre-training on a mix of long documents (books, codebases, multi-turn dialogue), using YaRN to adjust RoPE frequency. ~10B additional tokens.

The MLA architecture means that extending context during stage 2 does not significantly increase training memory — the KV cache compression applies during training too, since the compressed latent is what gets stored in the activation cache for BPTT across long sequences.

### Post-training: SFT → DPO

V2's alignment pipeline predates GRPO (which emerged with DeepSeek-R1 in 2025). The recipe:

1. **Supervised Fine-Tuning (SFT)**: 1.2M curated instruction samples covering general chat, coding, math, and tool use. Standard cross-entropy loss on completions.
2. **Direct Preference Optimization (DPO)**: preference pairs collected via rejection sampling from the SFT model, scored by a reward model. DPO optimizes directly on the preference pairs without a separate RL loop, using the reference SFT model as the KL anchor. See [DPO entry](../../foundational/dpo/).

The reward model used for preference data collection is trained on human-labeled comparisons. No online RL or rollout-based training appears in V2's pipeline — that iteration would come with R1.

## Engineering Tradeoffs

| Decision | Gained | Gave up |
|---|---|---|
| MLA + MoE together | KV cache and activated-param cost both reduced; complementary wins | Two novel components interacting — debugging training instability requires understanding both |
| 128k context via YaRN extension | Long context at low additional cost | YaRN extension quality degrades on extreme lengths; trained-from-scratch long context would be better |
| Dense first layer | Training stability; avoids routing chaos in early layers | One layer of wasted sparsity — ~1/60 of parameter savings foregone |
| Auxiliary load-balance loss | Prevents expert collapse | Accuracy cost from the loss term (V3 eliminated this with bias-based balancing) |
| DPO over PPO | Simpler pipeline; no rollout infrastructure needed | Less capable post-training for hard reasoning tasks; bounded by quality of offline preference pairs |
| 8.1T token training budget | Strong base model at reasonable cost | Dense models of similar activated size (21B) can train on more tokens for same cost; V2's win is inference, not training efficiency per se |

## Experiments & Results

**Versus DeepSeek 67B (dense predecessor):**
- Performance: V2 matches or improves on all major benchmarks (MMLU, HumanEval, MATH, C-Eval).
- Training cost: ~42.5% lower (fewer FLOPs per token due to sparse activation).
- KV cache: 93.3% smaller.
- Peak generation throughput: 5.76× higher (same hardware, far more concurrent sequences fit in memory).

**Versus contemporaries (mid-2024):**
- Competitive with GPT-4-class models on Chinese benchmarks (C-Eval, CMMLU).
- Comparable to or better than Llama-3 70B on English reasoning and coding.
- Strongest open-weight model at or near its activated-parameter class at release.

**Post-training delta:**
- DeepSeek-V2-Chat outperforms DeepSeek-V2 base by ~10 points on instruction-following benchmarks (MT-Bench, AlpacaEval 2).
- DPO provides a noticeable win over SFT-only on preference-sensitive tasks; the gain narrows on objective tasks (math, code) where SFT already performs well.

## Reproducibility Notes

- V2 model weights are publicly released on HuggingFace. The SFT checkpoint is also available separately.
- Training code is not released for V2 (same situation as V3).
- The preference data used for DPO is not released; practitioners adapting the recipe need to construct their own preference pairs.
- For MLA inference, use **FlashMLA** (DeepSeek OSW 2025) for optimized decode kernels. The absorption trick (W_UK folded into W_Q, etc.) must be done in the weight-loading step; naive implementations that re-project K, V at every step leave most of the performance gain on the floor.
- Expert parallel serving: vLLM and SGLang both support V2 serving with MoE; expert sharding across GPUs is handled automatically.

## Commentary

V2's significance is architectural and economic. The "236B model that costs like a 21B model to serve" framing is not marketing — the KV cache reduction and sparse activation are quantifiable, and the throughput numbers validate them. This was the moment the field recognized that MoE + attention compression could break the inference cost curve.

Three things V2 established that V3 then inherited:

1. **MLA works at scale**. The theoretical reduction in KV cache size translates directly to serving throughput with no quality regression. FlashMLA later showed this holds even with fully optimized kernels.

2. **Fine-grained MoE with shared experts is stable at 200B+ total scale**. Routing did not collapse; expert utilization remained balanced with the auxiliary loss. This de-risked scaling further to 671B.

3. **DPO is sufficient for a strong chat model**. The V2-Chat alignment, achieved without online RL, was competitive with models using more complex RLHF pipelines. This set expectations that V3/R1's GRPO leap was meaningful — they were improving over a strong DPO baseline, not a weak SFT baseline.

What V2 did not solve — and V3/R1 addressed:

- **Reasoning depth**: DPO-aligned V2 is a strong general assistant but not a strong multi-step reasoner. GRPO + thinking tokens (R1) unlocked that.
- **Training precision efficiency**: V2 trained in BF16; V3 moved to FP8, halving training cost again.
- **Pipeline bubble**: V2 used a standard pipeline schedule; V3's DualPipe nearly eliminated bubbles.
- **Load balancing without accuracy penalty**: the auxiliary loss in V2 was known to cost accuracy; V3's bias-based scheme removed that cost cleanly.

V2 is the "proof" run; V3 is the optimized production run of the same recipe.

## References

- [1] DeepSeek-AI. _DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts Language Model._ arXiv:2405.04434, 2024.
- [2] DeepSeek-AI. _DeepSeek-V3 Technical Report._ arXiv:2412.19437, 2024.
- [3] DeepSeek-AI. _DeepSeekMoE: Towards Ultimate Expert Specialization in Mixture-of-Experts Language Models._ arXiv:2401.06066, 2024.
- [4] Peng et al. _YaRN: Efficient Context Window Extension of Large Language Models._ arXiv:2309.00071, 2023.
- [5] Rafailov et al. _Direct Preference Optimization._ NeurIPS '23 / arXiv:2305.18290.
- [6] DeepSeek-AI. _DeepSeek-R1._ arXiv:2501.12948, 2025.
- [7] DeepSeek. _FlashMLA._ https://github.com/deepseek-ai/FlashMLA, 2025.
