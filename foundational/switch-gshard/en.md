# Switch Transformers & GShard — Scaling MoE to Trillions of Parameters

- **Authors / Org**: Fedus, Zoph, Shazeer et al. (Google Brain) — Switch Transformers; Lepikhin et al. (Google) — GShard
- **Published**: GShard: arXiv:2006.16668, 2020; Switch Transformers: arXiv:2101.03961, JMLR 2022
- **Links**: [Switch Transformers](https://arxiv.org/abs/2101.03961) | [GShard](https://arxiv.org/abs/2006.16668)

## TL;DR

GShard (2020) demonstrated the first 600B-parameter MoE model on TPU pods using top-2 routing with local group dispatching and a capacity factor to handle expert overflow. Switch Transformers (2022) simplified routing to top-1 (one expert per token), proved that this is stable with careful initialization and FP32 router precision, and scaled to 1.6T parameters (Switch-C) with 2048 experts — achieving a 7× wall-clock speedup over T5-XXL at the same compute budget. Together these two papers established the engineering vocabulary that every subsequent MoE system (DeepSeekMoE, Mixtral, Llama 4) builds on.

## Context & Motivation

Dense Transformers scale compute and parameters together: doubling a model doubles both FLOPs per token and memory. Mixture-of-Experts breaks this coupling by activating only a subset of parameters per token — a fundamentally different scaling axis. The idea dates to Jacobs et al. (1991) and Shazeer et al.'s 2017 Sparsely-Gated MoE, but the 2017 work suffered training instability and never reached full production deployment.

By 2020, TPU pods could host hundreds of devices interconnected at high bandwidth — the hardware moment for MoE had arrived. The two problems blocking practical MoE were:

1. **Load imbalance**: some experts attract most tokens ("rich-get-richer" routing collapse), leaving others idle and creating stragglers in distributed training.
2. **Training instability**: the routing network's discrete argmax is non-differentiable; large gradient updates when routing decisions flip can destabilize training.

GShard addressed both for top-2 routing at 600B scale. Switch Transformers re-examined the assumptions and found that top-1 routing sidesteps many instabilities while remaining competitive in quality.

## GShard: 600B with Top-2 Routing

### Architecture and TPU Layout

GShard augments every other Transformer FFN layer with MoE. The model is sharded so each TPU device hosts exactly one expert per MoE layer. With 2048 TPUs and 2048 experts, each expert lives on exactly one device — expert parallelism at its most natural.

The critical design is **local group dispatching**. Tokens in a batch are divided into groups (e.g., 4096 tokens → 32 groups of 128). Routing is computed locally within each group, meaning the argmax over 2048 experts is still done globally per token, but the *capacity accounting* is done per group. This limits the scope of cross-device communication: each device only needs to receive tokens from its group's routing decisions rather than the full global sort.

### Capacity Factor and Token Dropping

Each expert has a hard capacity limit:

```
Expert capacity C = (tokens_per_group / num_experts) × capacity_factor
```

At `capacity_factor = 1.0` and perfect balance, every expert gets exactly its share. In practice routing is uneven, so `capacity_factor = 1.25` is the GShard default — 25% headroom. If a token's first-choice expert is full, it is dispatched to its second choice. If the second choice is also full, the token is **dropped**: it bypasses the expert FFN and uses the residual stream unchanged (a pass-through / identity bypass). Dropped tokens still contribute to the model's output through the non-MoE layers.

GShard adds an **auxiliary load-balancing loss** to discourage routing collapse. It penalizes imbalance at the group level, pushing the router toward uniform utilization.

### Engineering Reality of Token Dropping

Token dropping is not a theoretical edge case — at production batch sizes with `capacity_factor = 1.0`, drop rates of 5–15% are common during early training. The key insight is that dropping is **graceful degradation**: the model still produces a valid output, just a less expert-processed one. During inference, setting a higher capacity factor (1.5–2.0) and accepting the extra padding cost is typical for quality-sensitive applications.

## Switch Transformers: Top-1 Routing at 1.6T Scale

### The Top-1 Simplification

Switch Transformers makes a surprising claim: route each token to exactly **one** expert. Prior work assumed top-2 was necessary for stability (the second expert acts as a backup). Switch shows this is not true, and top-1 has direct benefits:

- **Half the all-to-all communication volume** vs. top-2 (each token crosses one inter-device boundary instead of two).
- **Simpler capacity accounting**: a token either goes to its expert or is dropped — no second-chance dispatch.
- **Lower routing overhead**: the router computes a softmax over `n` experts and takes the argmax; for top-1 this is a simple `max` with no secondary dispatch logic.

The router computes a probability distribution over experts for token `x`:

```
h(x) = W_r · x          (linear projection, router weights)
p_i(x) = softmax(h(x))_i
route(x) = argmax_i p_i(x)
```

The token is sent to expert `route(x)` and the expert output is scaled by `p_{route(x)}(x)` before being added to the residual stream.

### Auxiliary Load-Balancing Loss

The central training mechanism for preventing routing collapse:

```
L_aux = α · n · Σᵢ fᵢ · Pᵢ
```

where:
- `n` = number of experts
- `fᵢ` = fraction of tokens dispatched to expert `i` in the current batch (computed with straight-through, not differentiable)
- `Pᵢ` = fraction of the router's total probability mass assigned to expert `i` (differentiable)
- `α` = loss coefficient (typically 1e-2; too large collapses to uniform routing, too small allows collapse)

The product `fᵢ · Pᵢ` is minimized when both the token fraction and the probability mass are uniformly distributed. `fᵢ` is a stop-gradient count — it provides the signal about which experts are over-used without making the count itself differentiable. `Pᵢ` is the differentiable handle: the router learns to redistribute probability mass away from crowded experts.

### Expert Capacity in Switch

```
C = (tokens_per_batch / num_experts) × capacity_factor
```

At `capacity_factor = 1.0`: zero-waste at perfect balance, but any imbalance causes drops.
At `capacity_factor = 1.25`: 25% overhead per expert in buffer allocation; tokens/experts ratio slack.
At `capacity_factor = 2.0`: generous headroom, high buffer memory cost.

For a model with 128 experts processing a batch of 512 tokens: `C = (512/128) × 1.25 = 5` tokens per expert. If expert 0 attracts 9 tokens, 4 are dropped.

### Training Stability Fixes

MoE models before Switch routinely diverged. Switch identifies two root causes and fixes both:

1. **Router precision**: Keep the router linear projection in FP32 even when the rest of the model is BF16. The router's softmax over hundreds of experts is numerically sensitive; BF16 produces underflows for small logit differences that cause routing collapse. Cost: negligible — the router is tiny relative to expert weights.

2. **Weight initialization**: Initialize router weights with a smaller scale (factor of 0.1× the default). Large initial router weights → large initial routing probabilities → a few experts dominate from step 1 → auxiliary loss fights an already-entrenched imbalance.

### Scale: Switch-C

Switch-C uses 2048 experts, 1.6T total parameters, ~1B activated parameters per token. It achieves a **7× wall-clock training speedup** vs. T5-XXL (11B dense) at the same FLOP budget. Quality is competitive on SuperGLUE benchmarks, though the sparse model requires more steps to reach dense-equivalent perplexity in some settings.

The "superposition hypothesis" implicit in Switch-C: more experts = more independent parameter groups = more distinct knowledge patterns can be stored. Each expert learns to handle a particular token type, domain, or syntactic pattern; the combinatorial space of routing paths lets the model store vastly more associations than a same-FLOP dense model.

## Engineering Tradeoffs

| Decision | Gained | Gave Up |
|---|---|---|
| Top-1 routing (Switch) vs. top-2 (GShard) | Half the all-to-all communication; simpler dispatch logic | Second-choice fallback for dropped tokens; slight quality gap vs. top-2 at small expert counts |
| Capacity factor > 1.0 | Graceful handling of routing imbalance; fewer dropped tokens | Extra buffer memory per expert; underutilized capacity at the tail of expert load distributions |
| Token dropping (bypass connection) | Bounded expert load regardless of imbalance; no straggler waits | Dropped tokens get degraded FFN processing; soft failure that is hard to observe at inference |
| FP32 router, BF16 weights | Numerically stable routing softmax over many experts | Two-precision bookkeeping; minor memory overhead for router activations |
| Auxiliary load-balance loss | Prevents routing collapse during training | Tuning `α` is finicky; too large → uniform routing (experts identical); too small → collapse |
| Expert parallelism (1 expert/device) | Near-linear scaling of total parameters with device count | All-to-all communication cost per MoE layer; network bandwidth is the new bottleneck |

## Experiments & Results

**GShard (600B):**
- 600B parameter MoE trained on 100+ languages for neural machine translation.
- Top-2 routing with capacity factor 1.25; 2048 experts across 2048 TPU cores.
- Outperforms 100B dense models on WMT multilingual benchmarks; near-linear quality improvement up to 600B.

**Switch Transformers:**
- Switch-Base (7B total, 12 experts): matches T5-Base with 7× fewer FLOPs per token.
- Switch-Large (26B total, 128 experts): faster convergence than T5-Large; roughly matches quality at 4× lower compute cost.
- Switch-C (1.6T total, 2048 experts): 7× training speedup vs T5-XXL; top score on SuperGLUE at the time.
- Ablation: top-1 vs. top-2 — within 1% on downstream benchmarks, top-1 consistently faster due to halved communication.

## Commentary

The lasting contribution of these two papers is not the specific routing algorithm — it is the **operational vocabulary**: capacity factor, token dropping, auxiliary loss, expert parallelism. Every MoE system built after 2021 uses these terms and these knobs. The capacity factor in particular is an underappreciated concept: it transforms a hard combinatorial allocation problem (assign tokens to experts without overflow) into a soft engineering tradeoff (choose how much memory to burn for headroom vs. how many tokens to drop).

From a systems perspective, the most important insight is that **token dropping is acceptable**. Pre-Switch, the assumption was that every token must receive full processing; dropping felt like a correctness violation. Switch demonstrated empirically that 1–5% token drop rates during inference are invisible in downstream metrics. This unlocks a whole class of bounded-cost routing algorithms that would otherwise require either dynamic batching of variable lengths or expensive load-balancing rebalancing passes.

The long-term limitation of the Switch/GShard approach is the auxiliary loss tuning problem. `α` is a scalar that must be swept for each new model scale — too small and you get routing collapse (convergence but bad generalization); too large and you get uniform routing (the model forgets to specialize). DeepSeek's later work on auxiliary-loss-free routing via bias terms (see [DeepSeekMoE](../../deepseek/moe/en.md)) is a direct response to this fragility. [Mixtral](../../mistral/mixtral/en.md) and [Llama 4](../../meta/llama4/en.md) both inherit the basic framework but tune the balance differently — Mixtral uses top-2 with no dropping, Llama 4 uses top-2 with interleaved MoE layers.

## References

- [1] Lepikhin et al. _GShard: Scaling Giant Models with Conditional Computation and Automatic Sharding._ arXiv:2006.16668, 2020.
- [2] Fedus, Zoph, Shazeer. _Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity._ JMLR, 2022. arXiv:2101.03961.
- [3] Shazeer et al. _Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer._ ICLR 2017.
- [4] DeepSeekMoE: see [../../deepseek/moe/en.md](../../deepseek/moe/en.md)
- [5] Mixtral of Experts: see [../../mistral/mixtral/en.md](../../mistral/mixtral/en.md)
- [6] Llama 4: see [../../meta/llama4/en.md](../../meta/llama4/en.md)
