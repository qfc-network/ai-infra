# MoE Routing Improvements — Expert Choice & Loss-Free Balancing

- **Key papers**:
  - **Expert Choice** — Zhou et al., Google. NeurIPS 2022.
  - **DeepSeekMoE** — Dai et al., DeepSeek AI. 2024-01.
  - **DeepSeek-V3** — DeepSeek AI. 2024-12.
- **Links**: [Expert Choice (arXiv:2202.09368)](https://arxiv.org/abs/2202.09368) · [DeepSeekMoE (arXiv:2401.06066)](https://arxiv.org/abs/2401.06066) · [DeepSeek-V3 (arXiv:2412.19437)](https://arxiv.org/abs/2412.19437)

## TL;DR

The canonical MoE routing problem is a balancing act: route semantically similar tokens to the same expert (for quality) while keeping all experts equally loaded (for hardware efficiency). Token Choice routing — each token selects its top-k experts via a softmax router — dominated early designs (Switch, GShard, Mixtral) but requires an auxiliary load-balancing loss that injects noise into the gradient and caps capacity by dropping overflow tokens. Two alternatives have since matured: **Expert Choice** routing (Zhou et al., 2022) inverts the assignment so each expert selects its own tokens, achieving perfect load balance by construction at the cost of variable per-token expert coverage and incompatibility with autoregressive decoding. **Loss-Free Balancing** (DeepSeek-V3) keeps Token Choice but replaces the auxiliary loss with a per-expert bias term updated by a simple sign rule, removing the gradient noise while preserving token-complete routing. DeepSeekMoE's fine-grained plus shared expert decomposition layers on top of either routing strategy and further improves specialization. The routing choice propagates directly into expert parallelism dispatch kernels, all-to-all communication patterns, and the inference scheduling graph.

## Context: The Routing Problem

A Mixture-of-Experts (MoE) layer replaces a dense FFN with `E` expert FFNs; only `k` of the `E` experts activate per token. The router is a learned linear projection followed by softmax that assigns routing weights. The gains are real — for the same active parameter count, you can scale total parameters by E/k, providing more capacity for the same compute budget. But the router creates an infrastructure problem: **expert load imbalance**.

If the router collapses — routing all tokens to the same two experts — most experts sit idle, effective capacity shrinks, and expert parallelism communication becomes unbalanced (some GPUs receive all the work, others nothing). During training this is recoverable via auxiliary loss. During inference, imbalance translates directly to latency: the step time is determined by the slowest expert, and unbalanced dispatch means wasted bubble time in expert parallel pipelines.

### Token Choice Routing (Switch / Mixtral Baseline)

Standard Token Choice: token `t` produces a logit vector `g_t = W_router · x_t ∈ R^E`, and selects the top-k experts by softmax score. Each expert has a hard capacity `C`:

```
C = floor((T / E) × capacity_factor)    # tokens per expert per batch
```

where `T` is the batch token count and `capacity_factor` is a tunable scalar (typically 1.0–1.25). Tokens routed to an expert that has already filled its capacity bucket are **dropped** — they skip the FFN entirely for that layer. Dropped tokens hurt quality; raising `capacity_factor` reduces drops but wastes compute on empty expert slots.

To discourage collapse, an auxiliary balancing loss is added to the training objective:

```
L_aux = α × E × Σ_i  f_i × p_i

where:
  f_i = fraction of tokens routed to expert i in the batch
  p_i = mean router probability assigned to expert i
  α   = loss weight (typically 0.01 – 0.1)
```

This loss is effective but introduces a persistent tension: `α` must be large enough to prevent collapse yet small enough not to dominate the language modeling loss. In practice, it adds noise to the gradient at every step, widening the gap between training loss and validation loss under aggressive scaling.

## Expert Choice Routing (Zhou et al., 2022)

Expert Choice inverts the selection direction. Instead of each token choosing its experts, **each expert chooses its tokens**.

Given batch size `T` and `E` experts with a capacity multiplier `k`, expert `i` independently selects the top-`B` tokens by router logit, where the budget `B` is fixed:

```
B = floor(k × T / E)       # tokens allocated to expert i

For each expert i:
  scores_i = softmax(x · W_router^T)[:, i]   # score of expert i for all T tokens
  selected_i = argsort(scores_i, descending=True)[:B]
  token_expert_matrix[i] = selected_i
```

Load balance is exact by construction: every expert processes exactly `B` tokens, no auxiliary loss needed, and no tokens are dropped at the expert level. Total FLOPs per layer are strictly `k × T × FFN_cost(d)` regardless of batch composition.

### The Variable Coverage Problem

The inversion has a fundamental consequence: **a given token may be selected by 0, 1, 2, or many experts**. Under Token Choice top-2, every token is guaranteed exactly 2 expert activations per layer. Under Expert Choice, coverage is variable — popular tokens get processed by many experts, rare tokens may be skipped entirely. On expectation each token is selected `k` times, but variance is non-trivial.

This creates a hard problem for **autoregressive (causal) decoding**. At inference time you need to know which experts process each query token before running the experts; under Token Choice the token decides, so the dispatch list is computable from the query in O(1). Under Expert Choice, you would need to run all E experts' selection passes against the full sequence context before dispatching — which is sequential and expensive. In practice, Expert Choice is **used almost exclusively for non-autoregressive settings**: encoder-only models (BERT-style), classification, and full-sequence inference where you have the entire input batch available. It has seen little adoption in autoregressive LLM decoding.

### Expert Parallelism Dispatch Difference

Expert parallelism (EP) splits the `E` experts across `P` GPUs. Routing requires an all-to-all: tokens on device `d` that are assigned to expert `e` on device `d'` must be sent over the interconnect. Under Token Choice, the dispatch kernel builds a per-token expert list and packs tokens into contiguous buffers ordered by destination expert. Under Expert Choice, the kernel builds a per-expert token list; tokens may arrive out of original sequence order and must be reordered for the gather. This requires a different reorder/scatter kernel and makes overlap with compute slightly harder.

## Loss-Free Balancing (DeepSeek-V3)

DeepSeek-V3 retains Token Choice (compatible with autoregressive generation) but removes the auxiliary load-balancing loss entirely, replacing it with a **per-expert bias correction** mechanism.

Each expert `i` maintains a scalar bias `b_i` (initialized to 0). At routing time, the expert selection is made on **biased logits**:

```
g_t^biased = g_t + b            # b ∈ R^E, added before top-k selection
top_k selection on g_t^biased   # determines which experts receive the token
routing weight computation on g_t (original, unbiased) for the weighted sum
```

After each forward pass (training step or micro-batch), biases are updated via a simple sign rule:

```
load_i  = fraction of tokens routed to expert i in the step
b_i ← b_i - γ × sign(load_i - target_load)

where:
  target_load = k / E   (uniform target)
  γ            = bias update step size (a small constant, e.g. 0.001)
```

The update is **not a gradient step** — it is a sign-based correction outside the autograd graph, operating only on the discrete routing decision. Overloaded experts have their selection bias reduced (making them less likely to be chosen in subsequent steps); underloaded experts have theirs raised. The system converges to approximate balance without any term in the training loss.

### Why This Helps Training Stability

The auxiliary loss `L_aux` is a task-level signal added to the language modeling loss. Its gradient flows through the router weights at every step, pushing those weights toward balance even when the LM gradient wants them to specialize. At large scale (DeepSeek-V3 is ~671B total parameters, 37B active), this tension is measurable: training loss curves under auxiliary-loss balancing exhibit more variance, and the final perplexity is marginally higher than under loss-free balancing with equivalent balance quality. DeepSeek-V3 reports that removing `L_aux` produces a smoother loss curve and a measurable improvement in final checkpoint quality.

The bias update rule is also simpler than alternatives (EMA-based load estimates, second-order corrections): the sign function makes it robust to outlier batches, and `γ` is insensitive to scale.

## Fine-Grained + Shared Expert Decomposition (DeepSeekMoE)

DeepSeekMoE introduces a structural change to the expert pool itself, orthogonal to the routing algorithm choice.

A standard MoE layer has `E` routed experts and selects top-`k`. DeepSeekMoE splits this into:

- `Ks` **shared experts**: always activated for every token; handle common, broadly applicable knowledge.
- `N` **fine-grained routed experts**: each expert's hidden dimension is reduced (they are smaller than standard experts); top-`Kt` are selected per token via the router.

The total active parameter count is:

```
active_params = Ks × d_model × d_expert_ffn          # shared, always active
              + Kt × d_model × (d_expert_ffn / m)    # routed, each 1/m the size
```

where `m` is the granularity factor (e.g., `m=4` → each routed expert is 1/4 the size of a standard expert, so you can have 4× more experts for the same active FLOPs).

**Knowledge redundancy reduction**: in standard MoE, routed experts tend to replicate common knowledge (e.g., syntactic structure) because every expert must handle whatever tokens it receives. With shared experts handling common knowledge, routed experts can specialize more narrowly, reducing redundancy and increasing the effective utilization of routed capacity. DeepSeekMoE-16B achieves performance comparable to LLaMA-7B while using only 2.8B active parameters.

**Memory implication**: shared experts are always in HBM (they are always needed). Routed experts are candidates for expert offloading — if `E` is large enough, only the top-`Kt` experts' weights need to reside on-device per step. Shared experts block this offload path and must be sized accordingly.

## End-to-End Memory and Latency Accounting

For a concrete reference: a 671B MoE model (DeepSeek-V3 scale) with 256 experts per layer, top-8 routing, BF16 weights:

- **Routed expert weights per GPU** (EP=64): each GPU hosts ~4 experts × `d_model × d_ffn × 2` bytes.
- **All-to-all volume per layer**: each token sends its hidden state (BF16, `d_model=7168` → 14 KB) to 8 experts across the EP group; with sequence length 4096 and batch size 1, dispatch volume is 4096 × 8 × 14 KB / 64 peers ≈ 7 MB per EP peer per layer, each way.
- **Imbalance overhead**: a 20% load imbalance (one expert sees 1.2× average tokens) stalls that GPU for 17% of expert compute time while other GPUs wait at the next all-to-all barrier.

Loss-Free Balancing keeps imbalance within ~2–3% at steady state (per DeepSeek-V3 ablations), versus ~5–8% under auxiliary-loss balancing with aggressive `α`.

## Engineering Tradeoffs

| Scheme | Load Balance | Training Stability | Inference Compatibility | Auxiliary Loss | Hardware Efficiency |
|---|---|---|---|---|---|
| Token Choice + aux loss (Switch, Mixtral) | Approximate; controlled by `α` and capacity factor | Moderate; aux loss adds gradient noise | Full; tokens always get exactly k experts | Yes; `L_aux` in training objective | Moderate; token dropping wastes capacity |
| Expert Choice (Zhou 2022) | Exact by construction; no overflow, no dropping | High; no aux loss | Autoregressive incompatible; encoder/batch-only | None | High at training; poor at AR decode |
| Loss-Free Balancing (DeepSeek-V3) | Approximate; converges to ~uniform via bias | High; cleaner gradient signal, smoother loss curve | Full; standard top-k token choice preserved | None; bias update outside autograd | High; no dropped tokens at reasonable γ |
| Fine-grained + shared experts (DeepSeekMoE) | Depends on routing algorithm used | Depends on routing algorithm used | Full | Depends on routing algorithm | High; reduces expert redundancy, improves utilization |

## Practical Deployment Notes

**Choosing routing scheme**: For any model requiring autoregressive generation, Expert Choice is off the table unless you add an inference-time approximation (e.g., fall back to Token Choice for decoding). Loss-Free Balancing is the current state of the art for LLM training and inference combined.

**Capacity factor tuning**: Even with loss-free balancing, setting `capacity_factor` remains relevant for handling adversarial input distributions at inference time. Set it to 1.0 for matched compute budgets; raise to 1.1–1.25 if you expect distribution shift.

**EP all-to-all cost**: On NVLink within a node, EP all-to-all at BF16 is fast enough not to be the dominant cost. Across nodes (InfiniBand), it becomes a real latency sink. Both Expert Choice and Token Choice send the same total bytes; the difference is in kernel shape and whether reordering is needed.

**Bias initialization and warmup**: DeepSeek-V3 initializes all biases to 0 and finds no warmup needed; the sign update converges to reasonable balance within a few hundred steps from random initialization.

## Cross-References

- `../switch-gshard/` — foundational Token Choice routing with auxiliary loss; prerequisite reading
- `../../deepseek/moe/` — fine-grained and shared expert pattern details from DeepSeekMoE
- `../../deepseek/v3-tech-report/` — Loss-Free Balancing as deployed in DeepSeek-V3 at 671B scale
- `../../mistral/mixtral/` — top-2 Token Choice with auxiliary loss; open-weight MoE baseline

## References

- [1] Zhou et al. _Mixture-of-Experts with Expert Choice Routing._ NeurIPS 2022 / arXiv:2202.09368.
- [2] Lepikhin et al. _GShard: Scaling Giant Models with Conditional Computation and Automatic Sharding._ ICLR 2021 / arXiv:2006.16668.
- [3] Fedus et al. _Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity._ JMLR 2022 / arXiv:2101.03961.
- [4] Dai et al. _DeepSeekMoE: Towards Ultimate Expert Specialization in Mixture-of-Experts Language Models._ arXiv:2401.06066, 2024.
- [5] DeepSeek-AI. _DeepSeek-V3 Technical Report._ arXiv:2412.19437, 2024.
- [6] Jiang et al. _Mixtral of Experts._ arXiv:2401.04088, 2024.
