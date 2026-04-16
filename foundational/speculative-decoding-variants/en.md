# Speculative Decoding Variants — Medusa / EAGLE

- **Authors / Org**: Cai et al. (Princeton + Together AI) · Li et al. (SJTU + Microsoft Research)
- **Published**: Medusa — 2024-01 (arXiv:2401.10774) · EAGLE — 2024-01 (arXiv:2401.15077) · EAGLE-2 — 2024-06 (arXiv:2406.16858)
- **Links**: [Medusa](https://arxiv.org/abs/2401.10774) · [EAGLE](https://arxiv.org/abs/2401.15077) · [EAGLE-2](https://arxiv.org/abs/2406.16858)

## TL;DR

The original speculative decoding methods (Leviathan, Chen) use a **separate draft model** that produces tokens in a linear chain. Medusa and EAGLE remove the separate model and switch to **tree-structured drafts**: a single forward pass of the target model verifies an entire branching tree of candidates simultaneously. Medusa attaches extra decoding heads that draft in parallel; EAGLE adds a lightweight autoregressive draft on the **feature level** (hidden states) rather than the token level, yielding higher acceptance rates. EAGLE-2 further introduces **adaptive tree expansion** — prune branches where draft confidence is low, spend budget where it is high. Both methods are now default-on in vLLM and TensorRT-LLM. See the [Speculative Decoding primer](../speculative-decoding/en.md) for the rejection-sampling foundation; this entry builds directly on it.

## Context & Motivation

Vanilla speculative decoding has two practical friction points:

1. **Separate draft model**: requires a matched, distilled companion model for each target. Maintaining two sets of weights and ensuring distribution alignment is operational overhead.
2. **Linear chain verification**: proposes `K` tokens in sequence; on the first rejection the rest are discarded. If acceptance rate `α = 0.7` and `K = 5`, expected accepted tokens ≈ `(1 - 0.7^5) / (1 - 0.7) ≈ 2.8`. The chain wastes the entire speculative budget when an early token misses.

A tree addresses both problems: propose a branching set of candidate continuations; verify all paths in one forward pass; accept the longest agreeing path. Self-draft methods eliminate the companion model entirely.

## Medusa — Parallel Multi-Head Drafting

### Architecture

Medusa attaches `H` extra LM heads to the **frozen or lightly fine-tuned** target model. Each head `h` is a two-layer MLP followed by a linear projection to vocabulary size, trained to predict the token at position `t + h` given the last hidden state at position `t`.

```
target hidden state at t
    │
    ├── head-1 → logits for t+1   (top-s₁ candidates)
    ├── head-2 → logits for t+2   (top-s₂ candidates)
    ├── head-3 → logits for t+3   (top-s₃ candidates)
    └── head-4 → logits for t+4   (top-s₄ candidates)
```

All four heads fire in the **same forward pass** — no autoregressive dependencies between them. This is the key difference from a separate draft model: there is no sequential draft generation step.

### Tree Construction

Candidate trees are built by taking the Cartesian product of head outputs, pruned to a fixed budget. In practice: head-1 emits `s₁ = 3` top tokens; head-2 emits `s₂ = 3`; head-3 `s₃ = 2`; head-4 `s₄ = 2`. This gives a tree with up to `3 × 3 × 2 × 2 = 36` paths but typically far fewer after pruning.

### Tree Attention

All candidate paths are verified in a **single target-model forward pass** using a custom attention mask:

- Every node can attend to all original prefix tokens (the verified context).
- A node can attend to its ancestors in the tree but **not** to nodes on sibling branches.

This is implementable as a block-sparse causal mask. The tree has `|nodes|` tokens total; the forward pass cost scales with that count, not with the number of paths.

After the forward, the target's token probabilities at each node position are compared against the Medusa head's proposals via the standard rejection-sampling rule. The **longest accepted root-to-leaf path** becomes the output, and the state updates to that endpoint.

### Training

Medusa heads are trained with a cross-entropy loss predicting the correct future tokens from the training data. The target model body can be frozen (Medusa-1) or jointly fine-tuned (Medusa-2). Medusa-1 adds < 1% of the parameter count and takes a few hours on a single node.

## EAGLE — Feature-Level Autoregressive Drafting

### The Feature-Level Insight

Medusa heads are non-autoregressive: head-2's prediction of `t+2` does not use the predicted token at `t+1`. This limits acceptance rate — the prediction of `t+3` is conditioned only on `t`'s hidden state, not on what `t+1` and `t+2` are.

EAGLE restores autoregressive structure but at the **hidden-state (feature) level**. A lightweight draft network predicts the hidden state `h_{t+1}` given `h_t` (from the target's last layer) and the current token embedding `e_t`. The target's own **LM head** is then applied to `h_{t+1}` to get token probabilities:

```
h_t (from target) ──┐
                     ├──► draft_net ──► ĥ_{t+1} ──► [shared LM head] ──► p̂(x_{t+1})
e_t (token embed) ──┘
```

This is cheaper than running the full target transformer because the draft network is a single transformer layer (or small MLP). The key observation: **features (hidden states) are more predictable than tokens** — a single layer operating on `h_t` captures far more context than a head operating on only `h_t` without seeing `h_{t-1}`.

### Tree Construction in EAGLE

EAGLE's draft is autoregressive, so tree construction expands level by level:

1. Run draft network once: predict `ĥ_{t+1}`, sample top-`s₁` tokens → level-1 nodes.
2. For each level-1 node, run draft network again: predict `ĥ_{t+2}`, sample top-`s₂` tokens → level-2 nodes (branching `s₁ × s₂`).
3. Repeat to depth `d`.

This produces a tree with structured branching. The same tree attention mask used in Medusa applies for verification.

### EAGLE-2 — Adaptive Tree Expansion

Fixed trees waste budget: a high-confidence prediction at depth 1 should fan out wider; a low-confidence prediction should be pruned early.

EAGLE-2 introduces a **draft confidence score** derived from the draft token's probability. During tree expansion, a node is expanded only if its probability exceeds a dynamic threshold; otherwise it is pruned. The total tree size is constrained to a fixed budget `B` (e.g., 60 nodes), but the shape adapts to where the draft model is confident.

The key equation is the expected speedup improvement: by concentrating the budget on high-`α` branches, EAGLE-2 achieves ~20% higher throughput than EAGLE-1 with the same compute budget.

## Tree Verification Algorithm

The core primitive shared by all tree-based methods:

```python
def tree_verify(prefix_kv, tree_nodes, target_model, draft_probs):
    # tree_nodes: list of (token, parent_idx) — the candidate tree
    # Build tree attention mask: node i attends to prefix + ancestors of i
    mask = build_tree_mask(tree_nodes)

    # Single forward pass over all tree nodes
    target_logits = target_model(prefix_kv, tree_nodes, attn_mask=mask)

    # Walk root-to-leaf paths; accept/reject each token via rejection sampling
    best_path = []
    for path in enumerate_paths(tree_nodes):
        accepted = []
        for depth, (token, node_idx) in enumerate(path):
            p_target = softmax(target_logits[node_idx])[token]
            p_draft  = draft_probs[node_idx][token]
            u = random.uniform(0, 1)
            if u < p_target / p_draft:
                accepted.append(token)
            else:
                # Sample replacement from residual distribution
                replacement = sample_residual(target_logits[node_idx], p_draft)
                accepted.append(replacement)
                break
        if len(accepted) > len(best_path):
            best_path = accepted

    return best_path
```

The mask construction is the implementation detail: node `i` at depth `d` on branch `b` may attend only to the `d` ancestors on branch `b` plus all `|prefix|` tokens. This is stored as a triangular mask over the concatenated (prefix + tree_nodes) sequence.

## Engineering Tradeoffs

| Decision | Gained | Gave up |
|---|---|---|
| Medusa heads (parallel, non-autoregressive) | Zero draft latency; simple training | Lower acceptance rate than sequential draft; head-2 doesn't see head-1's output |
| EAGLE feature-level draft (autoregressive) | Higher acceptance rate; shared LM head | One extra lightweight forward per draft step; training more complex |
| EAGLE-2 adaptive trees | Best throughput per compute budget | Dynamic shape complicates batching; threshold tuning needed |
| Tree attention mask | Verifies entire tree in one pass | Custom attention implementation required; not a stock causal mask |
| Fine-tuning Medusa heads only | Fast to add to any existing model | Suboptimal if target model not trained with Medusa in mind |
| Batching multiple requests | Throughput scales | Different requests accept different path lengths; scheduler must handle variable-length steps |

## Experiments & Results

| Method | Speedup (single stream) | Notes |
|---|---|---|
| Vanilla speculative (separate draft) | 2.0–3.0× | Baseline; separate 7B draft for 70B target |
| Medusa-1 (frozen) | 2.2–2.8× | On Vicuna / Llama-2 |
| EAGLE-1 | 2.5–3.5× | On Llama-2 / 3; outperforms Medusa at same tree budget |
| EAGLE-2 | 3.0–4.0× | Adaptive tree; current state of the art for lossless decode |

All methods preserve the target distribution exactly under the rejection-sampling guarantee.

In **batched serving** (vLLM production workloads), speedups are 1.3–1.8× — smaller than single-stream benchmarks but consistent, and they stack with PagedAttention and continuous batching.

## Production Integration

**vLLM**: Medusa and EAGLE are supported via `--speculative-model` flag (for external draft models) and `--num-speculative-tokens`; tree draft is enabled when Medusa/EAGLE weights are specified. The scheduler handles variable accepted-token counts by padding verification batches.

**TensorRT-LLM**: Medusa heads are built into the model engine at compile time; tree attention mask is fused into the attention kernel.

**Hugging Face TGI**: Medusa support added in v1.4.

Operationally: to add EAGLE to an existing Llama 3 70B deployment, download the published EAGLE draft weights (a single transformer layer, ~100M params), point the inference server at both, and enable tree decoding. No changes to the target model.

## Commentary

The Medusa → EAGLE → EAGLE-2 arc is a case study in tightening a specific bottleneck. Vanilla speculative decoding identified the right abstraction (draft + verify); the variants show that **the quality of the draft distribution** is the primary lever — and that features carry more signal than tokens. The tree structure is the natural extension once you accept that a single draft path is wasteful.

The deepest open question is **composability with disaggregated prefill/decode** (see [DistServe](../distserve/en.md) and [Prefix Caching](../prefix-caching/en.md)): in a disaggregated system, the prefill node and decode node run separately, potentially on different hardware. Tree verification requires the target model to see the tree — this ties verification to the decode node and complicates any scheme where prefill and decode share state asynchronously.

The second open question is **adaptive draft model selection**: for easy requests (high `α`), deeper trees with more speculation are optimal; for hard requests (low `α`), overhead dominates and it may be better to fall back to standard decoding. Runtime oracle methods for this are an active research area.

## References

- [1] Cai et al. _Medusa: Simple LLM Inference Acceleration Framework with Multiple Decoding Heads._ arXiv:2401.10774, 2024.
- [2] Li et al. _EAGLE: Speculative Sampling Requires Rethinking Feature Uncertainty._ arXiv:2401.15077, 2024.
- [3] Li et al. _EAGLE-2: Faster Inference of Language Models with Dynamic Draft Trees._ arXiv:2406.16858, 2024.
- [4] Leviathan et al. _Fast Inference from Transformers via Speculative Decoding._ ICML '23. → See [Speculative Decoding](../speculative-decoding/en.md).
- [5] Miao et al. _SpecInfer: Accelerating LLM Serving with Speculative Inference and Token Tree Verification._ MLSys '24.
