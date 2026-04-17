# Sequence Parallelism — Ring Attention, DeepSpeed Ulysses, and Megatron-CP

- **Key Papers**: Liu et al. (2023), Jacobs et al. (2023), NVIDIA Megatron-Core (2024)
- **Published**: Ring Attention — 2023-10 · DeepSpeed Ulysses — 2023-09 · Megatron-CP — integrated Megatron-Core 2024
- **Links**: [Ring Attention (arXiv:2310.01889)](https://arxiv.org/abs/2310.01889) · [DeepSpeed Ulysses (arXiv:2309.14509)](https://arxiv.org/abs/2309.14509) · [Megatron-Core](https://github.com/NVIDIA/Megatron-LM/tree/main/megatron/core)

## TL;DR

Single-GPU attention is bounded above by HBM capacity: at 128k tokens and BF16, a single attention layer's KV tensors consume tens of gigabytes, before accounting for query, output, or activations. Sequence parallelism (SP) distributes the sequence dimension across multiple GPUs, making long-context training linear in GPU count rather than quadratic in sequence length. Three architecturally distinct approaches have emerged: **Ring Attention**, which passes KV blocks around a device ring; **DeepSpeed Ulysses**, which transposes the problem into the head dimension via all-to-all collectives; and **Megatron-CP (Context Parallelism)**, which adds causal load balancing and integrates cleanly with Megatron's existing TP/PP/DP parallelism. The right choice depends on sequence length, head count, and the parallelism degrees already in use.

## Why Sequence Parallelism Is Necessary

Attention complexity is $O(S^2)$ in compute and $O(S)$ in memory for Q, K, V tensors, but the relevant constant is large. For a model with $d_{\text{model}} = 8192$, $n_{\text{heads}} = 64$, $n_{\text{kv\_heads}} = 8$, and BF16 weights:

**KV cache per layer per token**: 2 (K and V) × $d_{\text{head}}$ × $n_{\text{kv\_heads}}$ × 2 bytes = 2 × 128 × 8 × 2 = 4096 bytes = 4 KB.

**KV activations at training for one layer, sequence S = 1M**:

$$\text{KV memory per layer} = 2 \times S \times n_{\text{kv\_heads}} \times d_{\text{head}} \times \text{bytes} = 2 \times 10^6 \times 8 \times 128 \times 2 \approx 4 \text{ GB}$$

With 80 transformer layers, that is 320 GB of KV activations — more than four H100s worth of HBM — for a single training example, before considering query tensors, attention scores, feed-forward activations, or optimizer states.

FlashAttention (see [../flash-attention/](../flash-attention/en.md)) eliminates the $O(S^2)$ activation by recomputing attention tiles during the backward pass, but it does not reduce the $O(S)$ memory for Q, K, V themselves. At $S \geq 128$k tokens on typical models, Q/K/V storage alone saturates a single GPU's HBM even with FlashAttention. Sequence parallelism is therefore not an optimization but a hard requirement for long-context training.

Note that Megatron-LM v3's sequence parallelism (see [../megatron-lm/](../megatron-lm/en.md)) is a related but distinct technique: it shards the non-matmul activations (dropout, LayerNorm outputs) along the sequence axis within a tensor-parallel group. That approach reduces activation memory for non-attention operations but does not distribute the attention computation itself. The three approaches described in this entry all address the attention computation directly.

## Approach 1: Ring Attention

Ring Attention (covered in depth in [../ring-attention/](../ring-attention/en.md)) shards the sequence across $P$ devices using a ring topology. Device $i$ holds the $i$-th chunk of $S/P$ tokens from Q, K, V. To compute full attention, each device must eventually process its Q chunk against all K/V chunks.

The algorithm proceeds for $P$ rotation steps:
1. Device $i$ computes partial attention of its local Q against the K/V chunk currently in its buffer, using blockwise FlashAttention with online softmax to accumulate partial results.
2. Simultaneously, it asynchronously sends its current K/V buffer to device $(i+1) \bmod P$ and receives the next K/V buffer from device $(i-1) \bmod P$.
3. After $P$ steps, every Q chunk has attended to every K/V chunk; online softmax normalization produces exact full attention.

**Memory profile**: each device holds $S/P$ tokens of Q, K, V — total $O(S/P)$ per GPU, linear scaling with parallelism degree.

**Communication volume**: each step sends one K/V block of size $2 \times (S/P) \times n_{\text{kv\_heads}} \times d_{\text{head}}$ between neighboring devices. Over $P$ steps, total data sent per device is $2 \times S \times n_{\text{kv\_heads}} \times d_{\text{head}}$ — proportional to the full sequence KV, not the local chunk. This is a point-to-point send/receive, not a true collective, but the aggregate volume matches that of an all-to-all over the KV tensors.

**Causal attention and load balancing**: with a causal mask, device $P-1$ (holding the last sequence chunk) must compute attention against all $P$ K/V blocks, while device 0 (holding the first chunk) needs only one block. This creates severe load imbalance. Striped Attention and NVIDIA's Context Parallelism solve this by interleaving token positions across devices (even-odd interleaving): device $i$ holds tokens $\{i, i+P, i+2P, \ldots\}$ rather than a contiguous chunk $[iS/P, (i+1)S/P)$. Each device then has equal causal work at every ring step, doubling effective throughput for causal models at long context.

**Overlap requirement**: Ring Attention's defining performance property is compute-communication overlap. If the time to compute attention on one K/V block approximately equals the time to transmit one K/V block, the ring runs at compute-bound speed. This requires:
- Sufficient sequence length per device that one block saturates the tensor cores.
- Network bandwidth fast enough to move the K/V block before the next compute step starts (NVLink at 900 GB/s bidirectional, or InfiniBand HDR at 200 Gb/s, each with different tradeoffs at different block sizes).

## Approach 2: DeepSpeed Ulysses

Ulysses (from [../../microsoft/deepspeed/](../../microsoft/deepspeed/en.md)) takes a fundamentally different approach: rather than distributing the sequence dimension and rotating K/V, it transposes the data layout so that each GPU performs full attention on a subset of heads.

**Setup**: partition the $N$ attention heads across $P$ devices, with each device owning $N/P$ heads. The constraint is $P \leq N$ (head count is the upper bound on Ulysses parallelism degree).

**Forward pass**:
1. Before attention, each device holds the full sequence but only a subset of Q, K, V head projections from its TP shard (or from data-parallel input). An **all-to-all collective** redistributes the tensors: after the all-to-all, each device holds all heads ($N$ heads) but only $S/P$ tokens of Q, K, V.
2. Each device computes full self-attention on its $S/P$-token slice across all $N$ heads, using standard FlashAttention.
3. A second **all-to-all collective** transposes back: each device returns to holding full sequence $S$ but only $N/P$ heads of the output.

**Communication volume per all-to-all**: the tensor being redistributed is $S \times d_{\text{model}}$ (the Q or K or V projection output). Total bytes per all-to-all = $S \times d_{\text{model}} \times \text{bytes}$. With two all-to-alls per attention block (one before, one after), total communication per attention layer is $2 \times S \times d_{\text{model}} \times \text{bytes}$.

For $S = 128$k, $d_{\text{model}} = 8192$, BF16: $2 \times 128000 \times 8192 \times 2 \approx 4.2$ GB per layer — significant, but well-defined and independent of the parallelism degree $P$ (the volume stays the same; only the distribution changes).

**Overlap**: Ulysses all-to-alls are globally synchronizing collectives, not point-to-point transfers. They are harder to overlap with compute than Ring's asynchronous send/receives. In practice, implementations pipeline the all-to-all with the Q/K/V projections for partial overlap.

**Key constraint**: $P \leq n_{\text{heads}}$. For models with GQA and few KV heads (e.g., $n_{\text{kv\_heads}} = 8$ in Llama 3), the effective constraint is $P \leq n_{\text{kv\_heads}} = 8$ when applying Ulysses only to the KV heads. This limits Ulysses scalability on GQA models with aggressive KV head reduction.

**Memory profile**: after the first all-to-all, each device processes $S/P$ tokens with full head width. Activation memory for the attention layer is $O(S/P \times d_{\text{model}})$ — identical memory efficiency to Ring Attention per token, but the full sequence Q/K/V must be held simultaneously in the all-to-all buffers, creating a transient peak.

## Approach 3: Megatron-CP (Context Parallelism)

Megatron-CP integrates sequence parallelism as a first-class parallelism axis in Megatron-Core's 4D hierarchy: tensor (TP) × pipeline (PP) × context (CP) × data (DP). The CP axis handles the sequence dimension for attention in a way that composes cleanly with the other axes.

**Algorithm**: Megatron-CP uses an all-gather + reduce-scatter communication pattern rather than a pure ring rotation. The attention computation proceeds as follows:
1. Each device holds $S/P$ tokens (its CP shard) for Q, K, V.
2. An **all-gather** assembles the full K and V tensors (size $S$) on every device. Each device now has local Q ($S/P$ tokens) and full K, V ($S$ tokens).
3. Each device computes $\text{Attention}(Q_{\text{local}}, K_{\text{full}}, V_{\text{full}})$ using FlashAttention, producing $S/P$ output tokens.
4. No reduce-scatter on the output is needed because each device already produces its own $S/P$ output slice directly.

This is equivalent to Ring Attention but with explicit gather rather than rotation — the tradeoff is that the all-gather requires each device to hold $S$ tokens of K, V temporarily (not just $S/P$), creating a $P\times$ memory spike for K/V during the gather. In practice, this is managed by the fact that the K/V all-gather is fused with the FlashAttention kernel (using K/V chunking), so the full $S$-length K/V is never materialized simultaneously.

**Causal load balancing**: Megatron-CP implements even-odd interleaving of token positions to equalize the causal work per device, analogous to Striped Attention. Device $i$ owns tokens $\{i, P+i, 2P+i, \ldots\}$ and $\{S-1-i, S-1-P-i, \ldots\}$ (the forward and backward halves of the sequence interleaved). This ensures each device computes equal FLOPs under causal masking.

**Integration with 4D parallelism**: within a CP group, all devices share the same TP group. TP's all-reduce pattern for QKV projections and attention output projections operates within the CP group and is unaffected. PP operates across CP groups (each PP stage runs at one CP degree). DP is the outer replication axis. Llama 3 405B (see [../../meta/llama3/](../../meta/llama3/en.md)) uses $\text{TP}=8, \text{PP}=16, \text{CP}=2, \text{DP}=128$ for the long-context training phase, giving $8 \times 16 \times 2 \times 128 = 32768$ GPUs.

**Memory analysis with CP degree $N$**:

$$\text{KV cache per layer} = 2 \times \frac{S}{N} \times n_{\text{kv\_heads}} \times d_{\text{head}} \times \text{bytes per value}$$

$$\text{Total KV memory across all layers} = N_{\text{layers}} \times 2 \times \frac{S}{N} \times n_{\text{kv\_heads}} \times d_{\text{head}} \times \text{bytes}$$

For Llama 3 405B ($N_{\text{layers}} = 126$, $n_{\text{kv\_heads}} = 8$, $d_{\text{head}} = 128$, BF16, $S = 128\text{k}$, $N = 2$):

$$126 \times 2 \times \frac{128000}{2} \times 8 \times 128 \times 2 \approx 41 \text{ GB per GPU}$$

Without CP ($N=1$), this would be 82 GB — exceeding one H100's 80 GB HBM budget even before accounting for model weights, optimizer states, or feed-forward activations. CP=2 makes it feasible; higher CP degrees push to even longer contexts.

## Communication Comparison

| Dimension | Ring Attention | DeepSpeed Ulysses | Megatron-CP |
|---|---|---|---|
| Collective type | Point-to-point send/receive (ring step) | All-to-all × 2 per attention layer | All-gather on K/V per attention layer |
| Message volume per layer | $2 \times S \times n_{\text{kv\_heads}} \times d_{\text{head}}$ (total, amortized over $P$ steps) | $2 \times S \times d_{\text{model}}$ (per all-to-all) | $S \times n_{\text{kv\_heads}} \times d_{\text{head}} \times 2$ (gather) |
| Head-count constraint | None ($P$ can exceed $n_{\text{heads}}$) | $P \leq n_{\text{heads}}$ (or $n_{\text{kv\_heads}}$ under GQA) | None ($P$ can exceed $n_{\text{heads}}$) |
| Causal load balance | Requires Striped Attention variant | Balanced by design (heads, not sequence positions) | Even-odd interleaving built in |
| Overlap strategy | Async K/V send overlaps with compute | All-to-all overlapped with QKV projection | All-gather fused with FlashAttention tiling |
| Integration complexity | Standalone; requires custom comm kernel | Standalone; works with any framework | Tightly coupled to Megatron-Core TP/PP/DP axes |

## When to Use Which

**Use DeepSpeed Ulysses** when:
- Sequence length is moderate (32k–256k tokens) and head count is large enough to support the required $P$ (i.e., $n_{\text{heads}} \geq P$ or $n_{\text{kv\_heads}} \geq P$ under GQA).
- You are already using DeepSpeed for ZeRO or MoE and want minimal integration overhead.
- You prefer globally synchronizing collectives (easier to reason about correctness) over asynchronous ring communication.
- The model does not use aggressive GQA (few KV heads would cap $P$ too low).

**Use Ring Attention / Megatron-CP** when:
- Sequence length is extreme ($\geq 512$k tokens) and head-count-based parallelism is insufficient.
- You need $P > n_{\text{heads}}$, which Ulysses cannot provide.
- You are already inside the Megatron ecosystem and want CP to compose with existing TP/PP/DP configurations without rewriting the training loop.
- Causal models are the primary target (both support even-odd interleaving; load balance is critical at extreme lengths).

**Combining approaches**: Ulysses and Ring Attention are complementary and can be composed — Ulysses handles the head dimension, Ring handles the remaining sequence dimension. This allows $P = P_{\text{Ulysses}} \times P_{\text{Ring}}$ total parallelism degree, useful when head count is 64+ and sequences are millions of tokens. DeepSpeed has published results using hybrid Ulysses + Ring.

## Integration with 4D Parallelism

Long-context training in practice requires all four parallelism axes simultaneously:

```
total_gpus = TP × PP × CP × DP
```

CP is the axis that resolves the sequence memory bottleneck; the other three axes are determined by model size (TP/PP) and throughput (DP). The communication patterns are designed to be non-interfering: TP all-reduces operate within a node (NVLink), PP send/receives cross the pipeline boundary between node groups, CP ring or gather operations communicate within the CP group (often also within a node or across a small number of nodes), and DP gradient all-reduces operate across the entire DP axis.

FlashAttention (see [../flash-attention/](../flash-attention/en.md)) is a prerequisite for any sequence-parallel attention implementation: without tile-level attention and online softmax, the $O(S^2)$ attention matrix for the full sequence would have to be materialized even after splitting $S$ across devices, defeating the purpose. All three SP approaches described here assume FlashAttention or an equivalent IO-aware kernel.

## Reproducibility Notes

- Ring Attention is implemented in the open-source EasyContext library and JAX-based LongContext training repositories; the core algorithm is straightforward but causal load balancing requires Striped Attention.
- DeepSpeed Ulysses is implemented in DeepSpeed (see [../../microsoft/deepspeed/](../../microsoft/deepspeed/en.md)) and available as `deepspeed.sequence_parallel`.
- Megatron-CP is implemented in Megatron-Core (see [../megatron-lm/](../megatron-lm/en.md)) and requires Megatron's parallelism configuration. NVIDIA publishes validated recipes for Llama-class models.
- FlashAttention must be available for all three; FA2 is the minimum for stable training, FA3 is recommended on H100.
- Even-odd interleaving for causal load balancing is not optional at sequence lengths where one token chunk has significantly more causal work than another — without it, some GPUs will stall waiting for others, and effective throughput collapses.

## References

- [1] Liu et al. _Ring Attention with Blockwise Transformers for Near-Infinite Context._ arXiv:2310.01889, 2023.
- [2] Brandon et al. _Striped Attention: Faster Ring Attention for Causal Transformers._ arXiv:2311.09431, 2023.
- [3] Jacobs et al. _DeepSpeed Ulysses: System Optimizations for Enabling Training of Extreme Long Sequence Transformer Models._ arXiv:2309.14509, 2023.
- [4] NVIDIA Megatron-Core. Context Parallelism implementation. github.com/NVIDIA/Megatron-LM, 2024.
- [5] Meta AI. _The Llama 3 Herd of Models._ arXiv:2407.21783, 2024.
- [6] Related: [Ring Attention](../ring-attention/en.md), [Megatron-LM](../megatron-lm/en.md), [FlashAttention](../flash-attention/en.md), [Llama 3](../../meta/llama3/en.md), [DeepSpeed](../../microsoft/deepspeed/en.md)
