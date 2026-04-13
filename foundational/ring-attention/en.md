# Ring Attention (and Context Parallelism)

- **Authors / Org**: Liu, Zaharia, Abbeel (UC Berkeley)
- **Published**: 2023-10
- **Links**: [Ring Attention (arXiv:2310.01889)](https://arxiv.org/abs/2310.01889) · [Blockwise Transformers (arXiv:2305.19370)](https://arxiv.org/abs/2305.19370) · [Striped Attention](https://arxiv.org/abs/2311.09431)

## TL;DR

At very long context, attention's memory no longer fits on a single GPU **even with FlashAttention**. Ring Attention shards the **sequence axis** across GPUs and passes K/V blocks around a ring: each device computes partial attention against whatever K/V slice it currently holds, then forwards that slice to the next device while receiving the next one. Combined with blockwise softmax, the full attention is computed exactly, with compute and ring-communication fully overlapped. Used in Gemini 1.5's million-token context, adopted (with variants) in NVIDIA's Context Parallelism, Llama 3's CP axis, and most long-context training stacks today.

## Context & Motivation

FlashAttention solved the memory-bandwidth problem for attention on a single GPU. At `N = 128k` to `1M+` tokens, two new issues appear:

1. **Activation memory still scales with `N`** — even with online softmax, storing Q, K, V along the sequence exhausts a single GPU.
2. **No intra-sequence parallelism** — Megatron's TP splits the hidden dimension, not the sequence; FSDP splits parameters, not activations. Neither helps with long sequences.

Previous tricks (blockwise computation, gradient checkpointing) reduced memory but didn't distribute it. Ring Attention is the first clean way to **parallelize one sequence across multiple devices** while computing exact attention.

## Core Method

### Sequence sharding + ring rotation

Split the sequence into `P` equal chunks, place chunk `i` on device `i`. Each device holds its own slice of Q, K, V. To compute attention, each device must eventually see every K/V in the sequence.

- **Ring topology**: arrange devices `0 → 1 → ... → P-1 → 0`.
- At step `s`, device `i` holds its Q permanently, and a K/V block that originally belonged to device `(i − s) mod P`.
- Device `i` computes partial attention of its Q against the current K/V block using **blockwise FlashAttention with online softmax**.
- Meanwhile, asynchronously **send the current K/V block to device `i+1`** and **receive the next block from device `i-1`**.
- After `P` steps, every Q has seen every K/V; online softmax normalization across steps produces the exact attention.

### Overlap

If compute time per block ≈ communication time per block, the ring runs at compute-bound speed regardless of `P`. This is Ring Attention's defining property: **adding devices lengthens the context linearly, not the wall time**.

### Causal masking and variants

- Naive ring with causal mask leaves half the GPUs idle late in the sequence (upper-triangular blocks are zero).
- **Striped Attention** (follow-up from the same authors) permutes sequence positions so that each ring step has balanced work across devices. ~2× throughput for causal models at long context.
- **NVIDIA Context Parallelism** (used in Megatron-Core, Llama 3): production-quality ring with Striped-like balancing, integrated with TP/PP/FSDP.

### Composition with other parallelism

CP is **orthogonal** to TP/PP/DP/FSDP. The resulting parallel volume:

```
total_gpus = TP × PP × CP × DP
```

Llama 3 used `CP=2` at long-context training stage; frontier labs reportedly go further.

## Engineering Tradeoffs

| Decision | Gained | Gave up |
|---|---|---|
| Shard sequence across devices | Context scales with `P`; single-GPU memory constant per chunk | Requires `P` devices even for short seqs (not used until long-context phase) |
| Ring communication | Fully overlapped with compute when tuned | Sensitive to imbalance; naive causal masking halves efficiency |
| Exact attention | No quality approximation | Cannot be combined with sparse attention tricks without care |
| Orthogonal to TP/PP/DP | Drop-in 4th axis | Topology and schedule interact with other axes; config complexity up |
| Striped permutation | Balanced causal work | Position IDs require shuffling; some kernels need rewriting |

## Experiments & Results

- Original paper: exact attention at **millions of tokens** on 32-TPU v4 pod; near-linear scaling in sequence length as devices are added.
- Striped Attention: ~2× over naive ring for causal LMs at 128k–1M.
- Gemini 1.5 Pro reported 1M (and later 10M) context; architecture details not public but Ring/CP is the accepted mechanism.
- Llama 3 405B: `CP=2` used during the 8k→128k context-extension stage.

## Reproducibility Notes

- Reference implementations: original JAX code (Liu et al.), `ring-flash-attention` (PyTorch, Zilin Zhu), NVIDIA Megatron-Core CP.
- Correctness verification: compare against single-GPU FlashAttention on small sequences; numerical identity within FP16/BF16 tolerance.
- Production deployment lives inside large training frameworks (Megatron-Core, MaxText, torchtitan).

## Commentary

Ring Attention is the elegant answer to a question that looked algorithmically unsolvable: **parallelize one sequence across many GPUs without approximating attention**. The key insight — that online softmax is associative across K/V blocks, so blocks can arrive in any order — was already in FlashAttention; Ring Attention just exploits it with a ring topology. The technique is quietly essential: every credible "1M context" claim in 2024–2025 relies on CP under the hood. Looking ahead, the interesting frontier is **heterogeneous ring topologies** (mixing fast NVLink hops with slower cross-node links), **disaggregated long-context serving** (prefill on a big CP ring, decode elsewhere), and combining CP with long-context-friendly architectures like MLA. Anyone training or serving >32k context should understand this — there is no shortcut around it.

## References

- [1] Liu, Zaharia, Abbeel. _Ring Attention with Blockwise Transformers for Near-Infinite Context._ arXiv:2310.01889, 2023.
- [2] Liu, Abbeel. _Blockwise Parallel Transformer for Large Context Models._ arXiv:2305.19370, 2023.
- [3] Brandon et al. _Striped Attention: Faster Ring Attention for Causal Transformers._ arXiv:2311.09431, 2023.
- [4] NVIDIA. _Context Parallelism in Megatron-Core._ Docs, 2024.
