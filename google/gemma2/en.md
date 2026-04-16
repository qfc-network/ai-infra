# Gemma 2

- **Authors / Org**: Gemma Team, Google DeepMind
- **Published**: 2024-08
- **Links**: [paper (arXiv:2408.00118)](https://arxiv.org/abs/2408.00118) · [model (HuggingFace)](https://huggingface.co/collections/google/gemma-2-release-667d6600fd5220e7b967f315) · [code (keras-hub / transformers)](https://github.com/keras-team/keras-hub)

## TL;DR

Gemma 2 is Google DeepMind's second-generation open-weight model family, released in three sizes: 2B, 9B, and 27B. The 27B is trained from scratch; the 2B and 9B are trained with **knowledge distillation from the 27B teacher** — a deliberate choice to pack more capability into smaller models by training against soft probability distributions rather than hard token labels. Two architectural additions relative to Gemma 1 are notable from an infra standpoint: **logit soft-capping** (a cheap training-stability technique that prevents logit explosion in BF16 without loss scaling) and **alternating local/global attention** (every other layer uses sliding-window attention, bounding KV cache growth for long sequences). The result: Gemma 2 27B is competitive with Llama 3.1 70B at half the parameter count; Gemma 2 9B competes with models in the 13B–40B range.

## Context & Motivation

### The efficiency gap in open-weight models

By mid-2024, the Llama 3 family had established a strong benchmark: 8B and 70B dense models trained on ~15T tokens, with instruction-tuned variants that matched or exceeded GPT-3.5-class performance. The practical question for the open-weight ecosystem was: can you get comparable capability at lower parameter counts?

The answer depends on two things: (1) training data quality and quantity, and (2) how effectively that data's information is transferred into model weights. Gemma 2 attacks (2) directly — for the 2B and 9B models, training on hard next-token prediction targets is replaced (or augmented) by training on the **teacher's full probability distribution**, which carries substantially more information per token.

### Why distillation matters at the small-model frontier

A one-hot label says "the correct next token is X." A teacher's soft distribution says "the correct next token is X with probability 0.6, but Y is plausible (0.2), Z is somewhat plausible (0.1), ..." The entropy in the teacher distribution encodes uncertainty, near-synonyms, plausible continuations, and reasoning paths that a hard label discards. For small models, which have limited capacity to infer these distributions from data alone, teacher guidance at every training step is a significant learning signal amplifier.

The infrastructure consequence: distillation training requires the teacher to be resident (or inference-accessible) throughout the student's training run. For a 27B teacher and a 9B student, this means co-loading both models, running teacher forward passes on every training batch, and using the resulting logit distributions as the training target.

## Core Method

### Architecture: what changed from Gemma 1

Gemma 2 retains the Gemma 1 base (Transformer with RoPE position embeddings, RMSNorm, GeGLU FFN activation, GQA throughout) and adds three modifications:

**1. Alternating local/global attention**

Attention layers alternate between two modes:
- **Local (sliding window)**: each token attends only to a window of the preceding 4,096 tokens. KV cache per layer is bounded at `2 × window_size × d_head × n_kv_heads`, regardless of sequence length.
- **Global (full)**: standard attention over the full sequence. KV cache grows with sequence length as normal.

Half the layers are local, half are global (alternating). The practical effect: for a sequence of length L, total KV cache is approximately:

```
KV cache ≈ n_layers/2 × (2 × 4096 × d_kv)   [local layers, bounded]
          + n_layers/2 × (2 × L × d_kv)       [global layers, grows with L]
```

At L = 8,192 (Gemma 2's trained context), local and global layers contribute roughly equally. At longer contexts, the local layers' contribution stays flat while global layers' grows — roughly halving the KV cache cost versus full attention throughout.

The global layers handle long-range dependencies; the local layers handle local coherence. The alternating pattern means every token is within two hops of a global attention computation.

**2. Logit soft-capping**

Applied in two places: to attention logits before the softmax, and to the final output logits before the vocabulary softmax. The formula:

```
logits_capped = cap × tanh(logits / cap)
```

With `cap = 50.0` for attention logits and `cap = 30.0` for output logits. `tanh` compresses values smoothly toward ±`cap` — extreme logits are pulled back without truncation, and small logits are nearly unaffected (since `tanh(x) ≈ x` for small x).

The motivation: BF16 training is susceptible to logit explosion — a feedback loop where large softmax inputs produce very peaked distributions, which produce large gradients, which push logits further. Loss scaling mitigates this in the gradient path but not the forward path. Soft-capping addresses it directly in the forward pass, eliminating the need for activation clipping heuristics. The computation cost is one `tanh` per attention and output softmax — negligible.

**3. Post-normalization (in addition to pre-norm)**

Gemma 2 applies RMSNorm both **before** each sublayer (standard pre-norm) and **after** each sublayer. The post-norm keeps the residual stream bounded as depth increases. Combined with soft-capping, this produces notably stable training dynamics at 27B scale even in BF16 without extensive loss scaling tuning.

### Knowledge distillation

For the 2B and 9B models, each training step computes a combination of:

1. **Standard cross-entropy loss** against the true next-token label:
   ```
   L_CE = -log p_student(x_{t+1} | x_{≤t})
   ```

2. **KL divergence against teacher logits**:
   ```
   L_KD = KL(p_teacher(· | x_{≤t}) || p_student(· | x_{≤t}))
        = Σ_v p_teacher(v) log [p_teacher(v) / p_student(v)]
   ```

The total loss is a weighted combination: `L = α · L_CE + (1 − α) · L_KD`. The KL term is computed over the full vocabulary distribution — not just the top-k tokens — which preserves the teacher's full uncertainty signal.

The teacher (Gemma 2 27B) is held frozen throughout student training. Both teacher and student see the same token sequence; teacher logits are computed via a forward pass and cached per batch. Because the teacher is 3× larger than the 9B student, teacher inference adds ~33% to the per-step compute cost.

**Why not distill the 9B into the 2B?** Using a stronger, more capable teacher produces better students — the teacher's uncertainty estimates are more calibrated, and the probability mass over plausible alternatives is larger. Using the 27B as the teacher for both 2B and 9B maximizes the signal for both.

### Model sizes and training scale

| Size | Parameters | Layers | Training tokens | Context |
|---|---|---|---|---|
| 2B | 2.6B | 26 | ~2T | 8,192 |
| 9B | 9.2B | 42 | ~8T | 8,192 |
| 27B | 27.2B | 46 | ~13T | 8,192 |

The 27B is the only model trained purely on next-token prediction. The 2B and 9B training pipelines run a 27B teacher inference pass alongside each student forward. Token budgets are smaller than Llama 3 (which trained the 8B on 15T tokens), trading data scale for distillation signal.

## Engineering Tradeoffs

| Decision | Gained | Gave up |
|---|---|---|
| Distillation for 2B / 9B | Substantially more capability per parameter; teacher's uncertainty signal reduces need for data scale | ~33% higher training compute per step (teacher forward pass); teacher must co-reside or be inference-accessible; couples student training schedule to teacher availability |
| Alternating local/global attention | ~50% KV cache reduction at trained context length; local layers are faster (bounded sequence length for each attention op) | Global layers still scale with L; long-context quality depends on global layers being well-distributed; local-only windows can miss dependencies > 4,096 tokens apart |
| Logit soft-capping | Training stability in BF16 without loss scaling tuning; prevents attention logit explosion | Breaks the dot-product attention kernel optimization (scores must pass through tanh before softmax, preventing FlashAttention-style fused kernels that merge softmax with the attention output accumulation) |
| Post-norm + pre-norm | Stable residual stream at depth; better gradient flow | Added RMSNorm operations per layer; minor compute overhead (~2% per layer) |
| GQA throughout | Reduced KV cache vs MHA; fast decoding | Fewer KV heads means coarser key/value representations; quality tradeoff at very small head counts |

Note on the soft-capping / FlashAttention interaction: the `tanh` in the attention path prevents direct use of standard FlashAttention fused kernels, which assume the scaled dot-product can be softmax'd directly. Custom kernels (Pallas on TPU, CUDA kernels with tanh fused into the attention score path) are needed for full efficiency. This is a non-trivial implementation cost that downstream serving frameworks must absorb.

## Experiments & Results

**Cross-size quality (relative to parameter count):**
- Gemma 2 27B matches or beats Llama 3.1 70B on MMLU, GSM8K, HumanEval, and several reasoning benchmarks — at ~38% of the parameter count.
- Gemma 2 9B competitive with Llama 3 70B on many benchmarks, and clearly outperforms the Llama 3 8B — demonstrating the distillation gain.
- Gemma 2 2B is the strongest model in its parameter class at release; comparable to prior-generation 7B models on several tasks.

**Distillation ablations (from the tech report):**
- 9B model trained with distillation achieves significantly higher performance than 9B trained without (next-token prediction only), at the same token budget.
- The gain is largest on reasoning tasks (GSM8K, MATH) where the teacher's uncertainty over plausible solution paths provides dense signal the hard labels miss.

**Inference efficiency:**
- At batch size 1, the alternating attention design means local-layer KV cache fits in L2/L3 for modern GPUs at typical sequence lengths, reducing HBM reads for those layers.
- Throughput at 27B is competitive with Llama 3.1 70B despite the smaller parameter count — serving is faster because fewer parameters means less weight bandwidth per token.

## Reproducibility Notes

- All three Gemma 2 model weights (base + instruction-tuned) are publicly released on HuggingFace under the Gemma license (permissive for research and commercial use below certain thresholds).
- Training code is not released. The distillation pipeline details (α weighting, teacher caching strategy, exact training curriculum) are described at a high level in the tech report but not precisely enough to reproduce from scratch.
- **HuggingFace Transformers** supports Gemma 2 (`AutoModelForCausalLM.from_pretrained("google/gemma-2-9b")`). The soft-capping is implemented natively.
- **vLLM and SGLang** both support Gemma 2 inference. The attention soft-capping requires a modified attention kernel; both frameworks implement this. Throughput is within 10–15% of uncapped attention at typical batch sizes.
- **Serving note**: the alternating local/global attention pattern requires correct handling in the KV cache manager — local layers' cache can be evicted/bounded by window size, but this optimization is not uniformly implemented across serving frameworks. Naive implementations treat all layers as global, leaving the memory savings unrealized.

## Commentary

Gemma 2's most durable contribution is the proof that **distillation from a strong teacher is a better use of small-model training compute than just scaling up tokens**. The 9B model punches well above its weight class precisely because it learned from a 27B model's full probability distributions, not just from hard data labels. This is not a new idea (Hinton et al. 2015, "Distilling the Knowledge in a Neural Network"), but applying it at this scale — online, over the full vocabulary distribution, for billions of training steps — validates it as a practical recipe for efficient open-weight model development.

The logit soft-capping technique is underappreciated. It's architecturally cheap (one tanh per attention and output), addresses a real BF16 training fragility, and the tradeoff (breaks standard FlashAttention fusion) is manageable with custom kernels. Expect to see this or similar stabilization techniques in future models that push BF16 training without extensive loss scaling infrastructure.

The alternating local/global attention is a more contested design choice. The KV cache savings are real, but the window size of 4,096 creates a hard boundary — tasks that require attending beyond 4,096 tokens in a single hop (very long document reasoning, retrieval, extended code contexts) rely entirely on global layers. Whether this is adequate depends on task distribution. Mistral's sliding-window attention (from Mistral 7B v0.1, 2023) had the same design; Mistral 7B v0.2 removed it for full context, suggesting the tradeoff is task-sensitive. Gemma 2's 8k trained context means this is mostly academic for the use cases it targets.

For practitioners: Gemma 2 9B is currently one of the best choices for deployments where a 7B–13B footprint is required and quality matters — the distillation advantage is measurable and the serving efficiency (compared to a 70B) is substantial. The instruction-tuned variants are notably well-aligned for their size.

## References

- [1] Gemma Team. _Gemma 2: Improving Open Language Models at a Practical Size._ arXiv:2408.00118, 2024.
- [2] Gemma Team. _Gemma: Open Models Based on Gemini Research and Technology._ arXiv:2403.08295, 2024.
- [3] Hinton et al. _Distilling the Knowledge in a Neural Network._ arXiv:1503.02531, 2015.
- [4] Ainslie et al. _GQA: Training Generalized Multi-Query Transformer Models._ arXiv:2305.13245, 2023.
- [5] Beltagy et al. _Longformer: The Long-Document Transformer._ arXiv:2004.05150, 2020. (Sliding window attention)
- [6] Jiang et al. _Mistral 7B._ arXiv:2310.06825, 2023. (Prior use of alternating local/global attention)
- [7] Zhang et al. _RMS Norm._ arXiv:1910.07467, 2019.
