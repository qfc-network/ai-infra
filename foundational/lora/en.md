# LoRA / QLoRA — Low-Rank Adaptation for Parameter-Efficient Fine-Tuning

- **Authors / Org**: Hu et al., Microsoft (LoRA); Dettmers et al., University of Washington (QLoRA)
- **Published**: 2021-06 / ICLR 2022 (LoRA, arXiv:2106.09685); 2023-05 / NeurIPS 2023 (QLoRA, arXiv:2305.14314)
- **Links**: [LoRA paper](https://arxiv.org/abs/2106.09685) · [QLoRA paper](https://arxiv.org/abs/2305.14314) · [HuggingFace PEFT](https://github.com/huggingface/peft) · [bitsandbytes](https://github.com/TimDettmers/bitsandbytes)

## TL;DR

Full fine-tuning of a 7B-parameter model requires storing and updating 7 billion gradients and optimizer states — roughly 112 GB in AdamW at BF16, before activations. For 65B models, it is simply impossible on anything short of a multi-node A100 cluster. LoRA (Low-Rank Adaptation) sidesteps this by observing that **weight updates during fine-tuning have low intrinsic rank**: rather than updating the full weight matrix W, train a pair of small matrices B and A whose product BA approximates ΔW. At inference, merge BA back into W — zero overhead. QLoRA pushes further: quantize the frozen base model to 4 bits (NF4), keep the LoRA adapters in BF16, and page optimizer states to CPU when GPU memory runs short. The result: fine-tune a 65B Llama model on a **single 48 GB A100**. LoRA and QLoRA are now the default fine-tuning primitives for the open-source LLM ecosystem.

## Context & Motivation

### The cost of full fine-tuning

Fine-tuning a pre-trained LLM on a downstream task classically means updating all parameters: gradient computation, gradient storage, and optimizer state storage (Adam requires two moments per parameter) for every weight in the model. At BF16:

- 7B model: ~14 GB weights + ~28 GB optimizer states + activations ≈ 80+ GB total
- 13B model: ~26 GB weights + ~52 GB optimizer states ≈ 100+ GB total
- 65B model: ~130 GB weights + ~260 GB optimizer states ≈ impractical on single nodes

Even with gradient checkpointing and mixed precision, full fine-tuning at 65B requires 8× A100-80GB or more, which is inaccessible to most research groups and nearly all practitioners.

### Previous approaches and their problems

Before LoRA, the dominant parameter-efficient methods were:

- **Adapter layers**: insert small bottleneck MLP modules between Transformer layers, train only those. Problem: adds latency at inference (extra sequential operations); can't be easily removed.
- **Prefix tuning / prompt tuning**: prepend trainable soft tokens to the input or each layer's KV. Problem: reduces effective context length; can be unstable to optimize; doesn't merge away.
- **BitFit**: train only bias parameters. Very few parameters but limited expressivity; doesn't generalize well to diverse tasks.

All three either add inference overhead or sacrifice quality. LoRA's design goal was: **match full fine-tuning quality, add zero inference overhead, use a fraction of the parameters**.

### The low-rank hypothesis

The key empirical observation motivating LoRA: during fine-tuning, the gradient updates to weight matrices have low **intrinsic dimensionality**. That is, the optimization trajectory mostly lives in a low-dimensional subspace of the full parameter space. If this is true, a low-rank decomposition of ΔW should be sufficient to capture the fine-tuning signal without needing to represent all d×k dimensions.

Aghajanyan et al. (2020) provided empirical support for this via intrinsic dimension analysis, showing that many NLP tasks can be solved by optimizing fewer than 1,000 parameters in an appropriate subspace of a pre-trained model. LoRA operationalizes this observation into a practical training technique.

## Core Method

### LoRA: Low-Rank Decomposition of Weight Updates

For a pre-trained weight matrix W₀ ∈ ℝ^{d×k}, instead of learning the full update ΔW ∈ ℝ^{d×k}, LoRA learns a low-rank decomposition:

```
ΔW = BA,  where B ∈ ℝ^{d×r}, A ∈ ℝ^{r×k}, r ≪ min(d, k)
```

The modified forward pass during training is:

```
h = W₀x + (α/r) · BAx
```

where α is a scaling hyperparameter (typically set to r or 2r; α/r acts as a learning-rate-like scale on the LoRA update). W₀ is **frozen** throughout training; only A and B are updated.

Initialization: A is initialized from a Gaussian (so ΔW = BA ≈ 0 at step 0, preserving the pre-trained behavior); B is initialized to zero.

**Parameter count**: for a single weight matrix with d=k=4096 and r=8, full fine-tuning requires 4096×4096 = 16.7M parameters; LoRA requires 2×8×4096 = 65,536 parameters — **0.4% of the original**. Across all attention projection matrices in a 7B model (Q, K, V, O in each of 32 layers), LoRA with r=8 adds roughly 4–8M trainable parameters, less than 0.1% of total.

### Where to apply LoRA

LoRA is typically applied to the linear projections in the attention mechanism: Q (query), K (key), V (value), and O (output) projections. The original LoRA paper found that applying to Q and V alone was often sufficient; later work (including QLoRA) found that applying to all four plus the MLP layers (gate, up, down projections in Llama-style models) gives better quality, especially for instruction following and coding tasks.

### Merging at inference

At inference time, the update can be merged directly into the weight:

```
W = W₀ + (α/r) · BA
```

This is a one-time operation done before deployment. The merged model is identical in shape and latency to the base model — no extra compute, no extra memory. LoRA's inference overhead is exactly zero when merged.

Alternatively, the adapter can be kept **unmerged** and applied dynamically. This enables **LoRA multiplexing**: a single base model GPU deployment can serve multiple fine-tuned variants by swapping adapters per request (the adapter is small enough to load in milliseconds). Systems like S-LoRA (Sheng et al., 2023) implement this at scale.

### QLoRA: Quantized Base + BF16 Adapters

QLoRA extends LoRA by quantizing the frozen base model to 4 bits, dramatically reducing its memory footprint while keeping the LoRA adapters in BF16 for stable gradient computation.

**NF4 (Normal Float 4)**: a 4-bit data type specifically designed for normally distributed weights. The 16 quantization levels are placed at the quantiles of a standard normal distribution — optimal for weights that follow approximately N(0, σ²). NF4 achieves lower quantization error than standard INT4 for normally distributed data.

**Double quantization**: the quantization constants (one per block of weights, typically 64 parameters) are themselves quantized from FP32 to FP8, saving an additional ~0.37 bits per parameter on average.

**Paged optimizers**: QLoRA uses NVIDIA's unified memory (accessible via `cudaMallocManaged`) to transparently page optimizer states between GPU and CPU when the GPU runs short of memory. This handles memory spikes during the backward pass without crashing — critical for fitting 65B on a single GPU.

**Forward pass**: during training, the NF4 weights are dequantized to BF16 on-the-fly before matrix multiplication, then discarded. The actual compute happens in BF16; only storage is in NF4. Quantization errors accumulate only in the base model's frozen weights, not in the trained LoRA adapters.

## Engineering Tradeoffs

| Decision | Gained | Gave up |
|---|---|---|
| LoRA vs full fine-tuning | 10–100× fewer trainable parameters; fits in single-GPU memory; faster iteration | Slight quality ceiling on tasks requiring broad weight changes (e.g., domain adaptation); limited to directions captured by rank-r subspace |
| LoRA rank r (4 → 64) | Higher r: more expressivity, can represent more complex updates | Higher r: more parameters, more memory, slower convergence; beyond r=64 rarely helps; r=8–16 is the usual sweet spot |
| Which layers to apply LoRA | All projections (Q/K/V/O + MLP): better quality, especially for complex tasks | More adapters = more parameters; Q+V only is sufficient for simple classification/NLU |
| NF4 base (QLoRA) vs BF16 base | 4× memory reduction for base model; enables 65B on 1× A100-80GB | ~1–3% quality gap on some benchmarks vs BF16 full fine-tune; dequantization adds ~10% compute overhead |
| Merged vs unmerged at inference | Merged: zero latency overhead, identical to base model | Unmerged: adapter swapping, serving multiple tasks from one base; can't merge multiple adapters without arithmetic |

## Experiments & Results

**LoRA (Hu et al., 2022):**
- Evaluated on GPT-3 (175B) fine-tuned on GLUE, SuperGLUE, and E2E NLG benchmarks.
- With r=4, LoRA matches full fine-tuning on most tasks while training only 0.01% of parameters (as few as 4.7M out of 175B).
- Inference latency: identical to base model after merging (confirmed by profiling — zero overhead).
- Ablation: applying to Q+V outperforms Q-only; all four attention matrices (Q/K/V/O) give marginal additional gains at 2× the parameters.

**QLoRA (Dettmers et al., 2023):**
- Fine-tuned Llama-65B on a single NVIDIA A100-80GB (previously required ~8× A100). Training speed: roughly 24 GPU-hours per 10K steps on Alpaca dataset.
- Guanaco-65B (QLoRA fine-tune of Llama-65B on OASST1 dataset): achieves 99.3% of ChatGPT performance on Vicuna benchmark, per human evaluators.
- Guanaco-7B (QLoRA on Llama-7B, single 24 GB consumer GPU): competitive with models that required full fine-tuning on 8× A100.
- Quality comparison: BF16 full fine-tune of 7B > QLoRA of 7B by ~1–2% on MMLU; but QLoRA of 13B or 33B fits in the same GPU budget and generally outperforms full fine-tune of 7B — scaling wins.

## Reproducibility Notes

LoRA and QLoRA are both fully open-source and production-grade:

- **HuggingFace PEFT** (`pip install peft`): canonical LoRA implementation. `LoraConfig` specifies target modules, rank, alpha, dropout. Integrates directly with `Trainer`.
- **bitsandbytes** (`pip install bitsandbytes`): provides NF4 quantization, double quantization, and paged optimizers. Used by PEFT's QLoRA path.
- **Axolotl**, **LLaMA-Factory**, **Unsloth**: higher-level wrappers that configure PEFT/bitsandbytes and handle data preprocessing.

Typical QLoRA recipe for a 7B model on a single A100-40GB:
```python
from transformers import BitsAndBytesConfig
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
)
model = AutoModelForCausalLM.from_pretrained(base_model, quantization_config=bnb_config)
lora_config = LoraConfig(r=16, lora_alpha=32, target_modules=["q_proj","v_proj","k_proj","o_proj"])
model = get_peft_model(model, lora_config)
```

Common failure modes:
- **Forgetting to enable gradient checkpointing**: without it, activation memory causes OOM even with QLoRA on large models.
- **Wrong target modules**: different model families have different projection names; always check `model.named_modules()`.
- **Alpha too low**: if `lora_alpha` is much smaller than `r`, the adapter updates are too small; convention is alpha = r or 2r.

## Commentary

LoRA's impact comes from a property rare in ML papers: **it makes a large existing system strictly better with no downside at deployment time**. The merged model is indistinguishable from a full fine-tune in shape and latency. This is a stronger claim than "comparable quality with fewer resources" — it means you can adopt LoRA without any inference infrastructure changes.

QLoRA then did something more radical: it moved the memory frontier by a factor of four for the frozen base model, at the cost of a small quality penalty. The practical consequence is that the "who can fine-tune a 65B model" question went from "hyperscalers only" to "anyone with access to a rented A100." This democratization effect compounded: the open-source community could now experiment with 65B-scale models, driving the explosion of Llama-based fine-tunes in 2023–2024.

The residual limitations are real but narrow:

- **Rank limitation**: LoRA cannot represent updates that require more than r independent directions. For tasks that require broad distributional shift (e.g., adding a completely new language to a model), full fine-tuning or a higher-rank adapter is necessary.
- **NF4 quality gap**: for tasks sensitive to absolute model quality (e.g., competitive math, complex reasoning), the 1–3% QLoRA penalty matters. The practical answer is: use a larger QLoRA model that fits in the same budget — scaling usually covers the gap.
- **Adapter composition**: combining multiple LoRA adapters trained on different tasks is non-trivial. LoraHub, TIES-merging, and DARE explore this, but it remains an active problem.

For a practitioner in 2026: LoRA (r=8–16, applied to all attention + MLP projections) is the correct default for any fine-tuning task. QLoRA is the correct default if the model doesn't fit in GPU memory at BF16. Use full fine-tuning only if you have the compute budget and have confirmed that LoRA's rank constraint is the binding quality limit.

## References

- [1] Hu et al. _LoRA: Low-Rank Adaptation of Large Language Models._ ICLR 2022 / arXiv:2106.09685.
- [2] Dettmers et al. _QLoRA: Efficient Finetuning of Quantized LLMs._ NeurIPS 2023 / arXiv:2305.14314.
- [3] Aghajanyan et al. _Intrinsic Dimensionality Explains the Effectiveness of Language Model Fine-Tuning._ ACL 2021.
- [4] Sheng et al. _S-LoRA: Serving Thousands of Concurrent LoRA Adapters._ arXiv:2311.03285, 2023.
- [5] HuggingFace PEFT library. https://github.com/huggingface/peft
- [6] bitsandbytes. https://github.com/TimDettmers/bitsandbytes
- [7] Zhao et al. _TIES-Merging: Resolving Interference When Merging Models._ NeurIPS 2023 / arXiv:2306.01708.
