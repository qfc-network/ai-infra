# Knowledge Distillation at Scale

- **Foundational work**: Hinton et al., "Distilling the Knowledge in a Neural Network" (arXiv:1503.02531, 2015)
- **Key modern instances**: TinyBERT (Jiao et al., 2020) · DistilBERT (Sanh et al., 2019) · Gemma 2 (Google DeepMind, 2024) · DeepSeek-R1 distilled series (2025)
- **Links**: [Hinton et al.](https://arxiv.org/abs/1503.02531) · [TinyBERT](https://arxiv.org/abs/1909.10351) · [Gemma 2](https://arxiv.org/abs/2408.00118) · [DeepSeek-R1](https://arxiv.org/abs/2501.12948)

## TL;DR

Knowledge distillation trains a smaller student model to match the output distribution of a larger teacher, rather than training on ground-truth labels alone. The core insight is that a teacher's soft probability distribution over the vocabulary (or over classes) encodes far more information than a one-hot label: it assigns small but nonzero probability to semantically related outputs, conveying structural similarity the label never could. At scale, three distillation regimes exist — white-box feature distillation (match internals), black-box response distillation (match logits or text outputs), and sequence-level distillation (train on teacher-generated text). The infra consequences differ sharply: white-box requires co-resident teacher and student; black-box can run offline from a static dataset; sequence-level is cheapest operationally but loses the soft distribution entirely. Gemma 2's 27B model and the DeepSeek-R1 distilled series are the clearest recent evidence that distillation is now a first-class method for getting frontier-quality reasoning into deployable model sizes.

## The Core Distillation Loss

### Soft Label Loss

Given a teacher T and student S with logit vectors `z_T` and `z_S` over the vocabulary, the soft label loss is:

$$\mathcal{L}_\text{KD} = \tau^2 \cdot \mathrm{KL}\!\left(\sigma\!\left(\frac{z_T}{\tau}\right) \,\middle\|\, \sigma\!\left(\frac{z_S}{\tau}\right)\right)$$

where `σ` is the softmax function and `τ` is temperature. The `τ²` prefactor compensates for the fact that higher temperature flattens the distributions, reducing the magnitude of gradients — without the correction, the KD loss contribution would shrink as `τ` increases.

At `τ = 1`, the loss uses the teacher's unmodified output distribution. At `τ > 1`, the distribution flattens: tokens with probability 0.001 under `τ=1` become more distinguishable from tokens with probability 0.0001. This **amplifies "dark knowledge"** — the teacher's beliefs about wrong outputs. A language model assigning 0.002 probability to "feline" when generating after "the cat sat on the" and 0.0001 to "carburetor" is signaling structural similarity between "feline" and the correct continuation; that signal is washed out in a one-hot label. High temperature makes the signal legible to the student.

### Combined Training Loss

In practice, the student is trained on a weighted combination of ground-truth cross-entropy and the KD loss:

$$\mathcal{L} = (1-\alpha)\cdot\mathcal{L}_\text{CE}(\text{student}, \text{label}) + \alpha\cdot\mathcal{L}_\text{KD}(\text{student}, \text{teacher})$$

`α` is typically 0.5–0.9 for LLM distillation. Setting `α=1` discards ground-truth labels entirely and trains only on the teacher distribution; this is the standard configuration for sequence-level distillation where there may be no ground-truth labels at all (the student trains on teacher-generated chains of thought).

## Three Distillation Regimes

### White-Box / Feature Distillation

Full access to teacher internals. The student is trained to match not just logits but intermediate representations: hidden states layer by layer, attention matrices, or activation statistics.

**TinyBERT** (2020) distills BERT into a much smaller model by matching: the embedding layer output, each Transformer layer's hidden states and attention matrices (via MSE loss on the attention score matrices), and the final prediction layer. Layer mapping is required when student and teacher have different depths: a student with 4 layers is mapped to a subset of the teacher's 12 layers.

**Infra cost**: the teacher must be resident in GPU memory during student training. For a 13B teacher and 1B student, the teacher's activations need to be materialized at each distilled layer — not just the logits. This increases per-step memory by the teacher's activation size, which can be several GB depending on sequence length. Co-locating teacher and student on the same GPU requires large VRAM; alternatively, the teacher runs on a separate GPU and activations are transferred via NVLink/PCIe.

**When to use**: maximum transfer efficiency; the student can learn the teacher's internal reasoning structure, not just its surface behavior. Preferred when the student is small enough that logit-only supervision is insufficient.

### Black-Box / Response Distillation

Only the teacher's output distribution (logits or softmax probabilities) is used. No access to internals. The student is trained to minimize KL divergence between its output distribution and the teacher's, using the combined loss above.

**Gemma 2** uses response distillation for all sizes (9B, 27B). A key engineering addition is **logit soft-capping**: before computing softmax, logits are transformed as:

$$z_\text{capped} = \tanh\!\left(\frac{z}{\text{cap}}\right) \cdot \text{cap}$$

with `cap = 30`. This prevents logit spikes: if the teacher is highly confident on one token (logit 200), the raw distribution collapses to near-certainty and the soft label signal disappears — the KD loss degenerates to cross-entropy. Soft-capping compresses extreme logits smoothly, preserving the non-dominant probability mass that carries dark knowledge. This is both a stability mechanism and a distillation quality mechanism.

**Infra cost**: only the teacher's logit vector is required per token, not full activations. This can be generated offline — run the teacher over the training corpus once, save logit tensors to disk (or a compressed version), and train the student from the stored logits. This **offline distillation** decouples teacher inference from student training, simplifying the job topology: no co-resident teacher, no teacher inference server, no synchronization. The cost is storage: for a vocabulary of 256K tokens (Gemma 2), saving FP32 logits per token requires 1 MB per 256 tokens — manageable but non-trivial at the scale of trillions of tokens.

### Sequence-Level Distillation

The simplest operational form. The teacher generates text (completions, chains of thought, solutions) and the student is fine-tuned on that text using standard cross-entropy. No soft labels; no KD loss; just next-token prediction on teacher-generated data.

**DeepSeek-R1 distillation** (2025) is the clearest large-scale example. DeepSeek-R1, a 671B reasoning model trained via GRPO, was used to generate long chains of thought on mathematics, science, and coding problems. Models of 1.5B, 7B, 14B, 32B, and 70B parameters were then fine-tuned on this data. The 7B distilled model matches R1-Zero 70B on AIME 2024 math benchmarks; the 14B model matches or exceeds GPT-4o on several reasoning tasks. The "distillation" here is entirely sequence-level — no KD loss, no soft labels — which is why it is operationally trivial once the teacher generation step is done.

**What is lost**: the student sees only samples from the teacher's distribution, not the full distribution. It cannot learn the teacher's uncertainty structure (which tokens are close in probability); it can only imitate the teacher's most-likely trajectories. For most practical purposes at current scales, this loss is acceptable — the gap between sequence-level and full KD distillation is smaller than the gap between both and training from scratch.

**Infra cost**: the teacher generation step is the expensive part. Generating 100B tokens of chain-of-thought from a 671B model at 100 tokens/second requires ~10^9 seconds of 671B-model compute — roughly 1000 A100-days. This is a one-time job. The student fine-tuning is orders of magnitude cheaper. The decoupling is the point: teacher generation runs once on a cluster; student training runs independently, potentially on smaller hardware.

## Step-by-Step / Process Distillation

Sequence-level distillation of reasoning traces (chain of thought) is a specific form called **process distillation**. The student learns to generate not just the correct final answer but the reasoning process that leads to it.

A subtlety in loss direction: standard KD minimizes the forward KL divergence `KL(p_T || p_S)`, which is mean-seeking — the student tries to cover all of the teacher's probability mass. For sequence generation, **MiniLLM** (Gu et al., 2024) argues that reverse KL `KL(p_S || p_T)` is preferable: it is mode-seeking, forcing the student to commit to specific reasoning paths rather than averaging over multiple modes. Mean-seeking can produce incoherent outputs that blend multiple reasoning strategies. In practice, for chain-of-thought distillation, training on teacher-generated sequences (sequence-level) implicitly approximates reverse KL and avoids the mode-averaging problem.

## Infrastructure Implications

### Teacher Residence During Training

For online distillation (teacher generates targets during student training, or activations are matched live):

- **Same job, same GPUs**: requires VRAM for both teacher and student simultaneously. A 70B teacher + 7B student in BF16 requires ~140 GB (teacher) + 14 GB (student) = ~154 GB minimum, plus optimizer state for the student. On 8×H100 (640 GB aggregate), this is feasible but tight.
- **Separate teacher servers**: teacher inference runs on a separate set of GPUs; student training communicates with teacher inference servers via gRPC or shared storage. More flexible but adds latency and network overhead. Doubles the cluster footprint.
- **Offline distillation (preferred)**: run teacher inference separately to generate a static dataset of logits or sequences. Student training reads from this dataset. Zero teacher overhead during student training; full teacher cluster can be repurposed after generation.

### Memory Budget for Feature Distillation

For white-box distillation matching `k` layers with hidden dimension `d` and sequence length `L`:

Additional activation memory per step ≈ `k × L × d × sizeof(dtype)`. For `k=12`, `L=2048`, `d=5120` (13B BERT-class), FP16: 12 × 2048 × 5120 × 2 bytes ≈ 250 MB per batch element. At batch size 32, this is ~8 GB of additional activation storage from the teacher alone — significant.

### Online vs. Offline Trade-Off

Online distillation allows the teacher to adapt its targets based on the student's current state (curriculum adaptation: harder examples when student is stronger). This can improve final quality but requires maintaining a live teacher inference process throughout training. The GPU overhead typically doubles the compute cost. For most production distillation workflows, offline distillation is the pragmatic choice: generate once, store, train many students from the same teacher data.

## Scale of Results

The empirical case for distillation at scale:

- **Gemma 2 27B**: matches or exceeds Llama 3 70B on most benchmarks, despite being 2.5× smaller. Distillation from the 27B teacher and soft-capped KD are credited as primary contributors.
- **DeepSeek-R1 distilled 7B**: achieves 55.5% on AIME 2024, matching R1-Zero 70B (55.5%). An 8× parameter reduction with matched reasoning performance.
- **DeepSeek-R1 distilled 14B**: 69.7% on AIME 2024, exceeding QwQ-32B despite being 2× smaller.
- **TinyBERT 4-layer**: 96.8% of BERT-base on GLUE despite 7.5× fewer parameters and 9.4× inference speedup.

The pattern is consistent: distillation closes 60–80% of the performance gap between a small model trained from scratch and the teacher, at a fraction of the pretraining compute.

## Limitations

**Teacher ceiling**: the student cannot systematically exceed the teacher. If the teacher has systematic biases or failure modes, the student inherits them. A teacher that is weak on specific reasoning patterns will produce distillation data that encodes those weaknesses.

**Distribution shift**: the teacher was trained on one distribution; the student's fine-tuning data may not match the teacher's training distribution. For sequence-level distillation, the teacher generates responses to prompts that may be out-of-distribution for it, producing low-quality reasoning traces for exactly the hard cases where distillation would be most valuable.

**Mode collapse with KD direction**: forward KL (standard) is mean-seeking — for highly multimodal teacher distributions (e.g., two equally valid reasoning paths), the student may assign moderate probability to both and high probability to neither, producing incoherent outputs. Reverse KL forces commitment but risks ignoring valid modes entirely.

**Soft label storage at LLM scale**: saving full vocabulary soft labels for a trillion-token corpus at 256K vocabulary size requires petabyte-scale storage (assuming FP16: 512 bytes/token × 10^12 tokens = 512 TB). Practical solutions: top-K logit compression (save only the K highest-probability logits and their values), temperature-compressed distributions, or online generation without storage.

## Engineering Tradeoffs

| Approach | Pros | Cons | When to use |
|---|---|---|---|
| White-box feature distillation | Maximum transfer; student learns internal representations | Teacher must be co-resident; large activation memory overhead; requires layer mapping | Small student (≤1B) where logit supervision is insufficient; BERT-class models |
| Black-box response distillation (offline logits) | Decouples teacher and student jobs; clean infra; supports large teachers | Requires soft label storage at scale; no internal signal | Medium-large students (7B–70B); Gemma 2-style production distillation |
| Sequence-level distillation (offline text) | Operationally simplest; standard fine-tuning pipeline; no soft label storage | Loses distribution information; student sees only teacher's modes | Chain-of-thought distillation; R1-style reasoning transfer; largest teachers (671B+) |
| Online distillation | Curriculum adaptation; freshest teacher targets | Doubles GPU footprint; synchronization complexity | When student-teacher co-training is feasible; research settings with budget |
| Logit soft-capping (Gemma 2) | Stabilizes KD loss; preserves dark knowledge from peaked distributions | Adds a hyperparameter (cap value); changes teacher's effective distribution | Any response distillation from a large, confident LLM |

## Cross-References

- `../../google/gemma2/` — logit soft-capping and distillation from the 27B teacher are the two main Gemma 2 architectural choices; the distillation recipe is described in detail in the Gemma 2 technical report.
- `../../deepseek/r1/` — R1 distillation pipeline: GRPO-trained 671B teacher generates chain-of-thought data; 1.5B–70B students are fine-tuned on that data via standard cross-entropy.
- `../rlhf/` — RLHF's SFT stage uses human-generated demonstrations; distillation replaces human data with model-generated data, enabling scale that human annotation cannot match.
- `../dpo/` — DPO trains on preference pairs; can be combined with distillation: use teacher completions as the preferred response and random/earlier model completions as the rejected response, giving a distillation-guided DPO dataset without human raters.

## References

- [1] Hinton et al. _Distilling the Knowledge in a Neural Network._ arXiv:1503.02531, 2015.
- [2] Jiao et al. _TinyBERT: Distilling BERT for Natural Language Understanding._ EMNLP '20 / arXiv:1909.10351.
- [3] Sanh et al. _DistilBERT, a distilled version of BERT._ arXiv:1910.01108, 2019.
- [4] Gemma Team. _Gemma 2: Improving Open Language Models at a Practical Size._ arXiv:2408.00118, 2024.
- [5] DeepSeek-AI. _DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning._ arXiv:2501.12948, 2025.
- [6] Gu et al. _MiniLLM: Knowledge Distillation of Large Language Models._ ICLR '24.
- [7] Kim & Rush. _Sequence-Level Knowledge Distillation._ EMNLP '16.
