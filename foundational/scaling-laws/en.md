# Scaling Laws for Neural Language Models — Kaplan 2020 + Chinchilla 2022

- **Authors**: Jared Kaplan, Sam McCandlish, Tom Henighan et al. (OpenAI) · Jordan Hoffmann, Sebastian Borgeaud, Arthur Mensch et al. (DeepMind)
- **Published**: 2020-01 (Kaplan et al.) · 2022-03 (Hoffmann et al. / Chinchilla)
- **Links**: [arXiv:2001.08361](https://arxiv.org/abs/2001.08361) · [arXiv:2203.15556](https://arxiv.org/abs/2203.15556)

## TL;DR

Both papers ask the same question: given a fixed compute budget, how should you allocate it between model size and training data? Kaplan et al. (2020) were the first to rigorously characterize that language model loss follows smooth **power laws** in the number of parameters, training tokens, and compute — and concluded that you should preferentially scale parameters over data. Hoffmann et al. (2022) — the Chinchilla paper — ran a corrected set of isoFLOP experiments and showed that Kaplan's methodology had a subtle flaw: it did not fully converge the smaller models it tested, artificially making them look weaker. The corrected answer is that parameters and data should scale **equally**: for every doubling of compute, both model size and dataset size should roughly double. In practice this collapses to a rule of thumb: train each parameter on about **20 tokens**. Chinchilla (70B, 1.4T tokens) validated the prediction by beating the much larger Gopher (280B, 300B tokens) on nearly every benchmark, despite using the same total compute. Together, the two papers replaced ad-hoc guesswork with a principled framework for pre-training budget allocation that every major lab now uses as a baseline.

## Context & Motivation

Before Kaplan 2020, practitioners made compute allocation decisions largely by intuition and convention. GPT-2 and then GPT-3 were scaled by OpenAI researchers who had credible but informal beliefs: bigger models are better, data is plentiful, so scale parameters. There was no principled way to answer: "if I have 10× more compute next year, what is the right N and D?" The empirical precedents that existed — from image models, recurrent networks, even earlier language models — were fragmentary and did not carry over neatly to the transformer-based, next-token-prediction paradigm.

Meanwhile, compute costs were becoming the dominant training line item. Google's Megatron-LM, OpenAI's GPT-3, and their successors cost tens of millions of dollars per run. Getting the N/D split wrong meant burning months of GPU time on an under-performing model. There was strong practical motivation to find reliable extrapolation rules.

Kaplan et al. tackled this systematically: they trained hundreds of models spanning many orders of magnitude in parameter count (768 to 1.5B) and dataset size, carefully tracking cross-entropy loss at each checkpoint. The resulting smooth curves — not the noisy spikes typical of earlier scaling work — showed that loss follows consistent power-law behavior and that the scaling exponents are stable across architectures, context lengths, and model shapes.

## Core Method

### Kaplan's Power Laws

Kaplan et al. fit three empirical power laws. Holding all other factors at their compute-optimal level:

- **Loss vs. parameters**: `L(N) ∝ N^{-0.076}`
- **Loss vs. dataset size (tokens)**: `L(D) ∝ D^{-0.095}`
- **Loss vs. compute**: `L(C) ∝ C^{-0.050}`

Each relationship is clean and holds over many orders of magnitude. The combined prediction is:

```
L(N, D) ≈ (N_c / N)^{α_N} + (D_c / D)^{α_D} + L_∞
```

where `L_∞` is an irreducible entropy floor, and `N_c`, `D_c` are scale constants fitted from data.

A key corollary: if you have a fixed compute budget `C ≈ 6ND` (the factor of 6 arises from the forward + backward pass through a transformer), you want to choose N and D to minimize L. Kaplan's analysis found that **the loss improves faster by increasing N than D at fixed compute**, leading to the recommendation to scale parameters aggressively.

### IsoFLOP Methodology

To make the analysis rigorous, both papers use **isoFLOP curves**: pick a fixed compute budget C, train many models of different sizes (varying N), adjust D = C / 6N accordingly, and record the final loss. Plotting final loss vs. N at fixed C gives a U-shaped curve (in log space), whose minimum identifies the compute-optimal (N*, D*) pair at that compute level. Repeating across many compute budgets yields the **compute-optimal frontier**.

### Kaplan's Conclusion: Scale Parameters

Kaplan's isoFLOP curves found minima at relatively high N and low D, suggesting:

- `N_opt ∝ C^{0.73}` (parameters grow faster than data)
- `D_opt ∝ C^{0.27}`

This became the operative belief that drove GPT-3 (175B params, 300B tokens) and Gopher (280B params, 300B tokens): massive models trained for a relatively short time.

### Chinchilla's Correction

Hoffmann et al. noticed that Kaplan's smaller models were not trained to convergence — they used a learning rate schedule derived from the largest model in each run and applied it uniformly. Because the LR schedule was calibrated to the long training run of large models, the smaller, shorter-run models were still on the "warm" part of the schedule when training ended. Their loss was artificially elevated, making them appear weaker than they truly were at convergence. This introduced a systematic bias toward larger N.

When Chinchilla re-ran the isoFLOP experiments with **proper per-model LR tuning** (each model trained to convergence with its own schedule), the optimal N/D split shifted dramatically:

- `N_opt ∝ C^{0.49}`
- `D_opt ∝ C^{0.51}`

Parameters and data scale almost **identically** with compute. The practical rule of thumb: **train on approximately 20 tokens per parameter**.

### Chinchilla vs. Gopher

| Model | Parameters | Training Tokens | Compute |
|---|---|---|---|
| Gopher | 280B | 300B | ~6.3 × 10²³ FLOPs |
| Chinchilla | 70B | 1.4T | ~6.3 × 10²³ FLOPs |

Same total compute. Chinchilla's training tokens-per-parameter ratio (~20) matches the new optimal; Gopher's (~1.07) is massively undertrained. Chinchilla outperforms Gopher on the majority of the 57 tasks tested in MMLU and on BIG-bench, often by a wide margin, while being 4× smaller and substantially cheaper to serve at inference.

## Engineering Tradeoffs

| Decision | Gained | Gave Up |
|---|---|---|
| Scale parameters over data (Kaplan) | Larger model, stronger zero-shot performance at fixed training compute | Data efficiency; GPT-3 and Gopher were compute-suboptimal — a smaller model trained longer would have matched them |
| Balance parameters and data (Chinchilla) | True compute-optimal training; smaller model beats undertrained larger one | Requires 20× more tokens per parameter — significant data curation and deduplication effort |
| Inference-optimal over compute-optimal | Smaller model = cheaper per-token serving cost (Llama's explicit strategy) | Higher total training compute for the same benchmark score; training budget goes up |
| Power-law extrapolation for planning | Smooth, predictable loss curves enable pre-run budget planning; can extrapolate from small runs | Emergence is discontinuous; cross-entropy loss does not predict capability jumps on tasks like arithmetic or code |
| Single-epoch training (Kaplan assumption) | Maximum sample diversity; no data repetition; clean power-law fits | Frontier models regularly repeat data; Muennighoff 2023 shows diminishing but nonzero returns past one epoch |

## Experiments & Results

**Chinchilla vs. contemporaries on MMLU (5-shot accuracy):**

| Model | Params | Tokens | MMLU |
|---|---|---|---|
| GPT-3 | 175B | 300B | 43.9% |
| Gopher | 280B | 300B | 60.0% |
| Chinchilla | 70B | 1.4T | 67.5% |
| Megatron-Turing NLG | 530B | 270B | 46.8% |

Chinchilla's gains are not marginal — a 4× smaller model substantially outperforms the 280B Gopher trained on the same hardware budget. This demonstrated conclusively that the field had been operating in an undertrained regime.

**IsoFLOP curves** in the Chinchilla paper show smooth parabolas (in log-log space) at each compute level. The minimum shifts steadily toward larger D relative to N as compute increases — the opposite direction from what Kaplan implied. Crucially, the curves are well-behaved enough that you can fit them and confidently extrapolate the optimal N*/D* for a compute budget you have not yet run.

**Power-law extrapolation accuracy**: Both papers show that models trained on small compute budgets can predict, within a few percent, the loss of models trained on 100–1000× larger compute. This is the key practical value: small proxy runs inform large production runs.

## Reproducibility Notes

Neither paper releases training code or model weights in the standard open-source sense. Both are fundamentally **empirical theory papers**: the core claim is that power-law fits hold reliably, and the deliverable is the fitted exponents and the isoFLOP methodology.

Key caveats:

- **Data quality is assumed constant.** Both analyses hold data quality fixed and vary only quantity. In practice, the quality-quantity tradeoff is significant — lower-quality data at 20 tokens/param may underperform a higher-quality dataset at 10 tokens/param. The FineWeb and DCLM ablations (2024) have begun quantifying this.
- **LR schedule sensitivity.** Chinchilla's entire revision of Kaplan rests on a single methodological difference: per-model LR tuning. The community should treat any isoFLOP comparison that does not report per-model hyperparameter optimization with skepticism.
- **Architecture.** Both papers use decoder-only transformers of varying depths and widths. Scaling laws for mixture-of-experts, state-space models (Mamba), or other architectures are not directly transferable.
- **Context length.** The fits assume a fixed and relatively short context window. Scaling laws for long-context models (128K+) remain an active research area.
- **Single epoch assumption.** Kaplan explicitly assumes data is never repeated. This assumption breaks for frontier models, where high-quality web text is a finite resource. Muennighoff 2023 shows that repeating data up to 4 epochs degrades performance only modestly, which has practical implications for labs operating at the data frontier.

## Commentary

Chinchilla is now the operating assumption across every major lab. When a new model is announced with "Chinchilla-optimal" training, it means the token count is approximately 20× the parameter count. This is the baseline expectation; deviating from it requires justification.

However, **"Chinchilla-optimal" is not the same as "deployment-optimal."** The Chinchilla frontier is defined purely by training compute — it minimizes loss for a given FLOPs budget, ignoring inference. But for a production model serving billions of requests, the lifetime cost is dominated by inference, not training. A smaller, over-trained model (more tokens than Chinchilla-optimal) has lower per-token inference cost. This is the explicit argument Touvron et al. made for Llama 1 and 2: accept higher training compute in exchange for a model small enough to run cheaply at scale. Mistral 7B, Phi, and Gemma follow the same logic.

The result is a bifurcation: frontier capability labs (targeting benchmark leadership) train Chinchilla-optimal or near-Chinchilla-optimal models. Efficiency-first labs (targeting deployment economics) deliberately over-train smaller models. Neither camp is wrong — they are optimizing different objective functions.

There is also the **emergence problem**. Scaling laws predict smooth cross-entropy loss. But capability evaluations (MMLU, GSM8K, HumanEval) show discontinuous jumps at certain scales — behaviors that are absent at N and present at 10N, with no signal in the loss curve. This means scaling laws are necessary but not sufficient for capability prediction. The community has not yet reconciled these two empirical facts.

Finally, the scaling law framework assumes independent, identically distributed data. As datasets become more curated, deduplicated, and synthetic-augmented, the i.i.d. assumption weakens. Whether the exponents remain stable as data composition shifts significantly is an open question.

## References

- [1] Kaplan, J., McCandlish, S., Henighan, T., et al. (2020). "Scaling Laws for Neural Language Models." arXiv:2001.08361. https://arxiv.org/abs/2001.08361
- [2] Hoffmann, J., Borgeaud, S., Mensch, A., et al. (2022). "Training Compute-Optimal Large Language Models." arXiv:2203.15556. https://arxiv.org/abs/2203.15556
- [3] Muennighoff, N., Rush, A., Barak, B., et al. (2023). "Scaling Data-Constrained Language Models." arXiv:2305.13230. https://arxiv.org/abs/2305.13230
