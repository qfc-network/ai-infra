# Constitutional AI — Harmlessness from AI Feedback

- **Authors / Org**: Yuntao Bai, Saurav Kadavath, Sandipan Kundu et al. (Anthropic)
- **Published**: 2022-12
- **Links**: [paper (arXiv:2212.08073)](https://arxiv.org/abs/2212.08073) · [blog](https://www.anthropic.com/news/constitutional-ai-harmlessness-from-ai-feedback)

## TL;DR

Constitutional AI (CAI) replaces human feedback for harmlessness with AI-generated feedback guided by a written "constitution" — a set of natural-language principles. The method has two stages: (1) supervised learning from AI-revised responses (SL-CAI), in which the model critiques and rewrites its own outputs against the constitution before being fine-tuned on those rewrites; and (2) reinforcement learning from AI feedback (RLAIF), in which a separate feedback model scores response pairs against constitutional principles rather than relying on human harmlessness raters. The result is a training signal that is cheap to generate, explicit enough to audit, and easy to update by editing the constitution rather than re-running a labeling operation. CAI is the foundation of Claude's alignment approach and introduced two patterns — the critique-revision loop and the RLAIF reward pipeline — that became standard across the synthetic data and alignment literature almost immediately.

## Context & Motivation

RLHF as introduced by InstructGPT requires human preference labels for both the helpfulness dimension and the harmlessness dimension. The helpfulness side of this is tractable: labelers compare whether response A or B better answers the prompt, which is cognitively straightforward. Harmlessness is a different problem:

- Labelers must read and evaluate potentially graphic, toxic, or illegal content to assign harmlessness scores. This causes psychological harm at scale and drives turnover.
- Harmlessness criteria evolve — a new attack surface or a new regulatory concern means re-labeling or layering another annotation campaign on top.
- Human preference signals are implicit and opaque: you know which response a rater preferred, but not *why*, making it hard to audit whether the model has learned the right criterion or a correlated proxy.
- Scaling RLHF to a larger model, a new language, or a new risk domain requires proportionally more human labor for the harmlessness side, even if helpfulness labeling can be parallelized easily.

At the same time, Anthropic's HHH (Helpful, Harmless, Honest) framework had articulated what harmlessness *should* mean conceptually — and that conceptual content was articulable in natural language. The observation that motivated CAI: if harmlessness criteria can be written down as principles, and if a capable language model can understand and apply those principles, then the model itself can generate the harmlessness supervision signal. The human bottleneck is lifted; the criteria become explicit; and iterating on alignment criteria becomes an editing task, not a labeling campaign.

CAI's three stated goals:
1. Reduce human labeling cost for harmlessness without sacrificing quality.
2. Make the alignment criteria explicit and auditable (the constitution as a governance artifact).
3. Improve transparency of the training signal so behavior changes are traceable to specific principles.

## Core Method

CAI operates in two sequential stages, each producing a model that feeds into the next.

### Stage 1 — Supervised Learning from AI Feedback (SL-CAI)

**Input**: a "helpful-only" SL model — fine-tuned to follow instructions but without harmlessness training, so it will comply with most harmful requests. This is deliberate: you need a model that *can* produce harmful outputs to generate the revision training set.

**Critique-revision loop**:
1. Sample responses from the helpful-only model for a set of "red-team" prompts designed to elicit potentially harmful outputs.
2. Present each response to the same model with a critique prompt drawn from the constitution — for example: *"Identify specific ways in which the assistant's last response is harmful, unethical, racist, sexist, toxic, dangerous, or illegal."*
3. Ask the model to revise the response to eliminate the identified harms, using a separate revision prompt.
4. Repeat critique and revision N times per sample (the paper uses 1–4 rounds). Each round produces a less harmful response, and diversity in revision paths produces a richer training set.
5. Fine-tune on the final revised responses → the SL-CAI model.

The critique-revision loop is the key structural insight. A model strong enough to identify specific harms in its own output is strong enough to revise them away — the supervision signal is self-generated. The role of the human-written constitution is to *steer* the critique, not to provide labels directly.

**The constitution** contains approximately 16 principles in the paper's published version, drawn from: the UN Declaration of Human Rights, Anthropic's internal guidelines, and principles targeted at specific failure modes (e.g., choosing responses that are "least likely to contain false information," or "least likely to be misinterpreted as supporting a political party"). Principles are short — typically one to two sentences — and written in natural language. The full list is published in the paper's appendix, which is itself a form of alignment transparency absent from human-labeled RLHF.

### Stage 2 — RL from AI Feedback (RLAIF)

**Input**: the SL-CAI model from Stage 1 plus a set of red-team prompts.

**Generating comparison pairs**: sample pairs of responses to each prompt from the SL-CAI model.

**AI preference labeling**: present each pair to a feedback model (either a separately pretrained LM or the SL-CAI model itself) using a multiple-choice prompt that invokes one constitutional principle: *"Which of the following responses is less harmful? (A) [response A] or (B) [response B]?"* The feedback model's output log-likelihoods over the tokens "A" and "B" give a soft preference probability:

$$p(A \text{ preferred}) = \text{softmax}\bigl(\log P(\text{"A"} \mid \text{prompt, principle}),\ \log P(\text{"B"} \mid \text{prompt, principle})\bigr)$$

This soft label — a continuous probability rather than a hard 0/1 binary — is more informative per sample than a binary human label and directly plugs into the Bradley-Terry pairwise-comparison loss used to train the harmlessness Preference Model (PM):

$$\mathcal{L}_{\text{PM}} = -\mathbb{E}_{(A,B)}\bigl[p(A) \cdot \log \hat{r}(A) + p(B) \cdot \log \hat{r}(B)\bigr]$$

**Reward model training**: the harmlessness PM is trained on this AI-labeled preference data, learning to score (prompt, response) → harmlessness scalar.

**PPO**: standard Proximal Policy Optimization with a combined reward — the sum of the harmlessness PM score (from RLAIF) and a helpfulness PM score (from standard human-labeled RLHF, unchanged). The weighting between the two is a tunable hyperparameter. The KL penalty to the SFT policy is retained, as in InstructGPT, to prevent degenerate reward hacking.

The resulting model is called the RLHF-CAI or RL-CAI model. The critical design property: the harmlessness reward is fully determined by the constitution plus the feedback model's reading of it. There is no human harmlessness label anywhere in the RL loop.

### Constitution design considerations

The principles are not chosen arbitrarily. A single principle would overfit the PM to one failure mode (e.g., toxicity) while leaving others (e.g., deception, privacy violations) unaddressed. Diversity across principle types — physical safety, psychological harm, legal issues, epistemic harms, civil rights — is essential for a PM that generalizes. Principles can also specify positive criteria ("prefer the response that is most helpful to the human") alongside negative ones, which is how CAI avoids the evasion problem: the constitution itself instructs the model to be helpful when the request is not clearly harmful.

## Engineering Tradeoffs

| Decision | Gained | Gave up |
|---|---|---|
| AI feedback for harmlessness instead of human labels | Scales cheaply to new domains and languages; no harmful content exposure for labelers; easy to update by editing the constitution | Feedback model inherits base model biases; may miss novel harmful patterns outside the model's world knowledge |
| Explicit constitution vs implicit human preferences | Auditable training signal; behavior changes traceable to specific principles; straightforward to iterate | Constitution must be carefully designed and maintained; gaps or ambiguities in principles become blind spots in the PM |
| Soft AI preference labels vs hard binary human labels | More signal per sample (probability mass, not 0/1); faster PM convergence; calibrated uncertainty | Quality of soft labels depends entirely on feedback model calibration; a miscalibrated feedback model systematically biases the PM |
| Multi-turn critique-revision loop (SL-CAI) | Gradual harmlessness improvement per sample; diverse revision paths create a richer fine-tuning distribution | Compute cost multiplied by N revision rounds per sample; longer chains risk semantic drift away from the original helpful intent |
| Separate harmlessness and helpfulness PMs combined via reward weighting | Each PM optimized on its own signal; balance tunable at RL time without re-training either PM | Two reward models increase memory and serving cost in the PPO loop; Goodhart's law operates independently on each |

## Experiments & Results

Anthropic evaluates the RL-CAI model against a standard RLHF baseline trained on human harmlessness labels across two axes — harmlessness and helpfulness — using both human evaluators and automated Elo-style preference judgments.

**Harmlessness**: the RL-CAI model matches or exceeds the human-RLHF model on harmlessness ratings. Crowdworkers rate RL-CAI responses as less harmful in direct comparison.

**Helpfulness**: this is the more interesting result. Human-labeled harmlessness training tends to produce evasive models — ones that refuse borderline or benign-seeming requests to play it safe. RL-CAI is *less* evasive: because the constitution explicitly instructs the feedback model to prefer helpful responses when the request is not clearly harmful, the harmlessness PM does not penalize helpfulness unnecessarily. Human evaluators prefer RL-CAI responses approximately 72% of the time in head-to-head comparisons on combined harmlessness + helpfulness.

**Evasion reduction**: the qualitative improvement on evasion is arguably as important as the quantitative harmlessness improvement. A model that refuses "what household chemicals should I not combine?" is not safer than one that answers helpfully — it's less useful and trains users to rephrase prompts to circumvent the refusal. CAI's constitution can directly specify that such requests should be answered, making the training signal self-consistent in a way human labeling is not.

**SL-CAI alone**: fine-tuning on critique-revised responses (without the RL stage) already produces meaningful harmlessness improvement over the helpful-only baseline, suggesting the critique-revision loop is valuable even without a reward model.

**Scaling**: larger feedback models produce better harmlessness PMs. The feedback model quality is the primary quality bottleneck, not the constitution text itself, suggesting that CAI benefits from scaling at the feedback stage in a similar way that standard RLHF benefits from scaling the reward model.

## Reproducibility Notes

**What is published**: the full constitution (all ~16 principles) is in the paper's appendix. The critique and revision prompt templates are published verbatim in the supplementary material. The RLAIF preference labeling prompt format is specified in detail.

**What is not published**: model weights, the red-team prompt dataset, and the human helpfulness preference data. The feedback model architecture and size are described but the checkpoint is not released.

**Base model requirements**: meaningful critique-revision quality requires a base model capable of following multi-turn instructions and producing coherent critiques. Empirically, models below approximately 7B parameters produce critiques that are too shallow or generic to drive useful revisions. Larger base models (13B+) show substantially better critique specificity.

**Open-source approximations**: the critique-revision pattern has been widely replicated — Stanford Alpaca and its derivatives explored CAI-style self-critique prompting, and the Llama-2-Chat paper's "red-teaming with RLHF" work draws directly on the CAI framing. Full RLAIF (Stage 2) requires a trained harmlessness reward model as the starting point, which is the harder-to-replicate piece. The Google RLAIF paper (Lee et al. 2023) provides a detailed open comparison between human-labeled RLHF and RLAIF that serves as an independent reproducibility reference.

**Key hyperparameters**: number of critique-revision rounds (1–4), number of constitution principles sampled per batch, KL coefficient in PPO, and the weighting ratio between the harmlessness and helpfulness PM rewards. The paper reports sensitivity to the KL coefficient and the reward weighting; these should be treated as tunable for any new base model or domain.

## Commentary

CAI's lasting contribution is best understood by separating its two components.

**The critique-revision loop** (Stage 1) introduced a pattern that immediately generalized beyond alignment. The insight — that a model capable of identifying a flaw in its own output is capable of correcting it, and that self-correction iterated N times produces a training set better than the first draft — is now the foundation of virtually every synthetic data pipeline. STaR (self-taught reasoner), Self-Instruct, OpenHermes, Magpie, and dozens of post-2023 synthetic fine-tuning datasets use variants of the same loop. The alignment framing was the original application but not the limiting one.

**RLAIF** (Stage 2) turned the reward signal itself into something a language model generates. The "RLAIF" label was popularized by the Google follow-up paper (Lee et al. 2023), but the mechanism — use a language model to produce preference labels via a structured prompt, train a reward model on those labels, do RL — is CAI. The practical consequence for infrastructure is significant: RLAIF removes the human bottleneck from the reward modeling loop, making it possible to generate harmlessness preference labels at arbitrary scale for new domains, languages, and risk categories without a new labeling operation.

**The constitution as a governance artifact** is the third contribution, less often cited in the ML literature but important for anyone building production alignment systems. A constitution is a document — it can be versioned, diffed, reviewed by non-engineers, and held accountable. When a model behaves unexpectedly, you can ask whether the behavior is consistent with the constitution, and if not, add a principle. This auditability is qualitatively different from implicit human preference data, where the training signal is opaque by construction. For anyone building alignment infrastructure in 2026: the question of "what principle does your model's behavior derive from" is increasingly important for safety reviews, regulatory compliance, and internal accountability. CAI made that question answerable.

**The evasion-helpfulness tension** that CAI addresses is not fully solved by CAI, but CAI was the first system to explicitly encode the resolution in the training signal rather than relying on post-hoc RLHF to balance it. The pattern of having the constitution specify positive criteria ("be helpful when the request is not harmful") alongside negative ones ("don't assist with illegal activity") is now standard in instruction-tuning rubrics at every major lab.

For engineering: the key lesson from CAI is that **the reward signal is the load-bearing component of any RLHF system**, and that making it cheap, scalable, and auditable is an infrastructure problem at least as important as the RL algorithm itself. CAI solved the infrastructure problem for harmlessness. The field has spent the years since applying the same insight to reasoning (process reward models), factuality (grounding reward models), and instruction following (LLM-as-judge). The algorithmic framing differs across these applications, but the structural pattern — replace expensive opaque human labels with cheap explicit model-generated labels — traces directly back to CAI.

## References

- [1] Bai et al. _Constitutional AI: Harmlessness from AI Feedback._ arXiv:2212.08073, 2022.
- [2] Bai et al. _Training a Helpful and Harmless Assistant with Reinforcement Learning from Human Feedback._ arXiv:2204.05862, 2022.
- [3] Lee et al. _RLAIF: Scaling Reinforcement Learning from Human Feedback with AI Feedback._ arXiv:2309.00267, 2023.
- [4] Ouyang et al. _Training Language Models to Follow Instructions with Human Feedback (InstructGPT)._ arXiv:2203.02155, 2022.
- [5] Ganguli et al. _Red Teaming Language Models to Reduce Harms._ arXiv:2209.07858, 2022. (Anthropic red-teaming context)
