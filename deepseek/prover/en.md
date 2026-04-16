# DeepSeek-Prover — Formal Theorem Proving via RL and MCTS

- **Authors / Org**: DeepSeek-AI (Xin et al.)
- **Published**: 2024-05 (V1, arXiv:2405.14333); 2024-08 (V1.5, arXiv:2408.08152)
- **Links**: [V1 paper (arXiv:2405.14333)](https://arxiv.org/abs/2405.14333) · [V1.5 paper (arXiv:2408.08152)](https://arxiv.org/abs/2408.08152) · [code (V1.5)](https://github.com/deepseek-ai/DeepSeek-Prover-V1.5)

## TL;DR

DeepSeek-Prover applies LLM fine-tuning and reinforcement learning to **formal theorem proving in Lean 4**. The key infrastructure insight: the Lean 4 compiler is a **perfect binary oracle** — it tells you definitively whether a proof is correct, with no human labeling and no reward model. This makes formal theorem proving the cleanest possible RL environment for mathematical reasoning: rewards are ground truth, step-level verification is free (just type-check the tactic), and the problem space (Mathlib4) has tens of thousands of theorems of increasing difficulty. V1 generates synthetic training data by translating informal math to Lean 4 and filtering by compilation. V1.5 adds reinforcement learning with **RMaxTS** (Reward-Max Tree Search), a Monte Carlo Tree Search variant that concentrates exploration on the most promising proof branches. DeepSeek-Prover-V1.5 achieves state-of-the-art results on miniF2F-test among open models, and — more importantly for the repo — concretely instantiates the inference-time scaling and PRM ideas that R1/GRPO describe abstractly.

## Context & Motivation

### Why formal theorem proving matters for AI infra

Formal theorem proving in systems like Lean 4, Coq, or Isabelle requires producing machine-verifiable proofs: every logical step must be expressed as a **tactic** that the type-checker accepts. This is qualitatively different from informal math (where a human decides if reasoning is convincing) and from code generation (where tests check output, not proof structure).

For LLMs, this creates a uniquely clean RL problem:

1. **Binary, exact reward**: the Lean 4 compiler either accepts the proof or rejects it. No human preference, no learned RM, no distribution shift in the reward signal.
2. **Step-level verification for free**: each tactic in a Lean 4 proof can be individually type-checked. A partial proof that is valid up to step k and invalid at step k+1 is immediately identifiable — the compiler tells you precisely where the failure is. This is a free PRM.
3. **Infinite problem supply with difficulty gradient**: Mathlib4 (the main Lean 4 math library) contains over 150,000 theorems ranging from undergraduate to research level. Curriculum learning over this library is natural.
4. **No data contamination**: formal proofs in Lean 4 are not present in typical LLM pretraining data at scale; performance gains are less likely to be memorization artifacts.

The connection to the broader reasoning agenda: if GRPO + rule-based rewards can teach a model to reason on informal math (R1), can the same approach — with a compiler instead of a symbolic checker — teach it to produce **verifiably correct** mathematical reasoning?

### The gap V1 addresses

Pre-2024, frontier LLMs could generate informal mathematical arguments but could not reliably produce valid Lean 4 proofs for non-trivial theorems. The bottleneck was training data: Lean 4 proofs are rare in web crawls, and human-written Lean proofs require expert mathematicians. V1's hypothesis: **auto-formalization** (translating informal math → Lean 4) with compiler-based filtering can generate sufficient training data from existing informal math corpora.

## Core Method

### V1: auto-formalization and data synthesis

The V1 pipeline:

1. **Informal source**: collect high-school and olympiad math problems with known solutions from web sources and benchmarks (MATH, AMC, AIME, etc.).

2. **Auto-formalization**: prompt a base LLM to translate each informal problem + solution into a Lean 4 theorem statement and proof sketch. The translation is imperfect and often wrong.

3. **Compiler filtering**: run the Lean 4 compiler on each generated proof. Accept only those that fully compile (type-check). Rejection rate is high (~95%), but at scale even 5% of millions of generated proofs yields hundreds of thousands of valid training examples.

4. **SFT on valid proofs**: fine-tune the model on the accepted (problem, proof) pairs using standard next-token prediction. The compiler-filtering step ensures the training data is noise-free — every example in the SFT set is a verified correct proof.

5. **Iterative refinement**: re-run auto-formalization with the fine-tuned model (which is better at producing valid Lean 4), filter again, and repeat. Each iteration generates harder valid proofs.

The key insight: **the compiler replaces human annotation at every stage**. No mathematician is needed to validate training examples; the type-checker does it automatically and at scale.

### V1.5: RL with Monte Carlo Tree Search

V1.5 adds a reinforcement learning stage on top of the V1 SFT checkpoint, using proof search as the RL environment.

#### Proof search as an MDP

A formal proof is a sequence of **tactics** `t₁, t₂, ..., tₖ` that transform the initial goal into a closed proof. Each tactic takes the current **proof state** (the remaining goals and hypotheses) as input and produces a new proof state. This is a Markov Decision Process:

- **State**: current proof state (a structured Lean 4 term representing remaining goals)
- **Action**: next tactic (a token sequence generated by the LLM)
- **Reward**: +1 if the proof is complete (all goals closed), 0 otherwise
- **Terminal**: proof complete, or max depth / timeout reached

The LLM is the policy: given the proof state as context, generate the next tactic. The Lean 4 type-checker is the environment: apply the tactic, check it, and return the new proof state or a failure signal.

#### RMaxTS

Standard MCTS uses the **average** reward of rollouts through a node to estimate its value. For theorem proving, this underestimates the value of nodes on rare-but-valid proof paths: a node that leads to one successful proof in 10 rollouts is vastly better than a node that leads to 10 failed proofs, but their average rewards (0.1 vs 0) differ only slightly in noisy settings.

RMaxTS modifies the value estimate to use the **maximum** reward of any rollout:

```
V_RMax(node) = max(rewards of all rollouts through node)
```

The UCT selection formula becomes:

```
score(node) = V_RMax(node) + c · √(ln N_parent / N_node)
```

where `c` controls exploration. The max-value backup has two consequences:

1. **A single successful rollout marks a subtree as high-value**, directing future exploration toward it regardless of how many failed rollouts preceded it.
2. **Rare valid proof paths are not drowned out** by the majority of invalid ones — the max survives even when most rollouts fail.

This is well-suited to theorem proving because valid proofs are sparse (most random tactic sequences fail) but the search should commit heavily once a partial proof path succeeds.

#### Combining whole-proof and step-by-step generation

V1.5 uses two complementary generation modes:

- **Whole-proof generation**: generate the entire proof as a single token sequence, then compile. Fast; one LLM call per attempt; fails silently on long proofs where an error deep in the chain invalidates everything.

- **Step-by-step with MCTS**: generate one tactic at a time, check each step with the compiler, and use the proof state after each step as the next context. MCTS coordinates which partial proofs to extend. Slower per theorem but explores the proof space more systematically.

In practice, V1.5 uses whole-proof generation for simpler theorems (faster, lower overhead) and MCTS for theorems where whole-proof generation fails repeatedly.

#### RL training loop

The training loop alternates between search and learning:

1. **Rollout phase**: run RMaxTS on a batch of theorems, generating proof attempt trajectories.
2. **Reward assignment**: the Lean 4 compiler assigns binary rewards to complete trajectories.
3. **Policy update**: update the LLM using GRPO on the successful trajectories (positive reward) relative to the group of attempts. Failed attempts provide the negative baseline.
4. **Repeat**: the improved policy generates better proof attempts in the next rollout phase.

The Lean 4 compiler is the only reward source — no human labeling, no learned RM. The compiler serves simultaneously as the reward model and the step-level verifier (PRM).

### The Lean 4 infrastructure requirement

Running this training loop requires the Lean 4 toolchain as a co-process alongside the LLM:

- **Lean 4 + Mathlib4**: must be installed and loadable; Mathlib4 takes ~30 minutes to compile from scratch but is cached.
- **LeanDojo** (Yang et al., 2023): a Python interface that wraps Lean 4, extracts proof states programmatically, and applies tactics interactively. V1.5 depends on LeanDojo for its step-by-step MCTS environment.
- **Parallelism**: each theorem search is independent; running 256 or 1024 theorem searches in parallel requires 256–1024 concurrent Lean 4 processes, each with a loaded Mathlib4 environment. Memory per process is ~2–4 GB. A typical training server with 8 H100s runs LLM inference on the GPUs and Lean 4 verification on the CPUs.
- **Latency**: the Lean 4 type-checker for a single tactic is fast (1–50 ms per step), but loading the full Mathlib4 context (required for each new theorem) adds ~2–5 seconds. Caching the Lean environment per-theorem across MCTS rollouts is essential for throughput.

## Engineering Tradeoffs

| Decision | Gained | Gave up |
|---|---|---|
| Lean 4 compiler as reward | Perfect binary reward; no RM training; no reward hacking | Requires Lean 4 toolchain infrastructure; limits domain to formally expressible math; compilation latency in the RL loop |
| Auto-formalization + filtering (V1) | Scalable training data from informal math; no human annotation | ~95% rejection rate is expensive at scale; translation quality limits which informal proofs can be formalized correctly |
| RMaxTS (max reward backup) | Sparse valid proofs survive majority-invalid rollouts; faster convergence on hard theorems | Less stable than mean backup for easy problems where many paths succeed; hyperparameter c is sensitive |
| Whole-proof vs step-by-step | Whole-proof is fast; step-by-step enables MCTS search | Must choose mode per theorem or run both; step-by-step requires LeanDojo integration and per-step compiler calls |
| GRPO for policy update | No critic model; same infrastructure as R1 training; group relative advantage from proof outcomes | Proof reward is binary (0/1), not graded; GRPO advantage collapses when all rollouts fail or all succeed — exploration requires non-trivial pass@k |
| Mathlib4 as training domain | Difficulty gradient from undergraduate to research; theorems are diverse | Lean 4 + Mathlib4 is a specific formal system; does not generalize to Coq/Isabelle/HOL without retraining |

## Experiments & Results

**miniF2F benchmark** (high-school olympiad problems formalized in Lean 4/Isabelle; 244 test theorems):
- DeepSeek-Prover-V1 (whole-proof): 46.3% pass@32 (each theorem attempted 32 times)
- DeepSeek-Prover-V1.5 (whole-proof): 48.2% pass@32
- DeepSeek-Prover-V1.5 (RMaxTS, 600 rollouts): **63.5% pass@32** — a large jump from adding tree search

For comparison:
- GPT-4 (informal reasoning only, no Lean): ~50% on informal miniF2F, but cannot produce valid Lean proofs
- InternLM2-Math-Plus 7B (prior open SOTA): ~43% on miniF2F in Lean 4

**FIMO benchmark** (competition math at IMO difficulty):
- V1.5 solves 5/148 problems — a modest absolute number, but non-zero on IMO-level problems is notable for an open 7B-class model.

**Scaling with search budget**: pass rate on miniF2F increases log-linearly with the number of RMaxTS rollouts from 64 to 600, confirming that inference-time compute scaling applies to formal proof search.

## Reproducibility Notes

- V1.5 code, weights, and the formalized theorem datasets are released at [github.com/deepseek-ai/DeepSeek-Prover-V1.5](https://github.com/deepseek-ai/DeepSeek-Prover-V1.5).
- **LeanDojo** must be installed separately: `pip install lean-dojo`. Requires Lean 4 and Mathlib4; the first Mathlib4 build takes 20–40 minutes.
- **Reproducing the RL loop**: the training code is released. Key requirement: a machine with sufficient CPUs to run many parallel Lean 4 processes (recommend ≥64 CPU cores for reasonable throughput). Each RMaxTS rollout occupies one CPU core for the duration of the search.
- **Evaluation**: the released models can be evaluated on miniF2F with the provided scripts; the evaluation itself requires Lean 4 + Mathlib4 and takes several hours on a single machine for the full 244-problem test set.
- The **auto-formalization pipeline** (V1) is also released; it requires access to an LLM capable of informal→Lean translation, which can be the DeepSeek-Prover model itself in a bootstrapping setup.

## Commentary

DeepSeek-Prover is the clearest concrete example in the repo of the inference-time scaling thesis applied to a non-trivial domain. The [inference-time scaling entry](../../foundational/inference-time-scaling/) describes parallel search with a verifier as the general pattern; Prover is what this looks like when the verifier is a type-checker and the search space is the set of valid Lean 4 tactic sequences.

Three connections to other entries that are worth making explicit:

**Compiler as perfect PRM.** The [PRM entry](../../foundational/process-reward-models/) explains that PRMs trained on human labels are expensive and domain-specific, and that MC rollout labeling is an approximation. Lean 4 sidesteps both problems: every tactic application is a free, exact, step-level verification. The compiler is the PRM that the PRM community is trying to approximate for informal math.

**GRPO without a reward model.** The training loop uses GRPO with binary compiler feedback — no RM, no human preference data. This is the formal-math analogue of R1-Zero, and it works for the same reason: the reward is ground truth and unambiguous. The difference is that in R1-Zero, the "verifier" is a symbolic answer checker (compare numeric outputs); in Prover, the verifier is a full type-checker that validates logical structure.

**RMaxTS and the sparse reward problem.** The [inference-time scaling entry](../../foundational/inference-time-scaling/) discusses MCTS as a search strategy and mentions that PRMs provide instant value estimates. RMaxTS is the right MCTS variant when rewards are sparse and binary: if even one rollout succeeds, that node is worth exploring further, regardless of how many fail. Mean-backup MCTS would undervalue a node with 1-in-100 success rate; RMaxTS correctly identifies it as high-value. This is a design principle that applies whenever the reward signal is sparse and exact.

The limitations are real: formal proof requires expressing problems in Lean 4, which is a high barrier for most math. The model has no way to reason about problems that haven't been formalized, and formalization itself is a bottleneck. The long-term question is whether the formal proof capabilities transfer to informal reasoning — whether a model trained to produce valid Lean 4 proofs is better at informal multi-step math than one trained on informal math alone. Early evidence suggests yes (DeepSeek-Prover models perform well on informal MATH benchmarks too), but this is an active research question.

## References

- [1] Xin et al. _DeepSeek-Prover: Advancing Theorem Proving in LLMs through Large-Scale Synthetic Data._ arXiv:2405.14333, 2024.
- [2] Xin et al. _DeepSeek-Prover-V1.5: Harnessing Proof Assistant Feedback for Reinforcement Learning and Monte-Carlo Tree Search._ arXiv:2408.08152, 2024.
- [3] Yang et al. _LeanDojo: Theorem Proving with Retrieval-Augmented Language Models._ NeurIPS '23 / arXiv:2306.15626, 2023.
- [4] Zheng et al. _MiniF2F: a cross-system benchmark for formal Olympiad-level mathematics._ ICLR '22 / arXiv:2109.00110, 2021.
- [5] DeepSeek-AI. _DeepSeek-R1._ arXiv:2501.12948, 2025. (GRPO; rule-based rewards for reasoning)
- [6] Lightman et al. _Let's Verify Step by Step._ arXiv:2305.20050, 2023. (PRM; compiler as perfect step verifier)
- [7] Snell et al. _Scaling LLM Test-Time Compute Optimally._ arXiv:2408.03314, 2024. (Inference-time scaling; MCTS with value functions)
- [8] The mathlib4 Community. _The Lean 4 Mathematical Library._ github.com/leanprover-community/mathlib4, 2024.
