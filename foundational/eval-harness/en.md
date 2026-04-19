# LLM Evaluation Harness

- **Primary project**: EleutherAI `lm-evaluation-harness`
- **Maintained by**: EleutherAI; major contributors include HuggingFace, Allen AI
- **Links**: [GitHub](https://github.com/EleutherAI/lm-evaluation-harness) · [Open LLM Leaderboard](https://huggingface.co/spaces/HuggingFaceH4/open_llm_leaderboard) · [Paper: arXiv:2110.08207](https://arxiv.org/abs/2110.08207)

## TL;DR

`lm-evaluation-harness` is the canonical open-source framework for standardized LLM evaluation. It implements 60+ benchmark tasks — MMLU, HellaSwag, ARC, WinoGrande, GSM8K, HumanEval, TruthfulQA, and many others — under a unified interface. It powers the HuggingFace Open LLM Leaderboard, making it the de facto standard for comparing open-weight models. The framework abstracts two distinct evaluation modes — log-likelihood scoring and generation scoring — with task-level YAML configuration. Understanding its computational cost model is essential for anyone running serious evaluations: a full 5-shot MMLU evaluation on a 70B model consumes several hundred GPU-hours unless batched intelligently with a fast inference backend.

## Context & Motivation

Before `lm-evaluation-harness`, benchmark comparisons were inconsistently reproduced. Every lab ran its own evaluation scripts with different tokenization, prompt formatting, and few-shot selection strategies. A model could appear to score differently on "the same" benchmark depending on which implementation was used. The harness solved this by encoding each benchmark's prompt template, gold-answer extraction logic, and metric computation in a versioned, shared codebase. The Open LLM Leaderboard then applied this harness uniformly across submitted checkpoints, making cross-model comparison meaningful for the first time at scale in the open research community.

## Two Scoring Modes

The central design decision in `lm-evaluation-harness` is the split between log-likelihood scoring and generation scoring. These are not interchangeable — the choice is dictated by the task's answer structure.

### Log-Likelihood Scoring

Used for multiple-choice tasks: MMLU, HellaSwag, WinoGrande, ARC-Easy, ARC-Challenge, PIQA.

The model is never asked to generate text. Instead, for each answer choice the framework computes the log-probability that the model assigns to that choice given the context:

```
score(choice_i) = log P(choice_i | context)
               = sum_{t} log P(token_t | context, choice_i[:<t])
```

The predicted answer is `argmax_i score(choice_i)`. For a 4-choice question, this requires exactly 4 forward passes per example (one per choice). No sampling, no temperature, no beam search — just a single forward pass per candidate with teacher forcing.

Memory consequence: the forward pass for each choice processes `len(context) + len(choice)` tokens. For MMLU with 5-shot, context length is 500–800 tokens; choice length is 1–10 tokens. The bulk of compute is in the shared context prefix, which makes prefix caching highly effective here: compute the context KV cache once, then reuse it for all 4 choices.

Latency: at 100ms per forward pass on an H100 with a 70B model, 56,000 forward passes (MMLU 5-shot: 14,079 questions × 4 choices) takes approximately 93 minutes without batching. With efficient batching (the 4 choices of a single question share context and can be padded into a single batch), wall time drops to 20–30 minutes.

### Generation Scoring

Used for open-ended tasks: GSM8K, HumanEval, TruthfulQA-gen, LAMBADA.

The model generates a completion token-by-token using its standard decoding logic. The framework then applies an answer-extraction function — exact string match, regex match, or code execution — to the generated output.

Memory consequence: generation is autoregressive. Each output token requires one forward pass over `prompt_len + tokens_generated_so_far` tokens, with the KV cache growing incrementally. At 256 output tokens, a single GSM8K example requires 256 sequential forward passes after the initial prefill. Total generated tokens for all of GSM8K: 8,500 examples × 256 tokens = 2.176M tokens. At 5,000 tokens/second sustained throughput on H100, this is approximately 7 minutes. This is much faster per task than log-likelihood MMLU because the bottleneck is throughput, not forward-pass count.

## Task Definition Schema

Each task is defined by a YAML configuration file plus an optional Python class for complex preprocessing. The core fields:

```yaml
task: mmlu_abstract_algebra
dataset_path: hf://datasets/cais/mmlu
dataset_name: abstract_algebra
doc_to_text: "{{question}}\nA. {{choices[0]}}\nB. {{choices[1]}}\nC. {{choices[2]}}\nD. {{choices[3]}}\nAnswer:"
doc_to_target: "{{answer}}"
metric_list:
  - metric: acc
    aggregation: mean
num_fewshot: 5
fewshot_split: validation
output_type: multiple_choice
```

`doc_to_text` is a Jinja2 template rendering the prompt; `doc_to_target` extracts the gold label; `output_type: multiple_choice` routes the task to log-likelihood scoring. For generation tasks, `output_type: generate_until` is used with a `until` stop sequence list.

Registering a new task is purely a YAML authoring exercise for most benchmarks. The Python class hook is available for tasks requiring custom preprocessing (e.g., HumanEval's code execution environment setup). This schema design means task definitions are auditable — a diff to a YAML file is a diff to how a benchmark is evaluated.

## Few-Shot Evaluation

The `--num_fewshot N` flag prepends N in-context examples before each test example. Examples are drawn from the task's designated few-shot source (typically the training or validation split, separate from test). The few-shot examples follow the same `doc_to_text` template, concatenated with the few-shot delimiter (newline by default).

Prompt structure at 5-shot MMLU:

```
[Example 1 question + choices + "Answer: A"]
[Example 2 question + choices + "Answer: C"]
...
[Example 5 question + choices + "Answer: B"]
[Test question + choices + "Answer:"]
```

At 5-shot MMLU, each prompt is approximately 500–800 tokens depending on subject area (medicine questions run long; abstract algebra questions are shorter). This must fit within the model's context window. For models with 4k context windows, 5-shot is tight; for 8k+ context, no issue.

Context window violation: if the few-shot prompt exceeds the model's maximum context length, the harness truncates from the left by default. This can silently degrade accuracy for short-context models — the test question is preserved, but some few-shot examples may be dropped.

## MMLU Deep Dive

MMLU (Massive Multitask Language Understanding) is the most widely cited benchmark in LLM evaluation:

- 57 academic subjects: from high school mathematics to professional medicine, law, and ethics
- ~14,079 test questions total; approximately 100–300 per subject
- 4-choice multiple choice throughout
- Log-likelihood scored; no generation needed

The evaluation procedure: for each test question, compute `log P(A | context)`, `log P(B | context)`, `log P(C | context)`, `log P(D | context)`. Predicted answer = argmax. Compare against gold label. Report per-subject accuracy; aggregate as macro-average across all 57 subjects.

Compute for a 70B model, 5-shot:
- 14,079 questions × 4 choices = 56,316 forward passes
- At ~100ms per forward pass (H100, 70B, prompt ~700 tokens, batch size 1) = ~93 minutes
- With batching 32 questions at once (128 choices per batch): ~3 minutes

The 30× speedup from batching explains why the Open LLM Leaderboard uses vLLM as its backend rather than naive HuggingFace `generate()`.

Subject variance matters: models routinely score 80%+ on high school subjects but under 50% on professional law or moral scenarios. Reporting only the aggregate obscures this.

## GSM8K

GSM8K (Grade School Math 8K) is the standard arithmetic reasoning benchmark:

- 8,500 grade-school math word problems in the test split
- Open-ended generation task; no fixed answer choices
- Answer extracted by regex: the last number in the model's response
- Exact match against the gold numerical answer

The common failure mode is instructive from an engineering standpoint: the model generates correct reasoning steps but produces a wrong final number. The regex extractor picks up the last integer or decimal in the response, so a model that outputs "The answer is 42" when the correct answer is 42 passes, but a model that outputs "42 chickens remain, so the farmer has 15 eggs" would be scored against 15. This extraction brittleness motivates more sophisticated answer parsers, but the exact-match-on-last-number convention is standard for comparability.

Compute at 256 max tokens, greedy decoding:
- 8,500 examples × 256 tokens maximum = up to 2.176M tokens generated
- In practice most correct answers are under 200 tokens; mean generation ~150 tokens
- At 5,000 tokens/second sustained (H100 with vLLM, 70B): ~7 minutes total
- Memory: KV cache grows to peak at prompt_length + 256 tokens per sequence

Chain-of-thought prompting is standard for GSM8K: `--num_fewshot 8` with CoT examples that demonstrate step-by-step reasoning. Without CoT, even large models score substantially lower.

## HumanEval

HumanEval measures functional correctness on Python programming:

- 164 programming problems with function signatures and docstrings
- Model must generate the function body
- Graded by running against a private test suite; a solution passes if all test cases pass
- Reported as pass@k

The pass@k metric accounts for the stochastic nature of code generation by sampling multiple attempts:

```
pass@k = 1 - C(n - c, k) / C(n, k)
```

where `n` = total samples drawn per problem, `c` = number of samples that pass, `k` = the number of samples you are allowed to submit. For pass@1 with n=20 samples: you generate 20 completions and measure the fraction of problems where at least 1 of the 20 is correct, estimating the probability a single draw would be correct. The combinatorial formula provides an unbiased estimator without requiring exactly k samples.

Compute: at n=20 samples per problem, 164 problems, ~200 tokens per completion = 164 × 20 × 200 = 656k tokens generated. At 5,000 tokens/second: under 3 minutes. The bottleneck is code execution, not generation — running 3,280 test cases on potentially unsafe code requires a sandboxed execution environment (subprocess + timeout, Docker, or a microVM like Firecracker).

## Open LLM Leaderboard Infrastructure

The HuggingFace Open LLM Leaderboard runs `lm-evaluation-harness` on submitted model checkpoints with the following infrastructure setup:

- Hardware: single A100-80GB for most evaluations; multi-GPU for models requiring tensor parallelism
- Inference backend: vLLM for throughput; raw HuggingFace `transformers` as fallback for models that vLLM doesn't support
- Tasks (v1): MMLU (5-shot), ARC Challenge (25-shot), HellaSwag (10-shot), Winogrande (5-shot), GSM8K (5-shot, CoT), TruthfulQA (0-shot)
- Reproducibility: the exact harness commit hash and per-task config are logged alongside each model's results
- Model loading: BF16 precision by default; 4-bit quantized fallback for very large models on single GPU
- Throughput: a full leaderboard evaluation suite on a 7B model takes approximately 2–3 hours; on a 70B model approximately 12–15 hours with vLLM

The leaderboard's value derives from this reproducibility guarantee. Two models can only be meaningfully compared if evaluated identically. The harness makes this possible by encoding all evaluation decisions in version-controlled configuration.

## Engineering Tradeoffs

| Dimension | Option A | Option B | Key consequence |
|---|---|---|---|
| Scoring mode | Log-likelihood (multiple choice) | Generation (open-ended) | Log-likelihood: 4 forward passes, no sampling, prefix cache reuse; generation: autoregressive, slower per task, but necessary for tasks without fixed answer sets |
| Benchmark type | Static harness (MMLU, GSM8K) | Live evaluation (Chatbot Arena) | Static: fast, reproducible, contamination risk grows over time; live: ground-truth for user preference, slow to accumulate, expensive |
| Few-shot count | 0-shot | 5-shot | 5-shot improves accuracy 5–15 pts on MMLU; adds 500–800 tokens to every prompt; fails silently on short-context models |
| Inference backend | HuggingFace generate() | vLLM | vLLM delivers 10–30x higher throughput via paged attention; required for practical leaderboard operation; adds setup complexity |
| Answer extraction | Regex / exact match | Code execution (HumanEval) | Code execution requires sandboxed environment; adds infrastructure complexity but provides functional correctness signal |
| Evaluation scope | Single benchmark (GSM8K) | Full suite (6+ benchmarks) | Full suite: 12–15h per 70B model; enables multi-dimensional comparison; compute cost must be budgeted explicitly |

## Limitations and Benchmark Gaming

Static benchmarks have a fundamental contamination problem: as models are trained on increasingly large crawls of the internet, test set questions appear in training data. MMLU questions are widely reproduced in blog posts and study guides; GSM8K problems appear in math forums. A model that memorized MMLU test questions during pretraining will report higher accuracy than its reasoning ability warrants.

Detection is difficult. The standard approach — n-gram overlap between training data and test sets — catches exact copies but misses paraphrase contamination. Some labs report "contamination-filtered" results; the filtering methodology is inconsistently applied.

The practical consequence for infrastructure: high benchmark scores no longer reliably predict production quality. A model can achieve 85% MMLU while being noticeably worse than a competitor at 82% MMLU on real user tasks. This is why Chatbot Arena (see `../chatbot-arena/`) exists as a complementary signal: it accumulates fresh human preferences on new prompts that cannot be pre-contaminated.

For production model selection, use benchmark scores as a filter (eliminate clearly underperforming models) rather than a ranking (don't assume rank order reflects production quality). Reserve final decisions for internal evals on your specific task distribution.

## References

- [1] Gao et al. _A Framework for Few-Shot Language Model Evaluation._ arXiv:2110.08207, 2021.
- [2] Hendrycks et al. _Measuring Massive Multitask Language Understanding._ arXiv:2009.03300, 2020.
- [3] Cobbe et al. _Training Verifiers to Solve Math Word Problems._ arXiv:2110.14168, 2021. (GSM8K)
- [4] Chen et al. _Evaluating Large Language Models Trained on Code._ arXiv:2107.03374, 2021. (HumanEval)
- [5] Clark et al. _Think You Have Solved Question Answering? Try ARC._ arXiv:1803.05457, 2018.
- [6] Zellers et al. _HellaSwag: Can a Machine Really Finish Your Sentence?_ arXiv:1905.07830, 2019.

## Cross-References

- `../inference-time-scaling/` — scaling inference compute at test time; the harness measures the resulting output quality
- `../process-reward-models/` — PRMs are evaluated using math benchmarks (GSM8K, MATH) running inside this harness
- `../../meta/llama3/` — Llama 3 technical report cites MMLU, GSM8K, and HumanEval figures produced by lm-eval-harness
- `../chatbot-arena/` — complementary live evaluation; human preference signal that benchmarks cannot replicate
