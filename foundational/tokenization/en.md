# Tokenization — BPE, SentencePiece, and Tiktoken

- **Authors / Org**: Sennrich et al. (University of Edinburgh) · Kudo & Richardson (Google) · OpenAI
- **Published**: BPE: 2016 (ACL, arXiv:1508.07909) · SentencePiece: 2018 (EMNLP) · Tiktoken: 2023 (GitHub)
- **Links**: [BPE paper](https://arxiv.org/abs/1508.07909) · [SentencePiece paper](https://aclanthology.org/D18-2012/) · [SentencePiece code](https://github.com/google/sentencepiece) · [Tiktoken](https://github.com/openai/tiktoken)

## TL;DR

Tokenization converts raw Unicode text into integer sequences that a language model can process. Byte-Pair Encoding (BPE) learns a vocabulary of subword units by iteratively merging the most frequent adjacent pairs in the training corpus — avoiding both character-level inefficiency and the unseen-word problem of word-level vocabularies. SentencePiece extends BPE to work directly on raw Unicode without any pre-tokenization step, making it truly language-independent. Tiktoken is a Rust-based BPE implementation used in GPT-4 / o1 with a 100k–200k vocabulary. Vocabulary size is a first-order systems parameter: it sets the number of embedding rows, determines how many tokens a fixed compute budget can process, and shapes how multilingual data distributes across the model's conceptual bandwidth.

## Context & Motivation

### Why not characters or words?

**Character-level tokenization** is language-independent and handles any Unicode input without an out-of-vocabulary (OOV) problem. The cost: sequences become very long (a 512-word English text has ~2,500 characters), and the model must learn higher-level structure from scratch. Context window budget is wasted on individually uninformative tokens.

**Word-level tokenization** keeps sequences short and preserves natural linguistic units. The cost: a vocabulary large enough to cover a useful fraction of English requires 50k–200k entries; morphologically rich languages (Finnish, Turkish, Arabic) or specialized domains (code, math) multiply this further. Any unseen word at inference time becomes a single `<UNK>` token that loses all internal structure.

**Subword tokenization** occupies the sweet spot: common words remain intact as single tokens, rare words decompose into meaningful subword units, and an unknown word can always be decomposed into individual bytes (a guaranteed fallback). Modern LLMs universally use subword tokenization for this reason.

### Tokenization's systems-level role

Tokenization sits upstream of everything: it determines sequence length, and sequence length drives nearly all downstream costs — attention is quadratic, KV cache memory is linear, and prefill throughput is often bottlenecked by the number of tokens to process. A tokenizer that produces 30% more tokens for a given corpus forces every downstream component to handle 30% more work. For multilingual models, tokenizer fertility (tokens per character or word) is unevenly distributed, effectively giving some languages far less representational capacity per context window.

## BPE: Byte-Pair Encoding

### Algorithm

BPE starts with a base vocabulary of individual characters (or bytes) and iteratively merges the most frequent adjacent pair until a target vocabulary size V is reached:

```
Initialize: vocabulary = {individual characters in corpus}
For i = 1..V - |initial_vocab|:
    count all adjacent (a, b) pairs in current tokenized corpus
    (a*, b*) = argmax_{(a,b)} freq(a, b)
    merge every occurrence of (a*, b*) → new token "ab"
    add "ab" to vocabulary
```

**Key equation — merge selection:**

```
(a*, b*) = argmax_{(a,b)} freq(a, b)   over all adjacent pairs in corpus
```

The merge rules are learned on the training corpus and then applied deterministically at inference time using the same priority order. This makes BPE encoding a greedy rule-rewriting procedure, not a probabilistic model.

### Properties and engineering consequences

- **Vocabulary is a hyperparameter** chosen before training. Typical values: 30k–50k for early models (GPT-2: 50,257), 100k–200k for modern LLMs.
- **Merge rule order matters**: the same text tokenizes identically at inference time only if the same ordered list of merge rules is applied. Changing the tokenizer between pretraining and fine-tuning breaks the model.
- **Tokenization of code**: BPE learns to merge common code patterns (`def `, `if `, `return `, `self.`, `print(`) as single tokens, giving code models a compact representation.
- **Whitespace handling**: GPT-2 style BPE prepends a special whitespace byte `Ġ` (byte 0xC4 0xA0) before tokens that follow a space. This distinguishes " dog" from "dog" — important for model quality, confusing for debugging.

### Byte-level BPE

BPE can be initialized with all 256 byte values rather than Unicode characters. This guarantees that any byte sequence can be tokenized — there is no OOV. GPT-2 and Tiktoken use byte-level BPE. The tradeoff: CJK characters occupy 3 bytes in UTF-8 each, so the base representation before merges is 3 tokens per character. Merges on frequent CJK sequences reduce this, but a large vocabulary is needed to bring fertility down to acceptable levels.

## SentencePiece: Language-Independent Tokenization

### The pre-tokenization problem

Most BPE implementations split text on whitespace first (pre-tokenization), then learn merges within token boundaries. This embeds English-centric assumptions: whitespace as word boundary, no compounding, no agglutination. Japanese, Chinese, and Korean have no whitespace between words; Thai similarly; Arabic has clitics; German has long compounds.

SentencePiece (Kudo & Richardson, 2018) trains directly on raw Unicode text with no pre-tokenization. The entire sentence — including spaces — is treated as a sequence of Unicode code points. Space is represented as a special underscore character `▁` (U+2581), making it a reversible byte-level encoding of the original text.

### Unigram language model option

SentencePiece supports both BPE and a **unigram language model** tokenizer:
- The unigram model starts with a large candidate vocabulary and prunes it by iteratively removing tokens whose removal least increases total corpus log-likelihood.
- At inference time, the unigram model produces a probability distribution over segmentations; the MAP segmentation (Viterbi decoding) is used.
- SentencePiece BPE is deterministic; unigram can be sampled — `sentencepiece.encode(..., sample=True)` draws a segmentation from the posterior, which is a form of data augmentation during training.

### Usage in major models

- **Llama 1 / 2**: SentencePiece BPE, 32k vocabulary.
- **Llama 3**: SentencePiece BPE, **128k vocabulary**. The jump from 32k → 128k was motivated by multilingual coverage and code density. Encoding efficiency for Chinese text improves roughly 3× compared to the 32k vocabulary.
- **T5, mT5**: SentencePiece with 32k vocabulary, unigram model.
- **Gemma**: SentencePiece with 256k vocabulary.
- **DeepSeek V3**: Tiktoken cl100k_base-compatible BPE, 100k vocabulary.

## Tiktoken: Production BPE at Scale

Tiktoken is OpenAI's BPE tokenizer, implemented in Rust with Python bindings. It is designed for throughput: the Rust implementation is ~3–5× faster than the pure-Python HuggingFace tokenizer for equivalent vocabulary sizes.

### Vocabulary families

| Encoding | Vocabulary size | Used by |
|---|---|---|
| `r50k_base` | 50,257 | GPT-3, Codex |
| `cl100k_base` | 100,277 | GPT-3.5, GPT-4, DeepSeek |
| `o200k_base` | 200,019 | GPT-4o, o1, o3 |

`o200k_base` doubles the vocabulary relative to `cl100k_base`. For multilingual text, the increased vocabulary primarily buys better compression of non-Latin scripts. For English text, sequence length changes by only ~5%.

### Performance implications

A model with a 200k vocabulary has an embedding matrix of shape [200k, d_model]. At d_model = 7168 (DeepSeek V3 size) and bf16:

```
Embedding table size = 200,000 × 7,168 × 2 bytes = 2.87 GB
```

This is non-trivial on a single GPU but is shared between the embedding layer and the unembedding (lm_head) layer in weight-tied models. The unembedding projection is the most expensive single operation at decode time for small batch sizes because it is a [1, d_model] × [d_model, V] matmul — a GEMV dominated by bandwidth rather than compute.

## Token Budget Economics

### Vocabulary size tradeoff

| Vocabulary size | Tokens per character (English) | Tokens per character (Chinese) | Embedding params (d=4096) | Notes |
|---|---|---|---|---|
| 32k | ~0.27 | ~0.8–1.2 | 262M | Llama 1/2 |
| 50k | ~0.25 | ~0.7–1.0 | 410M | GPT-2/3 |
| 100k | ~0.23 | ~0.4–0.7 | 819M | GPT-4, DeepSeek V3 |
| 128k | ~0.22 | ~0.3–0.5 | 1.05B | Llama 3 |
| 200k | ~0.21 | ~0.25–0.4 | 1.64B | GPT-4o, Gemma |

Larger vocabulary → shorter token sequences → fewer attention operations, smaller KV cache → faster inference. But the embedding / unembedding matrices grow, and the final projection dominates latency for small batches.

### The multilingual fertility problem

For English text, 1 token ≈ 0.75 words ≈ 4 characters. For Chinese text with a small vocabulary tokenizer, 1 UTF-8 character can take 1–3 tokens (the 3-byte UTF-8 representation of CJK code points). This means:

- A 4,096-token context window holds roughly 2,700 English words.
- The same window holds only 800–1,500 Chinese characters with a 32k vocabulary.
- With a 128k vocabulary, Chinese fertility improves to ~1 token per 2–3 characters.

This asymmetry explains why multilingual models with small vocabularies systematically underperform on non-Latin-script languages: they see fewer semantic concepts per token and exhaust their context window faster.

### Connection to scaling laws

The "[20 tokens per parameter](../scaling-laws/en.md)" Chinchilla rule is tokenizer-dependent. A model trained on 300B tokens of Chinese text with a byte-level tokenizer is exposed to far fewer distinct words and concepts than a model trained on 300B tokens using a character-level tokenizer. Apparent data richness is a function of tokenizer fertility, not raw token count.

## Engineering Tradeoffs

| Decision | Gained | Gave up |
|---|---|---|
| Small vocabulary (32k) | Fewer embedding parameters; simpler model shards | Poor multilingual compression; Chinese/Japanese needs 3–4× more tokens per concept |
| Large vocabulary (128k–200k) | Better multilingual fertility; shorter sequences; faster inference for given text | Larger embedding + unembedding matrices; slower lm_head GEMV at small batch sizes |
| Byte-level BPE (Tiktoken, GPT-2) | Zero OOV guarantee; universal coverage of any byte sequence | CJK characters are expensive at base before merges (3 bytes each); requires large vocabulary to achieve good compression |
| SentencePiece (no pre-tokenization) | Language-independent; works on raw Unicode; reversible encoding | Slightly different tokenization behavior from BPE-with-pretokenization; requires retraining if switching |
| Unigram sampled tokenization | Segmentation diversity as data augmentation; useful for training robustness | Non-deterministic; not suitable for inference without argmax decoding |
| Shared embed / unembedding weights | Halve the parameter count for embedding + output | Constrains both matrices to be transposes; can slightly hurt quality vs. independent matrices |

## Experiments & Results

**BPE (Sennrich et al. 2016)**: demonstrated that NMT models using BPE subword units match or exceed word-level models on English-German and English-Russian translation, with no OOV errors. Vocabulary of 30k–60k suffices for most bilingual tasks.

**SentencePiece (Kudo & Richardson 2018)**: showed that treating space as a character (no pre-tokenization) produces tokenizations that are losslessly invertible — the original text can be exactly recovered from any token sequence. This is not guaranteed by whitespace-based BPE, which may destroy spacing information.

**Llama 3 vocabulary expansion (32k → 128k)**: Meta reported that the 4× vocabulary expansion reduced average sequence length for multilingual text by ~2.5×. Token efficiency for Chinese improved from ~0.8 tokens/character to ~0.3 tokens/character. Downstream task performance on multilingual benchmarks improved substantially, attributable partly to the tokenizer and partly to expanded multilingual training data.

**Tiktoken throughput**: The Rust implementation tokenizes at ~10 MB/s on a single CPU core for `cl100k_base`, versus ~2–3 MB/s for equivalent Python implementations. At dataset preprocessing scale (multi-terabyte corpora), this matters.

## Commentary

Tokenization is the first lossy compression step in any LLM pipeline, and its choices compound downstream in ways that are hard to reverse. A poor tokenizer for a target language effectively halves the model's context capacity for that language — the model has fewer tokens available per concept, exhausts its window faster, and appears weaker on benchmarks even if the transformer itself is well-trained. This is one reason why models like Llama 3, Gemma, and Qwen made significant vocabulary expansions compared to their predecessors.

The vocabulary size and embedding table form a classic space-compute tradeoff at inference time. The lm_head matmul ([batch × seq, d_model] × [d_model, V]) is the single most expensive operation per decoding step for small batches. With batch size 1 and d_model = 8192, V = 128k, this is a matrix-vector product requiring 2 GB of bandwidth per step — dominating decode latency on most hardware. Speculative decoding, which needs to verify draft tokens against the target model's logits, multiplies this cost further. Larger vocabularies make lm_head more expensive but reduce sequence length, so the net effect on end-to-end latency depends on sequence length and batch size.

The deeper engineering lesson is that tokenization is infrastructure that must be specified before training begins and is essentially impossible to change afterward. Models cannot be re-tokenized post hoc without full retraining. This makes the tokenizer a commitment analogous to the hardware choice for a training run: you will live with its consequences for the model's lifetime. The industry has converged on 100k–200k vocabulary BPE for production-grade multilingual models, but the right choice remains workload-dependent — code-heavy models benefit from different merge frequency distributions than conversation-heavy models.

## References

- [1] Sennrich et al. 2016, "Neural Machine Translation of Rare Words with Subword Units," ACL. arXiv:1508.07909.
- [2] Kudo & Richardson 2018, "SentencePiece: A simple and language independent subword tokenizer and detokenizer for Neural Text Processing," EMNLP.
- [3] OpenAI 2023, Tiktoken. https://github.com/openai/tiktoken
- [4] Dubey et al. 2024, "The Llama 3 Herd of Models," arXiv:2407.21783.
- [5] Liu et al. 2024, "DeepSeek-V3 Technical Report," arXiv:2412.19437.
- [6] Kudo 2018, "Subword Regularization: Improving Neural Network Translation Models with Multiple Subword Candidates," ACL. arXiv:1804.10959.
- [7] Hoffmann et al. 2022, "Training Compute-Optimal Large Language Models (Chinchilla)," arXiv:2203.15556. See also: [Scaling Laws entry](../scaling-laws/en.md).
- [8] [Llama 3 training details](../../meta/llama3/en.md).
