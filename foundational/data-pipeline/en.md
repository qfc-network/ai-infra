# Data Pipelines — FineWeb, MinHash, and Quality Filtering

- **Authors / Org**: Penedo et al. (HuggingFace) · Wenzek et al. (Facebook AI) · Soldaini et al. (AllenAI) · Together AI
- **Published**: FineWeb: 2024 (arXiv:2406.17557) · CCNet: 2020 (LREC) · Dolma: 2024 (ACL) · RedPajama-v2: 2023 (GitHub)
- **Links**: [FineWeb paper](https://arxiv.org/abs/2406.17557) · [FineWeb dataset](https://huggingface.co/datasets/HuggingFaceFW/fineweb) · [CCNet](https://github.com/facebookresearch/cc_net) · [Dolma](https://github.com/allenai/dolma) · [RedPajama-v2](https://github.com/togethercomputer/RedPajama-Data)

## TL;DR

Training data quality is a first-order determinant of LLM capability — arguably more important than architecture choices at a fixed compute budget. Building a high-quality pretraining corpus from Common Crawl requires a multi-stage pipeline: HTML-to-text extraction, language identification, URL and heuristic quality filtering, near-duplicate deduplication using MinHash locality-sensitive hashing, and learned quality classifiers. FineWeb extracted 15 trillion tokens from 96 Common Crawl dumps and demonstrated that a carefully tuned quality classifier outperforms perplexity-based filtering. Data mix — the proportion of web, code, books, math, and science data — is itself a training hyperparameter tuned via scaling-law experiments on smaller models before a full-scale run.

## Context & Motivation

### The Common Crawl gold mine (and sewer)

Common Crawl is a non-profit organization that has crawled the public web since 2008, archiving approximately 3–5 billion pages per monthly dump in WARC (Web ARChive) format. The total archive exceeds 1 petabyte of compressed data. It is the primary source for essentially every large-scale LLM pretraining corpus.

The problem: the raw crawl is dominated by low-quality content — spam, SEO-farmed articles, boilerplate navigation menus, adult content, duplicate pages scraped from other pages, machine-translated text, and content that is harmful or legally risky. Training on unfiltered Common Crawl produces models with visibly degraded capability relative to models trained on curated data. The engineering challenge is extracting a signal-to-noise ratio high enough to train competitive models, at trillion-token scale.

### Why this matters for compute

Training a 70B model on 1.4 trillion tokens at 6 FLOPs/token costs approximately:

```
6 × 70×10⁹ × 1.4×10¹² ≈ 5.9 × 10²³ FLOPs
```

This is roughly 5,000 A100-GPU-hours at peak utilization, or ~$5M at cloud pricing. The training run can only be done once or a small number of times. If the data has 10% near-duplicate contamination, or 20% of documents are low-quality SEO spam, that wasted compute cannot be recovered. Data pipeline quality multiplies directly into training efficiency.

## The Common Crawl Processing Pipeline

### Stage 1: WARC to text extraction

Raw WARC files contain HTTP headers, raw HTML, and binary assets. The first step is extracting clean text from HTML. Two popular tools:

- **trafilatura**: Python library; favors precision (extracts main content, discards navigation/ads); better for news and article content.
- **resiliparse** (CCNet / FineWeb): faster, better recall; used in high-throughput pipelines.

The extraction quality matters significantly. Navigation menus, cookie banners, advertisement text, and comment sections that leak through become training data. FineWeb uses a combination of resiliparse for extraction and custom post-processing rules to remove residual HTML artifacts.

### Stage 2: Language identification

`fastText` language ID models (specifically the `lid.176.bin` model from Meta, trained on Wikipedia/Tatoeba) classify each document's language from a short sample (~200 characters). Processing cost: ~200K documents/second on a single CPU.

FineWeb targets English and retains documents with English confidence ≥ 0.65. For multilingual corpora (mC4, ROOTS, CulturaX), the language ID threshold is applied per-language and the resulting documents are kept in language-specific shards.

### Stage 3: URL and heuristic pre-filters

Before expensive per-document processing, cheap URL-based filters eliminate known-bad sources:

- Blocklists of adult content domains (UT1, NSFW domain lists).
- URL patterns associated with spam farms (excessive numeric paths, domain generation algorithm patterns).
- `.onion` addresses, `.gov`/`.edu` spammy subdomains.

Then per-document heuristic filters (inspired by C4 and MassiveText):

- **Document length**: minimum 200 characters after extraction.
- **Line length distribution**: median line length > 30 characters (filters navigation-heavy pages).
- **Alphanumeric character ratio**: ≥ 70% of characters are alphanumeric or common punctuation.
- **Lines ending with punctuation**: ≥ 80% of lines end with `.`, `?`, `!`, `:`, `"` — filters bullet-point spam and keyword lists.
- **Symbol-to-word ratio**: filter pages with excessive `#`, `{`, `|`, `\` characters.
- **Repeated content**: filter pages where any line appears ≥ 10× (header/footer detection).

### Stage 4: MinHash deduplication

Heuristic filters remove obvious garbage but leave near-duplicates: the same article republished on 500 news aggregators, product descriptions scraped from manufacturer websites, boilerplate legal disclaimers, Wikipedia mirrors. Near-duplicates create memorization pressure and can dominate certain topic distributions in the training corpus.

**MinHash deduplication** is the standard approach at scale. The algorithm:

1. Tokenize each document into a set of n-grams (typically 5-grams of tokens).
2. For each of K independent hash functions h_k, compute the minimum hash value over all n-grams in the document. This K-dimensional vector is the **MinHash signature**.
3. Documents with similar content will have similar MinHash signatures — specifically:

**Key equation — MinHash Jaccard estimate:**

```
Pr[min_{h}(A) = min_{h}(B)] = |A ∩ B| / |A ∪ B| = Jaccard(A, B)
```

The probability that two documents have the same minimum hash under a random hash function equals their Jaccard similarity over n-gram sets. With K=128 hash functions, the estimate has standard deviation ≈ 1/√128 ≈ 0.09.

4. Group documents into buckets using **Locality-Sensitive Hashing (LSH)**: split the K-dimensional signature into B bands of R rows; two documents are candidate near-duplicates if they agree in at least one band. This gives a threshold behavior: at Jaccard ≥ 0.8, essentially all pairs are candidates; at Jaccard < 0.7, essentially none are.

5. For each bucket, form connected components and keep one representative. Documents shorter than ~200 tokens are excluded from deduplication (short texts match spuriously).

**Computational cost**: MinHash deduplication over 15 trillion tokens requires processing ~100 billion documents. The bottleneck is the LSH matching step, which is parallelized over Spark or Ray clusters. FineWeb reports ~$100K in cloud compute for the full deduplication pass.

### Stage 5: Quality filtering

After dedup, the remaining corpus still contains significant low-quality content. Two main approaches:

**Heuristic rules (C4-style)**: rules derived from manual inspection of Common Crawl quality. FineWeb applies 30+ rules including detection of lorem ipsum placeholder text, excessive non-UTF-8 code points, very short documents after extraction, and high fraction of lines that are URLs.

**Classifier-based quality filtering**: train a binary text classifier on:
- Positive examples: curated, high-quality text (Wikipedia, wikiHow, high-quality news, books).
- Negative examples: random sample of Common Crawl after heuristic filtering.

The classifier learns a quality score in [0,1] for each document. Documents above a threshold (typically 0.5) are retained.

**FineWeb's key finding**: in ablation experiments training 1.4B models on 350B tokens, the quality classifier significantly outperformed perplexity-based filtering (KenLM). The classifier-filtered data produced consistently better downstream benchmark scores across MMLU, HellaSwag, and ARC.

**Perplexity filtering (KenLM)**: an alternative: train an n-gram language model on high-quality text (Wikipedia) and retain documents with perplexity below a threshold. Fast and simple. Limitation: perplexity filtering tends to prefer formal, encyclopedic style and can over-remove valid informal text or specialized domain content.

## Data Mixing and Curriculum

### The mixing hyperparameter

Modern pretraining corpora combine multiple data sources with different characteristics:

| Source | Typical share | Quality signal |
|---|---|---|
| Web (filtered Common Crawl) | 60–80% | Breadth; risk of noise |
| Code (GitHub, StackOverflow) | 5–15% | High density reasoning patterns |
| Books / long-form text | 5–10% | Long-range coherence |
| Scientific papers (arXiv, S2ORC) | 2–5% | Technical precision |
| Math (MATH, AoPS, textbooks) | 1–5% | Structured reasoning |
| Curated Q&A, Wikipedia | 2–5% | Factual accuracy |

The proportion of each source is a hyperparameter that affects downstream capability in a non-obvious way. Code-heavy mixes improve reasoning and structured output. Math-heavy mixes improve arithmetic and logical tasks. The right balance depends on the target application.

### Scaling-law ablations for mix selection

Since a 70B training run cannot be repeated easily, data mix decisions are made via **scaling-law proxy experiments**: train multiple 1B models on 20B tokens each, varying the mix. Measure downstream benchmarks and fit scaling curves. Extrapolate the optimal mix to the 70B scale.

This is how Llama 3, DeepSeek, and other production models were tuned. The assumption is that the optimal mix is approximately scale-invariant — a mix that performs well at 1B also performs well at 70B. This assumption is imperfect but practically necessary.

### Curriculum ordering

Data ordering within training affects convergence. Common strategies:
- **Uniform shuffling**: baseline; prevent the model from seeing repetitions too close together.
- **Quality-based ordering**: lower-quality data early in training, higher-quality later. Intuitively, the model learns basic patterns from large noisy data and then refines on high-quality data.
- **Capability-specific upsampling near end of training**: Llama 3 upsampled math and code data in the final 10% of training to boost reasoning without affecting the rest of training.

## Engineering Tradeoffs

| Decision | Gained | Gave up |
|---|---|---|
| Heuristic-only quality filtering | Fast; interpretable; no training required | Can systematically over-remove valid content; rules require manual curation per language/domain |
| Classifier-based quality filtering | Better downstream model quality; adapts to content distribution | Requires labeled data; classifier quality depends on what "high quality" means; can amplify bias |
| Aggressive MinHash deduplication (Jaccard ≥ 0.8) | Removes near-duplicates; reduces memorization; flattens overrepresented topics | Can remove legitimate repetition (e.g., standard license headers in code); slightly reduces corpus size |
| Perplexity filtering (KenLM) | Simple; fast; no labels needed | Biases toward formal/encyclopedic text; hurts coverage of informal or technical domains |
| Larger CC dump coverage (more crawl snapshots) | More data; better temporal coverage; less repetition per document | Diminishing returns; older dumps have worse quality; processing cost scales linearly |
| Domain-specific data upsampling (code, math) | Improves targeted capabilities dramatically | Shifts distribution; can degrade performance on unsampled domains if done too aggressively |

## Experiments & Results

**FineWeb (Penedo et al. 2024)**: 15 trillion tokens from 96 Common Crawl dumps (2013–2024). After all pipeline stages, FineWeb retains approximately 2–3% of raw crawl volume. Quality classifier filtering produced the FineWeb-Edu subset (educational content), which outperformed the full FineWeb dataset on knowledge-intensive benchmarks despite being smaller (1.3T tokens). Training a 1.4B model on FineWeb-Edu for 350B tokens matched GPT-3.5-level performance on MMLU.

**CCNet (Wenzek et al. 2020)**: introduced the per-language pipeline with fastText language ID + KenLM perplexity filtering. Showed that perplexity filtering roughly tripled downstream task performance of a monolingual LM compared to unfiltered Common Crawl.

**Dolma (AllenAI 2024)**: 3 trillion tokens for training OLMo. Multi-source: Common Crawl (67%), C4, GitHub, books, scientific papers, Wikipedia. Full pipeline code and data provenance documented publicly, enabling reproducibility.

**RedPajama-v2 (2023)**: 30+ trillion tokens from 84 Common Crawl snapshots with computed quality signals (MinHash, perplexity, classifier scores) stored per-document, allowing downstream users to apply custom filtering thresholds without re-running the pipeline.

## Commentary

Data pipeline engineering is the least glamorous part of LLM development and arguably the most consequential. A model trained on a carefully curated 1T-token dataset regularly outperforms the same architecture trained on 5T tokens of poorly filtered web data. The filtering stages that seem like preprocessing housekeeping — language ID, heuristics, dedup, quality classifiers — collectively determine whether the model's weights encode useful world knowledge or memorized spam and duplicate boilerplate.

The MinHash deduplication step deserves more attention than it typically receives. Near-duplicate text in training data creates two distinct problems: it biases the model toward memorizing frequently occurring content (including undesirable content like common legal disclaimers and SEO-optimized article structures), and it distorts the effective distribution of topics seen during training (a news event covered by 10,000 syndicated articles is "10,000× more important" to the model than an equally significant event covered once). Aggressive deduplication at Jaccard ≥ 0.8 addresses both.

The data mix decision is where the scaling-law machinery from the [Chinchilla paper](../scaling-laws/en.md) becomes an operational tool. You cannot afford to train a 70B model multiple times to find the optimal mix, so you proxy the search at 1B and trust that scale-invariance holds approximately. This is imperfect — some capabilities (e.g., long-context reasoning) are not well-represented at 1B scale — but it is the best available approximation. The combination of compute-optimal sizing from scaling laws and empirical mix selection from proxy experiments defines the current state of the art for pretraining corpus construction. See also [DeepSeek V3](../../deepseek/v3-tech-report/en.md) and [Llama 3](../../meta/llama3/en.md) for how large-scale production runs implemented these principles.

## References

- [1] Penedo et al. 2024, "The FineWeb Datasets: Decanting the Web for the Finest Text Data at Scale," arXiv:2406.17557.
- [2] Wenzek et al. 2020, "CCNet: Extracting High Quality Monolingual Datasets from Web Crawl Data," LREC.
- [3] Soldaini et al. 2024, "Dolma: an Open Corpus of Three Trillion Tokens for Language Model Pretraining Research," ACL. arXiv:2402.00159.
- [4] Together AI 2023, "RedPajama: An Open Dataset for Training Large Language Models," GitHub.
- [5] Hoffmann et al. 2022, "Training Compute-Optimal Large Language Models (Chinchilla)," arXiv:2203.15556. See also: [Scaling Laws entry](../scaling-laws/en.md).
- [6] Broder 1997, "On the resemblance and containment of documents," Sequences '97 (MinHash original paper).
- [7] Raffel et al. 2020, "Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer (T5/C4)," JMLR. arXiv:1910.10683.
- [8] Dubey et al. 2024, "The Llama 3 Herd of Models," arXiv:2407.21783. See also: [Llama 3 entry](../../meta/llama3/en.md).
- [9] [DeepSeek V3 Technical Report](../../deepseek/v3-tech-report/en.md).
