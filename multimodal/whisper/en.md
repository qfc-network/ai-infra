# Whisper — Robust Speech Recognition via Large-Scale Weak Supervision

- **Authors / Org**: Alec Radford, Jong Wook Kim, Tao Xu, Greg Brockman, Christine McLeavey, Ilya Sutskever (OpenAI)
- **Published**: 2022-12-06
- **Links**: [arXiv:2212.04356](https://arxiv.org/abs/2212.04356) · [blog](https://openai.com/research/whisper) · [code](https://github.com/openai/whisper)

## TL;DR

Whisper trains a standard encoder-decoder transformer on 680,000 hours of weakly-supervised speech data scraped from the web, without any curated labels. The audio encoder processes 30-second chunks of 80-bin log-mel spectrograms — using the same self-attention mechanism as [ViT](../vit/en.md) — and the decoder autoregressively predicts token IDs. All tasks (transcription, translation, language detection, timestamps) are expressed through special control tokens, making Whisper a single multi-task model. The result is state-of-the-art robustness to accents, noise, and domain shift, achieved purely through scale rather than careful data curation.

## Context & Motivation

Prior ASR systems required carefully curated, hand-labeled audio datasets. High-quality labels are expensive: transcribing 1 hour of speech takes a trained transcriber 4–10 hours. This bottleneck kept ASR models small and domain-specific. Whisper asked: what if we collected audio-text pairs from the internet the same way [CLIP](../clip/en.md) collected image-text pairs — accepting noisy alignment in exchange for 1000× more data?

The scale hypothesis proved correct. Whisper-large-v2 (1.55B parameters) matched or beat fine-tuned Wav2Vec 2.0 on most benchmarks despite using weaker supervision. The key insight is that ASR is one of the few tasks where the "label" (a transcript) often appears naturally adjacent to the audio on the web — subtitles, podcast transcripts, lecture notes — making weak supervision surprisingly effective.

## Audio Preprocessing — Log-Mel Spectrogram

Audio is converted to a fixed-size 2D representation before entering the model:

1. **Resampling**: input audio resampled to 16 kHz.
2. **Short-Time Fourier Transform (STFT)**: applied with a 25ms Hann window at 10ms stride. At 16 kHz, 25ms = 400 samples; 10ms = 160 samples.
3. **Mel filterbank**: 80 triangular mel-spaced filters applied to the STFT magnitude spectrum.
4. **Log compression**: `log10(max(mel_spectrogram, 1e-10))`.
5. **Normalization**: subtract mean, divide by std across the chunk.

Result: for a 30-second audio chunk, the spectrogram has shape **80 × 3000** (80 mel bins × 3000 time frames, since 30s × 100 frames/s = 3000). This is the input to the encoder — a 2D "image" of the audio.

### Why 30-Second Chunks?

Whisper's architecture assumes fixed-length 30-second inputs. Shorter audio is zero-padded; longer audio is processed with a sliding window. 30 seconds was chosen to balance context length (longer chunks help the model resolve ambiguous sounds in context) against memory cost (the encoder must process all 3000 frames at once).

## Architecture

### Encoder — ViT-Style Attention on Mel Frames

The mel spectrogram (80 × 3000) is processed by a small convolutional stem followed by transformer blocks:

- **Two 1D conv layers** (kernel size 3, stride 1 and 2 respectively) reduce the 3000 time frames to **1500 frames** and project to the model dimension D.
- **Positional embeddings**: learned sinusoidal (fixed), shape 1500 × D.
- **L transformer encoder blocks**: standard MHSA + MLP, Pre-LN, no causal masking. The encoder sees all 1500 frames simultaneously (bidirectional).

The 1500-frame sequence is the output of the encoder — a rich contextualized representation of the full audio chunk. This is analogous to how ViT processes image patches: the 1D convolutions here play the role of patch projection.

### Decoder — Causal LM over Token IDs

The decoder is a standard autoregressive transformer:

- Causal self-attention over previously generated tokens.
- Cross-attention over the encoder output (1500 frames).
- Vocabulary: 51,864 tokens (multilingual BPE, ~99 languages).

The decoder generates text token-by-token using beam search or greedy decoding. Cross-attention makes every decoder step attend over all 1500 encoder frames, allowing the model to "re-read" any part of the audio at each prediction step.

### Multitask via Special Tokens

All tasks share the same decoder and are differentiated entirely by control tokens prepended to the decoder input:

```
<|startoftranscript|> <|en|> <|transcribe|> <|notimestamps|>
→  [text tokens...]  <|endoftext|>

<|startoftranscript|> <|de|> <|translate|> <|notimestamps|>
→  [translated English text...]  <|endoftext|>

<|startoftranscript|> <|en|> <|transcribe|>
→  <|0.00|> Hello, <|1.20|> world. <|endoftext|>
```

This multitask framing (borrowed from GPT-3's prompt-as-task-specification) means a single model handles speech-to-text, speech translation, language ID, and timestamp prediction. No task-specific heads or fine-tuning.

### Model Sizes

| Model | Encoder layers | Decoder layers | D | Heads | Params |
|---|---|---|---|---|---|
| tiny | 4 | 4 | 384 | 6 | 39M |
| base | 6 | 6 | 512 | 8 | 74M |
| small | 12 | 12 | 768 | 12 | 244M |
| medium | 24 | 24 | 1024 | 16 | 769M |
| large-v2 | 32 | 32 | 1280 | 20 | 1550M |

## Key Equation

The encoder's cross-attention at decoder step t:

$$\text{Attn}(Q_t, K_{\text{enc}}, V_{\text{enc}}) = \text{softmax}\left(\frac{Q_t K_{\text{enc}}^T}{\sqrt{d_k}}\right) V_{\text{enc}}$$

where Q_t ∈ R^{1×d_k} is the query from the current decoder state and K_enc, V_enc ∈ R^{1500×d_k} are the projected encoder outputs. This is standard cross-attention — the decoder "reads" any of the 1500 audio frames at each generation step.

## Streaming Inference

Whisper is not natively streaming — the encoder requires a full 30-second chunk. For real-time applications, a sliding window with overlap is used:

1. Buffer 30 seconds of audio.
2. Run Whisper → get transcript.
3. Advance by 15–25 seconds (keeping overlap for context).
4. The overlapping region helps the model correctly transcribe words that span the window boundary.

This adds ~30 seconds of latency inherently. Faster-Whisper and WhisperX implement optimized chunking with voice activity detection (VAD) to skip silent segments and align boundaries to natural speech pauses. For truly low-latency streaming, Whisper-streaming uses a prefix-based approach where partial transcripts from earlier chunks are fed as decoder context.

## Engineering Tradeoffs

| Decision | Gained | Gave up |
|---|---|---|
| Fixed 30s encoder input | Simple fixed-shape architecture; global audio context for ambiguity resolution | Inherent 30s latency; short clips waste compute (zero-padding); boundary artifacts |
| Weak supervision (web data) | 680k hours at near-zero label cost; extreme domain diversity | Noisier labels than human-annotated; hallucination on silent audio; language bias toward English |
| Multitask via special tokens | Single model for 5+ tasks; no task-specific heads | Inference cost same whether you need translation or not; special tokens add to vocabulary |
| Encoder-decoder (not CTC/attention-only) | Flexible output length; natural beam search; context-aware decoding | 2× parameters vs. encoder-only for equivalent encoder size; cross-attention adds latency |
| 1D conv stem (not pure patch embedding) | Mild temporal inductive bias; reduces 3000→1500 frames efficiently | Still not truly streaming; conv weights not reusable from ViT pretraining |
| BPE multilingual vocabulary | Single model covers 99 languages | Vocabulary dilution; minority languages underrepresented; token efficiency lower for non-Latin scripts |

## Experiments & Results

- **English ASR**: Whisper large-v2 achieves 2.7% WER on LibriSpeech clean — competitive with best supervised systems.
- **Robustness**: on difficult benchmarks (CHiME-6, TED-LIUM, CallHome), Whisper matches or outperforms models that were explicitly fine-tuned on those domains — despite Whisper using no domain-specific fine-tuning.
- **Multilingual**: supports 97 languages; average WER on MLS (multilingual LibriSpeech) is competitive with Wav2Vec 2.0 fine-tuned per language.
- **Translation**: speech-to-English translation quality is competitive with cascaded ASR+MT systems on CoVoST-2.

## Commentary

Whisper's architecture is deliberately boring, and that is the right call. The encoder-decoder transformer with cross-attention was already well-understood infrastructure; Whisper put essentially zero innovation into the model architecture and put all the innovation into data scale and collection. The lesson from CLIP applies here too: **past a certain compute budget, the bottleneck in specialized modalities (vision, speech) is data scale and diversity, not model architecture**. An exotic custom ASR architecture trained on 10k hours will lose to a standard transformer trained on 680k hours.

The 30-second chunking constraint is the dominant infrastructure concern for production deployment. Real applications rarely want 30-second latency. The solutions (VAD segmentation, streaming overlapping windows, speculative chunking) all introduce their own complexity and potential errors at chunk boundaries. Future ASR architectures will likely adopt streaming-compatible designs (causal encoders, chunk-wise processing) — the Whisper large model already serves as a teacher for distilling streaming students.

The multitask framing via special tokens is worth emphasizing for infrastructure engineers. Rather than building separate models for transcription, translation, and timestamp prediction, Whisper handles all three with a single model and vocabulary of control tokens. This is essentially prompt engineering at training time — the same philosophy used in instruction-tuned LLMs. The implication is that you can add new Whisper "tasks" by fine-tuning with new special tokens without changing the architecture, which is far cheaper than training a new model from scratch.

## References

- [1] Radford et al. _Robust Speech Recognition via Large-Scale Weak Supervision._ arXiv:2212.04356, 2022.
- [2] Bain et al. _WhisperX: Time-Accurate Speech Transcription of Long-Form Audio._ arXiv:2303.00747, 2023.
- [3] Peng et al. _Owsm v3.1: Better and Faster Open Whisper-Style Speech Models Based on E-Branchformer._ arXiv:2401.16658, 2024.
- [4] Related: [ViT](../vit/en.md), [CLIP](../clip/en.md)
