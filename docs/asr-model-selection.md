# Choosing the ASR model

All numbers below were measured with `tools/asr_model_bench.py` on DictatorFlow's
nine-clip corpus (`e2e/audio/ground_truth.json`), RTX 5090, greedy decoding, one
canonical normalizer shared with `tools/wer_bench.py`.

| Model | WER | Errors / 115 words | RTF |
|-------|-----|--------------------|-----|
| **openai/whisper-small.en** | **1.74%** | 2 | 0.18 |
| openai/whisper-medium.en | 3.48% | 4 | 0.25 |
| distil-whisper/distil-small.en | 6.09% | 7 | 0.18 |
| openai/whisper-base.en | 6.96% | 8 | 0.14 |
| distil-whisper/distil-medium.en | 12.17% | 14 | 0.13 |
| openai/whisper-tiny.en | 16.52% | 19 | 0.17 |
| nvidia/parakeet-ctc-0.6b | 100% | 115 | 0.46 |
| VibeVoice-ASR-BitNet (DictatorFlow local, Xeon CPU) | 60.45% | — | 1.53 |

`whisper-small.en` is the default. Its two remaining errors are ordinary
acoustic confusions (`using` -> `use`, `jumps` -> `dumps`), not a systematic
fault.

## Read this before trusting the table

**The corpus is 115 reference words.** One word is worth ~0.9% WER, so
differences of a few points are noise. That whisper-small beat whisper-medium
here (2 errors vs 4) is not evidence that small is the better model -- it is
evidence the corpus is too small to rank close models. Treat this table as
"which models are in the right league", and grow the corpus before making finer
calls. The same effect shows in the historical DictatorFlow runs, where an
unchanged Gemini path scored between 2% and 12% across repeat runs.

## Why parakeet scored 100%

`nvidia/parakeet-ctc-0.6b` was the previous default. Through the transformers
`automatic-speech-recognition` pipeline it decodes every frame to `<unk>` and
returns HTTP 200 -- a plausible-looking response that is entirely wrong. It
needs NeMo, not this code path. Passing `chunk_length_s` makes it raise
`'ParakeetCTCConfig' object has no attribute 'inputs_to_logits_ratio'` instead.

This is the failure mode worth designing against: not a crash, but a confident
wrong answer. `is_degenerate()` in `workers/asr_worker.py` now rejects
all-unknown-token transcripts with a 502 rather than serving them, and
`asr_model_bench.py` reports `n/a` instead of a WER when a backend scores no
clips at all -- it previously printed a flattering `0.0000`.

## Audio instruction models

`--backend name=audiolm:<model-id>` runs an audio-capable instruction model
(Gemma 3n, Qwen2-Audio, Phi-4-multimodal) with a plain "transcribe this audio
verbatim" prompt, and `--backend name=gemini:<model>` does the same against the
Gemini API.

Neither is measured here yet:

- `google/gemma-3n-E2B-it` is a gated repo and the access request on this
  account is still awaiting approval.
- Qwen2-Audio-7B needs ~14 GB and the box had 3.7 GB free.
- The Gemini key in the environment returns `API_KEY_INVALID`.

Historical DictatorFlow runs do show Gemini transcribing this corpus at 0% WER
on most clips, so the prompt approach clearly works; it is a cost and latency
question rather than an accuracy one. Re-run the bench once access is sorted.
