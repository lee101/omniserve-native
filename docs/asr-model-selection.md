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

**The corpus is 115 reference words, and the statistics say so.** Running the
bench with confidence intervals and a paired bootstrap against the leader:

| Backend | WER | 95% CI | vs leader |
|---------|-----|--------|-----------|
| whisper-small.en | 0.0174 | [0.000, 0.046] | leader |
| rover (4 systems) | 0.0174 | [0.000, 0.037] | p=1.000 |
| whisper-medium.en | 0.0348 | [0.000, 0.111] | p=0.720 |
| distil-small.en | 0.0609 | [0.012, 0.148] | p=0.254 |
| whisper-base.en | 0.0696 | [0.014, 0.169] | p=0.194 |

**Not one of these differences is significant.** whisper-small looks four times
better than whisper-base, and the paired bootstrap still cannot separate them
(p=0.194). The intervals are enormous because nine clips is nine independent
observations, whatever the word count suggests.

What that means in practice:

- The defensible claims from this corpus are the *large* ones: parakeet at 100%
  and VibeVoice at 60% are unambiguously broken/bad. Everything in the 1-7% band
  is one undifferentiated group.
- Do not re-rank the default on a two-error swing. Grow the corpus first --
  roughly 100+ clips before differences of a couple of points become detectable.
- The same effect is visible in DictatorFlow's historical runs, where an
  unchanged Gemini path scored between 2% and 12% across repeats. That spread
  was always noise; there was simply nothing reporting it as such.

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

## The statistical layer

`tools/asr_stats.py`, wired into the bench and covered by `tests/test_asr_stats.py`.

**Bootstrap confidence intervals.** Clips are resampled with replacement, not
words: nine clips are nine independent observations, and resampling words would
treat every word as its own experiment and report an interval several times too
narrow.

**Paired bootstrap.** Comparing two systems scores both on the *same* resampled
clips, so the corpus's shared difficulty cancels and only the between-system
difference is resampled. This is much more sensitive than eyeballing whether two
independent intervals overlap. A result is only called significant when the
p-value clears alpha *and* the difference interval excludes zero.

**ROVER combination.** `rover_combine` aligns every system into a confusion
network and votes per slot, NULL included, so a majority can delete a word as
well as correct one. On this corpus it ties the leader (2 errors) but with a
tighter interval -- [0.000, 0.037] against [0.000, 0.046]. That is the honest
result: voting reduces variance here rather than error, because these systems
share their blind spots. Where systems fail *differently* it does better, and
`test_voting_can_beat_every_input` pins that case.

**Agreement as a reference-free signal.** `agreement_rate` and
`disagreement_spans` run the same network without any ground truth, so they work
on production traffic. Mean agreement on this corpus is 0.939; the slots where
systems argue are where errors concentrate, which makes this usable for flagging
a transcript for review or for routing a low-confidence clip to a bigger model.

## Formatting normalization

`tools/asr_textnorm.py` is an opt-in second profile, reported as `lenient`
alongside the strict number, never instead of it.

It fixes how a word is *written* -- spelled-out numbers to digits, `mp3` to
`mp 3`, contractions expanded, British/American spellings -- and never touches
which word was *heard*. `jumps` -> `dumps` and `using` -> `use` keep counting as
errors, and a test asserts it. A normalizer that quietly absorbs recognition
errors makes every number downstream of it a lie.

The number parser earns its keep on this corpus specifically: `real_numbers.wav`
is referenced as "Testing 1 2 3 4 5 6 7 8 9 10", so a model that spells the
digits out would score near 100% on that clip while being perfectly correct.
Note the subtlety it has to get right -- a run of bare units is a *sequence*
("one two three" = 1 2 3), while "twenty five" is 25 and "two thousand five
hundred" is 2500. A greedy accumulator turns the first case into 6.

On the current table lenient scoring moves WER by ~0.0006, because whisper
already emits digits. It is insurance, not an improvement.
