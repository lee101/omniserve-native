---
language:
- en
license: cc-by-4.0
library_name: transformers
pipeline_tag: automatic-speech-recognition
base_model: nvidia/parakeet-ctc-0.6b
metrics:
- wer
tags:
- dictatorflow
- parakeet
- asr
---

# DictatorFlow Parakeet CTC variant

This is a fine-tuned variant of `nvidia/parakeet-ctc-0.6b`.

## Intended use

Low-latency English desktop dictation. It is not validated for medical, legal,
emergency, or safety-critical transcription.

## Training data

Trained on privately held speech whose speakers explicitly opted in to public
model weights and confirmed they had the right to contribute the speech.
Neither the recordings, transcripts, nor private manifest are published.

Before release, replace this paragraph with aggregate hours, speaker count,
collection dates, filtering, deletion/revocation cutoff, and the exact consent
version. Do not include identifying examples.

## Evaluation

Replace with candidate and base-model WER on:

- A speaker-disjoint held-out dictation set.
- At least one appropriately licensed public benchmark.
- Accented, noisy, short-command, and code/proper-noun slices.

Report normalization rules, sample counts, confidence intervals, and inference
hardware. A same-speaker training holdout is useful for iteration but is not a
release-quality generalization result.

## Limitations and risks

Speech models can produce omissions, substitutions, biased errors, and
confidently incorrect text. Fine-tuning can memorize rare phrases; release
review includes privacy tests and manual inspection for memorization.

## Base-model attribution

Derived from NVIDIA's `nvidia/parakeet-ctc-0.6b`, licensed CC BY 4.0. Preserve
the base model's attribution and document all modifications.
