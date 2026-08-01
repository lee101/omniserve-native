# Model Quality Report

Produced by `./scripts/quality_bench.sh <port>` (`tools/quality_bench.py`).
`performance/native-core.md` covers transport cost; this file covers whether
the served models answer correctly. Every number below was measured on this
host (RTX 5090, driver 595.84, llama.cpp b0.17.0) through the C gateway.

## Embedding artifact selection

Graded on paraphrase-vs-distractor sentence pairs: `sts_margin` is
`mean(cos) positives − mean(cos) negatives`, `ranking_accuracy` is the fraction
of the 25 positive/negative pair comparisons ordered correctly. 0.5 is chance.

| Artifact | Pooling | Dims | STS margin | Ranking accuracy | Warm latency |
| --- | --- | --- | ---: | ---: | ---: |
| `modernbert-base-q8_0` (MLM trunk) | mean | 768 | +0.0074 | 0.40 | 33 ms |
| `gte-modernbert-base-q8_0` | cls | 768 | **+0.2229** | **1.00** | 61 ms |
| `gte-modernbert-base-q8_0` | mean | 768 | +0.2476 | 0.96 | 33 ms |

`ModernBERT-base` is a masked-language-model trunk with no sentence-level
training objective. Mean-pooled, its vectors are dominated by a common
direction: every pair scores ≈0.96 regardless of meaning, so ranking is at
chance and any retrieval built on it is effectively random. Removing that
direction recovers a usable margin without changing the model — measured on the
same live endpoint:

| Post-processing | pos | neg | margin | ranking accuracy |
| --- | ---: | ---: | ---: | ---: |
| raw | 0.9657 | 0.9583 | +0.0074 | 0.40 |
| mean-centered | 0.5834 | 0.4768 | +0.1066 | 0.48 |
| dominant direction removed | 0.5861 | 0.4782 | +0.1079 | 0.48 |
| centered + z-scored | 0.5681 | 0.4258 | +0.1422 | 0.64 |

Post-processing alone does not reach usable ranking accuracy. Swapping to a
retrieval finetune of the same architecture does, at the same 768 dimensions,
the same 160 MB q8_0 artifact size, and the same runtime — so the index
dimension and the C encoder path are unchanged:

```bash
./scripts/convert_models.sh gte
OMNISERVE_NATIVE_EMBEDDING_GGUF=/nvme0n1-disk/models/omniserve-native/gte-modernbert-base-q8_0.gguf \
OMNISERVE_NATIVE_EMBEDDING_POOLING=cls
```

CLS pooling matches the checkpoint's training objective and is the recommended
setting; mean pooling on the same weights scores a marginally wider raw margin
but misorders one pair. Existing stored vectors are not comparable across the
swap and must be reindexed.

## KV cache quantization

Same prompts, same seeds, `qwen3-0.6b-q8` at `n_ctx=8192`, two contexts:

| `OMNISERVE_NATIVE_KV_TYPE` | KV buffer per context | Task accuracy | Cold determinism | Prefix agreement |
| --- | ---: | ---: | --- | ---: |
| `f16` (default) | 896 MiB | 0.375 | identical | 0.269 |
| `q8_0` | 476 MiB | 0.375 | identical | 0.269 |

q8_0 halves the KV cache with no measured quality change — identical accuracy,
identical contract checks, and the same greedy divergence point. Flash
attention is enabled automatically for a quantized cache because llama.cpp
requires it for a quantized V.

This table is why the `ampere` profile in `src/otune.c` now defaults to `q8_0`
as well. Two facts carry it across: the numbers above say the swap is
quality-neutral, and llama.cpp compiles `FATTN_VEC_CASES_ALL_D(Q8_0, Q8_0)`
outside the `GGML_CUDA_FA_ALL_QUANTS` guard, so the kernel exists on every
architecture with flash attention rather than only on the newer ones. What is
*not* claimed is a measurement on Ampere silicon — this host has a 5090 and no
3090 — so the accuracy claim is transferred from the table and the speed claim
is an inference from a 3090 being bandwidth-bound at decode. Re-run this bench
on a 3090 before treating either as measured there. On Gemma 4 the win is smaller in absolute terms
because sliding-window attention already keeps the cache small (448 MiB per
context at 8192), but it still buys roughly one extra parallel context per
GiB freed.

## Speculative decoding

`OMNISERVE_NATIVE_SPEC_DRAFT=k` drafts up to `k` tokens per round from the
context itself — the longest suffix of what has been written so far is looked up
in what came before, and whatever followed it last time becomes the guess — then
confirms them all in one decode of width `k+1`. Off by default; see the README
for why.

It is not a quality setting. Every token is sampled from the real model's
distribution at its own position, and a drafted token is only kept when the
sampler independently chose the same id, so nothing is ever emitted because the
drafter proposed it. What that buys is the right to reuse logits that were
computed anyway instead of running the model again.

The plumbing was verified by construction rather than by inspection, on CPU with
`qwen3-0.6b-q8` at temperature 0:

| Arm | Configuration | Result |
| --- | --- | --- |
| `base` | speculation off | reference |
| `never` | on, `MIN_NGRAM=999` so no draft is ever found | **byte-identical to base** |
| `loose` | on, `MIN_NGRAM=1` so nearly every round drafts and nearly every draft is rejected | coherent throughout |
| `spec` | on, shipping settings | copy-heavy prompt byte-identical; creative prompt diverged at char 217/348, stayed fluent |

`never` matching `base` exactly is what rules out a bug in the restructured
decode loop: every round took the same single-token path the old code took.
`loose` is the hardest test of the KV trim, because a rejected draft has to be
removed from the cache before the next round attends to it — text after a
rejection stays coherent, which a mis-trimmed cache would not produce.

### How much it lands

Acceptance is a property of the model and the prompt, so it is measurable on CPU
even though the speedup is not. `qwen3-0.6b-q8`, draft 4, 420 generated tokens
per row:

| Workload | Drafted | Accepted | Acceptance | Tokens that cost no model call |
| --- | ---: | ---: | ---: | ---: |
| copy-heavy (repeat / fix-a-typo / extract-then-repeat) | 98 | 65 | 0.66 | **15%** |
| open-ended (poem, invent a name, describe a planet) | 50 | 28 | 0.56 | **7%** |

That is the honest ceiling for drafting out of the context alone: it is free in
VRAM, and it is weak. 15% fewer model calls on the workloads it suits — which
are the shapes `/api/v1/summarization` and `/api/v1/autocomplete` actually see —
is worth having on a GPU and is nowhere near the 2-3x a draft model delivers. A
draft model is the upgrade, and it needs a VRAM lease before it can hold
anything, which is why the broker came first.

Wall clock on CPU for the same run: 20.4 tok/s off, 19.5 tok/s on. The calls
were saved and the seconds were not, because CPU decode is compute-bound and the
wider verify batch costs what it saves. This is the measurement that keeps
speculation off by default.

**Greedy output is not bit-reproducible when speculation fires**, and the
obvious claim that it should be is wrong on real hardware. Verifying `k` drafts
is one matmul of width `k+1` where there would have been `k+1` of width 1, and a
wider matmul sums in a different order; the logits differ in their last bits and
a near-tied argmax lands on the other token. The copy-heavy prompt above came
out identical because its logits are not near-tied; the poem diverged because
they are. This is the same effect as `llm.prefix_cache_agreement` below, and it
is why that is tracked as a score rather than asserted as determinism.

## Prefix cache and determinism

`llm.cold_determinism` passes: two full prefills of the same prompt at
temperature 0 are byte-identical.

`llm.prefix_cache_agreement` is 0.269 — an identical request served off a
reused KV prefix diverges from the cold answer at character 83 of 309. Both
continuations are valid; re-decoding one token against cached keys is not
bit-identical to a chunked prefill, so greedy near-ties flip. Confirmed by
construction: busting the cache between two calls reproduces the cold answer
exactly, while an immediately repeated call does not.

Consequence for callers: **temperature 0 is not reproducible across cache
states.** Anything that depends on byte-stable output (eval snapshots,
response caching keyed on output, golden tests) has to bust the prefix or
tolerate drift. The bench tracks this as a score rather than a failure, and
hard-fails only if agreement collapses below 0.05, which would mean the reuse
offset itself is wrong.

## Baseline gating

`tools/quality_bench.py` compares every score against
`performance/quality-baseline.json` and exits non-zero on regression beyond
`--tolerance` (default 5%; latencies get 25% because they are noisy). Record a
baseline against the production configuration once, then gate on it:

```bash
./scripts/quality_bench.sh 8791 --update-baseline   # record
./scripts/quality_bench.sh 8791                     # gate
```

The stored baseline also holds reference embedding vectors, so
`embedding.drift_cosine` catches a requantization or backend change that
silently moves the vector space.
