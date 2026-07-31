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
