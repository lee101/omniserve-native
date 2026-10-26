# consolidate-20261026 and llama.cpp upgrade (2026-10-26)

Branch `consolidate-20261026`: llm-optim-20261023, perf/agent-1024, llm-continuous-batching
(engine already in main; only docs unmerged, skipped), prod box main + uncommitted diff, local WIP.

llama.cpp: `llama.cpp-latest` (branch omni-latest-20261026) = upstream d854d98ba (2026-10-26, 1821 commits
past 571d0d540) + two omni patches: `GGML_CUDA_STREAM_PRIORITY`/per-shape CUDA graph key, and 2 rows per
mmvq block at one column for Q8_0 (upstream already carries equivalents of the rms_norm register-row patch).
API port in `src/backend_llama.c` (`LLAMA_DRY_MAX_SEQ_BREAKERS` selects the new signatures):
`llama_sampler_init_penalties(n_vocab, ...)`; `llama_sampler_init_dry(vocab, ...)` without n_ctx_train, and a
negative window now disables DRY, so the window is n_ctx_train.

## Method
RTX 3090 Ti (sm_86, not the 5090), shared at 100% utilisation by unrelated jobs during the whole session, so
wall clock varied up to 7x between runs. Contention-robust numbers: GPU kernel time per generated token from
nsys (sum of per-kernel median x instances). gemma-4-E4B-it-Q8_0 + MTP head, q8_0 KV, prod batching env.

| GPU kernel us / generated token | old (o1c libs) | latest | latest, LLAMA_ATTN_ROT_DISABLE=1 |
| --- | ---: | ---: | ---: |
| server, MTP off | 4044 | 4105 | |
| server, MTP on | 2271 | 2241 | 2239 |
| llama-bench tg128 (batch 1) | 5250 | 5174 | 5048 |
| llama-bench pp512 per run (ms) | 41.6 | 41.5 | 40.5 |

Kernels per token fall 4-5% (fusions) but decode is bandwidth bound, so GPU time is unchanged within 2%.
Best quiet-window llama-bench samples: pp512 7736 -> 8280 t/s (+7%), tg128 107.0 -> 107.7, pp3 336 -> 330.

Server, MTP on, aggregate tok/s at 1/4/8 streams, best quiet-window sample per variant (single windows, +-10%):
old 145/355/507, merge only on old libs 144/322/509, latest 133/361/476, latest rot off 153/393/543.
TTFT p50 at 1 stream 13-15 ms in all. Prefix path (3052-token prompt): cold 520-620 ms, warm 22-27 ms, same
in all builds. MTP acceptance 0.67-0.68 per draft in all builds.

## Parity (16 greedy prompts, sequential, MTP on)
- old vs merge on old libs: 16/16 identical.
- old vs latest: 14/16 identical, the other two diverge late (about 0.99 mean common prefix). Not bit-exact:
  upstream changed FA/MMQ kernels and the attention-rotation path. Task accuracy equal (0.667 on the 12 graded).
- With `LLAMA_ATTN_ROT_DISABLE=1` 11/16 (it changes KV q8_0 numerics; prod runs with rotation on, so it stays on).
- Old vs old with MTP on vs off already differ on 3/16 (batch-width fp noise), the same order as old vs latest.

## Tests
ctest 29/31 pass on the merged tree with both llama trees and the prod sd.cpp header; guard_holdout and
guard_multiturn need gitignored corpora (tests/data/guard-holdout.json, guard-multiturn.json) that exist only
on the box. The core suite needs `ONATIVE_TEST_PORT` when 18792 is busy. The ci preset (-Werror) builds after the
obroker.c buffer fix.

## Not done
- No sm_120 hardware here: the latest libs contain sm_120a SASS but were run only on sm_86.
- `LLAMA_ATTN_ROT_DISABLE=1` is about 2.5% less decode GPU time; left off to keep prod numerics.
