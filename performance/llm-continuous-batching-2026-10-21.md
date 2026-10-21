# LLM continuous batching (2026-10-21)

The embedded LLM used one `llama_context` per slot (`n_seq_max=1`), so every
concurrent request ran its own per-token decode and re-read all the weights.
Decode is bandwidth-bound, so that throws away almost all of the GPU. Prod
also admitted only 2 LLM requests at a time: `LLM_PERMITS=2` is a weight in
the `SLOTS=8` pool, not a concurrency limit, and there were 2 contexts.

This change runs every request as a sequence in one context, with one engine
thread doing the decoding. It borrows four ideas from DeepInfer
(`../DeepInfer`):

| DeepInfer | Here |
| --- | --- |
| Zero-overhead scheduling: CPU post-processing overlaps the GPU | Request threads assemble text, check stop strings and stream from a per-sequence token ring while the engine is already running the next step. A full ring drops the sequence out of the batch; it never blocks the step. |
| Sort-free sampler (radix-histogram top-k, `DEEPINFER_SAMPLER_FORCE_TOPK`) | The row is cut to top_k + penalty window with a 12-bit radix histogram plus a gather, then the request's own llama.cpp sampler chain runs on that set. The cut is exact whenever the chain can only lower logits (penalties ≥ neutral, no DRY, no min_probability); otherwise it samples the full vocabulary. |
| Prefill budget; only the tail prompt is chunked | FIFO prefill fills `LLM_STEP_TOKENS` (512) after the decode rows; a quarter of the budget is always kept for prefill. |
| Forks with refcounted blocks | A new request forks the longest computed prefix held by any sequence (`llama_memory_seq_cp`; with a unified cache that is cell metadata only). Busy sequences advertise their prompt as prefill lands. The source is pinned until the copy runs. |

The KV cache is unified. With per-sequence streams, llama.cpp's
`split_equal` gives every non-consecutive seq_id its own ubatch, which
removes the batching.

## Setup

- Host: RTX 3090 Ti 24 GiB (sm_86), llama.cpp `571d0d5` (the prod commit),
  built for sm_86 and sm_120.
- Model: `ggml-org/gemma-4-E4B-it-Q8_0`. It has the same architecture and
  size class as prod's `gemma-roleplay-v2-q8_0` (gemma4, 42 layers, 8.0 GB
  file). The prod file could not be copied over the prod uplink.
- Config: NGL=99, CTX=4096, KV q8_0, BATCH 1024, UBATCH 256, guard off.
- Workload: `scripts/llm_throughput_bench.py` sends roleplay-shaped streaming
  chats: the same ~400-token persona with different turns, 200 max tokens,
  temperature 0.8.
- Caveat: another session ran GPU jobs on this card for most of the day. The
  table uses runs with no foreign compute process (or the trace runs), and
  contended runs were discarded. Ordering held in every contended run too.

## Throughput

Aggregate output tok/s; stream tok/s is the per-request decode rate.

| Config | Streams | Aggregate tok/s | Stream tok/s | TTFT p50 / p95 |
| --- | ---: | ---: | ---: | --- |
| legacy, 2 contexts (prod config) | 1 | 97–103 | 99–105 | 27 / 29 ms |
| legacy, 2 contexts | 2 | 122 | 64 | 41 / 106 ms |
| legacy, 2 contexts | 8 | 58–62 | 28–31 | 14 / 28 s (queued) |
| legacy, 2 contexts | 16 | 43–49 | 22–24 | 25 / 29 s, half time out (503) |
| legacy, 8 contexts | 8 | 81 | 11 | 0.2 / 0.8 s |
| **batched, 16 seqs** | 1 | **104** | 106 | 24 / 25 ms |
| batched, 16 seqs | 4 | 280–286 | 87 | 36–39 / 52–56 ms |
| batched, 16 seqs | 8 | 237–472 | 37–69 | 68–75 / 84–286 ms |
| batched, 16 seqs | 16 | **562–660** | 47–53 | 47–94 / 88–125 ms |
| batched, 32 seqs | 32 | **1077** | 43 | 63 / 378 ms |

At 16 streams that is 4.6–5.4x the prod configuration's best aggregate and about
13x what it actually delivers under that load, where it queues and sheds
requests. Single-stream decode is no slower.

The single-stream number needed the sampler work. The first engine sampled
by running the chain over a full 262k-entry candidate array, which took
2.3 ms per row against 9.3 ms of GPU decode, and single-stream fell to
85 tok/s. The radix cut brings that to 0.35 ms (`LLM_ENGINE_TRACE=1`):

| | GPU decode / step | CPU sample / step |
| --- | ---: | ---: |
| 1 row, full-vocab sampling | 9.3 ms | 2.3 ms |
| 1 row, radix cut | 9.2 ms | 0.34 ms |
| 16 rows, radix cut, 8 sampling threads | 19.9 ms | 1.05 ms |

Sixteen rows cost 2.2x one row on the GPU, which is the whole case for
batching.

## Prefix forks

Five concurrent requests with the same persona and different turns:
1,617 prompt tokens were forked instead of prefilled, and the tokens computed
fell from 2,516 to 899. Each forked request finished in 1.39 s against
1.69–1.94 s without forking. At 16 streams the forks mainly help the tail:
TTFT p95 was 79–187 ms with forks against 1.3–4.9 s without, across three
paired runs.

## VRAM (exact, from llama.cpp's buffer logs; weights 4,788 MiB in all cases)

| Config | KV | Compute | Total beyond weights |
| --- | ---: | ---: | ---: |
| legacy, 2 contexts | 2 × 119 MiB | 2 × 279 MiB | 797 MiB |
| batched, 16 seqs | 1,904 MiB | 356 MiB | 2,260 MiB (+1.43 GiB) |
| batched, 32 seqs | 3,808 MiB | 676 MiB | 4,484 MiB (+3.6 GiB) |

Most of the KV is the full-size sliding-window cache (`swa_full`), which
prefix reuse and forks need. `LLM_SWA_FULL=0` cuts it to about
n_seq × (512 + ubatch) cells but turns off forks; reuse of the same sequence
then only lands when the window cells survived. Prod's 5090 had about
6.6 GiB free on 2026-10-21, so 16 sequences fit and 32 do not.

## Quality

`scripts/llm_batch_parity.py` sends 16 prompts (12 graded) at temperature 0.

- Batched engine, one request at a time, against legacy: **16/16
  byte-identical**.
- Batched with 6 filler streams against itself sequentially: same accuracy
  (8/12 both). 10/16 are byte-identical; the rest diverge late (mean shared
  prefix 0.83) into fluent near-synonyms. A row's logits depend in their last
  bits on the batch it was decoded with (wider matmuls sum in a different
  order), the same effect the speculation and prefix-cache sections of
  `quality.md` record. DeepInfer only removes it with batch-invariant
  kernels.
- Radix cut against the full vocabulary, seeded temperature 0.9 with top_p
  and top_k, plain and with repetition/frequency/presence penalties: **32/32
  byte-identical**. The cut is exact, not an approximation.
- Forked prefix against self-prefilled: replies use persona details that
  exist only in the forked cells. Divergence points match those of the
  never-forked control request, i.e. batch composition, not cache content.

## Rollout (5090)

1. Build against prod's llama.cpp (already sm_120); the binary is
   source-compatible with the 20261014 deploy tree.
2. Install `systemd/omniserve-native-llm-batching.conf`: `LLM_CONTEXTS=16`,
   `LLM_PERMITS=1` (8 concurrent LLM requests in the existing `SLOTS=8`
   pool). Raise `SLOTS` only after checking the other modalities' permits,
   because it widens their admission too.
3. Watch `/status.batching` (rows/steps is the mean batch), `/metrics`
   `omniserve_llm_engine_*`, and VRAM headroom for the co-resident image
   lanes.
4. Rollback: remove the drop-in, or set `OMNISERVE_NATIVE_LLM_BATCHED=0`.

## Not done

- 5090 numbers. Prod is the only 5090 and has no room for a second copy of
  the model beside the live services. The 3090 Ti results transfer by
  mechanism (bandwidth-bound decode), not by measurement.
- GPU-side sampling (llama.cpp backend samplers) to avoid copying 1 MB of
  logits per row to the host. At 16 rows the copy and the sampling together
  are about 5% of a step.
- Speculative decoding inside the batched engine. The ggml-org E4B MTP head
  (`mtp-gemma-4-E4B-it-Q8_0`, 94 MiB) is on disk. DeepInfer reports MTP only
  pays at ≥48 sequences per rank, so the priority is single-stream latency,
  not throughput.
- Preemption when the unified cache runs out. Cells are reserved at
  `CTX × LLM_CONTEXTS`, so it cannot run out.
