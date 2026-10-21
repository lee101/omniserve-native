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

## Prod (RTX 5090, 2026-10-21)

Deployed as `deploy-prod-20261021-llmbatch7` from branch `llm-batching-prod`
(prod's `penalty-window` commit `a420862` plus this work), with
`systemd/omniserve-native-llm-batching.conf`. Same bench, same box, minutes
apart, live co-tenants on the GPU; old prod is `deploy-prod-20261014-dry` with
its drop-ins (2 contexts, `LLM_PERMITS=2`, `SLOTS=8`):

| Streams | Old prod tok/s | Batched tok/s | Gain | TTFT p50 old -> new |
| ---: | ---: | ---: | ---: | --- |
| 1 | 78 | 142-151 | 1.9x | 209 -> 15-31 ms |
| 4 | 105 | 394 | 3.8x | 981 -> 30 ms |
| 8 | 130 | 621-642 | 4.9x | 2481 -> 32 ms |
| 16 | 143 | 798-901 | 6.3x | 5590 -> 37 ms |
| 32 | (143) | 1085-1174 | 8.2x | -> 210-230 ms |

What the 5090 needed beyond the 3090 Ti work (its host is a 2016 Broadwell
Xeon on PCIe 3.0, so CPU per token matters more than GPU):

- DRY was on in prod (`DRY_MULTIPLIER=0.8`). `llama_sampler_init_dry`
  detokenizes the whole 262k vocabulary once per sequence breaker, ~180 ms per
  request: that was nearly all of prod's 200 ms TTFT. It is now built once per
  parameter set and cloned.
- The top-k cut used to fall back to the full vocabulary under DRY. Penalties
  and DRY only lower logits of tokens already in the history, so the cut is
  top_k + distinct history tokens + max_new, still exact (32/32 seeded
  outputs identical to full-vocab sampling with prod's DRY settings).
- The radix histogram gather (scatter-heavy) became a sampled-threshold
  compare pass: 0.39 -> 0.14 ms per row here, sampling 3.95 -> 0.9 ms per
  step at ~13 rows on prod. One sampling thread per sequence up to 16.
- Background-tier LLM requests took the whole admission pool (one at a time,
  blocking everything); they now take `LLM_BACKGROUND_PERMITS` and batch.
- `SLOTS` 8 -> 32 with every other modality's permits scaled 4x, so only the
  LLM's admission widened; `WORKERS=96` since each stream holds a worker.
- KV pool decoupled from the sequence count (`LLM_KV_SEQS`): 32 sequences
  share 6 contexts' worth of cells (714 MiB KV + 305 MiB compute, vs 797 MiB
  for the old 2 contexts). No evictions in the 32-stream bench: forked
  personas share cells.

Prod engine trace at ~13 rows: steady 12.8 ms decode, 25-29% of steps
"reshaped" (row count or KV size changed, so llama.cpp runs them without a
CUDA graph, +3.5-4.5 ms), sampling ~0.9 ms. At 32 rows ~2 ms of each step is
the logits copy over PCIe 3.0.

Image lane: the broker refuses z-image requests needing 4096 MB headroom
because a 3584 MB lease is held once the diffusion model is loaded (it looks
like resident weights counted against free memory twice); the old process
had already logged 10,868 such denials in 6 days. The batched LLM adds
~220 MiB over the old 2 contexts.

## Image lane, diffusion and overflow (2026-10-21, later)

Image VRAM accounting. Prod refused most background images (10,868 denials
in the 6 days before): BiRefNet leased 3,584 MB with no pid, so the broker
never credited the weights it had already loaded and charged them twice; the
resident Z-Image path asked for 4,096 MB of headroom although a 1024x1024
generation grows the gateway by 1,666 MB (measured on the 5090); and the point
check re-demanded `SD_MIN_FREE_MB` after a lease had already reserved the
space. Fixed: workers lease with their pid, a granted lease skips the point
check, and the workspace is 2,304 MB scaled by pixels
(`systemd/omniserve-native-image-lease.conf`). Free- and background-tier
images now succeed right after a restart; broker denials since: 0.

Diffusion batching does not pay here. Z-Image Turbo (6B DiT, Q4_K) is
compute-bound even at 512x512 (1,024 tokens, ~12 TFLOP per step against
~3.5 GB of weights): `n=2` and two concurrent requests both take exactly 2x
one image (3090 Ti: 512^2 2.27 -> 4.57 s, 1024^2 10.2 -> 20.4 s), and the
denoise runs at ~60 TFLOPS sustained. Breakdown at 1024^2: conditioning
0.02 s, 9 steps 8.48 s, tiled VAE decode 1.50 s. Measured levers:

| Change | 512^2 | 1024^2 | Cost |
| --- | ---: | ---: | --- |
| sd.cpp with CUDA graphs | -6% | -4% | none, bit-identical; deployed |
| Q8_0 DiT instead of Q4_K | -8% | -6% | +2.7 GB VRAM; not deployed |
| untiled VAE (or 128 tiles) | | -8% (1.50 -> 0.68 s decode) | +7.2 GB peak; not deployed |
| direct-conv VAE | +134% | +275% | slower; off |

On the 5090 the graphs build measured within noise (512^2 1.42 vs 1.45 s).
More image throughput means more GPUs, i.e. overflow.

LLM CUDA graphs. `runtime/llama-cuda-graph-key.patch` keys llama.cpp's CUDA
graph cache on the graph's shapes and pointers instead of its first node, so
each batch shape keeps its own captured graph: 0 warm-up resets in a
16-stream run (was one per row-count change). Reshaped steps still cost ~6 ms
on prod because llama.cpp rebuilds and re-splits the host graph when the
ubatch shape changes; caching graph results per shape is the next step. Prod
links a private patched build (`/sdb-disk/code/llama.cpp-omniserve`), so the
shared llama.cpp other services use is untouched.

Overflow. When 32 local sequences are busy (or no slot is free) the gateway
relays chat/completions for `LLM_OVERFLOW_TIERS` to `LLM_OVERFLOW_UPSTREAM`,
the `tools/llm_overflow.py` sidecar (the gateway proxy has no TLS); a relay
that fails before any bytes falls back to the local queue. The sidecar routes
to a dedicated RunPod pod while one has capacity, else to a scale-to-zero
load-balancing serverless endpoint, and starts a pod only for sustained
overflow (see `performance/LLM_OVERFLOW.md`). Workers run `Dockerfile.llm`:
the same binary, LLM-only, KV pool auto-sized to the GPU (`LLM_KV_SEQS=auto`),
model pulled from a private HF repo at start.

## Not done

- 5090 numbers. Prod is the only 5090 and has no room for a second copy of
  the model beside the live services. The 3090 Ti results transfer by
  mechanism (bandwidth-bound decode), not by measurement.
- GPU-side top-k (llama.cpp backend samplers) to avoid copying 1 MB of
  logits per row over PCIe 3.0. With CUDA 12.9's CCCL there is no
  `DeviceTopK`, so each row would pay a full 262k argsort, about what the copy
  costs; worth it with CUDA 13.
- Fewer graph-less steps: llama.cpp keeps one CUDA graph and re-warms it on
  every shape change.
- Speculative decoding inside the batched engine. The ggml-org E4B MTP head
  (`mtp-gemma-4-E4B-it-Q8_0`, 94 MiB) is on disk. DeepInfer reports MTP only
  pays at ≥48 sequences per rank, so the priority is single-stream latency,
  not throughput.
- Preemption when the unified cache runs out. Cells are reserved at
  `CTX × LLM_CONTEXTS`, so it cannot run out.
