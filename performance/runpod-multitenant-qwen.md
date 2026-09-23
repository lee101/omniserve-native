# RunPod multitenant Qwen overflow: ra2 + Qwen Image Edit 2511 (2026-09-23)

One serverless worker image serves Qwen Image 2.1 ("ra2") and Qwen Image Edit 2511,
routed by input `task`. It replaces `tlofa06vj7iab7` as the ra2 overflow and adds a
remote backend for `qwen-edit`.

| item | value |
| --- | --- |
| endpoint | `d6opzplchn2w3a` `omniserve-qwen-mt-overflow`, template `8q4mmq15ex` |
| image | `ghcr.io/lee101/omniserve-native:qwen-mt-20260923b` (44 GB compressed, weights baked) |
| GPUs (priority order) | RTX 4090, then L40S / RTX 6000 Ada / L40 |
| scaling | workersMin 0, workersMax 3, idleTimeout 30 s, flashboot on, QUEUE_DELAY 2 s, exec timeout 600 s |
| worker code | `omniserve-native-ra2overflow/workloads/qwen_mt.py` |
| image build | `tools/qwen_mt_layer_builder.py` (in-datacenter weight layers), `tools/qwen_mt_image.sh` (code layer + env) |
| bench | `tools/qwen_mt_bench.py` (same corpus/seeds locally in-process or on an endpoint; SSIM/PSNR vs local) |
| adapter | `tools/runpod_overflow.py`: `model`/`task` in {qwen-edit, qwen-image-edit, qwen-image-edit-2511} -> workload `qwen-edit`; `RUNPOD_MT_ENDPOINTS` get `task` |

## Why the old RA2 endpoint was ~3x slower

Per-stage timings from the old worker's own `timings` plus RunPod delay/execution split,
1024², 20 steps, 2026-09-23:

| run | delay | exec | sample | rest of exec |
| --- | ---: | ---: | ---: | --- |
| old endpoint, fresh worker (4090) | 10.8 s | 55.8 s | 16.4 s | ~39 s: 11 GB HF download inside the first job |
| old endpoint, warm (same 4090) | 0 | 11.4-11.6 s | 10.9 s | encode 0.3 s |
| earlier 34.5 s "warm" / 60-145 s canary | | | | landed on A5000/L4/3090 workers and on fresh workers |

1. No network volume and no baked weights (`networkVolumeId` empty): every new worker
   downloads DiT + encoder + mmproj + VAE from Hugging Face during its first job.
2. GPU class variance: the pool mixed 4090 with A5000/L4/3090 (billed 0.27-0.74 $/h);
   the slow classes are what the 34.5 s warm and 60-145 s parallel numbers measured.
3. Worker settings stricter than local: EasyCache 0.05 (local 0.08) and VAE tiling at
   tile 32 / overlap 0.5 (local untiled when resident).
4. Not a precision gap: local prod and the worker both run the same Q4_K_M GGUF through
   stable-diffusion.cpp. The NVFP4/CuTe work (`cutedsl/research/qwen-edit-5090.md`) is a
   Diffusers path that production does not use; no torch.compile is involved on either side.

Fixes in the new worker: weights baked (sha256-pinned, byte-identical to the local files),
4090-first GPU list without the 24 GB Ampere/L4 classes, EasyCache 0.08 and no VAE tiling
(same as local), both contexts eager-loaded at boot before the first job is taken.

## Worker design

- `task` `ra2` (default): text-to-image, or ra2 reference edit when `image_base64` is present.
  `task` `edit`: Qwen Image Edit 2511 (`qwen_image_zero_cond_t=true`), `image_base64` required,
  20 steps, CFG 2.5, dense (no denoiser cache), flow shift 3.0. `task` `probe`: GPU/VRAM/residency.
- No shared weights are possible: ra2 uses a Qwen3-VL-8B encoder (+ mmproj) and the 2.1 VAE;
  2511 uses Qwen2.5-VL-7B (bf16, 16.6 GB) and the Qwen Image VAE. Each task owns one
  stable-diffusion.cpp context; sharing is at process/GPU level.
- Placement (`plan()`/`ensure()`): keep both resident if possible, preferring GPU text
  encoders for ra2; the fallback puts a text encoder's weights in host RAM (`te=cpu`
  params backend, GPU compute, streamed once per request). If the task does not fit beside
  the other, the least recently used context is freed. Weights are in page cache,
  so a swap costs 6-19 s instead of per-step streaming.
  - 96 GB: both fully on GPU. 48 GB: ra2 on GPU + edit with TE in RAM. 24 GB: ra2 preloaded;
    the first edit evicts ra2 (18.9 s). After that both stay resident with TEs in RAM
    (5.4 GiB free).
- Stage timings (`stages_ms`: condition, sampling, VAE decode) come from the sd.cpp log
  callback in every response. sd.cpp errors are returned instead of a bare failure.
- Library: the base image's sd.cpp `c678dfe`. The local lane runs `6dcb5bb`, which is
  `c678dfe` + 5 upstream bugfix commits with no header/ABI change.

## GPU class benchmark (same corpus/seeds as the local reference)

Reference: local RTX 5090 (shared with prod), same worker code and weights, PNG. Corpus:
3 ra2 prompts at 1024², 20 steps, EasyCache 0.08, seeds 424242/777/90210, plus the two
`qwen-edit-5090.md` instructions (seed 90908, 512², 20 steps, CFG 2.5). Sequence per GPU:
ra2 x3, edit x2, ra2, probe. Serverless $/h is from RunPod endpoint billing.

| GPU | $/h | cold ra2 wall (delay / boot) | warm ra2 exec | $/ra2 warm | edit exec (first / warm) | $/edit warm | ra2 SSIM vs local | edit SSIM | availability |
| --- | ---: | --- | ---: | ---: | --- | ---: | --- | --- | --- |
| RTX 4090 | 1.10 | 38.8 s (28.3 / 17.1) | 8.8-9.5 s | $0.0028 | 48.7 s (ra2 evicted, load 18.9 s) / 22.1 s | $0.0068 | 0.996 / 0.969 / 0.998 | 0.9959 / 0.9998 | ready 6.8 min after create |
| RTX 6000 Ada (48 GB pool) | 1.75 | 96.3 s (84.6 / 36.0) | 10.6-11.3 s | $0.0053 | 24.8 s / 23.8 s | $0.0116 | 0.984 / 0.978 / 0.999 | 0.9955 / 0.9998 | ready 9.2 min |
| RTX PRO 6000 Blackwell | 3.50 | 48.0 s (39.4 / 22.9) | 6.6-7.1 s | $0.0067 | 16.9 s / 16.6 s | $0.0161 | 0.994 / 0.982 / 0.999 | 0.9958 / 0.9999 | ready 5.8 min |
| RTX 5090 | 1.58 | not measured | | | | | | | no worker after 28 min (initializing/throttled) |
| local 5090 (shared) | ~0.10 | | 10.4-23.1 s sample (prod contention; lane p50 10.7 s) | ~$0.0003 | 45-117 s (weights streamed, `*=cpu`) | | reference | reference | |

Quality notes:
- Every repeat is pixel-identical on the same GPU (local and remote).
- Edit (dense) is tight on every class: SSIM >= 0.9955, and remote-to-remote >= 0.9996.
- ra2 prompt 2 (chalkboard sign text) drifts on every class, including remote-to-remote:
  4090 vs Ada 0.948, Ada vs PRO 6000 0.996. The composition and text are unchanged; the chalk
  details differ. This is cross-architecture float drift amplified by EasyCache skip
  decisions, not a lossier variant. Gate used: edit SSIM >= 0.99 (from
  `qwen-edit-offload-plan.md`); ra2 min SSIM >= 0.96 and mean >= 0.98 at the same
  settings, plus visual check. All three measured classes pass.
- Earlier the old endpoint was marked `quality: 1.0` without a measurement and ran EasyCache 0.05;
  the new worker matches the local settings exactly.

Decision: RTX 4090 first. It has the lowest $/image, the fastest cold start, and a warm ra2
render faster than the shared local lane. The 48 GB Ada pool is the fallback for
availability: about 15% slower, higher $/image, and both tasks stay resident with no swaps.
PRO 6000 is 25-35% faster but 2.4-3x the $/image, which does not pay off for overflow.
5090 had no capacity during the test.

## Weights: baked vs network volume

Weights are baked. A network volume pins the endpoint to one datacenter, and 4090 stock is
"Low" everywhere. Every cold worker would also read 11-42 GB over the volume's network
filesystem. The baked image's pull (44 GB) happens while a worker initializes, which is
not billed. Measured on hosts that already had the image: container start plus eager load of
the contexts was 17-36 s of boot. One worker on a fresh host spent 5 min in "loading
container image from cache" before it was ready (unbilled). The volume variant was not
measured: seeding a 42 GB volume needs a paid pod and the DC pin rules it out regardless.
The build host's uplink is about 1 MB/s, so the weight layers were built and pushed from a
RunPod worker (`qwen_mt_layer_builder.py`, 17 min, $0.32). The code layer is pushed from
here on its own (`qwen_mt_image.sh`).

## Not a general multi-workload worker

YuE (torch stack, needs 18 GiB) and Pixal3D (needs 23 GiB, 266 s jobs) are not merged in:

- Adding them would grow the 44 GB image.
- On 24-48 GB cards they would evict the Qwen contexts, and each eviction costs a 6-19 s reload.
- A single-concurrency worker would put 5-min 3D jobs in front of 10 s image jobs.
- Both already have their own endpoints that scale to zero, so merging saves no idle cost.

## idleTimeout

On a 4090, one cold start costs roughly 17-28 s of billed boot, about $0.005-0.009,
plus 30-40 s of user latency. Idling costs $0.0003/s. A 30 s idle window costs $0.009
per burst and makes any follow-up within the burst warm (10 s instead of 39-69 s).
Overflow is bursty (it happens when the local lane is saturated), so 30 s was kept.
With workersMin 0 the endpoint costs nothing at rest.

## Canary (production path, after the switch)

ra2 through the live `:8792` gateway, saturated with 3 paid + 2 sub requests, unique seeds:

| request | backend | wall | RunPod delay / exec |
| --- | --- | ---: | --- |
| paid 1-3 | local (frontier kept local) | 11.6 / 21.0 / 31.2 s | |
| sub 4 | RunPod `d6opzplchn2w3a` | 69.5 s | 58.3 s cold / 10.2 s |
| sub 5 | RunPod `d6opzplchn2w3a` (second worker) | 79.5 s | 68.6 s cold / 10.0 s |
| repeat of paid 1 | local | 10.4 s | full render, not a cache hit (no `cache: true`) |

`/status.overflow.saturated` went 0 -> 2, and ledger rows were written as `ra2 runpod d6opzplchn2w3a sub`.
Both overflows were cold starts on fresh workers (boot plus first eager load from disk).
The render itself took 10 s.

qwen-edit through the overflow adapter (`:18797 /v1/images/edits`, `model:
qwen-image-edit-2511`, paid), served on a warm RTX 6000 Ada worker: 29.2 / 25.0 / 25.5 s
wall (exec 28.1 / 23.8 / 23.9 s). SSIM vs local was 0.9955 / 0.9998 / 0.9998, and the remote
repeat was pixel-identical to the first run. It was a full render: the worker has no result
cache. Ledger rows were written as `qwen-edit`. There were no exact-cache hits in this
canary.

Gateway gap: `:8792` serves any edit locally as a ra2 reference edit, and the C gateway
(owned by another agent) has no `qwen-edit` model route yet. Until one is added, 2511 edits
reach RunPod only when a caller posts to the adapter with an edit model name. A route on
8791 or 8792 that sends `model: qwen-image-edit-2511` to the adapter is enough, because the
adapter already classifies it.

## Frontier / ledger

`frontier/seeds.json` changes:
- ra2 gains `runpod:d6opzplchn2w3a`.
- qwen-edit gains the same endpoint as its first remote candidate.
- `runpod:tlofa06vj7iab7` is marked unavailable (kept for rollback).
- Seed p50 is the cold wall (ra2 38.8 s, edit 80 s), because sparse overflow mostly lands
  on a cold worker. The ledger replaces the seed after 5 jobs.

Current routing: local stays on the frontier for both workloads, and the remote is used
when the local wait exceeds the policy threshold.

Adapter env (`/etc/omniserve-runpod-overflow.env`, backup `.bak-pre-qwen-mt`):

```
RUNPOD_RA2_ENDPOINTS=d6opzplchn2w3a,tlofa06vj7iab7
RUNPOD_MT_ENDPOINTS=d6opzplchn2w3a
RUNPOD_EDIT_ENDPOINTS=d6opzplchn2w3a
```

## Spend

About $2.1 of RunPod time for this work. Billing was not final at write-up, so this is an
estimate from endpoint billing plus measured seconds:
- weight-layer builder $0.32
- 4090 / Ada / PRO 6000 / 5090 benches ~$1.4, including ~$0.9 from a first image that
  crash-looped on a missing `libnccl.so.2` loader path (fixed in tag `b` via
  `LD_LIBRARY_PATH`)
- old-endpoint diagnosis ~$0.03
- canary ~$0.1

The bench endpoints have been deleted. The new endpoint is at workersMin 0.

## Rollback

```
sudo cp /etc/omniserve-runpod-overflow.env.bak-pre-qwen-mt /etc/omniserve-runpod-overflow.env
sudo systemctl restart omniserve-runpod-overflow
curl -sX PATCH https://rest.runpod.io/v1/endpoints/tlofa06vj7iab7 -H "Authorization: Bearer $RUNPOD_API_KEY" \
  -H 'Content-Type: application/json' -d '{"workersMax":3}'
git -C /nvme0n1-disk/code/omniserve-native revert <seeds commit> && python3 -m frontier cycle
```

`tlofa06vj7iab7` was not deleted (it is at workersMax 0). `d6opzplchn2w3a` can be set to
workersMax 0 at any time.

## Follow-ups

- Add a qwen-edit model route in the C gateway (see "Gateway gap").
- Re-run the 5090 arm when there is capacity. It is the class that matches the local
  reference architecture.
- Retune seed p50 from the ledger once 5 or more remote jobs per workload are recorded.
