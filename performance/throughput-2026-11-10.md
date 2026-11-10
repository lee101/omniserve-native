# Throughput pass, 2026-11-10

## Prod image lane (RTX 5090, :8791 embedded Z-Image turbo Q4_K)

24 h of `/var/log/omniserve/access.log`: 8,477 `/v1/images/generations` (576 GPU-minutes, the only route with
meaningful GPU time), 96% 1024^2 8-step with one routed LoRA, all `background` tier from the CuteDSL gallery
generator (`supervisor: cutedsl-gallery-generator` -> :8100 -> :8791), sent strictly one at a time.

| 1024^2, 8 steps | gen p50 | enc | cycle |
| --- | ---: | ---: | ---: |
| no LoRA | 3189 ms | 147 ms | |
| routed LoRA | 3665 ms | 166 ms | 4.08 s between requests |

Per-step fit (same prompts and seeds at 8/7/6/5 steps, min of 2): 0.33 s/step, ~0.5 s fixed (text encode, VAE,
LoRA setup). The DiT is ~88% of a request and has already had the kernel passes (`sdcpp-opt` perf/dit-fusion,
LoRA fusion, cuDNN VAE); F16/Q8 weights do not fit beside the other tenants (8 GB free).

### Gallery generator was discarding every image

Since 2026-11-03 01:54 `/opt/cutedsl-site/.env` is `root:root 0600`; the job runs as `administrator`, cannot
source it and uploads with stale R2 credentials: `PutObject ... Unauthorized` on every image
(run ending 2026-11-09: 2,166 stored, 28,691 failed, with a 30 s back-off after 10 consecutive failures).
Stopped with `supervisorctl stop cutedsl-gallery-generator`. Needs working R2 credentials for the job's user
before `supervisorctl update cutedsl-gallery-generator`.

Also in its log: 13% of prompts trip the bumpy-image detector (46k of 350k) and are regenerated with a shortened
prompt, while :8100 runs its own `quality_retry`; 16k prompts were rejected after both.

### Changes staged in the supervisor command (take effect on the next start)

- `--concurrency 2`: the client's decode, bumpy check, resize and R2 upload overlap the next render
  (cycle 4.08 s -> gen time; the gateway serialises the GPU itself).
- `--steps 7`: -10% gen (3146 -> 2827 ms min). 33 prompts x 2 seeds (`qualitybench/sweep_prompts_hard.json`,
  `prompts_v2.json`), `qualitybench/face_score.py`:

| steps | gen min ms | face IQA | image IQA | CLIP |
| ---: | ---: | ---: | ---: | ---: |
| 8 | 3146 | 0.457 | 0.677 | 0.379 |
| 7 | 2827 | 0.459 | 0.679 | 0.379 |
| 6 | 2479 | 0.446 | 0.666 | 0.379 |
| 5 | 2194 | 0.437 | 0.652 | 0.381 |

  Sheets (`qualitybench/out/zsteps-20261110/sheet_{a,b}.jpg`): 7 keeps the 8-step composition and detail; 6 breaks
  poses on some multi-figure prompts (extra arm on the dancers, malformed waving hand). 7 only.

Backup of the previous command: `/nvme0n1-disk/tmp/cutedsl-gallery-generator-supervisor.conf.bak-20261110`.

Route switches are free on this lane: gen p50 is 3665 ms whether the previous request used the same route
(n=3148) or a different one (n=585), so the MOSS `SD_LORA_KEEP` work has nothing to add here.

## MOSS fast lane (local 3090 Ti)

`performance/moss-fast-lane-fusion-2026-11-10.md`: paired, bit-identical, plain x0.845, LoRA-routed x0.80-0.82,
lane start ~9 min -> 12-15 s from `~/models-fast/moss`. sd.cpp `moss-i2i-reference` 61b59f6, ggml
`moss-dit-fusion` 7c2f2a11.

## BiRefNet worker (prod, live 2026-11-10, branch `perf/birefnet-20261110`)

Drop-in `zzzz-perf-20261110.conf` (worktree snapshot `omniserve-native-deploy/birefnet-perf-20261110`): strip-parallel
PNG, split-alpha WebP, native PNG/WebP decode, one source upload and one RGBA download per cutout, separable edge
band. PNG pixels identical, WebP colour bitstream identical (alpha stays lossless at lower effort: +0.4% bytes
<= 2 MP, +11% above). Rollback: delete the drop-in, daemon-reload, restart. Details: `matte/bench/README.md`.

| live worker, server ms min | before | after |
| --- | ---: | ---: |
| 1024 webp cutout | 280 | 188 |
| 1024 png cutout | 168 | 86 |
| 2048 webp cutout | 842 | 519 |
| 2048 png cutout | 663 | 305 |
| 1024 webp req/s at 1 / 2 / 4 | 3.7 / 5.4 / 8.6 | 4.7 / 7.1 / 11.2 |
| 2048 webp req/s at 1 / 2 / 4 | 1.04 / 1.74 / 2.53 | 1.74 / 3.23 / 5.40 |

Left off: `BIREFNET_GPU_PREPROCESS=1` (moves 0.1% of alpha pixels by > 8/255 on one upscaled image),
`BIREFNET_WEBP_METHOD=2`. The unit's WorkingDirectory is now the snapshot, so edits to main's `workers/` do not
reach this unit until the next snapshot.

### WebP encode (`OMNISERVE_NATIVE_SD_WEBP_METHOD`, opt-in)

1024^2 q85 on the prod Broadwell, `liboencode`: method 4 133 ms / 86.0 KB, method 2 54 ms / 88.3 KB but 3.3 dB
lower PSNR against the same source, method 1 40 ms / 97.5 KB. libwebp's lossy threading does nothing for opaque
images. Default unchanged; with client concurrency 2 the encode no longer holds the GPU idle anyway.

## YuE2 music (local 3090 Ti, not deployed)

`performance/yue2-music.md`, yue-cog `perf/yue-20261110` 71d81ce: 68.2 s clip, warm, 39.6 -> 29.4 s median
(27.7-28.5 s with `YUE_RESIDENT=1`), FLAC sha256 identical over 40 renders. Free AR offload from a CPU master
(-5.4 s of weight moves), in-graph sampling + shared Q/K norm (semantic 106 -> 133 tok/s), NAR untouched.
Prod worker never admits locally right now (8.9 GB free vs `YUE_MIN_FREE_MIB=18432`), so the gain lands on RunPod:
rebuild the image, new template, `scripts/benchmark_runpod.py`. Shared host: checkout with the `yue_runtime.py`
change plus `scripts/apply_yue2_patch.py` in the worker venv.

## Not touched

H3 video (RunPod image), Qwen ra2 (:8792, low traffic, tuned 2026-10).
