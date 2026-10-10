# Qwen Image 2.1 (ra2) latency, residency and a demand-driven residency policy — 2026-10-09

Branch `ra2-residency` (worktree `omniserve-native-residency`, base 40ef497 = the binary serving :8792).
Tools: `tools/ra2_residency_bench.py` (throwaway gateway on :8794 with the live unit env + overrides,
same seeds per arm, TE/sampling/VAE seconds from the sd.cpp INFO log, NVML peak, prod-5xx watch and a
device-free floor that kills the canary), `tools/ra2_compare.py` (per-image PSNR/SSIM/CLIP between arms),
`tools/image_parity_bench.py --reference-extra/--candidate-extra` (turbo vs base on one gateway).
Raw runs: `experiments/ra2-residency/`. Production was not restarted or reconfigured.

## What the box allowed today

Z-Image went fully resident on :8791 at 05:18 UTC (`zzzzzzz-zimage-resident.conf`). Over the next hour
(`freevram.log`, 360 samples) device free was 6.0-10.0 GiB (median 9.1), BiRefNet ran on CPU for lack of
headroom, and the broker's background headroom sat at 0.6-1.4 GiB. A streamed Qwen canary needs ~8 GiB:

| canary (prod binary + env, 1 request) | wall | TE | sampling | VAE | peak VRAM |
|---|---:|---:|---:|---:|---:|
| T2I turbo 512² | 3.57 s | 0.99 | 2.18 (6 steps) | 0.24 | 7.5 GiB |
| edit 512² (20 steps, EasyCache 0.08) | 4.75 s | 1.25 (+0.61 encode) | 2.52 | 0.24 | 7.9 GiB |
| T2I 1024², `MAX_VRAM=3` + `STREAM_LAYERS=1` | 42.7 s | 3.79 | 28.8 | 9.73 | 3.7 GiB |
| T2I 512², TE on CPU (`SD_BACKEND=te=cpu`) | killed | - | - | - | 7.3 GiB (DiT phase) |

Without a budget, sd.cpp stages the whole DiT (4.4 GB) or TE (5.4 GB) per call, so the streamed peak is
~7.5-10 GiB at any size; with a budget the segmented executor thrashes (14x slower sampling at 3 GiB, as
in the 09-23 `--max-vram` sweep). The 512² probe dipped device free to 1.6 GiB and the TE-on-CPU probe
tripped the 3.5 GiB floor guard (one internal background Z-Image 503 at 06:52 overlapped it). Under the
rule "free >= peak + 4 GiB" no 1024² Qwen canary could run, so the 1024² arms (prod breakdown, residency
partial/full, sd.cpp step caches on 20 steps, turbo parity on GPU) were queued behind a VRAM gate
(`queue_all.sh`, needs 15.3-21 GiB free) that never opened and were cancelled. Full Qwen residency
(~11.5 GiB standing) cannot coexist with resident Z-Image at all (VRAM math below).

## Current prod latency, 1024² (measured where possible)

| path | latency | breakdown | source |
|---|---:|---|---|
| T2I turbo (6-step v0.3 LoRA on Q4_K_M, streamed) | exec median 8.76 s (min 6.92, n=34) | sampling 6.4 s = first step 2.05 + 5 x 0.74 s/it; VAE 9 tiles 2.2 s | prod journal + ra2 ledger 05:02-05:21 today |
| T2I `turbo:false` (30 steps, EasyCache 0.15) | 10.3-11.0 s | - | qualitybench README (10-02, 5090 prod) |
| edit 1024 source (20 steps, EasyCache 0.08, prefix-KV) | 12.6 s min | - | 09-23 canary |
| DiT resident vs streamed (sd-cli, 8 steps) | 5.8 vs 9.4 s sampling | streamed step = 0.65 s PCIe upload + 0.45 s kernels | 09-23 phase8 |

Streamed steady state is PCIe bound (0.74 s/it vs ~0.45 s/it resident). Full residency would save about
0.3 s per DiT call plus the 0.8 s TE upload and most of the first-step setup: ~3.5-4 s on a turbo T2I
(8.8 -> ~5 s) and ~5 s on a 20-step edit (derived, not measured today).

Turbo vs base quality: GPU parity was blocked; a CPU proxy at 512² (same weights, CPU kernels) is in
the next section. Prior evidence: the runtime LoRA on Q4_K_M is visibly muddier than base and than the
pre-merged v0.3 GGUFs (Q5_K_M 9.8 s, Q6_K 10.4 s vs LoRA 11.5 s on the 3090 Ti, qualitybench README).

CPU proxy, 512², 4 corpus cases (`corpus_cpu512.json`, `cq_exec.sh`, `image_parity_bench.py` gates vs the
prod `turbo:false` path = 30 steps EasyCache 0.15; CPU wall times are only relative):

| candidate | CLIP image cos (mean / min) | prompt Δ | aesthetic Δ | gates | CPU wall vs base |
|---|---:|---:|---:|---|---:|
| prod turbo (v0.3 6-step LoRA r256 on Q4_K_M) | 0.903 / 0.728 | +0.010 | +0.029 | pass | 245 vs 332 s |
| merged turbo v0.3 Q5_K_M (no LoRA) | 0.914 / 0.805 | +0.012 | +0.068 | pass | 205 vs 332 s |

Contact sheet `experiments/ra2-residency/sheet_turbo_cpu512.jpg` (inspected): composition matches base on
all four; the runtime LoRA garbles small sign text ("BUILD FAST / TEST DEEP" becomes glyph soup) while the
merged Q5_K_M renders it cleanly; both are slightly softer than base. CLIP/aesthetic gates pass for both
and do not catch the text damage. Recommendation: serve T2I turbo from the merged Q5_K_M (16% faster here,
better text) once a second diffusion context is affordable, i.e. only when the Qwen lane is resident or
the TE can be shared; the runtime LoRA stays the zero-VRAM option.

## Step caches on 20-step Qwen

Not re-measured today (needs a 1024² canary). From 09-23 (20 steps, 1024², 4 prompts x 3): dense 13.98 s,
EasyCache 0.08 (prod for edits) 9.08 s SSIM 0.970, EasyCache 0.12 8.12 s SSIM 0.934, Spectrum 8.02 s SSIM
0.943. Unlike the 8-step Z-Image turbo schedule, skipping does engage at 20-30 steps. TaylorSeer, cache-dit,
dbcache and ucache arms are in `arms_cache.json`, ready for the next VRAM window.

## VRAM math (MiB, medians 06:30-07:30 today)

| tenant | idle | render |
|---|---:|---:|
| :8791 gemma q8 + embeddings + Z-Image resident | 14291 | 15616 |
| :8791 with Z-Image streamed (pre-05:18) | 6436-7272 | +~3000 |
| :8792 Qwen streamed | 1172 | +7500-10200 (lease 10240, edit 11264) |
| Qwen fully resident (TE 5408 + DiT 4391 + VAE 644 params + workspace, est.) | ~12600 | ~13500 |
| BiRefNet worker (GPU / CPU fallback) | 3394 / 1168 | +3584 lease |
| TTS 2576, bitbankmojo 1290, chronos 740, mojojojo 506, omniserve 498, cutedsl 498 | 6108 | - |
| total | 32607 | |

- Z resident + Qwen streamed + BiRefNet on GPU: 14291 + 1172 + 3394 + 6108 = 24965 -> 7.6 GiB free, less
  than one Qwen streamed render (7.5-10 GiB): this is the state that pushed BiRefNet to CPU and forces
  Qwen leases.
- Z resident + Qwen resident: 14291 + 12600 + 3394 + 6108 = 36.4 GB > 32.6. Never fits.
- Z streamed + Qwen resident: ~6900 + 12600 + 3394 + 6108 = 29.0 GB, 3.6 GiB free: no room for a Z-Image
  streamed render (~3 GiB) plus the 1 GiB keep-free floor. Fits only with BiRefNet on CPU or gemma evicted.
- Both streamed: ~17.6 GB, 15.0 GiB free.

So the box holds exactly one hot image model resident, and only while the other one is quiet.

## Demand-driven residency policy (implemented)

`src/oresidency.c` (arbiter, pure, unit-tested in `tests/test_residency.c`), hosted next to the VRAM
ledger on :8791 (`POST /v1/gpu/residency`, state in `GET /v1/gpu/status` -> `residency`). Each image lane
with `OMNISERVE_NATIVE_SD_RESIDENCY=demand` reports every 3 s: requests since last report, current state,
`resident_mb` (extra device MB when resident), `stream_mb` (one streamed render), `saved_ms` per request.

- benefit = EWMA request count (half-life 300 s) x saved_ms. Ranked greedily; the incumbent's benefit
  counts 1.5x and it cannot be displaced for `MIN_HOLD_S` (120 s).
- capacity = device free + resident_mb of the lanes resident now; need = admitted resident_mb + keep-free
  (1024) + the largest `stream_mb` of a lane left streamed that had a request in the last
  `RESERVE_IDLE_S` (180 s). A quiet Qwen lane reserves nothing, an active one reserves its 10 GiB render.
- Newcomers are judged on the minimum device free of the last 60 s minus 1 GiB hysteresis; holders on the
  current figure (co-tenant renders swing free by GBs). A revoked lane waits `MIN_HOLD_S`, doubled per
  consecutive short grant (max 32x), so a mis-sized claim cannot ping-pong.
- A winner promotes only after the memory is free on the device (revoked holder demoted first); a lane
  idle for `IDLE_RELEASE_S` (900 s) gives its VRAM back to the pool (LLM, BiRefNet).
- Switching rebuilds the sd.cpp context with the other placement profile (`SD_*` = streamed,
  `SD_RESIDENT_*` = resident; resident build first, so a failed promotion keeps serving streamed).
  It happens in a gap between generations or, if the lane is busy, at the start of the next request;
  requests size their broker lease from the pending state. Broker unreachable -> streamed; broker
  without the arbiter -> streamed, re-asked every 5 min. Heartbeats are not access-logged.

With today's mix (Z-Image ~1.9k/day, ~76% batch corpus; Qwen 282 requests/week in bursts of up to 114/h)
and the numbers above, the policy keeps Z-Image resident (3.9x) except while Qwen is active plus 180 s,
when Z-Image demotes so Qwen's streamed render has room instead of forcing into a 7.6 GiB gap. Qwen
itself is promoted only if both its benefit wins and ~11.5 GiB + 1 GiB + Z's 3 GiB stream reserve is free,
which today requires BiRefNet on CPU or gemma evicted.

Validation: unit tests (promotion, hold, idle release, stream reserve, revoke-before-promote, hysteresis,
noisy free, backoff, pressure, claim expiry); `onative_tests` core suite passes; CPU end-to-end
(`experiments/ra2-residency/e2e_cpu/run_e2e.sh`, Z-Image Q4 on CPU, gateway pointed at its own
arbiter over HTTP): promotion in a gap 0.9-1.1 s, demotion 0.6-0.9 s, generations succeed in both
states and across rebuilds; a deliberately mis-sized claimant flapped 16 times in 60 s before the
backoff and 8 after (min hold 5 s in the test, 120 s in the drop-in). Not validated on GPU: VRAM
release on demotion, resident LoRA (turbo) behaviour, real promotion time with eager upload.

## Deploy (not done) and rollback

Two binaries, as today: `build-res-main` (SD headers of `stable-diffusion.cpp`, the :8791 default lib)
and `build-res` (SD headers of `sdcpp-qwen-prefixkv`, :8792 pins `libstable-diffusion-9bc7b35-release.so`),
copied to `omniserve-native-deploy/deploy-residency-<sha>/`. Drop-ins: `systemd/residency/` (replace
`@DEPLOY@`). Order: :8791 first (arbiter + Z-Image demand; its base `SD_*` restore the streamed profile,
`SD_RESIDENT_*` mirror `zzzzzzz-zimage-resident.conf`), then :8792. Before go-live: a GPU canary of the
:8791 binary for Z-Image parity (it adds the ra2-tiers commits on top of ea3072c) and one forced
demote/promote cycle watching NVML. Rollback: remove `zzzzzzzz-residency.conf` on either unit,
`daemon-reload`, restart; the other unit keeps working (Qwen falls back to streamed on `no_arbiter`).
