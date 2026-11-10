# YuE2 music serving

The shared Cog runtime lives in `../yue-cog/yue_runtime.py`. The native gateway
relays authenticated `POST /v1/music/generations` requests to
`workers/yue_worker.py`. NetWRCK uses its existing asynchronous account/job API:
`POST /api/music-generator` with `model: "yue2-3b"`, followed by the returned
`status_url`. The studio is `/tools/yue-music`, listed under Audio & Voice.

NetWRCK radio shares this gateway via `OMNISERVE_YUE_URL` and
`OMNISERVE_YUE_KEY`. Its seed briefs cover 32 genres/styles; adding briefs
does not invoke inference. New studio/radio results are indexed with YuE2
tags; radio also records omniserve-native provenance, while the paid studio
currently submits directly to RunPod. See `../netwrck/docs/music-radio.md`
for compact mobile playback controls, indexing and bounded validation.
Generation presets, admission and exact-result caching are unchanged.

## Quality and execution

The release pins YuE2 inference to `bd90e4ccae671d869b3ecaca6d7e893927d29442`,
model weights to `14fc6c6f146441b1dd6363fcb2e01e82a6914cb7`, and the recommended
VAE to `9a94e1d0ea9f8087e98f77fa88df4a4068104d2a`. Public requests use BF16
AR/NAR, FP32 VAE, full musical planning, 32 ODE steps, and no quantization.
FLAC is the default. MP3 and WAV are also available. Clip and song tiers differ
only in the semantic token limit (1500 / 9000), approximately 60 / 360 seconds;
these are caps, not exact durations. Truncation metadata survives R2 caching
and is shown in the player.

Local admission requires 18 GiB available on the GPU (including the worker's
own resident allocation for a warm request), with at least 2 GiB free. The
pipeline has an 18 GiB memory budget, of which upstream reserves 2 GiB. It runs
in a separate child so a failed/idle model can release its entire CUDA context
without restarting the gateway. A successful process stays warm for 30 seconds,
then exits; memory pressure also releases it. Local inference is serialized.
A busy lane or CUDA OOM uses the configured YuE RunPod serverless endpoint.
OOMs increase the admission floor and apply backoff. Unknown local failures
and ambiguous submission failures are not automatically resubmitted. RunPod
execution is capped at 3 minutes for clips and 8 minutes for songs; queue wait
is bounded separately, and unfinished remote jobs are cancelled at deadline.

The router allows four distinct requests in flight; simultaneous identical
requests share one render. A bounded 2 GiB / 24-hour disk cache includes the
model, VAE, runtime, backend, quantization and complete normalized input.
NetWRCK stores audio plus result metadata in R2. Successful exact saved results
are served without another charge. This is a cache benefit, not faster model
inference. Runtime/model changes require bumping the public cache revision in
both `yue_runtime.py` and `yue_music.go`.

No CPU fallback is used in the interactive service. Explicit offline CPU
inference is supported by `YUE_DEVICE=cpu`; the existing quality samples took
roughly an hour each on the shared CPU. On 2026-09-20 the RTX 5090 had about
10 GiB free, below the safe admission floor. No existing GPU service was stopped.

The `quality-v2` runtime caches verified safetensor digests against the resolved
file path, device, inode, size, nanosecond mtime and ctime. First access and any
change rehash the entire file; upstream manifest checks remain enabled. Cache
files belong to the service user and must stay private. Cached pinned revisions
are resolved offline first; absent files trigger the normal download, while
integrity failures stay fatal. A real check of both pinned model files measured
26.746 seconds on first verification and 0.000635 seconds with verified digest
reuse; model/VAE identities matched. The cache handles Hugging Face snapshot
symlinks to extensionless blobs. These timings cover hashing only. Full startup
also includes imports and tokenizer setup and varies with host contention.
After cache priming, a fresh-process CPU pipeline load took 3.136 seconds with
identical weight identities, compared with 43.86 seconds before the symlink fix.
These changes save startup work only; they do not accelerate the denoiser or
change generated samples. Dedicated RunPod workers retain AR weights;
the shared-host worker offloads AR to reduce peak VRAM.

A real CPU canary with full planning, 32 ODE steps and a 200-token cap produced
7.999 seconds of valid stereo FLAC in 910.61 seconds. Its semantic sequence was
correctly marked truncated. Stage timings were 119.45 seconds planning, 84.29
seconds semantic generation, 577.68 seconds NAR and 8.36 seconds VAE. This is a
functional check, not a GPU speed or subjective audio-quality result.

## Exact acceleration (2026-11-10)

`../yue-cog` branch `perf/yue-20261110` carries `patches/yue2-accel.patch` for the
pinned upstream package plus `scripts/apply_yue2_patch.py` (idempotent, refuses
anything but the pinned pristine files, `--check`, `--revert`). Every default
produces the same ABC tokens, semantic tokens, latents and FLAC bytes as the
unpatched package: 40 patched renders on the local RTX 3090 Ti matched the
upstream FLAC byte for byte, and three further prompts/seeds plus CFG,
`cot=off` and `cot=melody` matched token for token. The quality preset is
unchanged (BF16 AR/NAR, FP32 VAE, full planning, 32 steps, no quantization), so
`RUNTIME_VERSION` and the result caches are not bumped.

| Change (env, default) | What it removes |
| --- | --- |
| `YUE_FAST_PLACEMENT=1` | Upstream moved 6.8 GiB of weights four times per request with AR offload (upload, AR down, AR up, everything down for the VAE; 2-3.5 s each). Weights now keep an immutable CPU master, so offload is free and AR is not restored after NAR: one 1.9 s upload per request, same peak VRAM. |
| `YUE_FUSED_SAMPLING=1` | Logits processing, the multinomial draw and the penalty history ran as about 25 eager ops with several host round-trips per token. They now run inside the decode CUDA graph, with one sync per token. Checked bitwise against the eager sampler once per process; CFG and `cot=off` keep the eager sampler. |
| `YUE_LEAN_DECODE=1` | Q and K heads share one RMSNorm+RoPE pass per layer, so each token launches fewer tiny kernels. Checked bitwise (logits and KV) once per process. |
| `YUE_RESIDENT=0` (image: 1) | Opt-in on the shared host: LM and VAE stay on the GPU. Peak allocation 9.6 GiB instead of 7.6 GiB, idle process VRAM 8.4 GiB instead of 1.5 GiB. |
| `YUE_PIN_WEIGHTS=0` | Opt-in: pinned masters upload in 0.5 s instead of 1.9 s, for 8 s once per process and about 9 GiB locked host RAM. Loses with the 30 s idle exit. |

Local RTX 3090 Ti, Electric Gold request (seed 831002, 3000-token cap, 32
steps, FLAC), 68.2 s of audio, 723 ABC + 1706 semantic tokens, warm, shared-host
settings (`YUE_OFFLOAD_AR=1`, 18 GiB budget), same process, min / median of 3,
no other GPU tenant. "Upstream paths" is the patched package with every switch
off, which runs the unmodified upstream code:

| Stage | Upstream paths | Defaults | `YUE_RESIDENT=1` |
| --- | ---: | ---: | ---: |
| ABC planning | 7.04 / 7.07 s | 5.39 / 5.53 s | 5.49 / 6.22 s |
| Semantic AR | 16.01 / 16.05 s | 12.80 / 12.85 s | 12.78 / 12.82 s |
| NAR, 32 steps (incl. AR offload) | 9.95 / 10.32 s | 8.17 / 8.22 s | 8.22 / 8.22 s |
| VAE (incl. LM offload) | 3.67 / 3.75 s | 1.03 / 1.03 s | 0.87 / 0.92 s |
| Other (upload, FLAC, base64) | 2.38 / 2.40 s | 1.66 / 1.82 s | 0.29 / 0.33 s |
| Wall | 39.20 / 39.57 s | 29.26 / 29.38 s | 27.65 / 28.54 s |
| Semantic tok/s | 106 | 133 | 133 |
| Peak allocated | 7744 MiB | 7758 MiB | 9831 MiB |

With another process holding the same GPU at 100% utilisation the unpatched
package took 77.35 / 77.96 s (own process, 3 renders) and the defaults 63.8 s
(one interleaved render; 63.45 / 63.92 s for two more before the lean decode
existed). The solver itself (NAR 7.9-8.0 s, VAE tiles 0.75-0.83 s) is untouched;
the NAR and VAE rows shrink only because weight moves left them.
Records: `../yue-cog/examples/accel-ab-20261110.json`.

Cold start is the largest remaining local cost. With the page cache hot, a fresh child spent 10.7 and 17.3 s outside the
model stages on its first request (imports, mmap load, first upload) against
2.2 s for a warm request, and the 30 s idle exit makes
most sporadic requests cold. The idle non-resident child holds 1.5 GiB of
VRAM. This host's HF cache is on the spinning `/vfast` pool: the first read of
the 7.3 GB checkpoint took 133 s and its first integrity hash 436 s, so local
serving wants `YUE_HF_HOME` on NVMe. Raising `YUE_IDLE_SECONDS` is a policy
choice (VRAM and about 7 GiB of page cache held); it is unchanged here.

Measured and rejected: LM-head row slicing to the allowed token range (BF16
GEMM on the slice is not bitwise equal); cuDNN autotuning for the VAE (1.5 ->
1.03 s on repeated shapes, 115 dB SNR, not exact, and it retunes for every new
last-tile length); one 2048-frame VAE tile (112 dB SNR, no gain, +1.4 GiB);
folding VAE weight norm (exact, no gain); pinning by default (above). The
per-request CUDA-graph capture costs 0.2-0.4 s and was left alone.

Rollout. Shared-host worker: update `../yue-cog`, run
`.venv/bin/python scripts/apply_yue2_patch.py` in the worker virtualenv (needs
`patch`), restart `omniserve-yue-worker`; no env change is required and
`--revert` restores upstream. RunPod: rebuild and push the image (the
Dockerfile applies the patch and sets `YUE_RESIDENT=1`; not built or tested
here), save a new template, then rerun `scripts/benchmark_runpod.py`: the
4090 render should reproduce FLAC sha256 `509a973d...d83c` from
`gpu-benchmark-20260920.json`; a different hash on the same GPU type blocks the rollout. `scripts/accel_ab.py` runs the same-process
legacy/accel comparison on any GPU and fails on any difference. `cog build`
images stay unpatched and keep the upstream paths. No cache revision bump in
`yue_runtime.py` or `yue_music.go` is needed while outputs stay identical.

## Configuration

Install `systemd/omniserve-yue-worker.service`. Put secrets in
`/etc/omniserve-yue.env` with mode 0600:

```
YUE_WORKER_SECRET=<same bearer secret used by the native gateway>
RUNPOD_API_KEY=<runpod key>
YUE_RUNPOD_ENDPOINT_ID=<dedicated YuE endpoint>
R2_ACCOUNT_ID=<R2 account id>
CLOUDFLARE_BUCKET=<music bucket>
CLOUDFLARE_R2_ACCESS_KEY_ID=<bucket access key>
CLOUDFLARE_R2_SECRET_ACCESS_KEY=<bucket secret>
CLOUDFLARE_CDN_DOMAIN=netwrckstatic.netwrck.com
```

Install `systemd/omniserve-native-yue.conf` as a gateway service drop-in.
Install `../yue-cog/requirements-serving.txt` into the worker virtualenv.
Remote requests carry a one-hour presigned PUT for one audio object. RunPod
uploads directly to R2 and returns its public URL plus metadata, avoiding
large base64 job results without giving the container bucket credentials.
Local generation can still return inline audio through the native gateway.
The worker binds only to loopback port 9106. Health is `GET /health`.
The gateway has a dedicated 25.5-minute relay deadline; it doesn't hold the
shared GPU semaphore while waiting for RunPod. The worker owns GPU admission.

NetWRCK configuration:

```
OMNISERVE_YUE_URL=http://127.0.0.1:8791
OMNISERVE_YUE_KEY=<same gateway bearer secret>
YUE_COMMERCIAL_LICENSE_CONFIRMED=1
```

The project owner confirmed existing commercial permission on 2026-09-21.
`YUE_COMMERCIAL_LICENSE_CONFIRMED=1` is recorded in NetWRCK's ignored
`search_server_go/.env`. Licensing is no longer a rollout blocker. Keep the
explicit deployment flag so unconfigured installations remain disabled.
The public model card's default license is separate from this project-specific
permission; no inference about its scope beyond this hosted service is needed.
Source: https://huggingface.co/m-a-p/YuE2-3B

Use `../yue-cog/scripts/runpod_deploy.py` with the built image and a private
registry auth ID. Defaults are zero minimum and maximum workers. Explicitly
set maximum workers to one during an authorized experiment or service rollout;
idle timeout is 10 seconds. Check existing resources first. Reset the YuE
endpoint maximum to zero after experiments and cancel unfinished jobs. Do not
clean up unrelated services on this shared RunPod account.

The dedicated template now points `YUE_HF_HOME` at RunPod's cached-model mount
`/runpod-volume/huggingface-cache/hub`, while retaining the HF token. This is a
cold-start and download optimization only; it does not change model weights,
sampling, denoising, or audio output.

## Example

`../yue-cog/examples/electric-gold.json` records the exact original prompt,
lyrics, seed, quality settings, CPU timings, and public MP3/FLAC URLs. The
existing synth-pop render is 60.9 seconds, 48 kHz stereo, full planning, 32 steps,
and neither symbolic nor semantic generation was truncated. It was generated
before this integration, not presented as a new GPU benchmark.

The studio preloads its original lyrics and style with seed 831002. The sample
was capped at 3000 tokens and ended at 1522. The public full-song preset allows
9000 tokens so the same lyrics can finish naturally. A 1500-token clip is not
promised to reproduce the complete sample.

## Launch pricing

The 2026-09-20 RTX 4090 benchmark used the Electric Gold prompt, seed 831002,
full planning, 32 steps, BF16, no quantization and a 3000-token cap. Both runs
used the same worker and performed inference (no result cache). Each produced
64.479 seconds of stereo 48 kHz FLAC, with neither stage truncated. Files were
byte-for-byte identical, finite, and unclipped (peak 0.986763).

| Measurement | First render | Warm render |
| --- | ---: | ---: |
| Runtime including pipeline startup | 63.987 s | 24.889 s |
| Provider execution including upload | 66.912 s | 25.780 s |
| Provider queue / worker initialization | 178.195 s | 0.016 s |
| Caller wall time including polling/download | 249.821 s | 32.231 s |
| Peak PyTorch allocation | 7420 MiB | 7695 MiB |
| Estimated GPU execution at $1.10/hour | $0.02045 | $0.00788 |

Warm stage timings: planning 5.10 s, semantic generation 11.73 s, NAR 3.50 s,
VAE 3.44 s. This establishes a measured serving baseline, not a denoiser
speedup against another GPU implementation. Listening and lyric-adherence
checks are still required before adopting approximate optimizations.
Complete timing, request and checksum records live in
`../yue-cog/examples/gpu-benchmark-20260920.json`; immutable image and endpoint
details live in `deploy/yue-runpod.json`. The benchmark reset both worker
limits to zero and endpoint health confirmed zero workers and queued jobs.
Studio preview: https://netwrckstatic.netwrck.com/static/tools/yue-music-preview.html .
The NetWRCK route and native gateway changes are prepared in the worktrees;
the shared running services were not restarted. Commercial permission was
confirmed on 2026-09-21; live generation still requires the worker/gateway
configuration and deployment of the isolated YuE changes. The required
`op-contributor` review executable was unavailable; no substitute model was used.

The short sample's memory footprint does not establish a safe full-song bound.
A proposed constrained local GPU canary was declined before model loading when
free VRAM fell from 10148 to 8642 MiB. Shared-host defaults remain conservative;
the active GPU services were left running.

The same-seed clip comparison measured full planning at 32 steps in 28.77 s,
full planning at 16 steps in 21.23 s, and melody planning at 16 steps in 22.54 s
on the warm 4090 worker. All were finite 48 kHz clips, but all hit the 1500-token
clip cap and the denoising alternatives were not listening-validated. Therefore
32 steps and full planning remain the default. The 16-step variants are measured
optimization candidates, not silently applied quality reductions.
The raw request/output records are in
`../yue-cog/examples/quality-variants-20260921.json`.

| Tier | Studio and API | Credits | Maximum audio |
| --- | ---: | ---: | --- |
| Clip | $0.08 | 8 | about 60 seconds |
| Song | $0.20 | 20 | about 6 minutes |

Both tiers retain the full quality preset. No subscription or separate API
markup; failed jobs use the existing refund path. Quotes from VN media use the
same base prices. These are the configured launch prices. Commercial permission
and sample provider execution costs are confirmed; long-song costs and the full
operating margin still need monitoring after rollout.

RunPod's published flex rate checked 2026-09-20 is $1.10/hour for RTX 4090 and
$1.58/hour for RTX 5090: https://www.runpod.io/pricing . Upstream reports a
71.04-second warm render for 214.85 seconds of audio on a 4090. This implies
about $0.022 GPU execution cost for that upstream benchmark, not our measured
cost. A 180-second billable render at $1.10/hour costs $0.055; a 360-second
render costs $0.11. Include cold start, model downloads, idle tail, retries,
bucket traffic, payment costs and any license fees before calling the margin
profitable. The router never selects a larger, more expensive GPU implicitly.

Do not label a speedup until cold and warm renders have been timed on the same
hardware. Keep FP8, shortened planning, fewer ODE steps and alternate decoders
out of the default service unless listening and lyric-adherence comparisons
with the same prompt/seed support the change.

## Validation

```
python3 -m unittest discover -s tests -p test_yue_worker.py -v
python3 tests/test_yue_routing.py build-dev/omniserve-native
ctest --test-dir build-dev --output-on-failure
cd ../netwrck/search_server_go
go test . -run 'Test(Yue|MusicGenerator|NormalizeMusic|ToolsEndpoints)' -count=1
```

Browser checks cover desktop/mobile layout, default sample playback, style
presets, full-song request payload, asynchronous polling, truncation messaging,
and credit errors. Site tests render through Pongo without restarting the
shared running Go service.
