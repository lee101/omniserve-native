# Z-Image residency sweep — 2026-10-09

Roadmap item 1 from `zimage-acceleration-roadmap-2026-08-30.md`, measured on the shared 5090.

## Why

`/var/log/omniserve/access.log`, 7 days to 2026-10-09: `/v1/images/generations` is
13.3k requests, p50 14.8 s, p95 28.6 s, 61.9 GPU-hours, about 95% of gateway busy time.
~76% is the cutedsl batch corpus (`generate_images.py` -> :8100 -> :8791 native fallback),
the rest are site requests that queue behind it on the single image permit.
The prod journal shows 8 steps at 1.65 s/it: the DiT streams Q4_K weights from CPU
(`SD_PARAMS_BACKEND=cpu`, `SD_STREAM_LAYERS=1`, `SD_MAX_VRAM=2`) over PCIe Gen3 x8.

## Method

Same prod binary (`deploy-perf-ea3072c`), same models and seeds, canary on :8793
(`OMNISERVE_ACCESS_LOG=0`, LLM/judge off), run under `monitoring/run_agent.py`, prod traffic live.

## Results (1024x1024, 8 steps, N=4)

| config | latency | peak VRAM | output vs resident |
| --- | --- | --- | --- |
| prod streamed (cpu params, 2 GiB cap) | 17.9-21.1 s | ~3.0 GiB | byte-identical |
| resident (TE+DiT+VAE on GPU, VAE tile 64) | 4.7-4.9 s | 9.7 GiB | reference |
| resident, 768x1344 | 4.6 s | 9.7 GiB | - |
| resident, TE on CPU | 6.6-7.4 s | 6.2 GiB | PSNR 20-27 dB (different images) |
| resident, VAE untiled | OOM | - | - |
| resident + spectrum cache | 4.35 s (-8%) | 9.7 GiB | PSNR 28-33 dB |
| resident Q8_0 | not measured: 6.3 GiB alloc OOM beside live tenants | ~12.4 GiB | - |

Step-cache modes on the 8-image parity corpus (`tools/image_parity_bench.py`, 512-768 px):
easycache 0.1/0.2, taylorseer interval 2 and cache-dit 0.08 never skip a step on this
8-step turbo schedule (identical output, latency within noise of a repeat run, 1.74 s).
Spectrum skips: 1.47 s (-16% at 512 px), CLIP image cosine 0.976-0.995, aesthetic delta
-0.06..+0.09, passes gates, but only -8% at 1024 px. Not recommended.

## Recommendation

Make the embedded Z-Image fully resident: about 3.9x per image with no change to output,
for +6.6 GiB standing VRAM (steady box use ~17.8 -> ~24.4 GiB of 32.6). Drop-in:

```
[Service]
Environment=OMNISERVE_NATIVE_SD_STREAM_LAYERS=0
Environment=OMNISERVE_NATIVE_SD_PARAMS_BACKEND=
Environment=OMNISERVE_NATIVE_SD_MAX_VRAM=
```

Remaining exposure: 3D/training swaps already unload embedded models; a Qwen edit burst on
:8792 plus gemma reload can approach the limit, and the broker's 4096 MB image lease still
gates each render. Rollback is removing the drop-in and restarting.
