# Prod RTX 5090: Z-Image residency A/B (2026-09-30)

Canary on port 8795 (new gateway build, VRAM broker off), Z-Image Turbo Q4_K, 9 steps,
WebP out, distinct seeds, `tools/prod_image_bench.py`. GPU shared with the live tenants.

| profile | 1024x1024 p50 | 512x512 p50 | 512 img/s | peak VRAM |
| --- | --- | --- | --- | --- |
| prod today: STREAM_LAYERS=1, PARAMS_BACKEND=cpu, MAX_VRAM=2, tile 64 | 15.5 s | 8.2 s | 0.12 | 5.4 GB |
| resident, VAE tile 64 | 5.1 s | 1.16 s | 0.86 | 12.2 GB |
| resident, no VAE tiling | 4.6 s | 1.11 s | 0.89 | 17.6 GB |

The first request on a resident profile pays a one-time ~11 s weight load.
Exact-repeat cache hits are 1-2 ms on every profile. The native C changes in
67b8284 do not move GPU time (cold 1024x1024: 15.5 s before, 16.0 s after); they
cut CPU-side request cost (parse 3.4 ms to 0.4 ms, decode 35 ms to 21 ms, webp cache hit 172 us to 2 us).

Cost of residency: about +7 GB VRAM held by the image lane, against the shared-GPU
budget in `zimage-residency.conf`.
