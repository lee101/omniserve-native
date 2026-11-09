# Native image upscaler (`POST /v1/images/upscales`)

In-process Real-ESRGAN through stable-diffusion.cpp's ggml ESRGAN (`new_upscaler_ctx` / `upscale`), dlopened from the same `libstable-diffusion.so` as the diffusion lane. Replaces fal creative-upscaler for plain 2x/4x.

## Contract
```
POST /v1/images/upscales           Authorization: Bearer <OMNISERVE_NATIVE_SECRET or keyed tier>
{"image_url": "https://... | data:image/...;base64,...", "scale": 2|4, "model": "general"|"anime", "output_format": "png"|"webp"|"jpeg"}
```
Defaults: scale 2, general, png. 200 = raw bytes, `Content-Type: image/png|image/webp|image/jpeg`, headers `X-Upscale-Width`, `X-Upscale-Height`, `X-Upscale-Scale`, `X-Upscale-Model`, `X-Upscale-Ms` (GPU run time). Errors are `{"error":{"message":..}}`: 400 bad JSON/field/URL/undecodable image or SSRF-blocked URL, 401 bad secret, 413 input > 4,194,304 px (scale 2) or > 2,100,000 px (scale 4) or output side > 8192 (checked from the header before decoding), 503 weights missing, lane saturated, admission timeout or VRAM lease denied (with `Retry-After`). `GET` on the same path returns loader status JSON.

- Inputs: PNG, JPEG, WebP. Odd sizes are fine (edge-padded, cropped back).
- Alpha: preserved for png/webp (RGB through ESRGAN, alpha bilinear-resized). jpeg output flattens alpha onto white. Fully opaque alpha is dropped (RGB output).
- Scale 2 general uses RealESRGAN_x2plus natively when `OMNISERVE_NATIVE_ESRGAN_GENERAL_X2` is set (needs the pixel_unshuffle patch in `stable-diffusion.cpp-prod/src/model/upscaler/esrgan.hpp`), else x4plus + exact 2x2 box downscale. anime always runs x4plus_anime_6B (+ box downscale for scale 2).
- https fetch: libcurl is dlopened (`libcurl.so.4`); https only (also on redirects, max 3), 20 s timeout, 48 MB cap, no proxy, and every connect address is checked in an open-socket callback (blocks loopback, RFC1918, CGNAT, link-local/metadata, multicast, ULA, mapped/NAT64/6to4 forms), so DNS rebinding and redirects to private hosts fail with 400. Without libcurl, https URLs get 503 and data URLs still work.

## Env
| var | default | meaning |
| --- | --- | --- |
| `OMNISERVE_NATIVE_ESRGAN_GENERAL` | unset | RealESRGAN_x4plus.pth (required for `general`) |
| `OMNISERVE_NATIVE_ESRGAN_ANIME` | unset | RealESRGAN_x4plus_anime_6B.pth (required for `anime`) |
| `OMNISERVE_NATIVE_ESRGAN_GENERAL_X2` | unset | RealESRGAN_x2plus.pth; enables native 2x |
| `OMNISERVE_NATIVE_ESRGAN_VRAM_MB` | 4096 | compute-buffer budget that drives tiling |
| `OMNISERVE_NATIVE_ESRGAN_TILE` | 0 | 0 auto, >0 force tile edge (input px), <0 force untiled |
| `OMNISERVE_NATIVE_ESRGAN_OVERHEAD_MB` | 512 | added to the VRAM lease |
| `OMNISERVE_NATIVE_ESRGAN_PRELOAD` | 1 | load + warm configured nets at startup (background thread) |
| `OMNISERVE_NATIVE_ESRGAN_THREADS` | 4 | CPU threads for sd.cpp |
| `OMNISERVE_NATIVE_UPSCALE_PERMITS` | 1 | tier-scheduler permits per upscale |
| `OMNISERVE_NATIVE_UPSCALE_QUEUE` | 4 | requests in flight before 503 "lane saturated" |
| `OMNISERVE_NATIVE_UPSCALE_QUALITY` | 92 | jpeg quality (and webp when oencode is missing) |
| `OMNISERVE_NATIVE_ESRGAN_CUDNN` | 1 | use ggml's cuDNN conv (direct conv, f32 accumulate) when `libcudnn.so.9` loads; 0 forces im2col |
| `GGML_CUDA_CUDNN_LIB` | `libcudnn.so.9` | cuDNN path (sibling `libcudnn_*.so.9` resolve from its directory) |
| `OMNISERVE_NATIVE_OENCODE_LIB` | `liboencode.so` (next to the binary) | libdeflate PNG + threaded WebP encoder; stb / WebPEncodeRGB fallback |
| `OMNISERVE_NATIVE_UPSCALE_PNG_LEVEL` | 1 | libdeflate level |
| `OMNISERVE_NATIVE_UPSCALE_WEBP_METHOD` | 2 | webp method (2 at q94 beats 4 at q92 on PSNR, 2x faster, +27% bytes) |
| `OMNISERVE_NATIVE_UPSCALE_WEBP_QUALITY` | 94 | webp quality |
| `OMNISERVE_NATIVE_UPSCALE_ENCODE_THREADS` | 4 | encoder threads |
| `GGML_CUDA_DISABLE_GRAPHS` | unset | set to 1 in a dedicated upscale process: captured tile graphs otherwise keep ~0.8 GB per net resident |

`OMNISERVE_NATIVE_SD_LIB` selects the library as for the diffusion lane; no diffusion model needs to be loaded.

## Admission and VRAM
Request -> queue cap -> fetch + header probe + limit check -> tier scheduler (`UPSCALE_PERMITS`) -> VRAM lease through the same path as image generation (local ledger / broker, or a cudaMemGetInfo point check when unbrokered) sized `tile_px * factor^2 * 1700 B + OVERHEAD_MB` -> one global mutex around sd.cpp (contexts are not thread-safe). ggml's compute buffer is ~1.7 KB per output pixel (im2col of the 64-channel convs at output resolution) for all three nets, so auto tiling runs untiled when the whole output fits the budget, else picks the smallest tile count per axis (sd.cpp overlaps tiles 25%) whose square tile fits. The `sd_upscaler_set_tile_size` symbol (added to sd.cpp-prod) makes this per request; an older library keeps the budget's largest tile. Once loaded the process holds ~1.1 GB (CUDA context, cuBLAS, ggml pool, 3 nets ~75 MB); in a process that already runs the diffusion lane the extra is the nets + cuBLAS/pool.

## Weights
sd.cpp loads the upstream `.pth` files directly (zip/pickle reader, converted to f16 at load), no conversion step:
```
mkdir -p models/esrgan && cd models/esrgan
curl -LO https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth             # 4fa0d389...
curl -LO https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2.4/RealESRGAN_x4plus_anime_6B.pth  # f872d837...
curl -LO https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth             # 49fafd45...
```

## Measured (RTX 3090 Ti, f16; GPU shared with another user's job, ranges are light..heavy contention)
| case | net / tile | run ms | process VRAM peak |
| --- | --- | --- | --- |
| 512x512 -> 1024 (2x) | x2plus untiled | 480-900 | 2.7 GB |
| 512x512 -> 2048 (4x), budget 4096 | x4plus tile 294 | 2050-4700 | 3.3 GB |
| 512x512 -> 2048 (4x), budget 8192 | x4plus untiled | 1460 | ~7.7 GB |
| 1344x768 -> 2688x1536 (2x), budget 4096 | x2plus tile 768 | 1750-3900 | 4.8 GB |
| 1344x768 -> 2688x1536 (2x), budget 8192 | x2plus untiled | 1480 | ~8 GB |
| 1344x768 -> 2688x1536 (2x), no X2 env | x4plus tile 336 + box | 9800 | 3.3 GB |
| 512x512 -> 1024 anime (2x) | anime_6B tile 294 + box | 800-1600 | 3.3 GB |

Process VRAM includes ~1.1 GB baseline. PNG encode + transfer adds ~0.3-0.5 s at 4 MP out (webp is similar, jpeg faster). First request after a cold start pays CUDA kernel loading unless preloaded (preload warms each net with a 64x64 run).

Accuracy vs PyTorch fp32 (spandrel, 512->1024 x2plus): 61.5 dB PSNR untiled, 58.7 dB tile 256; x4plus 256->1024: 59.2 dB.

Bottleneck (im2col path): ~67-79% of GPU time is ggml-cuda `im2col`. The default is now the cuDNN path (2.1-2.6x GPU, closer to PyTorch), plus an in-place dense concat in `esrgan.hpp` and a fused conv+bias+leaky/residual epilogue in ggml-cuda; libdeflate PNG cuts encode from ~1 s to ~0.13 s at 4 MP. Startup logs `[upscale] conv=cudnn png=libdeflate webp=oencode`. Numbers and the remaining ideas are in manifoldgen-site `docs/performanceideas.md`.

## Tests
`tests/test_core.c::test_upscale_contract` (parse, limits, header probe, data URLs, SSRF classifier, resize helpers) and `tests/test_upscale_route.py` (401/400/503 without a GPU; with `OMNISERVE_TEST_ESRGAN_DIR=models/esrgan` and a GPU: 2x/4x, odd-size alpha, jpeg/webp, anime, 413, SSRF).
