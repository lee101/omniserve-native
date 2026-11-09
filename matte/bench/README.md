# 2k matte bench (BiRefNet + guided upsample + key + recolour)

`gs2k.webp`: 2048² Qwen render, curly hair on a chroma green screen with heavy backlit spill (`png` via PIL for the scripts).

    uv run python matte/bench/guided_bench.py      # methods A/D/E/K/N, timings, crops (out/real_sheet.png), synthetic metrics
    BIREFNET_VRAM_BROKER_URL= python3 matte/bench/worker_e2e.py   # the worker path end to end, refine off vs on (FMT=png|webp)

Methods (3090 Ti, 2048², GPU only):

| | what | ms |
|---|---|---:|
| A | old: bilinear alpha, omatte at 2k | 150 |
| D | guided-filter alpha, F = I + (1-a) up(F_lo - B_lo) | 155 |
| K | D + colour-difference key in the edge band + despill (shipped, `workers/matte_refine.py`) | ~170 |
| N | BiRefNet at 2048 input | 545 |

- BiRefNet at 1024 is 140 ms of every row; refinement is 10-30 ms.
- Guided alpha vs bilinear (synthetic GT): SAD 32.8k vs 35.2k, gradient error 1.25 vs 1.63. Native 2048 is still best on alpha (28.0k) at 3.6x the time.
- The network marks the spill-lit hair halo opaque (alpha 1). No colour solve can fix that; the full-res key does (`out/e2e_crop.png`). The synthetic GT inherits the network alpha, so it penalises K for removing the halo: judge K visually.
- Despill limits the key channel to mean(other two), then returns half the removed light to all channels: lime rim -> warm highlight, not yellow (max limit) or saturated orange (no restore).
- Non-screen backdrops: `_key` returns None (backdrop excess share < 0.6), only the guided alpha and recolour run.
- Worker end to end 2048², cutout + composite: PNG 4.7 s -> 0.64 s (encode was 95%: `optimize=True` 5.8 s/image -> level 3 0.67 s), WebP 0.65 s (method 2 above 2 MP, concurrent encodes).

Bug found: `omatte.estimate_foreground_torch` passed torch's default stream (handle 0) through; the library read 0 as "none" and ran on a private non-blocking stream, racing torch's queued writes (NaN / stale foreground right after a BiRefNet call). Fixed by passing cudaStreamLegacy; `tests/test_matte_decontamination.py::test_device_path_waits_for_queued_torch_work`.

## Worker latency pass (2026-11-10)

    python3 matte/bench/stage_prof.py [--workers OTHER_REV/workers]      # per-stage, CUDA-synchronised
    python3 matte/bench/ab_identity.py OTHER_REV/workers [image ...]     # outputs + interleaved timing, old vs new
    python3 matte/bench/http_probe.py --url http://127.0.0.1:9094        # a running worker: latency, req/s at 1/2/4
    python3 matte/bench/preprocess_eval.py [image ...]                   # BIREFNET_GPU_PREPROCESS mask gate
    python3 matte/bench/forward_bench.py                                 # forward: compile modes, batch sizes

(env as for the other scripts: `BIREFNET_VRAM_BROKER_URL= OMATTE_LIB=... OENCODE_LIB=... BIREFNET_REFINER=...`)

Where the time went, 3090 Ti (shared), refiner v2, PNG data-URL source, best of N, ms:

| stage | 1024² before | after | 2048² before | after |
|---|---:|---:|---:|---:|
| decode (base64 + image) | 18 | 9 | 82 | 44 |
| network input (resize, upload) | 2.5 | 0.4 | 24 | 24 (4 with `BIREFNET_GPU_PREPROCESS=1`) |
| BiRefNet forward (eager fp16) | 130 | 130 | 130 | 130 |
| refiner / alpha upsample / key | 21 | 18 | 42 | 33 |
| colour solve | 3.5 | 2 | 54 | 12 |
| RGBA pack + download | 3 | 1 | 25 | 3 |
| encode WebP (m4 / m2 above 2 MP) | 60 | 56 | 199 | 167 (105 with alpha method 0 + filter 1) |
| encode PNG | 28 | 11 | 124 | 42 |

What changed (decoded pixels identical on every row unless stated; `ab_identity.py` over gs2k, nonkey, flowers, inner at
1024/2048, with and without the refiner, 4 request shapes x 2 formats: WebP byte-identical, PNG pixel-identical):

- PNG: `oenc_png_mt`, one filter + zlib-deflate strip per thread joined by sync flushes into one zlib stream
  (`BIREFNET_PNG_PARALLEL`, `BIREFNET_PNG_ZLEVEL=4`). 2048² RGBA 98 -> 26 ms at 8 threads, 18 at 16; +0.5..2% bytes on
  cutouts, +2..4% on opaque composites (z6 is size-neutral at 1.6x the time).
- WebP: `oenc_webp2` converts RGBA -> YUVA in parallel strips and encodes the alpha plane as a separate job (beside a flat
  dummy luma), then muxes VP8X/ALPH/VP8 by hand: same bytes as before. The alpha plane (lossless) is what bounds the
  encode at method <= 2 and on slow cores at any method; `BIREFNET_WEBP_ALPHA_METHOD[_LARGE]` lowers only its effort:

  | 2048² cutout, colour method 2 | 5950X ms | E5-2697v4 ms | bytes |
  |---|---:|---:|---:|
  | before | 188 | 451 | 627 k |
  | split, same alpha effort (default) | 174 | 419 | same file |
  | alpha method 0, alpha filter 1 | 111 | 248 | +11% |
  | alpha method 0, alpha filter 0 | 103 | 192 | +33% |

  At 1024² with colour method 4 the colour planes bound it (62 ms; prod 140 -> 125 with alpha method 1, +0.4% bytes).
  Method 2 instead of 4 for small images: 62 -> 38 ms, +1.7% bytes, visible-RGB PSNR -0.13 dB (33.03 -> 32.90).
- Decode: stills go through `oenc_webp_decode` (threaded libwebp; 2048² 70 -> 25 ms) and `oenc_png_decode` (libdeflate +
  SSE2 Paeth, 8-bit RGB/RGBA non-interlaced; 76 -> 37 ms), pixel-identical to Pillow; animation, EXIF, palettes, tRNS and
  every other format stay on Pillow, which no longer copies twice through `exif_transpose` + `convert`.
- One upload of the source per request, shared by the network input, refiner, alpha upsample and colour solve (was three
  host copies and three uploads at 2048²); the cutout is packed to RGBA on the device and handed to the encoder as an
  array (was device -> PIL RGB -> RGBA -> putalpha -> array).
- `matte_refine._edge_band`: the 193x193 max pool is two 1-D pools, same tensor, 33 -> 0.6 ms.
- `BIREFNET_GPU_PREPROCESS=1` (off by default, not identical): antialiased bilinear on the device instead of Pillow's.
  Mask mean |delta| 1.1e-4..4.0e-4, binary agreement >= 0.99966 on 7 images (the compiled-model gate is 5e-3 / 0.99);
  final alpha differs by more than 8/255 on 0..0.1% of pixels.

Not taken: pinned staging buffers (the 12 MB upload is 0.65 ms pageable, 0.49 ms pinned; the RGBA download 3 ms); torch.compile `max-autotune` (no result after 25 min of autotuning on the shared card); batching concurrent
jobs through one forward (see `forward_bench.py`; a new batch size makes a compiled model recompile under the model lock).

## Learned refiner (matte/train, 2026-10-06)

Pipeline: `gen.py` (Qwen renders of hair/fur/veils/dandelions on green/blue screens + scenes) -> `teacher.py`
(network alpha + full-res key, hue/translucency-gated vector despill, auto-reject) -> `build.py` (originals +
composites on scenes/colours/spill-lit screens, BiRefNet on each) -> `train.py` (2.5M-param refiner,
(image, prob) -> (alpha, F)) -> `compare.py` (real renders side by side). Data in `~/matte-data`.

v2 (295 subjects, 2950 samples, 20k it, 3090 Ti 53 min), held-out SAD / fg error:

| | BiRefNet | K (hand-built) | v2 |
|---|---:|---:|---:|
| scenes | 27.2 / 8.5 | 26.8 / 5.3 | **22.9 / 4.4** |
| colours | 11.6 / 4.0 | 10.6 / 3.6 | **10.1 / 2.2** |
| synthetic screens | 22.7 / 12.7 | 22.7 / 8.7 | **12.5 / 5.5** |
| real screen renders | 18.3 / 7.7 | **5.3 / 1.2** | 8.9 / 2.7 (GT is teacher, K-like: biased to K) |

On real renders v2 still leaves strong backlit lime halos that K removes, so the worker (BIREFNET_REFINER)
uses the refiner only when no keyable screen is detected and the K path otherwise. Refiner: 22 ms at 1k.
Next: more real halo data (more backlit prompts), train on originals only for the screen branch.
