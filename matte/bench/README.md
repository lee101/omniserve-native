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
