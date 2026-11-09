# MOSS fast lane: bit-identical DiT / LoRA fusions (2026-11-10) — INTERNAL

4-step merged Q4_0, CFG 1, 1024x1024, RTX 3090 Ti shared with two other compute jobs (about 1.75x slower than an idle
card during the final run). sd.cpp `moss-i2i-reference` 1f71d21+ / ggml `moss-dit-fusion`; lane via `qualitybench/serve_moss.sh`.

## Method

The GPU load from other jobs changes on a minutes scale, so separate-session A/B is useless. `GGML_AB_FILE=<file>` makes
ggml re-read an integer bitmask (bit set = that change off: 1 situ, 2 strided norm+rope, 4 modulate, 8 K F16, 16 LoRA fuse,
64 LoRA keep; bit 32 turns the lossy LoRA option on), so one resident server alternates variants per request. Same prompt and
seed for every variant, PNG pixels hashed, ratio taken per pair. `GGML_CUDA_OP_PROF=1` prints per-op time per graph (sync per
node; min-based column is the usable one on a shared card).

## Stage breakdown (baseline, idle-card moments)

| Stage | Time |
| --- | ---: |
| text encode, new prompt (Qwen3.5 Q8 resident on GPU; 0 on conditioning-cache hit) | 0.08-0.13 s |
| DiT, 4 steps | ~2.8 s (0.71 s/step) |
| VAE decode (untiled, 2.8 GB buffer) | 0.3-0.46 s |
| PNG encode | 0.06-0.09 s |
| runtime LoRA, same route as previous request | +0.9 s (+0.22 s/step) |
| runtime LoRA, route changed | another +0.4 s hot page cache, +1 to 20 s when the file is cold on /vfast |

DiT step by op (min-based ms, before -> after): situ activation 213 -> 19 (two slice copies + 4 elementwise passes, one of
them a 2.7 ms broadcast MUL, now one kernel), modulate 42 -> 13, Q/K attention prep 50 -> 14, Q4_0 GEMMs (MMQ) ~340 and
flash attention ~120 unchanged.

## Result (final paired run, 10 pairs per row, contended card)

| Request | baseline min / med | fused min / med | paired ratio (med) | pixels |
| --- | ---: | ---: | ---: | --- |
| plain, new prompt | 6.92 / 7.09 s | 5.80 / 5.99 s | 0.845 | identical |
| plain, same prompt | 6.96 / 7.05 s | 5.86 / 5.98 s | 0.846 | identical |
| LoRA route, route changes every request | 8.92 / 9.51 s | 7.25 / 7.69 s | 0.818 | identical |
| LoRA route, same route | 8.94 / 9.08 s | 7.20 / 7.27 s | 0.804 | identical |

40/40 images hash-equal to the baseline graph in this run (and to the pre-change library in earlier runs, 0 mismatches in
every run). Idle-card baseline measured the same day: plain 3.27-3.39 s, LoRA same route 4.27 s; the ratios put the fused
lane at about 2.85 s and 3.45 s (not re-measured idle).

## Changes

- `GGML_GLU_OP_SITU` (ggml, CPU + CUDA) used by `moss_image.hpp` on CUDA: FFN activation on the packed hidden|gate rows.
  `SD_MOSS_NO_FUSE=1` reverts.
- RMS_NORM+MUL+ROPE_APPLY fusion accepts row-strided views, so Q joins upstream's norm/rope/flash-attention fusion and K
  is written as F16 in the same pass (`GGML_CUDA_NO_STRIDED_NORM_ROPE`, `GGML_CUDA_NO_NORM_ROPE_F16`).
- RMS_NORM+MUL+MUL+ADD+ADD (adaLN modulate) in one kernel (`GGML_CUDA_NO_NORM_MOD_FUSE`).
- Runtime LoRA linear: SCALE, CONCAT (q|k|v) and ADD replaced by one combine pass per part (`GGML_CUDA_NO_LORA_FUSE`).
- `SD_LORA_KEEP=N` (sd.cpp; `LORA_KEEP`, default 8, in the fast lane): idle LoRA files stay registered and their
  LoraModels stay on the GPU, ~0.9 GB for all 9 route files. Without it every route change re-parses and re-uploads.
- `serve_moss.sh` takes model files from `~/models-fast/moss` (NVMe copy, 12.5 GB) when present: lane start 12 s vs ~9 min.

Together situ + strided rope gave 0.87, modulate + K F16 another ~0.97 on plain requests.

## Rejected / not done

- Lossy, opt-in `GGML_CUDA_LORA_GEMM_ACC=1`: up projection as SGEMM alpha=scale beta=1 into the result. LoRA requests
  0.75-0.76 vs 0.80-0.82 of baseline, but PSNR vs exact 14.5-23.8 dB (med 21.7, 20 images): same composition, details drift
  over the 4 steps. Not default, quality not scored.
- F16 dataflow between GEMMs (Z-Image style epilogues): the Q4_0 GEMMs run through MMQ with F32 in/out here, there is no
  F16 conversion to remove.
- CUDA graphs (build has them off): ~600 launches/step, estimated ~1%.
- Flash attention is already the CuTe kernel; VAE tiling is already off; text encoder is already resident.
- VAE: bias-add into conv and GroupNorm+affine+SiLU fusion estimated ~50 ms together (<2% each); cuDNN convs are not
  bit-identical (57 dB on Z-Image) and were not ported.
- Remaining LoRA cost is two rank-r SGEMMs plus one combine pass per target (~0.13 s/step); cheaper needs the lossy path.
