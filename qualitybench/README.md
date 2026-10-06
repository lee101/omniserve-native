# qualitybench

Image quality bench for the Qwen "ra2" lane. Needs a server on :8792 (`qualitybench/serve.sh`, env via `SDENV="K=V ..."`).

    qualitybench/variants.sh 'tag|steps|{"cache_threshold":0.1}' ...   # ONLY=fox,trio SEED=0
    uv run --with pillow python qualitybench/zoom.py x0,y0,x1,y1 out.png img1 img2 ...
    uv run --with pillow --with numpy python qualitybench/spec.py a.png b.png   # FFT lattice metric

`prompts.json`: fox (the cross-hatch repro), fox2, trio / duo / quad (multiple full-body characters, different poses).

## Findings (RTX 3090 Ti, Q4_K_M, 1024², seed 0)

- Cross-hatching = a 2 px grid: the fox crop has its strongest FFT peak at exactly Nyquist (0.5 cycle/px, x and y) in every variant, including dense sampling, VAE tiling on/off and 20/30 steps. The user's prod image has 2x the Nyquist energy of local output. EasyCache (0.05, 0.1, end 0.7) and tiling are not the cause.
- Nyquist notch (`SD_NOTCH=1`, request `notch:true`): Nyquist energy 9.2e-3 -> 2.5e-5, no visible detail loss, 22 ms CPU per 1024² image. Applied before WebP encode.
- Remaining: faint horizontal seams at 8/16 px periods (~1 grey level), not addressed.
- EasyCache 0.1 vs dense: 15.7 s vs 25.6 s with the same lattice; 0.05 is 19 s. 30 steps (24-29 s) improves stockings/hand detail slightly on the duo prompt over 20 (20-22 s).
- Multi-character: trio/duo/quad compose correctly with distinct poses; weak spots are shared faces/hair across characters and small, low-detail faces when 4 figures share a 1024² frame (use a taller/wider canvas or fewer figures per image).

## Frontier: steps x EasyCache (3090 Ti shared, 4 prompts fox2/trio/duo/quad, LPIPS-alex vs dense 30 steps, seed 0)

| config | s/img | LPIPS | PSNR |
|---|---:|---:|---:|
| 30 steps, EasyCache 0.30 | 17.4 | 0.069 | 29.5 |
| 30 steps, 0.20 | 18.2 | 0.041 | 30.8 |
| **30 steps, 0.15 (deployed)** | 18.7 | 0.030 | 31.6 |
| 20 steps, 0.05 (old default) | 19.4 | 0.084 | 23.8 |
| 30 steps, 0.10 | 21.4 | 0.020 | 33.4 |
| 24 steps, 0.05 | 21.6 | 0.055 | 26.8 |
| 30 steps, 0.05 | 23.8 | 0.010 | 36.3 |
| dense 30 (reference) | 36.0 | 0 | - |
| 40 steps (0.3 / 0.5) | 26.9 / 25.6 | 0.063 / 0.078 | - |

- More steps with a looser cache beats fewer steps with a tight cache: 30 steps at 0.15 costs the same as 20 at 0.05 and is 3x closer to dense 30.
- No gain from other samplers (res_multistep, ipndm, er_sde is broken), schedulers (beta, sgm_uniform) or 40 steps. `flow_shift` is ignored on the text-to-image path.
- Per-prompt LPIPS at 20 steps vs dense 30 is 0.06-0.10 for every prompt (trio is not an exception), so prompt-dependent step counts do not pay off.
- Adaptive early exit on the x0-prediction delta (`early-exit-experiment.patch`, sd.cpp `sample_euler`): the relative change per step stays at 1-2% down to the last step (it grows for the multi-character prompts), so a tolerance either never fires or damages detail (fox2 LPIPS 0.027 -> 0.074 at tol 0.02). EasyCache already is the adaptive mechanism. Not shipped.
- The server floors `steps` to `OMNISERVE_NATIVE_SD_MIN_STEPS` (30 in `serve.sh` and the prod unit); set it to 0 when sweeping steps.

## Lib, VAE and kernel A/B (2026-10-05, 3090 Ti shared, 1024², seed 0, fox2/trio/duo/quad, 30 steps EasyCache 0.15)

`qualitybench/ab.sh TAG [ENV=VAL ...]` (`LIB=` picks the sd.cpp build) starts a private lane with the prod config, renders the four prompts, appends to `out/ab.log`; `LOG=ab.log uv run --with torch --with lpips --with pillow --with numpy python qualitybench/score.py base` scores every tag against `base`. `OMNISERVE_NATIVE_SD_LOG=1` prints sd.cpp's per-stage timings (sampling, decode) to the server log.

Per-image stages: text encode 0.04 s (cached), sampling 15.1-16.3 s, VAE decode 3.7 s tiled. The decode is the only stage with slack.

| variant | decode | LPIPS vs base | PSNR | note |
|---|---:|---:|---:|---|
| tiled 32 (old default) | 3.7 s | 0 | - | |
| tile overlap 0.25 | 3.7 s | 0 | 96 | overlap is ignored, output bit-identical |
| untiled, master lib | 1.6 s | 0.006 | 56 | OOM and 7 s decode when another tenant holds VRAM |
| untiled, prod lib (`sdcpp-qwen-prefixkv`) | 1.3 s | 0.009 | 43 | its OOM retry falls back to smaller tiles (one 6.1 s decode in 8 images) |
| tile 64, prod lib | 1.4 s | 0.008 | 56 | |
| CUDA graphs (`GGML_CUDA_GRAPHS=ON`) | 3.8 s | 0 | 96 | bit-identical, sampling not faster (15.9 vs 15.1-16.0) |
| `GGML_CUDA_FORCE_CUBLAS=ON` | 3.9 s | 0.041 | 31 | sampling not faster, output drifts (fp16 dequant path) |
| turbo LoRA (runtime) | 3.7 s | 0.218 | 21 | 12.9 s; different distilled style, see the tier notes |

- Shipped: `tools/start-ra2-lane.sh` prefers the prod-parity lib and runs it untiled (about 2.4 s, 13%, per 1024² image at LPIPS < 0.01, far inside the 0.03-0.045 that EasyCache 0.15 already costs). The master lib keeps tiling.
- Rejected: CUDA graphs and cuBLAS give no sampling speedup; the sampler is bound by Q4_K/Q6_K `mul_mat_q` (62% of kernel time) and elementwise bcast/cpy/concat kernels (about 25%). The elementwise chain (modulate = repeat + mul + add) is the only fusion target left, worth under 1 s per image, and needs sd.cpp graph changes.
- Lane warm-up: `OMNISERVE_NATIVE_SD_WARMUP=1` renders one 1024² step before listening (sd.cpp stages encoder, DiT and VAE tensors and builds each graph on the first request). Start to ready 13.7 s (was 12.5 s with a 256² probe), and the first real 1024² request drops from 22.4 s to 17.6 s.
