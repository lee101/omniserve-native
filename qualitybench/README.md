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

## Turbo lane (prod since 2026-10-01: Viggle v0.3 6-step LoRA, runtime-applied on the Q4_K_M base)

- The cross-hatched, desaturated fox came from this lane: with the LoRA on the Q4_K_M base the fur is muddy, and the 2 px lattice is as strong as in the base model (Nyquist energy 6.7e-3 without the notch, 2e-5 with it).
- `turbo` request field (`false` = base path: floored steps, `HQ_EASYCACHE_THRESHOLD`, no LoRA); `notch` applies to both paths.
- The model card says to keep the LoRA unmerged at scale 1.0 because merging into quantized weights loses part of the update. The pre-merged v0.3 GGUFs (fp32 merge, quantized once) are better and faster than runtime LoRA on Q4: Q6_K 10.4 s, Q5_K_M 9.8 s, runtime LoRA 11.5 s (3090 Ti), and the fur and fox shape are visibly cleaner. Serving them next to the base model needs a second diffusion context (the text encoder would be loaded twice), so it is not wired into the unit yet.
- LPIPS against dense 30 is meaningless for turbo (different distilled style; 0.14 to 0.40).

## Prod canary (RTX 5090, 2026-10-02, drop-in `zzzzzzz-tiers.conf`, binary `omniserve-native-deploy/deploy-tiers-20261002`)

1024², steady state through the live unit: default (turbo 6-step LoRA) 8.5 to 9.2 s, `"turbo":false` (30 steps, EasyCache 0.15) 10.3 to 11.0 s. Both tiers return notched images. The base tier costs about 2 s more on the 5090 and avoids the muddy LoRA-on-Q4 look; making it the default means deleting `OMNISERVE_NATIVE_SD_TURBO_NODES` in `zzzzzz-turbo.conf`.

## Image format

The server answers WebP (q85) whatever `output_format` asks for, so a writer that trusts the request saves WebP bytes under `.png` (viewers other than a browser refuse those). Sweep tools now pick the extension from the bytes, and `qualitybench/fix_ext.py DIR...` renames existing files. WebP q85 is the default format everywhere (`OMNISERVE_NATIVE_SD_WEBP_QUALITY`, deploy script now 85).

## Acceleration sweep (2026-10-02, 3090 Ti, 1024², seed 0, 7 prompts, 36 configs)

`sweep.py` (server variants + request grid, resumable), `score_sweep.py` (LPIPS/PSNR vs dense 30, CLIP, sharpness, Nyquist), `build_report.py` (HTML gallery; `embed` for a single file). Results: `results/sweep-20261002/{results.jsonl,summary.csv,summary.json}`; clean single-prompt timings (no GPU contention): `results/sweep-20261002/retime.jsonl`. Full images are not committed; rerun `sweep.py`.

Clean timings (fox2) and mean LPIPS vs dense 30 (36.8 s):

| config | s | LPIPS |
|---|---|---|
| 16 steps, EC 0.15 | 13.5 | 0.168 |
| 20 steps, EC 0.15 | 15.9 | 0.106 |
| 24 steps, EC 0.15 | 17.2 | 0.078 |
| 30 steps, EC 0.30 | 18.0 | 0.077 |
| 30 steps, EC 0.15 (deployed) | 19.0 | 0.045 |
| 30 steps, EC 0.08 | 21.1 | 0.031 |
| 30 steps, EC 0.05 | 24.3 | 0.023 |
| 30 steps, EC 0.15, cache_end 0.7 / 0.5 | 25.6 / 31.7 | 0.031 / 0.024 |

- 30 steps with EasyCache 0.15 stays the knee: composition matches dense 30 (visually checked on quad, fox2). At 24/20/16 steps the composition drifts (poses, a figure changes) even where LPIPS is low-ish.
- ucache is unsupported for this model (log: "only UNET models"), output equals dense. cache-dit, dbcache and taylorseer (skip 2/3) give identical output to each other, are slower than EasyCache and worse in LPIPS (0.044 at 25.8 s vs 0.045 at 20.8 s). spectrum 0.057 at 20.4 s. EasyCache is the only useful cache here.
- `cache_end` (stop caching late) costs more time than it buys.
- `notch` off: Laplacian sharpness x1.97 (base) / x2.85 (turbo) vs the dense reference, lattice returns. Keep on.
- Turbo: LoRA scale 1.3 oversharpens (x2.57), 0.7 is softer and closer to the base look (LPIPS 0.20 vs 0.24). 4-node schedule is fastest but weaker; 8-node is no better than 6. Pre-merged Q6/Q5 turbo cost less than runtime LoRA and have the highest CLIP delta (+0.017/+0.018); fox2 shows a different, cleaner composition.
- Teleport was not benchmarked here (master sd.cpp lib has no latent API); unseeded teleport doubled HQ time on the 5090 (19.3 s vs 9.9 s) and is no longer sent without a seed.

## Hard-prompt sweep and latent trajectories (2026-10-03, 3090 Ti, sd.cpp = prod tree 9bc7b35 rebuilt for sm_86)

`sweep_prompts_hard.json`: 11 prompts (men and women duo/trio/quad, shirtless, group of five, hands, couple, witch, landscape). `sweep.py --prompts ... --tags ...`; `SWEEP_PROMPTS` env points `score_sweep.py`/`build_report.py` at the same file. The local lib now matches prod (latent replay API, VAE OOM retry, `sd_ctx_release_vram`): `/vfast/data/code/sdcpp-qwen-prefixkv/build-86` (source copied from prod `sdcpp-qwen-prefixkv`; prod tree also carries an uncommitted ggml and `SD_RESIDENCY_EVICT_MRU` diff).

Visual read (88 images, 8 configs; premerged_q5 server hung at load and was not rerun, the 2026-10-02 sweep had q5 trio at LPIPS 0.35):
- `base30_ec15` matches dense 30 on every prompt. `base24_ec15` changes hair and clothing colour on m_trio.
- Turbo (6-step LoRA) keeps anatomy on duos, trios, hands, and group-of-five, but changes look/colour vs base (darker, more contrast) and breaks the crouching/jumping pose in m_quad; LoRA scale 0.5/0.7 keep the base composition on m_duo but tangle the crouching figure in m_quad; premerged Q6 folds a head in m_quad. Scale and premerge do not fix pose failures; the base tier does.
- netwrck now routes multi-person prompts to the base tier at turbo price (`ra2NeedsBaseTier`, commit 9e78328d, deployed as netwrckprod177).
- Zero HTTP 500 in the 88 images on the prod-parity lib; the master lib produced 14 VAE-decode OOMs on the same box when another process held VRAM. Not isolated whether the retry or the quiet GPU explains it.

Latent trajectories (`traj.py` saves the sampler latent at every step with `sd-cli --latent-save-steps`; `traj_analyze.py` recovers x0 predictions `x0_k = z_k - sigma_k * v_k` from the Flux schedule). 5 of 11 prompts finished (capture was stopped by memory pressure):
- x0-prediction relative error to the final decays smoothly: 0.10-0.16 at step 20, under 2% only at step 28-29 for every prompt, step-to-step change flat at 1.2-2.6% after step 8. Landscape converges slower than the character prompts, so "simple scenes stop earlier" does not show up in latent distance.
- Conclusion: no usable plateau for a convergence cutoff or an adaptive step count; EasyCache already skips the redundant evaluations. Not tested: LPIPS of truncated runs per prompt, a cutoff learned from the first 3-5 steps.

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
