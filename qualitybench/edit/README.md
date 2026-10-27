# Reference-edit eval (Qwen Image 2.1, ra2 lane)

10 cases from GEdit-Bench (stepfun-ai/GEdit-Bench, English, one per category: background, colour, motion, portrait retouch, style, add, remove, replace, text, tone). Scored by a Gemini judge (SC instruction compliance, PQ perceptual quality, PR preservation) plus side-by-side visual review.

    uv run --with pyarrow --with pillow python qualitybench/edit/prep.py data-0000{0..6}-of-00009.arrow   # sources -> src/ (snowstorm: rows API)
    uv run --with pillow python qualitybench/edit/run.py --url http://127.0.0.1:8793 [--only cfg,..]       # out/<cfg>/<case>, resumable
    uv run --with pillow python qualitybench/edit/rank.py --configs a,b,c                                  # comparative judge (use this)
    uv run --with pillow python qualitybench/edit/judge.py                                                 # absolute judge (scores cluster at 7)
    uv run --with pillow python qualitybench/edit/report.py                                                # out/report.html grid

Lane: `tools/start-ra2-lane.sh` with `RA2_PORT=8793 RA2_EASYCACHE=0.08 RA2_VAE_TILING=1 OMNISERVE_NATIVE_SD_REFERENCE_EDIT=1 OMNISERVE_NATIVE_SD_PARAMS_BACKEND='*=cpu' OMNISERVE_NATIVE_SD_VAE_TILE_X=64 OMNISERVE_NATIVE_SD_VAE_TILE_Y=64` (prod parity; resident weights OOM on the shared 3090 Ti). `cfg_end` needs the patched sd.cpp lib (`RA2_SD_LIB=.../sdcpp-qwen-prefixkv/build-86-cfgend/bin/libstable-diffusion.so`). Config names: `s<steps>_c<cfg>`, `ec` EasyCache threshold, `ce` cfg_end, `fs` flow shift.

## Findings (2026-10-23, seed 7, 1 MP at source aspect)

Comparative judge, mean over 10 cases x 2 shuffles (0.75 * sqrt(SC*PQ) + 0.25 * PR); scores from separate rank sets are not directly comparable, the ordering within a set is.

| config | model evals | rank score | note |
|---|---:|---:|---|
| square 1024, 9 steps, CFG 1 (old ManifoldGen call) | 9 | 5.37 | reframes / crops / stretches non-square sources |
| 9 steps, CFG 1 (old lane default) | 9 | 6.28-6.37 | timid edits: "younger", "snowstorm", beach background leave most of the source |
| 20-30 steps, CFG 1 | 20-30 | 6.45-6.49 | sharper but same weak edits; beach keeps the dirt infield |
| 8-12 steps, CFG 4 | 16-24 | 6.28-6.36 | strong edits, under-converged: washed-out, hazy backgrounds |
| 14-20 steps, CFG 4 | 28-40 | 6.58-6.76 | knee; CFG 3/6, negative prompt, flow shift 2, EasyCache 0.15 all within judge noise |
| **16 steps, CFG 4, cfg_end 0.5 (deployed)** | **24** | 6.47 (vs 6.49 full CFG, same set) | same edit strength as full CFG; younger-face edit keeps identity better |
| Lightning 4-step LoRA | 4 | 6.02 | 2.5x faster, but "make her look younger" does nothing and unrelated details drift |

- CFG > 1 is what makes edits happen; steps fix convergence. CFG 6 washes out skies.
- `cfg_end=F` (sd.cpp patch, `sdcpp-cfg-end.patch` in the prod deploy dir) skips the unconditional pass after the first F of the steps. Without the arg the lib is bit-identical to 9bc7b35.
- Gemini absolute scores barely separate configs; the side-by-side rank judge and crops do.

## Prod (RTX 5090, shared with the 8791 gateway)

Drop-in `zzzzzzzzz-editcfg.conf`: binary `deploy-editcfg-20261023` (1397c30 + edit defaults), lib `libstable-diffusion-9bc7b35-cfgend.so`, `OMNISERVE_NATIVE_SD_EDIT_STEPS=16`, `OMNISERVE_NATIVE_SD_EDIT_MIN_GUIDANCE=4`, `OMNISERVE_NATIVE_SD_EDIT_EXTRA_ARGS=cfg_end=0.5`. Steps apply when a request omits them; the guidance floor also lifts netwrck's hardcoded CFG 1; extra args apply when a request sends none. Rollback: delete the drop-in, daemon-reload, restart.

Wall time on the contended card: 9-step CFG 1 edits 37-53 s, 16-step full CFG 73-92 s; about 30 s per edit is fixed overhead (weight streaming, VAE decode OOM retries, broker waits), which is the next thing to cut.
