# MOSS style LoRAs and hires pass (2026-11-06) — INTERNAL

4-step merged Q4_0 lane, 1024 px, CFG 1, seeds 0-1, RTX 3090 Ti. Renders and sheets in `qualitybench/out/{lora-ab,hires-ab}` (gitignored).
Reproduce: `qualitybench/moss_lora_ab.py --out qualitybench/out/lora-ab` (config `moss_lora_ab.json`), `moss_hires_ab.py`,
then `face_score.py <dir>` (RetinaFace crops -> pyiqa `topiq_nr-face`, Laplacian; whole image `topiq_nr`; CLIP ViT-B/32).

## LoRAs

| File (`/vfast/data/models/moss/lora/`) | What it is | Trigger |
| --- | --- | --- |
| `moss-digital-painting-dora.safetensors` | DoRA r32, detailed digital painting / concept art | none |
| `moss-nai5-painterly-cine.safetensors` | ai-toolkit r32, NAI5-like painterly cinematic anime | `@painterly cine` |
| `moss-r32-v2-kohaku.safetensors` | Kohaku LyCORIS r32, photographic realism / detail (skin texture, less AI gloss) | none |

sd.cpp ignores `dora_scale`. Checked against the SFT Q8 weights: `m / ||W + BA||` is 0.999 (p5 0.996, p95 1.002) on every target, so
loading the DoRA as a plain LoRA is effectively exact.

## Scores (mean over prompts x seeds; IQA higher is better)

| Style | Variant | face IQA | img IQA | CLIP | s/img |
| --- | --- | ---: | ---: | ---: | ---: |
| people | route (suffix only) | 0.571 | 0.489 | 0.362 | 3.2 |
| people | **r32 0.8** | 0.635 | 0.611 | 0.362 | 4.0 |
| people | r32 0.6 / 1.0 | 0.673 / 0.685 | 0.598 / 0.615 | 0.364 / 0.359 | 4.0 |
| people | beauty 0.7 | 0.664 | 0.550 | 0.376 | 4.0 |
| anime | route (suffix only) | 0.396 | 0.623 | 0.396 | 3.2 |
| anime | **r32 0.8** | 0.475 | 0.689 | 0.401 | 4.0 |
| anime | nai5 0.8 + trigger | 0.525 | 0.704 | 0.388 | 4.3 |
| painting | plain | 0.650 | 0.536 | 0.361 | 3.2 |
| painting | **dp 0.8** | 0.673 | 0.569 | 0.370 | 3.9 |

Seen in sheets: dp adds detail and a cleaner painterly finish; r32 is near-identity in style on anime but sharper; nai5 fixes some
compositions (one duplicated head in the route render) but drifts toward revealing clothing (bare shoulders, cleavage) on neutral
prompts, so it is opt-in only. Full-body anatomy (extra leg) varies by seed with or without LoRAs; 2 seeds are not enough to rank it.

## Hires pass (base -> lanczos upscale -> img2img on the same 4-step model)

| Variant | s/img | face IQA | img IQA |
| --- | ---: | ---: | ---: |
| 1024 base | 3.4 | 0.540 | 0.556 |
| 1024 -> 1536, strength 0.35 / 0.50 | 12.1 / 15.8 | 0.545 / 0.499 | 0.495 / 0.479 |
| 1024 -> 1280, strength 0.40 | 10.1 | 0.482 | 0.502 |
| 768 base / -> 1152 strength 0.40 | 1.9 / 7.2 | 0.507 / 0.466 | 0.570 / 0.464 |

Finer skin/hair texture at 1.5x, no structural face fix, 3.5-4.5x cost: not adopted (the BBOX V3 HiRes workflow does this with the
8-step DMD LoRA at 0.9 and CFG 1.8 on the dense model; our 4-step model already sits at that quality point).

## Routing changes

`animepaint` (nai5 0.8, opt-in phrasing: painterly/cinematic anime, anime painting), `anime` + r32 0.8, `digitalpainting` (dp 0.8),
`people` + r32 0.8. New per-route `keyword_bonus` (default 0.15): long prompts embed with low similarity, so an exact style phrase
needs more weight; both new routes use 0.3 with `min_score` 0.45. Route eval: 53/54 tuning (was 45/46), 19/21 held-out (was 14/16); the
remaining misses are pre-existing people/anime paraphrases.
