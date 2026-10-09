# Z-Image LoRA routing toward Qwen 2.1 quality - 2026-10-09

Goal: get embedded Z-Image (:8791, Q4_K, sd.cpp) closer to Qwen Image 2.1 (:8792) at Z-Image cost by
routing prompts through catalog LoRAs.

## Method

- 42 prompts sampled from the last 30 days of `generated_images` (cutedsl DB, read-only), 7 each:
  portrait, anime, product, landscape, text, SFW characters. Celebrity/NSFW prompts dropped.
  `experiments/zimage-lora-routing/prompts.jsonl`.
- Z-Image: prod :8791 (resident, `SD_STREAM_LAYERS=0`), background tier, 1024^2, 8 steps, seed 1234,
  explicit `loras`. Qwen 2.1: prod :8792 background tier, prod defaults (6-step turbo LoRA), same seed.
- Scores: CPU CLIP ViT-L/14 prompt cosine + LAION aesthetic head (`cutedsl-site/inference/aesthetic_score.py`),
  CLIP image cosine to the Qwen render. No pairwise preference judge exists in-repo
  (`latentteleport/judge.py` is CLIP-only); contact sheets were reviewed by hand.

## Results (all 42 prompts)

| variant | aesthetic | CLIP prompt | beats Qwen aes | p50 render |
| --- | --- | --- | --- | --- |
| Z-Image plain | 5.90 | 0.292 | 14/42 | 4.51 s |
| current keyword `auto_lora` (cutedsl lora_search) | 5.88 | 0.260 | 14/42 | 5.90 s |
| aesthetic_base 1.0 | 5.94 | 0.300 | - | 5.2 s |
| the_look 0.6 | 6.25 | 0.302 | 28/42 | 5.92 s |
| the_look 0.8 | 6.39 | 0.293 | 29/42 | 5.94 s |
| the_look 1.0 (28 prompts) | +0.61 vs plain | -0.013 vs plain | - | 5.95 s |
| **routed** (below) | **6.23** | **0.299** | **27/42** | 5.9 s routed, 4.5 s text |
| Qwen 2.1 | 6.07 | 0.283 | - | 10-18 s inference |

Per category, routed vs plain (aesthetic/CLIP): anime +0.52/+0.007, character +0.45/+0.009,
portrait +0.45/+0.008, landscape +0.32/+0.003, product +0.18/+0.015, text 0/0.

Rejected:
- Current keyword `auto_lora`: CLIP -0.033 (wrong style LoRAs, e.g. pop-art on a Giotto portrait, orcs on a wine glass).
- Trigger-word templates: `zit v6` leaked into rendered text ("zitv6" logo); aesthetic trigger no better than none.
- the_look on text prompts: aesthetic up but lettering degrades (typos, lost words), so text routes stay plain.
- Multi-LoRA (the_look 0.6 + anime_serenity 0.4): +0.41 vs +0.52 single on anime, +1.4 s more. Category style
  LoRAs (serenity, medieval landscape, vintage poster, smartphone, z_art) were weaker or negative.

## LoRA format defects in native sd.cpp

sd.cpp applies `diffusion_model.*` (lora_A/B) keys fully. It drops the FFN of kohya `lora_unet_*`
files ("Only (180 / 450) LoRA tensors have been applied" for aesthetic_base: `feed_forward` becomes
`feed.forward`), and PEFT `*.lora_A.default.weight` / `base_model.model.*` files are silent no-ops
(z_aesthetic_anime output byte-identical to plain). Affected: aesthetic_base, pixel_art, anime_artistic,
cosplay_nsfw (partial); z_aesthetic_anime, app_logo_maker, product_screenshots (no-op). The registry also maps
app_logo_maker and product_screenshots to the same `adapter_model.safetensors`. Routes only use dm-format LoRAs
(tested in `tests/test_lora_routes_config.py`).

## Routing (implemented, flag default off)

`OMNISERVE_NATIVE_LORA_AUTO_ROUTE=1` + `OMNISERVE_NATIVE_LORA_ROUTES=deploy/zimage-lora-routes.json`.
Applies only to `/v1/images/generations` text-to-image requests with no `loras`/`lora_id`, no
automatic NSFW LoRA, and `auto_lora` not false (netwrck RA1 sends `auto_lora:false`, so it is unaffected).
First matching route wins: people/anime keywords -> the_look 0.8; everything else -> the_look 0.6;
text/logo/typography/poster/sign/UI prompts are excluded and render plain. No prompt prefix.
LoRA ids resolve through the existing cache/registry path; a missing file logs and renders plain.

## Latency

sd.cpp applies LoRAs at runtime for Q4_K weights: +1.4 s per 1024^2 image (4.51 -> 5.92 s p50, +31%),
independent of strength; two LoRAs +2.8 s. Swapping LoRA/no-LoRA between consecutive requests showed
no extra cost beyond that (min 5.85 s while interleaved with plain batch renders). This is still
~2-3x cheaper than Qwen. Baking the_look 0.6 into a second Q4_K GGUF would remove the runtime cost
but needs a second resident DiT for text prompts.

Caveat: the aesthetic head rewards the_look's illustrative detail; composition is still more cinematic
on Qwen, and the_look pulls some full-body prompts toward close-ups.
