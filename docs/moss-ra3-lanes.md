# MOSS-Image lanes (ra3) — INTERNAL

Internal-only: do not describe the style routing, LoRA set or model in public docs, API responses,
tool pages or marketing. Public name is "ra3"; callers see one art generator with a quality choice.

## Lanes (`qualitybench/serve_moss.sh`, needs the sdcpp-moss build and the Q8_0 Qwen3.5 text encoder)

| Lane | Model / recipe | Time @1024 | Port (netwrck default) |
| --- | --- | ---: | --- |
| `LANE=fast` | 4-step merged Q4_0, CFG 1, keyword style routing | 3.4-6 s | 8793 (`RA3_LOCAL_IMAGE_URL`) |
| `LANE=hq` | dense SFT Q4_K, 30 steps, CFG 3, EasyCache 0.15, routing with negatives | 27-40 s | 8794 (`RA3_LOCAL_HQ_IMAGE_URL`) |
| `LANE=nsfw` | 4-step + adult LoRA 1.0 on every request (no routing) | ~4.4 s | separate, never shared with public routes |

`hq` is a different look (softer, more natural, more compositional variety), not a strict upgrade: the
4-step lane is more saturated and spelled lettering better in our sheets (`results/moss-routes/fast_vs_hq.jpg`).
Sell it as "natural look", priced 3x in netwrck (`ra3HQPriceFactor`).

## Routing (`deploy/moss-lora-routes.json`, `OMNISERVE_NATIVE_LORA_AUTO_ROUTE=1`)

First matching route wins; each can add a trigger prefix, a prompt suffix, a negative (only effective at
CFG > 1, i.e. the hq lane) and LoRAs by absolute path. Routes: psychedelic, gouache, impressionist,
3d animated, anime (suffix only: `anime character, dynamic composition, clean anatomy` fixed the
three-legs failure on 4/4 seeds without a LoRA), beauty (0.7), people (anatomy suffix). Lane steps/CFG are
pinned server side (`SD_LANE_STEPS`, `SD_LANE_GUIDANCE`) so a client's hardcoded 20 steps cannot over-sample
the distilled model. `SD_PUBLIC_NAME` (default `ra3` in the script) hides the checkpoint name from
responses, `/health` and `/v1/models`; the route name only appears in stderr logs.

## Evaluated, not adopted

- Banding-fix LoRA (`MATLOWAI/MOSS-Image-SFT-DeBand`, 1.1 GB): no stripes seen in our sheets.
- Fizgig look LoRA (Base only), `darask0` 4-step adapter (custom sampler), lodestones DMD (custom sampler).
- Celebrity likeness LoRAs and copyrighted-character LoRAs: not used.
- MOSS I2I LoRAs: need sd.cpp reference-latent support for MOSS; the decensor LoRA is deliberately excluded.
