# MOSS-Image SFT: dense vs 4-step on the shared RTX 3090 Ti

MOSS-Image-SFT (5B DiT, Qwen3.5-9B text encoder, FLUX.2 VAE) runs in
stable-diffusion.cpp upstream (`77b72a2`, built separately in
`../sdcpp-moss`; the `../stable-diffusion.cpp` checkout used by this repo
predates MOSS support). A 4-step distilled variant merged into a Q4_0 GGUF
(`mischiefaudio/MOSS-Image-SFT-4step-merged-GGUF`, CFG 1, stock Euler) is a
drop-in speedup, the same role Lightning plays for Qwen.

## Steady-state results (1024x1024, seed 42, `--offload-to-cpu`)

Weights load once per process (`-b 4` for the 4-step row); times exclude model
load, which on this host is dominated by ZFS reads (see below).

| Configuration | DiT evaluations | Sampling | VAE decode | Sampling + decode |
| --- | ---: | ---: | ---: | ---: |
| SFT Q4_K, 50 steps, CFG 3 (layer-22 skip on negative) | 100 | ~70 s (1.37-1.41 s/step) | 0.8 s | **~71 s** |
| **SFT 4-step merged Q4_0, 4 steps, CFG 1** | 4 | **2.84 s** (0.71 s/step) | 0.46 s | **~3.3 s** |

About **21x** fewer seconds per image than dense, before text encoding (not
measured warm separately; one encode per distinct prompt). Qwen Image 2.1
Lightning was 4.6x.

## Quality bench (qualitybench/moss_sweep.py + moss_score.py, 1024x1024, seed 0, resident server)

10 SFW prompts (text, poster, portrait, hands, ...) and 8 adult prompts; median end-to-end latency
including text encode and WebP. Raw data: `results/moss-20261029/`. LPIPS/PSNR vs the 50-step render are
not a quality measure across model families (the 4-step model composes differently); judged by CLIP and
contact sheets.

| Family | Config | s/image | CLIP | Notes |
| --- | --- | ---: | ---: | --- |
| SFW | dense 50 steps CFG 3 | 76.8 | 0.405 | reference |
| SFW | dense 30 steps | 46.4 | 0.405 | LPIPS 0.10 vs d50 |
| SFW | dense 30 + EasyCache 0.15 | 26.6 | 0.410 | best dense tradeoff |
| SFW | **4-step CFG 1** | **3.4** | 0.402 | text correct on sign/poster; **pick** |
| SFW | 4-step, 6 / 8 steps | 4.9 / 6.4 | 0.396 / 0.394 | no gain |
| SFW | 4-step CFG 1.5 | 6.5 | 0.401 | 2x cost, no gain |
| NSFW | dense 50 + LoRA 0.9 | 101 | 0.355 | reference |
| NSFW | dense 30 + EasyCache 0.15 + LoRA | 35.2 | 0.353 | |
| NSFW | **4-step CFG 1 + LoRA 1.0** | **4.4** | 0.340 | coherent anatomy/skin; **pick** |
| NSFW | 4-step + LoRA 0.8 / 1.2 | 4.3 / 4.4 | 0.347 / 0.344 | within noise; 1.0 is the LoRA's intended strength |
| NSFW | 4-step + LoRA, 6 steps | 6.2 | 0.343 | no gain |

The runtime LoRA costs about 1 s over the plain 4-step model. Chosen lanes: SFW 4-step (~22x faster than
dense 50), NSFW 4-step + LoRA (~23x). Launch with `qualitybench/serve_moss.sh` (`LANE=sfw|nsfw`).
Single seed per prompt: text accuracy needs more seeds before treating this as final.

Bench pitfall: the first NSFW sweep silently ran without the LoRA because the server binary predated
`DEFAULT_LORA_ALWAYS`; verify LoRA effect by comparing output hashes with/without it.

## Pitfalls found

- **Qwen3.5-9B `Q4_K_M` text encoder produces pure noise images** (every
  prompt, including empty, on CPU and GPU, with Q4_K/Q8_0/safetensors DiTs).
  Same build with `Q8_0` renders correctly. Upstream's docs show Q4_K_M; it
  did not work here. Use `Qwen3.5-9B-Q8_0.gguf` (9.5 GB) until a smaller quant
  is verified (Q6_K / Q5_K_M untested).
- Ruled out as causes: flash attention, DiT quantization, VAE file, CUDA
  kernels (CPU backend identical), sampler/scheduler/shift, the 15 later
  upstream MOSS commits (`95b3090` identical output).
- On this host the ZFS ARC is capped at ~12.9 GB, so Q8 text encoder (9.5 GB)
  plus DiT (3.3 GB) thrash it: cold conditioning took 190-320 s and the DiT's
  first step up to 180 s. A resident server avoids this; `sd-cli` per image does
  not.

## Reproduce

```sh
sd-cli --diffusion-model MOSS-Image-SFT-4step-v2.0-merged-Q4_0.gguf \
  --llm Qwen3.5-9B-Q8_0.gguf --vae flux2-vae.safetensors \
  --tokenizer tokenizer.json   # OpenMOSS-Team/MOSS-Image-Base/tokenizer/tokenizer.json
  --cfg-scale 1.0 --steps 4 -H 1024 -W 1024 --offload-to-cpu -s 42 -p "..."
```

Model SHA-256 (4-step Q4_0): `f23faecd6fe0ddb3e6fedb7dd31f760c5c67f917b5078e8fee343d2635be556d`.

## Not yet done

- NSFW LoRA (`axiomofmind/Moss-image-nsfw-uncensored-lora-GGUF`) on top of the
  4-step model, and the lodestones 8-step DMD LoRA (needs a custom sampler).
- Smaller text-encoder quant, VAE/DiT kernel tuning, 5090 numbers.

## Style LoRAs and prompt presets (2026-10-30, `qualitybench/moss_style_sweep.py`)

One resident 4-step server, per-request loopback `loras`; beauty, anime, gouache and psychedelic GGUF LoRAs
(`axiomofmind/Moss-image-*-lora-GGUF`, F16, ~55-74 MB). Variants per prompt/seed: plain, + positive suffix,
+ negative at CFG 1.5, LoRA, LoRA + suffix, LoRA + suffix + negative (CFG 1.5), LoRA 0.7, alternate checkpoint.
Sheets: `results/moss-style-20261030/sheets/`. Presets: `qualitybench/moss_style_presets.json`.

- **Anatomy (anime "three legs" prompt, seeds 0-3):** the plain 4-step render often crouches with ambiguous
  legs (same failure at dense 20/30 steps and CFG 1.5). Adding the positive suffix
  (`anime character, dynamic composition, clean anatomy`) gave clean two-leg renders on 4/4 seeds at no extra
  cost (3.4 s). The anime LoRA is more dynamic but keeps some leg ambiguity in crouched poses.
- **Negative prompt at CFG 1.5** (6.5 s) looked no better than the suffix alone: not worth 2x. Negatives are
  ignored at CFG 1, so presets default to positive suffix only.
- **LoRAs transfer to the distilled model** at scale 1.0 (~4.4 s): beauty (richer skin/freckles; alt 1750 more
  contrasty), gouache (flat painterly style, lettering correct), psychedelic (denser, cleaner detail).
- The suffix alone also fixed garbled lettering on the gouache cafe prompt ("DISTOUR CAFE").
- Not run: MOSS I2I LoRAs (`sorryhyun/MOSS-I2I-LORAs`) need the reference-latent path (`ref_isolate`,
  `ref_timestep`) that sd.cpp's MOSS support lacks. The decensor LoRA is deliberately not integrated.
