# Qwen-Image-Edit-2511 on our GPUs

Status 2026-09-05: feasibility only. Nothing is wired into omniserve-native yet. Today the product surface (`/tools/prompt-image-editor`, the editor's right-click "Edit with prompt", and the `/fal` proxy models `fal-ai/qwen-image-edit-2511`, `.../lora`, `.../multi`) runs on fal.ai. This note is the plan for moving the base model and the style LoRAs onto our own hardware with fal as overflow.

## What the vendored engine already supports

`/home/lee/code/cutedsl/external/stable-diffusion.cpp` is at commit `5ea20c1` (2026-01-27, "auto detect z-image-omni"). Its `docs/qwen_image_edit.md` documents Qwen Image Edit, Edit 2509 and Edit 2511. 2511 needs `--qwen-image-zero-cond-t`, otherwise quality degrades. Reference command from that doc:

```
sd-cli --diffusion-model qwen-image-edit-2511-Q4_K_M.gguf --vae qwen_image_vae.safetensors --llm qwen_2.5_vl_7b.safetensors --cfg-scale 2.5 --sampling-method euler --offload-to-cpu --diffusion-fa --flow-shift 3 -r input.png -p "..." --qwen-image-zero-cond-t
```

LoRAs load from `--lora-model-dir` and are selected per request with `<lora:name:scale>` prompt tags (`docs/lora.md`), with `--lora-apply-mode` immediate or at_runtime. That maps directly onto the lazy-load UX of the reference HF space: one resident base model, LoRA weights swapped per request.

## Weights

- Diffusion model: `unsloth/Qwen-Image-Edit-2511-GGUF` (Q8_0 about 21GB, Q4_K_M about 12GB; sizes are from the repo listing, not measured here).
- VAE: `Comfy-Org/Qwen-Image_ComfyUI` `split_files/vae/qwen_image_vae.safetensors`.
- Text encoder: Qwen2.5-VL-7B (`Comfy-Org/Qwen-Image_ComfyUI` text_encoders, or `mradermacher/Qwen2.5-VL-7B-Instruct-GGUF` Q8 about 8GB).
- Speed LoRA: `lightx2v/Qwen-Image-Edit-2511-Lightning` 4-step and 8-step bf16 safetensors.
- Style LoRAs (all public, verified on the HF API 2026-09-05, the same list `static/data/image-edit-loras.json` serves to the tool): prithivMLmods Ultra/Hyper Realistic Portrait, Anime, Pixar-Inspired-3D, Noir-Comic-Book-Panel, Polaroid-Photo, Midnight-Noir-Eyes-Spotlight, Object-Adder, Object-Remover, Unblur-Upscale; starsfriday Upscale2K; fal Multiple-Angles; dx8152 Gaussian-Splash.

## Fit

Estimates, not measurements:

| GPU | diffusion | text encoder | headroom | verdict |
| --- | --- | --- | --- | --- |
| RTX 5090 32GB (prod, leaf-gpu) | Q8_0 21GB | Q8 GGUF 8GB, or `--offload-to-cpu` | tight with Z-Image resident | swap with Z-Image, or Q4_K_M + Q8 encoder to coexist |
| RTX 3090 Ti 24GB (this box) | Q4_K_M 12GB | offloaded to CPU | none while liveportrait, omniserve :8791 and other jobs hold about 21GB | not viable without freeing the card |

The memory note "keep 5090 zimage-only" was a throughput decision for the art gateway. Edit requests are rarer and higher value, so the realistic shape is: Z-Image resident, Qwen edit loaded on demand through omniserve's workload swap, unloaded after an idle timeout.

## Speed

sd.cpp on a 5090 for a 20B MMDiT at 1024x1024: Lightning 4-step is the only path to interactive latency. Expect low single-digit seconds per step for Q8 without measured numbers; a measurement on the 5090 is the first thing to do. Quality ranking to verify with the same edit prompts: fal 2511 (reference) vs local Q8 + Lightning 8-step vs Q4_K_M + Lightning 4-step.

## Plan

1. Measure on the 5090: download Q8_0 + Q8 encoder + Lightning 8-step, run the doc command with `--qwen-image-zero-cond-t` and a Lightning LoRA tag, record wall time and peak VRAM at 1024 and 1536.
2. Add an omniserve workload `qwen-image-edit` exposing `POST /v1/images/edits` (image data URL, prompt, optional mask, steps, seed, `lora`) with the same bearer secret as `/v1/images/generations`, registered in `workloads/workloads.json` and admitted through `gpu_admission.py` so it swaps cleanly with Z-Image.
3. Point the netwrck `/fal` proxy provider chain at omniserve first for `qwen-image-edit-2511*`, fal second. RunPod serverless with a 30s scale-to-zero worker (the `vrmrig/worker` and `app-site/comfy-worker` patterns) as the third tier, promoted to a persistent pod only under sustained load.
4. Publish the API in `/api-docs` beside `qwen-image-editor`.
