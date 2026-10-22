# Shared-GPU tuning: chat priority, LoRA path, VAE decode (2026-10-22)

Prod (RTX 5090, Broadwell host) runs the batched LLM and resident Z-Image in one
process. Real traffic: chat is mostly one stream at a time; images are a sequential
background batch job (198/h, all `background` tier, near-zero queue wait), almost all
with the `the_look` LoRA from auto-routing.

## Chat starved by images -> LLM streams at high CUDA priority

LLM and diffusion share one CUDA context, so DiT kernels queued ahead of each LLM
decode. `runtime/llama-cuda-stream-priority.patch` adds `GGML_CUDA_STREAM_PRIORITY`
(`high`/`low`/int) to llama.cpp's ggml-cuda; Z-Image's ggml copy keeps the default.

| 3090 Ti, gemma-4-E4B Q8 + Z-Image 1024^2 looping | default | `high` |
| --- | ---: | ---: |
| 1 stream tok/s (idle: 110) | 11.4 | 49.4 |
| 1 stream TTFT p50 | 335 ms | 47 ms |
| 4 streams tok/s per stream | 8.5 | 44.4 |
| image while chat decodes | 10.0 s | 13.7 s |

Prod under live image load: 1 stream 146 tok/s, TTFT p50 46 ms (idle prod: 142-151);
before, single-row decode took ~18 ms (~55 tok/s) whenever an image was running. Images
only slow while chat decodes (~1 min of LLM GPU time per 11 h at current traffic).

## Runtime LoRA cost 30% -> 16% (`runtime/sdcpp-lora-fast.patch`)

nsys, 1024^2, 4 steps, FA on: the LoRA added 1010 ms of kernel time on 4917, of which
the rank-32 GEMMs were 92 ms. The rest: BF16 activation converts (281), output-sized
`scale` (255), q/k/v diff `concat` (104), the final add (285).

- linear LoRA factors load as F32 (cuBLAS SGEMM, no activation converts)
- q/k/v pieces of the fused qkv weight: one GEMM through stacked downs, one through a
  block-diagonal up, so the diff is produced already concatenated
- scale folded into the rank-r up factor; the w2 activation pre-scale (an MMQ q8_1 fp16
  overflow guard) is skipped for the SGEMM LoRA path
- same files with new multipliers (route 0.6 <-> 0.8) rescale the loaded tensors instead
  of re-reading 480 tensors (bit-identical to a fresh load)

LoRA overhead: 1010 -> 453 ms kernel (remaining: the add 269, GEMMs 211). 3090 Ti 1024^2
8 steps sampling: none 8.17 s, old LoRA 10.23 s, new 9.07 s. Output vs the BF16 path:
PSNR 27 dB from rounding amplified over the trajectory; same composition and quality.
`SD_LORA_FAST_LINEAR=0` restores the old path.

The patch also carries `SD_ZIMAGE_LORA_NAME_FIX=1` (default off): kohya Z-Image LoRAs
lose `feed_forward_wN`/`adaLN_modulation` keys to the underscore->dot pass (Aesthetic Base
and TartarusPixel apply 180/450 tensors), and unprefixed PEFT keys are not recognised.
Fully applied LoRAs change established looks, so prod keeps it off.

## Broken LoRAs fixed (gateway prefers converted siblings)

Registry audit (54 ids) against sd.cpp's applied-tensor counts:
- kohya `lora_unet_*` (aesthetic_base, pixel_art, cosplay_nsfw): 180/450 applied (FFN dropped).
- PEFT `*.lora_A.default.weight` / `base_model.model.*` (z_aesthetic_anime, app_logo_maker,
  product_screenshots): 0 applied, byte-identical to plain.
- Not Z-Image at all: anime_artistic (SDXL, 0/792), mecha_klein (Flux Klein `double_blocks`, 0/160).
  Need replacement files or removal from the cutedsl-site registry.

`tools/zimage_lora_convert.py` had already written `<stem>.sdcpp.safetensors` copies (bitwise renames of
the kohya tensors, alpha kept; verified 450/450 equal). `resolve_cached_lora_file` now prefers that sibling
whenever it exists in the LoRA dir, for lora_id, lora_filename, registry and auto-route lookups. The
registry maps app_logo_maker and product_screenshots to one shared `adapter_model.safetensors`; id-named
symlinks (`app_logo_maker.safetensors -> app_logo_maker_adapter_model.safetensors`, same for
product_screenshots) restore the separate adapters. Prod logs no partial-apply warnings since
(`deploy-prod-20261022-lorafix`). Fully applied LoRAs look stronger: Tartarus at 1.0 turns dark-fantasy pixel
art. `SD_ZIMAGE_LORA_NAME_FIX` stays off (not needed with converted files).

## VAE decode -22% (`runtime/ggml-im2col-tiled.patch`) and tile 80

ggml-cuda im2col mapped threads along IC*KH*KW, so reads hopped channel planes. The
tiled kernel stages input rows in shared memory (coalesced along IW) and writes each
pixel's IC*KH*KW run contiguously. Bit-identical images. 1024^2 decode on a 3090 Ti:
tile 64 1.68 -> 1.29 s, tile 80 1.25 -> 0.98 s.

VAE tile 80: 1024^2 is 2x2 tiles instead of 3x3. The VAE compute buffer (the generation
peak) grows 1664 -> 2600 MB, so `SD_MIN_FREE_MB` 2304 -> 3200. 0 broker denials after.

## Prod image times (median gen ms, 8 steps)

| | start | +LoRA path | +tile 80 | +im2col | +gemm-add, cuDNN VAE |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1024x1024 LoRA | 6037 | 5437 | 5215 | 4984 | 4454 |
| 1024x1024 none | 4600 | | 4471 | 4201 | 4129 (n=5) |
| 1024x768 LoRA | 4509 | | 3936 | 3902 | 3726 (n=1) |

Per-image timing is logged as
`[image] tier WxH s= l= tp= hit= q= lease= lock= gen= enc= ok=`.

## Public chat path caps

Public `/v1/chat/completions` goes gateway -> text-generator compat worker (:9083, auth and
billing; `CHAT_COMPAT_PROXY=1`) -> back to the gateway with `X-Omniserve-Internal: local`.
- The compat worker held its GPU semaphore (`INFERENCE_CONCURRENCY=2`) for every proxied chat,
  so public chat ran 2 at a time in front of a 32-sequence engine. Patched in text-generator
  (`VLLM_PROXY_CONCURRENCY`, default 64, own semaphore; local-model eviction still takes the GPU
  gate; requests pool 64; default thread pool +32 since each stream parks a thread per chunk).
- The outer gateway hop held an LLM slot and the inner request another: 2 of 32 slots per
  chat. `CHAT_COMPAT_PERMITS=0` (prod) leaves admission to the inner request.

## LoRA diff add fused into the up-GEMM (`runtime/ggml-cuda-gemm-add-fuse.patch`)

`out = add_inplace(W x, up (down x))`: when an F32 x F32 MUL_MAT feeds only an ADD of its shape, ggml-cuda
runs one SGEMM with beta=1 straight into the ADD's destination (no diff tensor, no add pass). Outputs stay
within the run-to-run hash set of the unfused path. 3090 Ti 1024^2 9 steps the_look 0.8: 10.6 -> 10.2 s
median; plain images unchanged. `GGML_CUDA_FUSE_GEMM_ADD=0` reverts.

## VAE convs on cuDNN (`runtime/ggml-cuda-cudnn-conv.patch`)

`OMNISERVE_NATIVE_SD_VAE_CONV_DIRECT=1` makes sd.cpp emit GGML_OP_CONV_2D; with `GGML_CUDA_CUDNN_CONV=1`
ggml-cuda runs it through cuDNN (dlopen'd, F16 x/w/y, heuristic algo cached per shape, created during the
CUDA-graph warmup run) instead of ggml's naive direct kernel. Per-tile (80^2 latent) conv time in a torch
microbench: cuDNN F16 60 ms vs im2col+GEMM 98 ms. 3090 Ti 1024^2: 9.08-9.35 -> 8.78-8.83 s, VAE output
57 dB PSNR vs the im2col path (max pixel diff <= 8), generation peak VRAM 10428 -> 9668 MB.
`GGML_CUDA_CUDNN_CONV_F32ACC=1` (60 dB) costs ~0.2 s.

Prod: pinned copy of the nvidia-cudnn-cu12 9.10.2 wheel in `omniserve-native-deploy/cudnn-9.10.2`
(`GGML_CUDA_CUDNN_LIB`). The first deploy pointed at `~/.local/.../nvidia/cudnn`, whose `libcudnn_ops.so.9`
is truncated (55.7 of 127.9 MB, an interrupted pip install): the loader SIGBUSes mapping it, which took the
gateway down twice before rollback. The pinned copy is md5-checked against an intact wheel.

## Z-Image Q8_0 DiT: rejected on prod

+2.7 GB resident. On the shared 5090 the broker then refused 5 of 6 1024^2 images ("needs at least 3200 MB
free"), so prod stays on Q4_K.

## Not done

- MTP speculation in the batched engine: the prod model is a gemma-4-E4B finetune and the
  E4B head fits, but single-stream chat is already ~146 tok/s.
