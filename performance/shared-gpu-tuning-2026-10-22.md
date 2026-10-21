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

## VAE decode -22% (`runtime/ggml-im2col-tiled.patch`) and tile 80

ggml-cuda im2col mapped threads along IC*KH*KW, so reads hopped channel planes. The
tiled kernel stages input rows in shared memory (coalesced along IW) and writes each
pixel's IC*KH*KW run contiguously. Bit-identical images. 1024^2 decode on a 3090 Ti:
tile 64 1.68 -> 1.29 s, tile 80 1.25 -> 0.98 s.

VAE tile 80: 1024^2 is 2x2 tiles instead of 3x3. The VAE compute buffer (the generation
peak) grows 1664 -> 2600 MB, so `SD_MIN_FREE_MB` 2304 -> 3200. 0 broker denials after.

## Prod image times (median gen ms, 8 steps)

| | start | +LoRA path | +tile 80 | +im2col |
| --- | ---: | ---: | ---: | ---: |
| 1024x1024 LoRA | 6037 | 5437 | 5215 | 5070 |
| 1024x1024 none | 4600 | | 4471 | 4279 |
| 1024x768 LoRA | 4509 | | 3936 | 3696 |

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

## Not done

- Implicit-GEMM conv for the VAE: im2col still materialises ~1 GB per full-res conv
  (~4% of an image).
- LoRA diff add (269 ms/4 steps) needs a GEMM with beta=1 or an add->rms_norm fusion.
- Z-Image Q8_0 DiT: -6-8% but +2.7 GB resident next to the Qwen edit sibling.
- MTP speculation in the batched engine: the prod model is a gemma-4-E4B finetune and the
  E4B head fits, but single-stream chat is already ~146 tok/s.
