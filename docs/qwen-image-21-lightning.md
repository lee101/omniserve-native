# Qwen Image 2.1 Lightning native inference

The opt-in `lightning` profile runs the LightX2V four-step LoRA through the
resident stable-diffusion.cpp backend. It supports both text-to-image and
reference editing without a Python inference server. Existing Viggle and base
lanes keep their defaults. See the measured results in
`performance/qwen21-lightning-2026-10-14.md`.

## Weights and schedule

Use a **base Qwen Image 2.1** DiT GGUF, Qwen3-VL-8B text encoder and vision
projector, and the Qwen Image 2.1 VAE. Do not put Lightning on top of an already
Viggle-merged GGUF. Keep the small Lightning adapter in its original precision.

The [upstream model card](https://huggingface.co/lightx2v/Qwen-Image-2.1-Lightning)
specifies Euler sampling, four raw sigma nodes `[1, .75, .5, .25]`, dynamic
exponential shifting, CFG 1, and no terminal stretching. Native inference
shifts those nodes using output sequence length `(width / 16) * (height / 16)`,
with `(base_len, max_len, base_shift, max_shift) = (256, 8192, .5, .9)`, then
appends zero. It bypasses denoiser skipping for these four steps.

Adapter revision: `6b0a1176327c01522a44276c5da67331a025b6fc`.
File: `pytorch_lora_weights.safetensors`, 67,144,280 bytes.
SHA-256: `53c69320d4163da51032f90275074028cb098556039c72b45f8dce9f0f9e34a3`.

```sh
hf download lightx2v/Qwen-Image-2.1-Lightning pytorch_lora_weights.safetensors \
  --revision 6b0a1176327c01522a44276c5da67331a025b6fc --local-dir /models/lightning
sha256sum /models/lightning/pytorch_lora_weights.safetensors
```

## Native lane

Build with `WITH_SD=ON` and an sd.cpp checkout supporting Qwen Image 2.1.
Use absolute paths for all weights. These settings suit the measured shared
24 GB GPU; `*=cpu` stores parameters in host memory and streams computation
on CUDA, rather than running the model on the CPU.

```sh
export OMNISERVE_NATIVE_BIND=127.0.0.1
export OMNISERVE_NATIVE_SD_MODEL=/models/qwen-image-2.1-Q4_K_M.gguf
export OMNISERVE_NATIVE_SD_DIFFUSION_MODEL=$OMNISERVE_NATIVE_SD_MODEL
export OMNISERVE_NATIVE_SD_LLM=/models/Qwen3VL-8B-Instruct-Q4_K_M.gguf
export OMNISERVE_NATIVE_SD_LLM_VISION=/models/mmproj-Qwen3VL-8B-Instruct-F16.gguf
export OMNISERVE_NATIVE_SD_VAE=/models/qwen_image_2.1_vae_bf16.safetensors
export OMNISERVE_NATIVE_SD_ACCEL_PROFILE=lightning
export OMNISERVE_NATIVE_SD_DEFAULT_LORA=/models/lightning/pytorch_lora_weights.safetensors
export OMNISERVE_NATIVE_SD_PARAMS_BACKEND='*=cpu'
export OMNISERVE_NATIVE_SD_VAE_TILING=1
export OMNISERVE_NATIVE_SD_VAE_TILE_X=64
export OMNISERVE_NATIVE_SD_VAE_TILE_Y=32
export OMNISERVE_NATIVE_SD_VAE_TILE_OVERLAP=0.5
export OMNISERVE_NATIVE_SD_CACHE_MODE=off
export OMNISERVE_NATIVE_SD_TELEPORT=0
export OMNISERVE_NATIVE_SD_NOTCH=0
./build-qwen/omniserve-native --port 8793
```

Send ordinary `/v1/images/generations` JSON. For editing, send
`/v1/images/edits` with `image_base64`. The profile selects reference editing,
four steps, CFG 1 and Euler automatically. `turbo:false` selects the base
schedule without the adapter. Requests supplying their own `loras` also use
the base schedule; compose and validate adapters separately rather than
silently stacking them with Lightning. A missing Lightning LoRA refuses to
initialize the diffusion backend. Check `/status.diffusion.ready` before use.

For a GPU with sufficient free memory, test resident weights by clearing
`SD_PARAMS_BACKEND`; do not assume a configuration that fits denoising also
fits VAE decoding. Keep model files on NVMe. Slow storage plus host swapping
can turn lazy first-load time into minutes.

## Overflow worker

Set `RA2_ACCEL_PROFILE=lightning`, `RA2_PARAMS_BACKEND=*=cpu`,
`RA2_VAE_TILE_X=64`, and `RA2_VAE_TILE_Y=32`. The worker downloads the pinned
adapter into a separate `lightning/` subdirectory and validates its SHA-256;
it fails rather than falling back to a misleading four-step base render.
`RA2_TURBO=0` or request `turbo:false` opts out. Lightning defaults notch
filtering off, while Viggle keeps its previous default.

The worker's ctypes ABI must match its Docker-pinned sd.cpp version. Run
`python workloads/qwen_image.py --selftest` against that library before use.
The local adaptive-Taylor sd.cpp library has a different cache struct and is
intentionally rejected by the current worker ABI guard; the C native lane
compiles against that checkout's header instead.

## Reproduce and review

```sh
python tools/qwen_lightning_bench.py \
  --dit /models/qwen-image-2.1-Q4_K_M.gguf \
  --llm /models/Qwen3VL-8B-Instruct-Q4_K_M.gguf \
  --vision /models/mmproj-Qwen3VL-8B-Instruct-F16.gguf \
  --vae /models/qwen_image_2.1_vae_bf16.safetensors \
  --lora /models/lightning/pytorch_lora_weights.safetensors \
  --params-backend '*=cpu' --vae-tiling --vae-tile 64x32 \
  --repeat 3 --out /tmp/qwen-lightning
ctest --test-dir build-qwen -R qwen_lightning --output-on-failure
```

The harness owns an isolated loopback gateway, disables result/latent caches,
and terminates that gateway on completion or failure. It saves source requests,
decoded RGB PNGs, native stage logs, encoded-response hashes and wall times.
Warm medians exclude the first request for each case when repeats exist.
Use `--profile base --steps 30` without `--lora` for a dense baseline;
`--edit-source /tmp/qwen-lightning/portrait-0.png` holds the edit input fixed.
Use `--output-format png` to remove lossy response encoding from comparisons.

Review faces, fingers, full-body proportions, lettering, edit compliance and
unchanged background/identity at full resolution. Image similarity between
different samplers is not a substitute for this review. This profile does not
alter the NSFW router or distill other model families.
