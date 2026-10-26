# Qwen Image 2.1 Lightning speed and visual review

Qwen Image 2.1 with the LightX2V four-step LoRA works for native generation
and reference editing. On the shared RTX 3090 Ti, the selected Q4 configuration
takes about **8.1 seconds per 1024x1024 generation** and **15.5 seconds per
edit**, including HTTP and image encoding. It is opt-in; running services,
the default Viggle lane and the NSFW router are unchanged.

## Measurements

All cases use seed 42, CFG 1, Euler, CUDA flash attention, host-resident
parameters streamed to CUDA (`*=cpu`), the same Q4 Qwen3-VL text encoder,
vision projector and BF16 VAE, and WebP quality 85 responses. Result caching,
latent replay, denoiser skipping and notch filtering are disabled. Model files
are on local NVMe. Approximately 7.6 GiB VRAM is free before the benchmark
beside the existing services. Runs execute sequentially, not concurrently.

| Configuration | Portrait | Full body couple | Lettering | Jacket edit |
| --- | ---: | ---: | ---: | ---: |
| Lightning Q4, 32x32 VAE tiles | 9.25 s | 9.24 s | 9.21 s | 16.54 s |
| **Lightning Q4, 64x32 VAE strips** | **8.10 s** | **8.12 s** | **8.08 s** | **15.46 s** |
| Lightning Q5 candidate, 64x32 strips | 8.79 s | 8.48 s | 8.53 s | 16.23 s |
| Base Q4, 30 dense steps, 64x32 strips | 39.62 s | 37.29 s | 37.82 s | 85.75 s |

Lightning Q4 strips have three repeats per case; medians exclude each case's
first request. Other Lightning rows have two repeats and report the second.
The dense baseline has one measurement per case, including first-use costs;
its portrait is colder than the later generation cases. These are small,
shared-host measurements, not production p95 claims. For the later couple
and lettering cases, Lightning is about **4.6x faster** than dense generation.
The observed edit ratio is **5.5x**. All edit comparisons use the same Q4
portrait input except the initial 32x32 ablation, whose portrait has a tiling
artifact described below.

Warm Q4 strip generation spends approximately 0.42 s conditioning, 5.2 s
sampling and 2.3 s decoding. The edit spends approximately 1.3 s conditioning,
11.0 s sampling and 2.35 s decoding, plus reference encoding and HTTP/encoding.
Each repeated case returns identical encoded-image hashes with caches off.

The first Q4 strip request takes 10.82 s; the first Q5 request takes 18.03 s.
The roughly 0.5 s process startup does not include lazy weight loading.
An initial attempt using the slower `/vfast` storage under heavy host swap
pressure times out after 300 s while loading weights; it is not an inference
latency measurement. Use NVMe and sufficient host RAM.

## Visual review and quant choice

Full-resolution review covers an adult portrait holding a cup with both hands,
an adult couple walking on a beach, a cafe sign, and a red-to-blue jacket edit.

- Q4 Lightning produces natural-looking faces, skin and clothing in these
  samples. The full-body subjects remain proportionate and the handholding is
  plausible, although fine finger/foot detail still merits review.
- The sign reads `Qwen Coffee, $2 per cup`; Q5 also follows the lettering.
- The jacket edit changes the color to dark blue while retaining the person,
  pose, cup and background. Small facial/texture changes remain: this is not a
  pixel-preserving masked edit.
- Q5 changes composition and fine detail but does not show a consistent visual
  improvement across these three prompts. Keep **Q4_K_M as the speed preset**;
  Q5 costs about 4-8% more latency in these cases and is an optional candidate,
  not a proven universal quality upgrade.
- The base 30-step generation samples are also coherent. Its jacket edit is
  visibly over-sharpened/textured under the tested CFG-1 base schedule; the
  timing comparison does not establish that this is an optimal base-edit setup.

The initial 32x32 tiled portrait has small white rectangular artifacts on the
cup at the central tile region. Explicit 64x32 strips remove them in the tested
portrait and edit and reduce decode time from about 3.5 s to 2.3 s. This is a
configuration workaround, not a general VAE kernel fix.

Disabling tiling requests a roughly 10.8 GB decode allocation, which does not
fit the available VRAM. The local sd.cpp build retries with 32x32 spatial
tiling, reproduces the cup artifacts and takes 12.41 s generation / 19.54 s
editing warm. Explicit strips avoid that failed allocation and retry. Keeping
the DiT resident with only `te=cpu` offload also fails VAE decoding on this host;
it is not a usable fast preset here.

Local contact sheet: `artifacts/qwen-lightning/comparison.webp`.
Columns are Lightning Q4, Lightning Q5, and base Q4; rows are portrait, couple,
lettering, and edit. Full-size PNGs, requests and logs live under
`/vfast/data/code/qwen-image-2.1-bench/comparison/`, in `lightning-q4-final`,
`lightning-q5-final`, and `lightning-base30`. Generated media stays outside
versioned source.

## Provenance and implementation

Machine-readable observations, response hashes, stage medians, full local
configuration and artifact paths are in `qwen21-lightning-2026-10-14.json`.
The source checkout is `c0d6c1edb2f66733af0d4be228e96a028444e4d8`; the locally
built sd.cpp library reports `c678dfe` and carries adaptive-Taylor ABI additions.
The native gateway compiles against the local header. The overflow ctypes
worker retains its Docker-pinned ABI and rejects this different local ABI.

Lightning adapter revision and digest are pinned in
`docs/qwen-image-21-lightning.md`. The Q5 candidate is
`abenzerps/Qwen-Image-2.1-Uncensored-GGUF` revision
`7959e658915f52274897fb573ffeda49de8dc16b`, file
`qwen-image-2.1-UC-Q5_K_M.gguf`, SHA-256
`af0bf278cf16d204fb31c384dc82fd41dca82d976b15fe9305a60c726fd6f821`.
The [publisher describes it as a quantization of the upstream base weights](https://huggingface.co/abenzerps/Qwen-Image-2.1-Uncensored-GGUF/blob/7959e658915f52274897fb573ffeda49de8dc16b/README.md).
This compares published quant artifacts, not two controlled conversions made
from a locally verified identical BF16 checkpoint.

The new native `lightning` profile fixes the training sigma nodes and Euler
sampler, enables distilled reference editing and requires a readable adapter.
It preserves `turbo:false` and caller-supplied-LoRA base paths. The overflow
worker has matching schedule/profile mapping, a pinned and checksum-validated
adapter download, explicit missing-adapter failure and rectangular VAE tiles.
The existing defaults are unchanged unless Lightning is selected.

Validation: native build succeeds; 10 Qwen Python tests pass, including the
real C gateway driven by a capturing SD stub. Eight targeted CTest tests pass
(Lightning, core, image decode, cache concurrency and image routing). A full
pre-registration CTest run passes 26/28; the two failures are missing existing
`tests/data/guard-holdout.json` and `tests/data/guard-multiturn.json` fixtures.
The GPU generation/edit runs above exercise the native path; the overflow
worker's new mappings are CPU-tested, not GPU-benchmarked in its pinned image.

## Next speed limits

Sampling dominates now. A dedicated GPU can test resident DiT weights and
untiled decoding with enough free VRAM; the shared-host failures should not
be extrapolated to an empty 24/32 GB GPU. Blackwell FP4 or W4A8 kernels need a
separate backend/hardware benchmark rather than assuming GGUF Q4 uses native
FP4 tensor cores. Avoid adding denoiser skipping to an already four-step model
without a fresh quality gate.

Distilling other models needs a separately trained and validated student or
distillation adapter; reducing their step count alone is not equivalent to
this LoRA. NSFW-router training, adapter composition and broader adult anatomy
coverage remain a separate pass.
