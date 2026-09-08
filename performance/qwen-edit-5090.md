# Qwen Edit / shared RTX 5090 evaluation — 2026-09-08

The production gateway on `127.0.0.1:8791` remained running throughout these
local canaries. No service restart, cloud pod, or production weight replacement
was performed. The host also runs image, search, audio, and forecasting tenants;
these are shared-host measurements, not dedicated-card peak throughput.

## Verified native improvements

- Qwen Image Edit 2511 Q4_K_M works through `/v1/images/edits`, using its image
  reference conditioning, the Qwen 2.5 VL 7B encoder, and Qwen VAE. The required
  `qwen_image_zero_cond_t=true` flag is set. It is instruction editing, unlike
  Z-Image's existing img2img strength-based transformation.
- GPU encoder/VAE placement initially aborted inside graph-cut allocator
  measurement. `stable-diffusion.cpp/src/core/ggml_graph_cut.cpp` now includes
  the backing tensors of cut views in segment leaves. The focused CPU
  regression fails against the original library and passes against the
  candidate. Six subsequent full Qwen renders completed successfully.
- At 512px, 12 steps, CFG 2.5 and fixed seed, CPU encoder/VAE renders took
  290.97 / 290.49 seconds. With the patched GPU placement: first request
  87.21 seconds, then 72.48–74.99 seconds across original/changed-seed/changed-source
  requests. This is approximately 3.9x faster warm generation.
- For each of those three cache-key cases, uncached and cache-prime pixels
  match exactly. Three subsequent result-cache hits per case take 39.9–76.5 ms.
  These are CPU result lookups, **not** faster diffusion evaluations.
- The native cache has a separate mutex and a lookup before GPU admission.
  A test with a delayed backend proves three exact cache hits complete while
  an uncached render owns the only image permit. Authorization, LoRA validation,
  source decoding, and request validation still precede this fast path.
- EasyCache is an explicit, approximate per-process experiment, disabled by
  default. Responses disclose the requested threshold separately from exact
  result caching. Teleport requests bypass both EasyCache and the normal
  result-cache namespace so dense replay cannot retrieve an approximate result.

## Evaluation contract and limitations

The dense sweep uses two instructions (replace sign text; replace background),
one source image, seed 90908, 20/12/8 steps, CFG 2.5, and two uncached renders per
configuration with alternating profile order. Separate cache checks use a second
seed and a one-pixel source mutation. PNG output avoids lossy-codec differences.
Every dense repeat is pixel-identical. This is a screening corpus, not a broad
editing-quality certification. Global SSIM/PSNR measure retention; saved image
pairs were also inspected for the requested edit and preservation of the subject.

Twelve steps retained global SSIM 0.99739 / 0.99682 versus the 20-step reference
(PSNR 34.82 / 31.77 dB). Eight steps retained 0.99589 / 0.98414 (32.84 / 24.04 dB),
including a visibly different blue background shade. Twelve steps is the more
conservative fast candidate. The first request includes lazy weight loading;
`load_seconds` in early canary manifests means time to `/status` readiness,
not complete weight loading. All settings and repetitions are retained.

EasyCache threshold 0.1 at 20 steps took 67.06 seconds warm for the text edit
and 67.66 / 74.07 seconds for the background edit (first request: 78.12 seconds).
Global SSIM versus dense 20-step output was 0.99244 / 0.99676, PSNR
30.39 / 31.51 dB. Both edits were visually correct in this example, and repeat
pixels matched exactly. This does not establish equivalence to dense inference
or accuracy on other inputs; keep approximation opt-in.

Threshold 0.2 took 68.29 seconds warm for text and 68.55 / 75.09 seconds for
background (78.11 seconds first request), with SSIM 0.99277 / 0.99746.
Repeats were exact and both edits visually correct, but there was no meaningful
speed advantage over 0.1. There is no evidence here to prefer the higher
threshold. Both canaries stopped successfully and restored baseline GPU usage.

Recommended next evaluation profile: GPU encoder/VAE with the graph-cut fix,
12 dense steps, CFG 2.5, 8 CPU threads, and the 4 GiB graph budget. Keep 20 dense
steps as the quality reference and EasyCache 0.1 as a separate opt-in arm.
Expand to multiple real sources, seeds, faces, detailed text, and high-resolution
edits before promoting a quality/performance default. This experiment does not
establish maximum throughput under simultaneous production tenant traffic.

Raw configuration, per-request timings, quality metrics and pixel hashes are
retained in [evals-2026-09-08](evals-2026-09-08), including dense step sweeps,
EasyCache A/Bs, exact result-cache checks, and the CPU-thread sweep.

The dense canary sampled a maximum process GPU allocation of 4640 MiB with
a 4 GiB graph budget. Sampling every five seconds can miss transient peaks.
It returned the host to approximately 19.04 GiB device usage on exit. The
graph budget is not a hard process-memory limit or a GPU reservation.

## CPU query sweep

Same model, zero temperature, three requests per case, separate canaries:

| CPU threads | Warm short request | Warm long-context retrieval | Warm code request |
|---|---:|---:|---:|
| 4 | 8696 ms | 487 ms | 11372 ms |
| 8 | 7472 ms | 351 ms | 7994 ms |
| 16 | 8161 ms | 495 ms | 10657 ms |

Warm entries are the median of repeats 1 and 2. Outputs match across thread
counts; the code answer is truncated by the 48-token benchmark cap and is not
a passing code-generation quality test. Eight threads is a measured candidate,
not an automatically deployed setting. The live gateway separately passed
16 checks, with 7/8 task accuracy and 1.825x measured prefix-cache speedup.

## Reproduce

Build the native gateway with `WITH_SD=ON`. Use an isolated patched SD library;
`--sd-library` selects it without replacing the production library. The canary
checks available memory and port ownership, records settings/telemetry, and
stops its child server in `finally`.

For a clean patched SD build, from `stable-diffusion.cpp`:

```bash
cmake -S . -B build-qwen-canary -DSD_CUDA=ON -DSD_BUILD_SHARED_LIBS=ON \
  -DCMAKE_CUDA_ARCHITECTURES=120 -DCMAKE_BUILD_TYPE=Release
cmake --build build-qwen-canary -j 6
c++ -std=c++17 -DGGML_MAX_NAME=160 -I. -Isrc -Iggml/include \
  scripts/test_graph_cut_views.cpp -Lbuild-qwen-canary/bin \
  -Wl,-rpath,build-qwen-canary/bin -lstable-diffusion -o /tmp/test-qwen-views
/tmp/test-qwen-views
```

Use `build-qwen-canary/bin/libstable-diffusion.so` below. The measured canary
instead incrementally rebuilt the changed graph-cut object and relinked the
existing CUDA objects into a separate temporary library; it did not overwrite
the running worker's library. CUDA 12.9 was used for the new CUDA C extension.

From `omniserve-native`:

```bash
cmake -S . -B build-cache-perf -DWITH_SD=ON -DWITH_LLAMA=ON -DONATIVE_LTO=OFF
cmake --build build-cache-perf -j 6
ctest --test-dir build-cache-perf --output-on-failure

.venv/bin/python tools/image_canary.py --model qwen \
  --binary build-cache-perf/omniserve-native --sd-library /path/to/libstable-diffusion.so \
  --text-backend cuda0 --vae-backend cuda0 --budget-gib 4 \
  --output /tmp/qwen-canary-new --timeout 2400 -- \
  .venv/bin/python tools/image_edit_sweep.py \
  --source ../stable-diffusion.cpp/assets/flux/flux1-dev-q8_0.png \
  --output /tmp/qwen-sweep-new --steps 20 12 8 --repeats 2
```

For approximate A/Bs, use `--easycache-threshold 0.1` on the canary and
`--expect-easycache 0.1 --reference-dir /tmp/qwen-sweep-new --steps 20` on the
sweep. Use a fresh output directory and server for each threshold. Exact
result-cache timings must never enter the denoising-speed comparison.

The gateway can proxy edits to a separate configured
`OMNISERVE_NATIVE_IMAGE_EDITOR_UPSTREAM`; the Qwen server itself uses
`OMNISERVE_NATIVE_SD_REFERENCE_EDIT=1`. A full mixed-tenant rollout remains
separate from these stopped canaries. Do not assume Qwen and Z-Image share
denoiser, text-encoder, or VAE weights: their checkpoint components differ.
Z-Image text-to-image and native img2img do reuse the same loaded context.

Upstream references: [Qwen Edit native usage](https://github.com/leejet/stable-diffusion.cpp/blob/master/docs/qwen_image_edit.md),
[Diffusers Qwen pipelines](https://huggingface.co/docs/diffusers/main/api/pipelines/qwenimage).
