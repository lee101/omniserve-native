# H3 latent-teleportation feasibility plan

The implementation in `../cutedsl` is evidence for a method, not a reusable H3
checkpoint. Its published adapter is tied to Z-Image Turbo at 512px and 16
steps. H3 has a different joint audio/video latent, sampler, conditioning path,
and resolution/duration grid, so its coefficients and 7,312-parameter adapter
must not be loaded into H3.

Spectrum v0.1.8 is the nearest working H3 analogue today: it forecasts the
post-transformer hidden feature within one request. Cross-request retrieval is
a separate experiment that should correct a query's local forecast rather than
copy a neighbour trajectory.

## Measurement contract

Every arm uses the same checkpoint, sampler, prompt, seed, canvas, duration,
and step schedule. Report model-load/warm-up, text encode, denoiser, VAE/audio
decode, encode, end-to-end time, peak VRAM, and actual transformer calls.
Compare decoded video with `scripts/compare_videos.py`; retain audio and inspect
speech/foley timing. PSNR and SSIM measure same-seed retention, not aesthetic
quality, so promotion also requires blind prompt-adherence and pairwise review.

The dense path is the reference. `torch.compile` may be called quality-neutral
only when decoded hashes match, or when a documented numeric tolerance passes
across all evaluation clips. Spectrum, EasyCache, FirstBlockCache, Sol-Attn,
quantization, and teleportation are approximate until proven otherwise.

## Staged experiment

1. Capture 64 complete H3 sampler trajectories: 32 T2V, 16 I2V, 8 first/last,
   and 8 audio-driven; two seeds; preview and balanced; 12 and 20 steps. Record
   the initial noise, every post-scheduler latent, timestep/sigma, prompt
   embedding, conditioning mode, and tensor metadata.
2. On held-out prompts, evaluate identity, raw momentum, one scalar fitted per
   `(schedule, step, horizon)`, and aligned anchor schedules. Do not build a
   retrieval index unless scaled momentum beats identity and raw momentum.
3. Add the `../cutedsl` residual-retrieval rule: prompt top-k, prune by the
   query's observed motion sketch, retrieve eight residuals, then apply
   variance shrinkage. Split by prompt family so gallery variants cannot leak
   across train/test.
4. Run a live sampler whose approximate state is fed into the next real H3
   call. Offline true-anchor resets are only a proxy. Compare 12/8, 20/10, and
   20/6 real-call budgets with dense 12- and 20-step references.
5. Measure a learning curve at 64, 256, 2k, and only then 20k trajectories.
   Proceed to 20k only if retrieval improves held-out decoded retention over
   scaled momentum and the live path improves end-to-end latency by at least
   15% without a blind-preference regression.

## Dataset shape and ownership

Keep one append-only manifest shared by gallery generation and experiments.
Each row includes prompt and normalized prompt hash, seed, model/weight hashes,
mode, dimensions, frame count, steps, scheduler, source gallery ID, artifact
URLs, capture status, and split. `h3-cog` owns capture and inference; `app-site`
owns RunPod orchestration and gallery publication; `manifoldgen-site` and
`cutedsl-site` consume the manifest rather than duplicating tensors.

Store normalized prompt embeddings in a resident vector index. Store full
trajectories in sharded object storage, not worker disks or the web repos.
Before committing to 20k, measure bytes per trajectory from the first 64:

```
total_bytes = sum(shard sizes)
projected_20k = total_bytes / completed_trajectories * 20_000
```

Full multi-step video latents can reach terabytes at 20k. The serving index
should therefore keep only embeddings, compact motion descriptors, confidence
statistics, and the residual records needed by top-k retrieval. Never retain
Spectrum's full hidden-feature history across requests; upstream reports
multi-gigabyte transient history even around 0.5 MP.

## Immediate acceleration matrix

Run cache-busted, warm A/Bs in this order:

1. dense W4A8;
2. built-in ComfyUI `TorchCompileModel` (`compile` in
   `scripts/spectrum_bench.py`), with cold compile and second-run timing;
3. Sol-Attn only;
4. Spectrum only and Sol-Attn + Spectrum;
5. EasyCache and FirstBlockCache only as explicitly approximate arms;
6. the best non-teleport arm plus scaled momentum, then retrieval.

Do not combine Spectrum with EasyCache/LazyCache: Spectrum v0.1.8 deliberately
self-disables because it cannot obtain actual history when the outer cache
skips the H3 call.

## 2026-08-09 evidence

`experiments/results/existing-spectrum-s7-quality.json` compares decoded saved
artifacts with the dense seed-7 reference. Spectrum-Sol scored PSNR 19.77267 /
SSIM 0.760391; Sol-Attn + EasyCache scored 22.897064 / 0.785265. Both decoded
video and audio hashes differ from dense. These metrics do not judge aesthetics,
but they disprove lossless or zero-impact labels for those samples.

A separate earlier local 5090 sweep has internally matched timing and video
artifacts in `experiments/results/local-accel-s7-quality.json`:

| profile | speedup | PSNR vs dense | SSIM vs dense |
| --- | ---: | ---: | ---: |
| Sol-Attn | 1.123x | 22.721451 | 0.783956 |
| EasyCache balanced | 1.209x | 34.973096 | 0.967884 |
| Sol-Attn + EasyCache | 1.222x | 22.954348 | 0.789190 |

On this one prompt, EasyCache balanced is the useful Pareto candidate: it gives
nearly all of the stacked speedup with much higher same-seed retention. This is
not enough prompts to promote a default, and none of the decoded hashes match.

The corrected RunPod sweep now sends the public `accel_profile` Cog input,
checks the deployed OpenAPI schema before spending on inference, changes graph
inputs between arms to avoid ComfyUI cache hits, records wall and generation
time, and always terminates its pod. The earlier `_tuning`-only sweep is invalid:
the old Cog schema stripped those private fields, leaving identical graphs.

Three corrected RTX 5090 attempts (community pod `s3qratqozpfpd5`, secure pods
`weqy0wxx04ez9g` and `5h1aeueic8k922`) never reached container uptime while
pulling/creating the image. The first two were terminated by the eight-minute
zero-uptime guard; the compile-image retry still had zero uptime after a raised
twelve-minute guard. The compile image is 22.9 GB unpacked / 12.75 GB compressed
and its GHCR manifest is publicly readable, so image size/layer delivery or the
RunPod create path—not registry visibility—is the next startup investigation.
No new speed number was produced and no RunPod pod was left running. Do not
infer an acceleration result from those attempts. Slim or repair the image
startup path before retrying; keep `compile` in a separate named result set so
its cold and warm results are not confused with Spectrum.
