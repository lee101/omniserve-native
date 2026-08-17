# OmniServe Native

A native C front door for local model serving. It exposes a browser model
selector and admits a model only when its configured weight and runtime budget
fit live NVIDIA GPU memory after retaining a 2 GiB safety reserve.

It binds only to `127.0.0.1:8080` by default. Selection is deliberately
separate from inference: an inference backend must consume the admitted
`active_model` state. This prevents a failed or overloaded backend from
bypassing VRAM admission.

## Run

```bash
make test
make run
```

Use `build/omniserve --models models/models.csv --port 18080` when port 8080
is already occupied.

Open `http://127.0.0.1:8080`. The service calls `nvidia-smi` on every status
and selection request. If NVIDIA telemetry is missing or errors, all model
admissions fail closed rather than risking VRAM exhaustion.

## Chat Front Door

`POST /v1/chat/completions` is an OpenAI-compatible JSON pass-through. It first
tries `OMNISERVE_LOCAL_UPSTREAM` when an admitted model still fits live VRAM.
On connection failures or 5xx responses, the local circuit opens after three
failures for 30 seconds. It then tries `OMNISERVE_REMOTE_UPSTREAM` with the
optional `OMNISERVE_REMOTE_BEARER_TOKEN`; that upstream has an independent
circuit breaker. Requests are rejected when both paths are unavailable.

Set `OMNISERVE_FRONTDOOR_TOKEN` to a long random value. Every chat request must
include `Authorization: Bearer <token>`. The status and model-selector routes
are intentionally unauthenticated for local operational visibility, so expose
them only through Cloudflare Access or a private tunnel.

The service invokes the installed `curl` binary for outbound HTTPS. This keeps
the C build dependency-free while supporting HTTPS remote providers. The Docker
image includes `curl` and CA certificates.

## Tunnel Deployment

1. Copy `.env.example` to `.env` and set the front-door and Cloudflare Tunnel
   tokens. Configure the tunnel's public hostname to route to `http://omniserve:8080`.
2. Set the local upstream to the OpenAI-compatible server serving the selected
   QLoRA/base model. In Docker on Linux, use a reachable host address instead
   of `host.docker.internal` unless you configure that hostname explicitly.
3. Run `docker compose up -d --build`.

The compose file does not publish port `8080` to the host; the Cloudflare Tunnel
is its only network ingress. Keep Cloudflare Access in front of the hostname.

## Distribution

GitHub Actions builds, tests, and uploads a Linux x86_64 tarball plus
`SHA256SUMS` on every push to `main`. Download the artifact from the workflow,
verify it, then run `omniserve --models models/models.csv`. A future bucket
mirror should copy both files unchanged and preserve the checksum manifest.

## Model Catalog

`models/models.csv` contains `id,label,weights_mib,runtime_mib`. The runtime
budget must include KV cache, compute workspace, and backend overhead for the
intended context length. Models that do not fit remain visible in the selector
but cannot be admitted.

The catalogue includes `anima-gemma-qlora`, the tiny Gemma QLoRA used by
text-generator.io. It is admitted only when a request needs it; OmniServe does
not preload it. `qwen2.5-7b-qlora` remains for chat. Large GGUF Q4 profiles and
the 24 GB Anima diffusion offload/compile rows were removed from this front
door so a quiet card is not reserved for unused weights.

## Shared GPU job queue

`build/omni-job` is a native SQLite/WAL queue for GPU work across service
types. Content-derived keys are unique, claims are transactional, abandoned
leases are recovered, a stale worker cannot settle a re-claimed job, and only
one live job may hold a named GPU. Required VRAM and priority are stored per
job; workers filter the kinds they implement. This lets an idle GPU choose
video matting, image generation, or another queued service without two workers
performing the same content-addressed request.

```bash
make build/omni-job
build/omni-job init /var/lib/omniserve/jobs.sqlite
build/omni-job submit /var/lib/omniserve/jobs.sqlite "$KEY" video-matting request.json 1800 10
build/omni-job claim /var/lib/omniserve/jobs.sqlite host-pid host:0 12000 120 video-matting,image
```

The `video-matting-rvm` catalog entry is the measured 1.8 GiB admission
profile for the sibling Manifold video-matting proof. The queue is engine
agnostic; production can replace the GPL research backend without changing
the scheduling or cache contract.

The production video-matting workload samples eight frames with a resident
YOLO11n COCO person gate. Clips containing a person use local RVM; clips with
no detected person, or any detector uncertainty, return a structured
`fallback_required` result so the Manifold service can move the same durable
job to its general-matting standby without treating that decision as an
endpoint outage. The H3 image also has a face-refinement detector, but that
face-specific weight is not used as a proxy for full-person routing.

RVM uses the ResNet50 backbone in native PyTorch FP16 with eight-frame temporal
chunks. It measured faster than MobileNetV3 on the production RTX 5090;
`RVM_BACKBONE=mobilenetv3` remains available for other GPU profiles.
Production benchmarks on the RTX 5090 found eager mode slightly faster once
warm and avoided Inductor's 45-110 second cold start, so compile is opt-in with
`RVM_TORCH_COMPILE=1`. A lazy Inductor failure falls back to the already
resident eager module. Metrics report the actual engine, compile fallback reason,
chunk size, person confidences, and per-job throughput.

The worker writes RGBA directly to `libvpx-vp9`, then forces the libvpx decoder
through `alphaextract` after the final audio remux. A result is never cached or
uploaded unless that separate WebM alpha plane is decodable. Plain `ffprobe`
may still report the visible VP9 stream as `yuv420p`; use a forced libvpx decode
when inspecting these files.

## Multi-workload GPU runtime

OmniServe owns the deployable GPU image in `Dockerfile.runpod`; `Dockerfile`
continues to build the lightweight authenticated front door. The RunPod entry point
loads `workloads/workloads.json`, admits a workload against live free VRAM, and
releases the active engine before switching kinds. The current image provides
detail-preserving `video-matting` and an adapter for the H3 runtime inherited
from the shared base image. New services add a manifest row and a module with
`handler(job)` plus an optional `release()` hook; product sites do not need a
new GPU server or scheduling implementation.

MiniMax-Music3 is deployed from this repository as a separate scale-to-zero
endpoint with `scripts/deploy-music3-runpod.sh`. The RunPod adapter is native C
in `music3c/`; it lazy-starts `sgl-omni` on the first request and will not
replace the ~38 GiB checkpoint unless `MUSIC3_FORCE_REDOWNLOAD=1`. Prefer H100
80GB (H200 remains allowed). The worker keeps the quality-preserving upstream
defaults: backbone and RVQ CUDA graphs, compiled DIT blocks, compiled DAV
decoder, 30 DIT steps, and the measured-fastest `torch_sdpa` acoustic attention
path.

Requests should set an explicit workload:

```json
{
  "input": {
    "workload": "video-matting",
    "video_url": "https://cdn.example/input.webm",
    "preserve_audio": true,
    "output_upload_url": "https://presigned-upload.example/...",
    "output_public_url": "https://cdn.example/output.webm"
  }
}
```

For compatibility, a request containing `video_url` and
`output_upload_url` is inferred as `video-matting`. RunPod supplies the
distributed endpoint queue. A local mixed-use GPU instead drains the native
SQLite queue through the same registry:

```bash
build/omni-job submit /var/lib/omniserve/jobs.sqlite "$KEY" video-matting /srv/requests/$KEY.json 1800 10
PYTHONPATH=. python3 runtime/local_worker.py --kinds video-matting,h3-video
```

Deploy or update the scale-to-zero RunPod endpoint with:

```bash
OMNISERVE_IMAGE_REPOSITORY=ghcr.io/lee101/omniserve-native \
  scripts/deploy-runpod.sh
```

The deployment script accepts the previous `VIDEO_BACKGROUND_RUNPOD_*` names
as compatibility aliases, so the live Manifold endpoint can migrate without a
new public job contract. RVM remains GPL-3.0 research software and must be
replaced or approved before commercial launch; that replacement is isolated to
the workload module.

## Wan-Animate-2 and global GPU admission

`wan-animate-2` accepts a character image plus a driving video and uses the
official distilled Wan-Animate-2 Diffusers checkpoint by default. Its ordered
execution frontier is `small` (20 GiB, NF4 + sequential offload), `balanced`
(28 GiB, NF4 + model offload), and `throughput` (56 GiB, resident BF16 plus
`torch.compile`). `execution_profile=auto` selects the fastest lane that fits
after reclaiming idle workloads. A CUDA OOM unloads every engine, clears the
allocator, and retries once on the next smaller lane.

On a shared production GPU, set `OMNISERVE_VRAM_BROKER_URL` to the native front
door, such as `http://127.0.0.1:8791`. The Python runtime then composes its live
NVIDIA memory checks with the front door's cross-process `/v1/gpu/lease` API.
Set `OMNISERVE_VRAM_BROKER_REQUIRED=1` to fail closed if that broker is down.

```json
{
  "input": {
    "workload": "wan-animate-2",
    "character_image_url": "https://cdn.example/character.png",
    "driving_video_url": "https://cdn.example/dance.mp4",
    "prompt": "Person appearance: ... Background: ...",
    "duration": 5,
    "execution_profile": "auto",
    "output_upload_url": "https://presigned-upload.example/...",
    "output_public_url": "https://cdn.example/result.mp4"
  }
}
```
