# Qwen Image 2.1 ("ra2") overflow — cost, wiring, rollback

Local lane: a second `omniserve-native` (`omniserve-native-qwen.service`,
`127.0.0.1:8792`, branch `qwen-ra2`, `build-qwen`, stable-diffusion.cpp master)
serving `POST /v1/images/generations` and `/v1/images/edits` from the Q4_K_M
Qwen Image 2.1 GGUF set. This document covers only what happens when that lane
cannot answer: where the request goes, what it costs, and how to turn it off.

## Architecture

```
client (netwrck, cutedsl.cc, ...)
  │  POST /v1/images/generations | /v1/images/edits   (paid tier)
  ▼
omniserve-native  :8792                      ┌─ local: libstable-diffusion.so, one permit
  ├── osd_try_cached_result ─────────────────┤  (DiT resident, TE on CPU)
  │        hit → serve from the result cache └─ miss ↓
  ├── osched_try_acquire_n(image) ── free? ──► generate locally
  │        busy / no permit        ──┐
  ├── local generate failed ────────┤
  └── no SD context loaded ─────────┘
                                    ▼
        oproxy_target_relay(OMNISERVE_NATIVE_IMAGE_OVERFLOW_UPSTREAM
                            + OMNISERVE_NATIVE_IMAGE_OVERFLOW_PATH=/predict-sync)
        Authorization: Bearer <OMNISERVE_NATIVE_IMAGE_OVERFLOW_API_KEY>
        (caller's Authorization/X-API-Key/X-Rapid-API-Key/secret dropped)
                                    ▼
        app.nz  POST /api/cogs/<ra2 cog id>/predict-sync
          │  decideTier() over a 5-minute window of requests × learned seconds
          ├── local GPU headroom on the app.nz host (free)
          ├── RunPod serverless, workersMin=0  ← sparse traffic, pay per second
          └── RunPod pod (promoted only while sustained)  ← continuous billing
                                    ▼
        RunPod worker: ghcr.io/lee101/omniserve-native:ra2
          runtime/handler.py (serverless) or runtime/cog_pod.py (pod)
          workloads/qwen_image.py → libstable-diffusion.so, weights on the volume
                                    ▼
        the worker's own output JSON is relayed back to the client unchanged
```

The gateway never talks to RunPod and holds no provider credential: app.nz owns
provisioning, per-second billing, the idle reap and the orphan reconciler, so a
gateway crash cannot leak a rented worker. The response body is the image JSON
the client already parses, because the worker returns the OmniServe
`/v1/images/generations` shape (`data[0].b64_json`) alongside app.nz's `outputs`.

## Cost model

Published RunPod list prices (runpod.io/pricing, page updated 2026-09-13, read
2026-09-22):

| pool | rate | per second |
| --- | ---: | ---: |
| Serverless RTX 4090 24 GB | $1.10/hr | $0.000306 |
| Serverless 48 GB (L40S/L40/6000 Ada) | $1.75/hr | $0.000486 |
| Pod RTX 4090 | $0.74/hr | $0.000206 |
| Pod L40S 48 GB | $1.09/hr | $0.000303 |

Serverless is billed per second from worker start to full stop
(docs.runpod.io/serverless/pricing), so the cold worker — not the sampling — is
what a sparse lane actually pays for. app.nz models that premium as
`serverlessPriceFactor = 1.4` over the same GPU's pod rate (`server/comfy.go:46`)
and bills per second as `perHour × factor / 3600 × elapsed`
(`server/cogs.go:1079-1082`).

Break-even: a pod is cheaper once its busy fraction exceeds
`1 / serverlessPriceFactor`. That is the `1 / (1.20 × seconds_per_request)` rule
in `README.md` evaluated with the live factor:

| seconds/request | requests/hour to justify a pod (÷1.4) | (÷1.20, older rule) |
| ---: | ---: | ---: |
| 5 | 514 | 600 |
| 13 | 198 | 231 |
| 25 | 103 | 120 |

The running router implements this as utilisation, not a raw rate: a 5-minute
request window (`rpsWindow = 5m`, `server/gpu_router.go:66`), `busyFraction =
rate × EMA predict seconds` (`:190`), promote at 0.70 busy, demote at 0.40
(`:141-152`, `1/1.4 ≈ 0.714`), with the per-cog runtime learned as an EMA
(`server/cog_stats.go:21`). Hysteresis is what keeps a lane from flapping
between the two pools.

Per-request cost for a 1024×1024, 20-step EasyCache request, taking 13 s on a
4090:

| placement | cost per request | when it applies |
| --- | ---: | --- |
| serverless, warm worker | $0.0040 | bursty traffic |
| serverless, cold worker (+10-20 s load) | $0.0070-0.0100 | first request after idle |
| pod, continuously billed | $0.0027 | only above ~198 req/hr |
| local `ra2` lane | $0 (sunk cost) | until the device is busy |

Two consequences follow, and they are the whole design:

1. **Scale to zero must be fast, and the first request must be cheap.** The
   weights live on the RunPod network volume (models are ~11 GB: 4.6 GB DiT +
   4.7 GB text encoder + 1.1 GB mmproj + 0.6 GB VAE); the worker loads them by
   mmap in well under a second and streams the rest on first use, so a cold
   start costs seconds, not a download. `workersMin = 0` and a short idle
   timeout keep an idle lane at zero spend.
2. **Free and background traffic never reach the remote.** The overflow is
   gated on `OMNISERVE_NATIVE_OVERFLOW_TIERS` (default `paid`) before the
   request is looked at, so the only way to spend money is a paid request that
   the local device could not take.

## Shared image admission on :8791

Configure the main Z-Image gateway on `:8791` to route Qwen requests to the
loopback sibling on `:8792`:

```bash
OMNISERVE_NATIVE_IMAGE_MODEL_UPSTREAMS=ra2=http://127.0.0.1:8792,qwen-image-2.1=http://127.0.0.1:8792,qwen=http://127.0.0.1:8792
OMNISERVE_NATIVE_IMAGE_MODEL_UPSTREAM_SECRET=<8792 instance's OMNISERVE_NATIVE_SECRET>
```

Send all image traffic through `:8791`, with `"model":"ra2"` for Qwen; keep
`:8792` private and do not configure a reciprocal mapping there. The main
scheduler holds its image permits throughout each sibling response, balancing
Qwen against local Z-Image and other admitted GPU work. This includes edits and
img2img even if Z-Image lacks reference-edit support. The sibling retains its
own overflow policy below. Watch `/status.image_model_upstreams` and
`omniserve_image_model_relay_total{model="ra2"}` on the main gateway. Removing
the mappings restores existing routing after restart. These are configuration
instructions only; this change does not modify production services.

## Production environment

`ra2` instance (`/etc/omniserve-qwen.env`, service `omniserve-native-qwen`):

```bash
OMNISERVE_NATIVE_IMAGE_OVERFLOW_UPSTREAM=http://127.0.0.1:8787/api/cogs/<RA2_COG_ID>
OMNISERVE_NATIVE_IMAGE_OVERFLOW_PATH=/predict-sync
OMNISERVE_NATIVE_IMAGE_OVERFLOW_API_KEY=<app.nz API key of the cog's owner>
OMNISERVE_NATIVE_IMAGE_OVERFLOW_TIMEOUT_MS=600000
OMNISERVE_NATIVE_OVERFLOW_TIERS=paid
```

* The key is an app.nz credential with the `api` scope for the user that owns
  the cog — a `papers_api` API key (`api_keys.app_id = 'papers_api'`) or an
  OAuth token carrying the `api` scope (`server/main.go:2154-2196`). The gateway
  strips the caller's own `Authorization`/`X-API-Key`/`X-Rapid-API-Key`/`secret`
  whenever this key is set, so a client credential is never replayed to app.nz.
* `<RA2_COG_ID>` is the deployed cog's id, not the template name: the seam is
  addressed per deployment (`POST /api/cogs/{id}/predict-sync`). Deploy it once
  and pin the returned id.
* The remote path defaults to `/predict-sync`; the timeout has to cover a
  serverless cold start plus ~11 GB of weight streaming, which is why it is 10
  minutes rather than the 600 s default for chat-shaped upstreams.
* `/v1/images/edits` overflows through the same path: `handle_images` serves
  both routes when the model advertises reference-edit support
  (`OMNISERVE_NATIVE_SD_REFERENCE_EDIT=1`, which the prod unit already sets), and
  the body is relayed unchanged, `image_base64` included.

Worker-side defaults are set in `Dockerfile.runpod-sdcpp` (EasyCache 0.05, 20
steps, guidance 1.0, 1024×1024, WebP). app.nz's cog template supplies the rest.

## Deploy

```bash
# 1. Worker image (from a checkout with the stable-diffusion.cpp CUDA build)
docker build -f Dockerfile.runpod-sdcpp -t ghcr.io/lee101/omniserve-native:ra2 .
docker push ghcr.io/lee101/omniserve-native:ra2

# 2. Deploy the cog on app.nz and capture its id
curl -sS -X POST https://app.nz/api/cogs/run \
  -H "Authorization: Bearer $APPNZ_API_KEY" -H 'Content-Type: application/json' \
  -d '{"template":"qwen-image-2.1","input":{"prompt":"a red fox in snow"}}' \
  | jq -r '.model.id'

# 3. Seed the shared network volume once (first cold start would otherwise pay
#    for the 11 GB download); the worker fills it itself on first start.
#    app.nz attaches the volume; /runpod-volume/omniserve/qwen-image-2.1 is the
#    path the image reads.

# 4. Pin the id in /etc/omniserve-qwen.env (above), then
systemctl restart omniserve-native-qwen
curl -sS http://127.0.0.1:8792/status | jq '.overflow'

# 5. Smoke: a paid request on a saturated lane must come back with the remote's
#    body while status counters move.
curl -sS http://127.0.0.1:8792/v1/images/generations \
  -H "Authorization: Bearer $OMNISERVE_NATIVE_SECRET" \
  -H 'X-Omniserve-Tier: paid' -H 'Content-Type: application/json' \
  -d '{"prompt":"overflow smoke","width":1024,"height":1024,"steps":20}'
curl -sS http://127.0.0.1:8792/status | jq '.overflow'
```

`scripts/deploy-runpod.sh` is the direct-RunPod path (template + endpoint from
`deploy/runpod.json`) and is not needed for the app.nz placement; it is still
how a standalone endpoint is stood up for a worker smoke test. Two env knobs
were added for that: `OMNISERVE_DOCKERFILE` (default `Dockerfile.runpod`) and
`OMNISERVE_RUNPOD_CONFIG` (default `deploy/runpod.json`), plus
`deploy/runpod-ra2.json` — pinned to `allowedCudaVersions: ["12.8"]`, because
the ra2 image builds against CUDA 12.8 while the video endpoint's config asks
for 13.0.

```bash
OMNISERVE_DOCKERFILE="$PWD/Dockerfile.runpod-sdcpp" \
OMNISERVE_RUNPOD_CONFIG="$PWD/deploy/runpod-ra2.json" \
OMNISERVE_IMAGE_REPOSITORY=ghcr.io/lee101/omniserve-native \
RUNPOD_API_KEY=... bash scripts/deploy-runpod.sh
```

## Rollback

Local failure and saturation are already fail-safe: without the upstream the
lane behaves exactly as it did before (queue, then 503), and free traffic never
touches the remote either way.

```bash
# Stop spending, keep the lane: drop the upstream and restart the ra2 unit.
sudo sed -i '/IMAGE_OVERFLOW_/d' /etc/omniserve-qwen.env
sudo systemctl restart omniserve-native-qwen

# Stop the remote side entirely (app.nz releases the pod/worker; scale to zero
# is free at rest, so this is for incidents, not for idle):
curl -sS -X POST https://app.nz/api/cogs/<RA2_COG_ID>/sleep \
  -H "Authorization: Bearer $APPNZ_API_KEY"
```

Nothing in the gateway holds a provider credential, so rollback cannot strand a
rented worker: the app.nz idle reap and orphan reconciler own that, and the
per-second billing stops when the last worker stops.

## Observability

* `/status.overflow` — `image` (remote configured), `image_path`, `tiers`,
  `saturated`, `local_failed`.
* `omniserve_overflow_total{cause="saturated"|"local_failed"}` — the two causes
  call for opposite responses. `saturated` is a capacity statement (no permit, a
  broker refusal, or a headroom refusal) and says buy local capacity;
  `local_failed` means the lane itself could not run the request at all (no SD
  context loaded, or generation failed before anything was written) and says fix
  a backend.
* `X-AppNZ-Execution-Tier` on the app.nz response (`serverless`, `pod`, or
  `local`) — which placement actually served the request, mirroring what
  `omniserve_capacity_spend_rate_usd_hr` reports for the rented lanes.

## Measurements behind the defaults

| what | measured | where |
| --- | ---: | --- |
| resident weight load (worker ctx) | 410-680 ms | `tools/ra2_worker_smoke.py` on a shared 3090 Ti |
| warm t2i, 512², 4 steps, TE on CPU | 2.0 s | same run |
| first (cold) denoise step vs warm | 2.68 s vs ~0.27 s | weight streaming on first use |
| local prod lane, 1024², 20 steps, EasyCache 0.05 | 10-11 s wall | `qwen21-ra2-rollout-2026-09-22.md` |
| reference edit, 512², 4 steps, TE on CPU | 39.6 s sampling | shared box, CPU text encoder |

The edit path is text-encoder bound when the TE is on CPU, so a dedicated
serverless worker should keep the encoder on the GPU (`RA2_PARAMS_BACKEND` unset,
the image default) and the paid-only tier gate is what keeps that decision
paying for itself.
