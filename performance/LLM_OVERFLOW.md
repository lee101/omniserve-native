# LLM overflow to RunPod (`tools/llm_overflow.py`)

```
client ─► omniserve-native :8791 (embedded batched LLM)
            │ local LLM saturated
            ▼ OMNISERVE_NATIVE_LLM_OVERFLOW_UPSTREAM=http://127.0.0.1:18796  (same path + body, plain http)
          llm_overflow.py :18796
            ├─ ready pod with in-flight < POD_CAPACITY   https://{podId}-8000.proxy.runpod.net   secret: <per-pod secret>
            └─ else LB serverless endpoint                https://{ENDPOINT_ID}.api.runpod.ai     Bearer $RUNPOD_API_KEY
```

Both remotes run the same `lee101/omniserve-llm:<tag>` image (omniserve-native in
LLM-only mode, `$PORT`=8000, `/ping` 204 loading / 200 ready).

## Router

- Proxies `POST /v1/chat/completions` and `/v1/completions`; `GET /health`, `GET /status`
  (router, per-backend in-flight/failovers, scaler EWMAs, spend, last decisions, non-secret config).
- Responses stream through unbuffered (SSE chunk by chunk); status and `Content-Type` pass
  through, plus `X-Overflow-Backend: pod:<name>|serverless:serverless`. Upstream requests carry
  `Accept-Encoding: identity` (RunPod's LB gzips otherwise, which also buffers SSE).
- Caller credentials (`Authorization`, `X-API-Key`, `X-Rapid-API-Key`, `secret`, `Cookie`) are
  dropped; the backend's own auth is added. `X-Omniserve-Tier` and other headers pass.
- Order: least-loaded ready pod under capacity → serverless. Before any byte reaches the
  caller, a connect error, timeout, 5xx, 401/403/404/408/429 or a "no workers" body fails
  over: each pod once, then serverless retried every `SERVERLESS_RETRY_S` until
  `OVERFLOW_DEADLINE_S` (cold start). A pod that frees up meanwhile is preferred. Other 4xx
  pass through. Once bytes are sent there is no retry; a mid-stream upstream error closes
  the caller's connection. Nothing works → `503 {"error":{"type":"overflow_unavailable"}}`,
  `Retry-After: 5`.
- A pod is taken out of routing after `POD_FAILS_UNREADY` consecutive failures and comes back
  on the next good `/ping`.

## Scaler (every `SCALE_INTERVAL_S`)

Signal: overflow concurrency (in-flight relays), time-averaged between ticks, folded into
EWMAs whose mean age matches a `SHORT_WINDOW_S` (2 min) and `LONG_WINDOW_S` (15 min) window
(tau = window/2).

| action | condition |
| --- | --- |
| start pod | long ≥ `POD_PROMOTE_CONCURRENCY` + Σ live pods `capacity × POD_PROMOTE_UTIL`, short ≥ `POD_PROMOTE_CONCURRENCY`, live pods < `MAX_PODS`, cooldown passed, spend + price ≤ `MAX_USD_HR` |
| drain + terminate | pod long-window concurrency < `POD_DEMOTE_CONCURRENCY` for `POD_IDLE_S` (cooldown applies) |
| | age > `POD_TTL_S` |
| | live spend > `MAX_USD_HR` (least-utilised first) |
| | not ready within `POD_BOOT_TIMEOUT_S`; status EXITED/ERROR |
| warm serverless | no free pod, short ≥ `WARM_CONCURRENCY` and ≥ 1.5 × long: one `GET /ping` to the endpoint per `WARM_INTERVAL_S` |

Promote 8 / demote 2 is the hysteresis; `POD_PROMOTE_UTIL` 0.70 mirrors app.nz's 0.70
busy-fraction promote (`RA2_OVERFLOW.md`, 1/serverlessPriceFactor 1.4). A pod costs
$0.6-1.2/hr against $1.1-1.75/hr serverless for the same card, and one pod absorbs
`POD_CAPACITY` (32) streams, so 8 sustained concurrent overflow streams is where a pod pays
for itself. Draining pods stop taking new requests and are terminated when empty or after
`POD_DRAIN_S`.

Pods: `POST /v2/pods` per GPU in `POD_GPU_TYPES` × `POD_CLOUD` until one returns 201
(skipping types whose catalog price would break the cap; 402/429 stop the round), name
`POD_NAME_PREFIX` + random hex, `registry` `REGISTRY_AUTH_ID`, env `HF_TOKEN`, `PORT`, `OMNISERVE_NATIVE_SECRET` =
HMAC(`POD_SECRET_KEY` or the RunPod key, pod name) plus `REMOTE_ENV_JSON`, port `8000/http`,
`minCudaVersion` `POD_MIN_CUDA`. Ready = `/ping` 200 through the proxy URL. Every
`RECONCILE_S` the sidecar lists `/v2/pods`, adopts prefixed pods it does not know (so a
restart never leaks one; the secret is re-derived) and refreshes status/cost. SIGTERM leaves
pods running; `serve --terminate-on-exit` terminates them. Nothing is written to disk.

## Endpoint setup

```bash
# dry run: prints the POST/PATCH body (secrets redacted) and the endpoint id if it exists
uv run --no-project --with aiohttp tools/llm_overflow.py setup --image lee101/omniserve-llm:<tag>
# create/update (workersMin 0, free at rest)
uv run --no-project --with aiohttp tools/llm_overflow.py setup --image lee101/omniserve-llm:<tag> --apply
uv run --no-project --with aiohttp tools/llm_overflow.py pods   # pods owned by the sidecar
```

`POST /v2/serverless` (or `PATCH /v2/serverless/{id}` when an endpoint with `--name` exists):
`type: LOAD_BALANCER`, `scaling: {type: REQUEST_COUNT, requestCount: 24}`,
`workers: {min: 0, max: 2, idleTimeout: 90}`, `gpu.pools` ADA_24, ADA_32_PRO, AMPERE_48,
ADA_48_PRO with `minCudaVersion: 12.9`, `ports: [8000/http]`, env `PORT`/`PORT_HEALTH`=8000,
`HF_TOKEN` (from env or `~/.cache/huggingface/token`), `flashboot: FLASHBOOT`,
`gpu.excludedTypes` the RTX PRO 6000 MIG 2g.48gb slice (it sits in ADA_48_PRO and was picked
first), `registry: $REGISTRY_AUTH_ID` (private GHCR). v2 takes the image inline, so no separate
template is created.

## Live run 2026-10-21 (image `ghcr.io/lee101/omniserve-llm:20261021a`)

Endpoint `k5k2rka63edkxp` (`omniserve-llm-overflow`), registry auth
`<registry-auth-id>` (`appnz-ghcr-lee101-v2`).

| | result |
| --- | --- |
| serverless, fresh worker (image pre-pulled, 8 GB HF download + load) | TTFT 56 s |
| serverless, FlashBoot resume after scale to 0 running | TTFT 2.1-2.2 s |
| serverless warm, 1 stream (RTX 4090) | 93-120 tok/s decode, TTFT 0.3 s |
| serverless warm, 16 / 32 streams (one 4090) | 704 / 690 tok/s aggregate, 43-103 tok/s per stream, TTFT p50 0.29 s |
| scale to 0 running | 90 s after the last request (idleTimeout) |
| pod RTX A6000 SECURE ($0.59/hr; COMMUNITY had no placement) | create → `/ping` 200 in 151 s |
| pod, 1 / 16 streams | 94 tok/s / 492 tok/s aggregate, TTFT 0.6-0.8 s via proxy |
| sidecar kill -9 + restart | pod adopted, secret re-derived, routing resumed; SIGTERM with `--terminate-on-exit` terminated it |

RunPod lists `idle`/`initializing`/`throttled` workers beyond `workers.max` (pre-pulled hosts
per release); only `running` is billed.

## Prod

`/etc/omniserve-llm-overflow.env` (root, 0600):

```bash
RUNPOD_API_KEY=...
HF_TOKEN=...
RUNPOD_LLM_ENDPOINT_ID=k5k2rka63edkxp
POD_IMAGE=ghcr.io/lee101/omniserve-llm:20261021a
REGISTRY_AUTH_ID=<registry-auth-id>
# optional: MAX_PODS=1 MAX_USD_HR=2.0 POD_PROMOTE_CONCURRENCY=8 POD_DEMOTE_CONCURRENCY=2
#           POD_IDLE_S=1200 POD_TTL_S=21600 POD_CAPACITY=32 POD_CLOUD=SECURE,COMMUNITY
#           REMOTE_ENV_JSON='{"OMNISERVE_NATIVE_LLM_CONTEXTS":"32"}' SERVERLESS_SECRET=...
```

```bash
sudo cp systemd/omniserve-llm-overflow.service /etc/systemd/system/
sudo systemctl daemon-reload && sudo systemctl enable --now omniserve-llm-overflow
curl -s 127.0.0.1:18796/status | jq '.router, .scaler'
journalctl -u omniserve-llm-overflow -f | rg 'scale|failover'
```

Without `POD_IMAGE` the scaler only reports; without `RUNPOD_API_KEY` it is router-only.
Rollback: unset `OMNISERVE_NATIVE_LLM_OVERFLOW_UPSTREAM` on the gateway, stop the unit, then
`tools/llm_overflow.py pods` and terminate anything left.

## Env reference

| var | default | |
| --- | --- | --- |
| `BIND`/`PORT` | 127.0.0.1 / 18796 | listen |
| `RUNPOD_LLM_ENDPOINT_ID` / `SERVERLESS_URL` | - | LB endpoint (URL overrides id) |
| `SERVERLESS_SECRET` | - | set on the endpoint as `OMNISERVE_NATIVE_SECRET` and sent as `secret:` |
| `REGISTRY_AUTH_ID` | - | RunPod container registry credential for pods and the endpoint |
| `OVERFLOW_DEADLINE_S` / `SERVERLESS_RETRY_S` | 90 / 6 | serverless cold-start retry budget |
| `CONNECT_TIMEOUT_S` / `READ_TIMEOUT_S` | 10 / 330 | per attempt; read = idle gap between chunks |
| `POD_CAPACITY` / `MAX_PODS` / `MAX_USD_HR` | 32 / 1 / 2.0 | |
| `POD_PROMOTE_CONCURRENCY` / `POD_PROMOTE_UTIL` / `POD_DEMOTE_CONCURRENCY` | 8 / 0.70 / 2 | |
| `POD_IDLE_S` / `POD_TTL_S` / `POD_BOOT_TIMEOUT_S` / `POD_DRAIN_S` | 1200 / 21600 / 900 / 120 | |
| `SCALE_INTERVAL_S` / `SCALE_COOLDOWN_S` / `RECONCILE_S` | 15 / 300 / 60 | |
| `SHORT_WINDOW_S` / `LONG_WINDOW_S` | 120 / 900 | |
| `POD_GPU_TYPES` | RTX 4090, RTX 5090, L40S, A40, RTX A6000 | v2 catalog `id`s, preference order |
| `POD_CLOUD` / `POD_MIN_CUDA` / `POD_DISK_GB` / `POD_PORT` | SECURE / 12.9 / 40 / 8000 | |
| `POD_NAME_PREFIX` | omniserve-llm-overflow- | ownership tag for adoption |
| `POD_SECRET_KEY` | RunPod key | HMAC master for per-pod secrets |
| `WARM_CONCURRENCY` / `WARM_INTERVAL_S` | 2 / 60 | |

Tests: `uv run --no-project --with pytest --with aiohttp pytest tests/test_llm_overflow.py -q`.
