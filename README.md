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
