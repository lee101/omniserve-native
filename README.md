# omniserve-native

Native C inference server for one GPU: llama.cpp LLM/ModernBERT embeddings + stable-diffusion.cpp diffusion, or existing Gemma/vLLM, image, 3D, multimodal, TTS, and STT workers behind one tier-priority scheduler. OpenAI-compatible HTTP, zero Python in the gateway hot path. Every OmniServe-owned gateway runtime source file is C; llama.cpp, stable-diffusion.cpp, CUDA, and optional model-specific 3D workers remain external runtimes behind stable HTTP boundaries.

The source is available under the [Apache License 2.0](LICENSE). Model weights
and external runtimes retain their own licenses.

- `src/ohttp.c` — epoll reactor + worker pool HTTP/1.1 (SO_REUSEPORT, keep-alive, chunked/SSE streaming, TCP_NODELAY)
- `src/ojson.c` — allocation-light JSON tokenizer + escape helpers
- `src/oproxy.c` — resolved-once HTTP/1.1 connection pools + framed raw relay; preserves chunked/SSE TTFT and binary responses
- `src/osched.c` — paid/sub/free/background weighted admission (FIFO within tier, background only on idle) + NVML VRAM via dlopen
- `src/oscale.c` — pure cost/priority decision engine for rented overflow capacity (paid-only, scale-to-zero, spend caps, hard TTL)
- `src/ocapacity.c` — controller thread: samples admission pressure, asks oscale, warms cogs through app.nz on loopback
- `src/otune.c` — per-device batch/KV/flash-attention profiles (blackwell, hopper, ada, ampere, turing, cpu)
- `src/backend_llama.c` — embedded libllama (CUDA sm_120), prefix-aware parallel context pool, chunked long-prompt prefill, Gemma 4/Qwen chat formatting, streaming token callback, and encoder-only embeddings with an independent CPU/GPU placement policy
- `src/backend_sd.c` — embedded stable-diffusion.cpp (`-DWITH_SD=ON`), allocation-efficient PNG output
- `/docs` + `/openapi.json` — self-contained API docs (text-generator.io pattern: static spec, human page)

## Build

Presets carry the right flags for each purpose; use `dev` while working and
`release` to ship.

```bash
cmake --preset dev && cmake --build --preset dev && ctest --preset dev
cmake --preset release && cmake --build --preset release   # LTO, -march=native
cmake --preset sanitize && ctest --preset sanitize         # ASan + UBSan
cmake --preset ci                                          # portable, -Werror
```

`release` expects `../llama.cpp` built with `-DGGML_CUDA=ON
-DCMAKE_CUDA_ARCHITECTURES=120`; `sanitize` and `ci` build with `WITH_LLAMA=OFF`
so they need no CUDA and no external checkout.

The presets use Ninja, and CMake picks up `mold`/`lld` and `ccache` when they are
installed. `dev` differs from `release` only in dropping LTO, which is the whole
incremental cost: touching `src/main.c` rebuilds in **2.5s** under `dev` versus
**3.8s** under `release`. Warnings are broader than `-Wall -Wextra` alone
(`-Wshadow`, `-Wstrict-prototypes`, `-Wmissing-prototypes`, `-Wvla`,
`-Wpointer-arith`, `-Wwrite-strings`, `-Wundef`, `-Wold-style-definition`,
`-Wredundant-decls`) and the tree is clean under all of them; vendored llama.cpp
and stable-diffusion.cpp headers are included as `SYSTEM` so third-party code
cannot fail our build.

CI (`.github/workflows/ci.yml`) runs the portable build under both gcc and clang
with `-Werror`, the test suite under ASan+UBSan and under TSan (the scheduler,
the capacity controller, and the HTTP reactor are all threaded), plus `cppcheck`
and `clang-tidy`. The clang-tidy check list lives in `.clang-tidy` and is
curated so a clean run means something: every disabled check records why, and
`clang-analyzer-optin.performance.Padding` stays on because it caught a real
8-byte hole in a per-thread job struct.

## Run

```bash
OMNISERVE_NATIVE_LLM_GGUF=/nvme0n1-disk/models/qwen3-0.6b-q8.gguf \
OMNISERVE_NATIVE_EMBEDDING_GGUF=/nvme0n1-disk/models/omniserve-native/modernbert-base-q8_0.gguf \
OMNISERVE_NATIVE_SD_MODEL=/models/sd-turbo.safetensors \
OMNISERVE_NATIVE_SECRET=... \
./build/omniserve-native --port 8791
```

Split Z-Image artifacts use `OMNISERVE_NATIVE_SD_DIFFUSION_MODEL`,
`OMNISERVE_NATIVE_SD_VAE`, and `OMNISERVE_NATIVE_SD_LLM`. Diffusion and general
flash attention default on. Constrained co-residency can additionally set
`OMNISERVE_NATIVE_SD_PARAMS_BACKEND=diffusion=cpu`,
`OMNISERVE_NATIVE_SD_MAX_VRAM`, and `OMNISERVE_NATIVE_SD_STREAM_LAYERS=1`.
Exact-prompt latent replay is opt-in per request with `"teleport": true`.
`OMNISERVE_NATIVE_SD_TELEPORT_CACHE_SIZE` bounds the in-process LRU (default
64, maximum 256), and `OMNISERVE_NATIVE_SD_TELEPORT_START_STEP` selects the
default resume step (7). Keys include the complete prompt, negative prompt,
dimensions, steps, guidance, seed, and resume step; approximate prompt matches
are deliberately excluded.

Keep `OMNISERVE_NATIVE_IMAGE_UPSTREAM` configured for LoRA/catalog and auxiliary
image routes while selecting the embedded stable-diffusion.cpp generator with
`OMNISERVE_NATIVE_IMAGE_PREFER_EMBEDDED=1`. Trusted, non-relayed loopback
callers may pass up to eight normalized LoRAs as
`{"path":"/path/adapter.safetensors","scale":0.8}`. Public callers must use
cache-validated `lora_id`, optionally with `lora_filename` and `lora_scale`, and
`OMNISERVE_NATIVE_LORA_DIR` set. Filename lookup is cache-only, rejects path
components, and verifies the resolved file remains below that directory; the C
request path never downloads weights. Images are WebP by default (falling back
to PNG if libwebp is unavailable), controlled by
`OMNISERVE_NATIVE_SD_IMAGE_FORMAT` and `OMNISERVE_NATIVE_SD_WEBP_QUALITY`.
`OMNISERVE_NATIVE_SD_MAX_BATCH` is deliberately 1 by default and may be raised
to at most 8 only after a VRAM/latency canary.

Or use it as the fast shared gateway in front of the current production workers:

```bash
OMNISERVE_NATIVE_LLM_UPSTREAM=http://127.0.0.1:8300 \
OMNISERVE_NATIVE_IMAGE_UPSTREAM=http://127.0.0.1:8100 \
OMNISERVE_NATIVE_IMAGE_WORKER_UPSTREAM=http://127.0.0.1:8100 \
OMNISERVE_NATIVE_BIREFNET_UPSTREAM=http://127.0.0.1:9094 \
OMNISERVE_NATIVE_TTS_UPSTREAM=http://127.0.0.1:9083 \
OMNISERVE_NATIVE_STT_UPSTREAM=http://127.0.0.1:9083 \
OMNISERVE_NATIVE_FORECAST_UPSTREAM=http://127.0.0.1:8101 \
OMNISERVE_NATIVE_EMBEDDING_UPSTREAM=http://127.0.0.1:9083 \
OMNISERVE_NATIVE_MULTIMODAL_UPSTREAM=http://127.0.0.1:9083 \
OMNISERVE_NATIVE_ANIMATION_UPSTREAM=http://127.0.0.1:9092 \
OMNISERVE_NATIVE_3D_UPSTREAM=http://127.0.0.1:9093 \
OMNISERVE_NATIVE_AUX_UPSTREAM=http://127.0.0.1:9083 \
OMNISERVE_NATIVE_SLOTS=4 \
OMNISERVE_NATIVE_LLM_PERMITS=1 \
OMNISERVE_NATIVE_IMAGE_PERMITS=4 \
./build/omniserve-native --port 8791
```

`OMNISERVE_NATIVE_UPSTREAM` sets one unified fallback; modality-specific values override it. Upstream DNS is resolved once and `OMNISERVE_NATIVE_UPSTREAM_IDLE` controls the per-modality idle connection pool (default: worker count). Chronos-2 is exposed through `/forecast`, `/forecast_batch`, and the canonical `/v1/forecasts` alias; `OMNISERVE_NATIVE_FORECAST_PATH` can remap that alias for another worker API.

`OMNISERVE_NATIVE_IMAGE_UPSTREAM` is the canonical OpenAI/control-plane image API. `OMNISERVE_NATIVE_IMAGE_WORKER_UPSTREAM` optionally sends legacy CuteDSL routes such as `/generate_image`, `/caption`, and `/aesthetic_score` directly to their contract-compatible worker while retaining native admission, pooling, and metrics.

For distilled pipelines whose public API uses zero as a guidance sentinel while the native scheduler expects a nonzero CFG, `OMNISERVE_NATIVE_SD_ZERO_GUIDANCE` maps only an exact request value of `0.0`. Z-Image-Turbo uses `1.0`; explicit nonzero request guidance is never changed.

Constrained GPU VAE decode can use `OMNISERVE_NATIVE_SD_VAE_TILING=1` without
moving the VAE to CPU. Latent tile width and height default to 32 and are controlled by
`OMNISERVE_NATIVE_SD_VAE_TILE_X`, `_Y`, and `_OVERLAP` (default 0.5). Keep this
behind the image parity gate because tiling changes the decode graph.

Other tuning: `OMNISERVE_NATIVE_PORT`, `BIND`, `SLOTS`, `SECRET`, `LLM_GGUF`, `LLM_SWAP_DIR`, `LLM_CONTEXTS`, `NGL`, `NGL_AUTO_KEEP_FREE_MB`, `CTX`, `BATCH`, `UBATCH` (both accept `auto`), `KV_TYPE` (`f16` default, `q8_0` halves the KV cache at no measured quality cost — see `performance/quality.md`), `FLASH_ATTN` (auto; forced on for a quantized cache because llama.cpp requires it for a quantized V), `EMBEDDING_GGUF`, `EMBEDDING_NGL` (defaults to CPU), `EMBEDDING_CTX`, `EMBEDDING_POOLING` (`mean` default, `cls` for retrieval finetunes like gte-modernbert), `EMBEDDING_THREADS`, `SD_MODEL`, `ADMISSION_TIMEOUT_S`, `UPSTREAM_TIMEOUT_MS`, `REACTORS`, `WORKERS`, and per-modality `LLM_PERMITS`, `IMAGE_PERMITS`, `TTS_PERMITS`, `STT_PERMITS`, `EMBEDDING_PERMITS`, `MULTIMODAL_PERMITS`, `ANIMATION_PERMITS`, `3D_PERMITS`, and `AUX_PERMITS`.

## Local-first ASR and background fine-tuning

Point `OMNISERVE_NATIVE_STT_UPSTREAM` at the `omniserve-asr-router` C binary to
prefer a lazy local Parakeet/Whisper worker on NVIDIA CUDA, AMD ROCm, or CPU.
The router checks worker health and free VRAM, and replays only transient
429/502/503/504 responses to the configured managed STT fallback. Caller 4xx
responses are returned without retry. The systemd unit runs the native binary;
`workers/asr_router.py` remains as a readable reference implementation while
the cutover is canaried. DictatorFlow can use the gateway first via
`OMNISERVE_STT_URL`; its existing provider chain remains available if the
whole local route is down.

The native router reuses the gateway's epoll HTTP server, bounded request body,
resolved-once upstream targets, and keep-alive pools. It adds
`X-Omniserve-ASR-Backend` and `X-Omniserve-ASR-Attempts` to every relayed
response, and marks worker calls with `X-Omniserve-Internal: local`. Build and
exercise it independently with:

```bash
cmake --build --preset dev --target omniserve-asr-router
OMNISERVE_ASR_ROUTER_BIN=build-dev/omniserve-asr-router \
  python tests/test_asr_router_native.py
```

Fine-tuning runs through `OMNISERVE_NATIVE_TRAINING_UPSTREAM` and the forced
background `/v1/training/jobs/run` route. It owns every scheduler permit only
while the machine is idle. When interactive work queues, the trainer saves a
checkpoint and exits rather than pausing while retaining VRAM. See
[`docs/asr-training.md`](docs/asr-training.md) for setup, explicit public-model
consent, WER release gates, and Hugging Face publication.

Worker APIs sometimes use different paths for the same operation. Configure `IMAGE_GENERATE_PATH`, `TTS_PATH`, `TTS_OPENAI_PATH`, `STT_FILE_PATH`, `STT_URL_PATH`, `STT_OPENAI_PATH`, and `CAPTION_PATH` (all with the `OMNISERVE_NATIVE_` prefix) to rewrite only the upstream request path while preserving the public path and query string. This avoids false-positive route support from blind same-path proxying.

## GPU placement is checked, not assumed

A CUDA init failure makes llama.cpp load the weights on CPU and keep reporting
ready, which serves text at a fraction of GPU throughput and holds the whole
model in host RAM. When `OMNISERVE_NATIVE_NGL` asks for offloaded layers and no
GPU backend device is registered, the gateway logs the reason and exits so the
supervisor retries; `OMNISERVE_NATIVE_ALLOW_CPU_FALLBACK=1` overrides that for
deliberate CPU deployments. `/status` reports `gpu.placement`, `gpu.device`,
`gpu.kv_type`, and `gpu.degraded`; `/health` stays 200 with a `status` of `ok`
or `degraded` so a liveness probe cannot restart-loop the process, while
`/readyz` returns 503 when the embedded model is on CPU so a load balancer can
drain it. `vram_available` distinguishes "the driver says 0 free" from "the
driver is unreachable" — the NVML handle is retried with a backoff instead of
being resolved once, so a transient driver outage no longer poisons VRAM gating
for the process lifetime.

The packaged unit orders after `nvidia-persistenced.service` and waits for
`nvidia-smi -L` before starting. Without that ordering a cold boot reaches
`network.target` before the driver stack is usable.

## Observability and automated on-call

`GET /metrics` is Prometheus text and `GET /errors` is the recent-failure ring.
Both exist so a monitor never has to scrape the journal: on a box sharing one
with a dozen services, a log line is ambiguous about which process produced it,
while a counter you can diff between two polls is not. The access log below is
the per-request record, kept in its own bounded file for forensics — it does not
feed alerting.

- `/metrics` — responses by status class, `omniserve_gpu_degraded`,
  `omniserve_vram_free_gib` (`-1` when the driver is unreachable), admission
  slots, per-tier admission timeouts, worst queue wait, and per-lane rented-
  capacity spend rate.
- `/errors` — the last 32 5xx responses with method, path, and age in seconds.
  Own responses are recorded where the status is written; **proxied** responses
  are recorded by reading the status line of the first relayed write, so an
  upstream 500 is not invisible just because the bytes passed through verbatim.

- `access.log` — one line per request: timestamp, peer, method, path, model,
  status, duration, the internal/external verdict, and which trust-relevant
  headers were *present*. Never header values, never the body, never the query
  string. It answers "who called and with what markers", which a counter cannot;
  it is not an alerting path. Formatting is allocation-free, the write happens on
  a drain thread, and retention is capped in-process at **4 x 32 MiB = 128 MiB**,
  roughly a month at this gateway's measured rate.
  See [`docs/access-log.md`](docs/access-log.md).

`monitoring/` (gitignored, host-local) drives those signals into an autonomous
on-call loop: `oncall.sh` polls every 120s and wakes a coding agent **only** when
new 5xx appeared since the last poll, `/readyz` went 503, or the process stopped
answering. A cooldown prevents agent storms and a lockfile prevents overlapping
runs. `monitoring/ONCALL.md` is the agent's brief — what the service is, how to
attribute a 5xx (own bug vs upstream vs capacity vs degradation), the
build/test/restart/quality-bench sequence that must pass before anything counts
as fixed, and the hard rules: never weaken a check to silence an alert, never arm
a billable capacity lane, never restart another service unless `/errors`
attributes the failure to it, never commit.

Deliberately not escalated: 4xx at any volume (clients sending bad requests is
not an outage), admission timeouts alone (that is load, not a defect), and a
single transient 5xx below the threshold.

## Per-device tuning

Batch geometry that saturates a 5090 stalls a T4, so `src/otune.c` keys the
defaults off the backend's device description and `/status.tune` reports which
profile was applied. Anything set explicitly always wins, so
`OMNISERVE_NATIVE_BATCH`/`UBATCH`/`KV_TYPE`/`FLASH_ATTN` still override.

| Class | Devices | batch/ubatch | KV | Flash attn | Contexts |
| --- | --- | --- | --- | --- | ---: |
| blackwell | RTX 5090, B200 | 4096 / 1024 | q8_0 | on | 4 |
| hopper | H100, H200 | 4096 / 1024 | q8_0 | on | 6 |
| ada | RTX 4090, L40S | 2048 / 512 | q8_0 | on | 3 |
| ampere | RTX 3090, A100, A40 | 2048 / 512 | q8_0 | on | 3 |
| turing | T4, RTX 2080 | 1024 / 256 | f16 | off | 1 |
| cpu | no GPU device | 512 / 128 | f16 | off | 1 |

On this 5090, a 5.6k-token prefill improved from a median of 30.9 ms to 21.0 ms
(1.47x, n=45 per configuration, interleaved A/B/A/B so host-load drift cannot be
mistaken for the result). Best case barely moved (14.3 ms to 13.0 ms): wider
batches cut the number of prefill iterations, so the gain is in robustness under
contention rather than in peak throughput.

The ampere row carries a quantized KV cache for the same reason the ada row
does: a 3090 hits the same 24 GB wall, and llama.cpp compiles the `q8_0`/`q8_0`
flash-attention case for every architecture that has flash attention at all, so
there is no kernel gap between the two classes. `performance/quality.md`
measures that swap as quality-neutral, and a 3090 is bandwidth-bound at decode,
so halving KV traffic is a throughput win on top of the extra context it fits.
Unlike the blackwell row, this one is reasoned from the measured KV table and
llama.cpp's kernel matrix rather than benchmarked on Ampere silicon — this host
has none. `OMNISERVE_NATIVE_KV_TYPE=f16` restores the old behaviour without a
rebuild.

## Speculative decoding

Decoding one token at a time is bandwidth-bound, not compute-bound: the whole
weight matrix is read to produce a single token, and reading it for a batch of
five costs barely more. A decode step therefore has spare token-slots in it that
are already paid for. `src/ospec.c` spends them on guesses.

The guesses come from the context, not from a second model: the longest suffix
of what has been written so far is looked up in what came before, and whatever
followed it last time becomes the draft. That costs no VRAM, which matters on a
device where the LLM shares memory with an image model — a draft model would
need a lease from the broker above before it could hold anything. The verify
loop is the same one a draft model would drive, so adding one later changes the
draft source, not the design.

It is not a quality setting. Every token is sampled from the real model's
distribution at its own position and a drafted token is kept only when the
sampler independently chose the same id, so nothing is emitted because the
drafter proposed it. What a hit buys is the right to reuse logits that were
computed anyway. A miss costs a wider batch that produced one token; the
governor in `ospec.h` turns speculation off for a request that keeps missing and
probes occasionally in case that changes.

**Off by default** (`OMNISERVE_NATIVE_SPEC_DRAFT=0`), because the premise is
hardware-dependent and only half of it is measured here:

- **CPU: measured, and it does not pay.** 20.4 tok/s off, 19.5 tok/s on
  (`qwen3-0.6b-q8`). The calls were saved and the seconds were not, because CPU
  decode is compute-bound and the wider verify batch costs what it saves.
- **GPU: unmeasured on this host.** This is the case the technique exists for,
  but all 32 GB is committed to co-tenants (an image server, two search
  servers, and the gateway's own Gemma), so there was no room to load a model
  and measure. Shipping it on by default would be asserting a speedup nobody
  measured.

How much it can pay is measurable regardless, because acceptance is a property
of the model and prompt rather than the device. On `qwen3-0.6b-q8`, 15% of
tokens cost no model call on copy-heavy prompts and 7% on open-ended ones. So
the honest expectation on a GPU is single-digit to mid-teens percent, not a
multiple — drafting out of the context is free but weak. A small draft model
corrected by the big one is the version that gets a multiple, and it needs a
lease from the VRAM broker before it can hold anything, which is the order these
two landed in.

`./scripts/spec_bench.sh <model.gguf> [ngl] [draft]` settles it in one run: it
reports wall clock, the acceptance rate the model achieved, and how many model
calls speculation avoided, for both arms. Acceptance is a property of the model
and prompt; whether saved calls become saved seconds is a property of the
hardware. If it pays, the default is one value in `ollm_init`.

`performance/quality.md` records how the implementation was verified, and why
greedy output is *not* bit-reproducible when speculation fires. Knobs:
`SPEC_DRAFT` (tokens per round, capped by the micro-batch), `SPEC_MIN_NGRAM`,
`SPEC_MAX_NGRAM`, `SPEC_PROBE`, and `SPEC_PATIENCE` — how many consecutive
missed rounds to sit through before giving up. Patience is the one setting that
depends on the hardware rather than the model: quitting after one miss is right
where a miss is expensive and forfeits most of the win where it is not. `/status.speculation` and
`omniserve_llm_spec_*` report drafted, accepted and calls saved — separately,
because an acceptance rate alone cannot distinguish speculation that is off from
speculation that is landing perfectly.

## Cost- and priority-guided overflow capacity

The local GPU is sunk cost, so it is always tried first. Renting a remote
4090/5090 cog is only worth it when paid traffic is queueing behind a saturated
local device *and* the overflow it would absorb is worth more than the instance
costs. `src/oscale.c` is the decision engine and is pure — it takes an
observation and a clock and returns hold/up/down — so the expensive-if-wrong
logic is unit tested without a GPU, a network, or a provider account.
`src/ocapacity.c` is the controller that samples the admission scheduler and
drives the control plane.

Invariants, in the order they are enforced:

1. **Every lane defaults to zero instances and to disabled.** A lane has to be
   armed per modality, and an armed lane with no template or no per-request
   value is refused at startup with a log line rather than silently arming.
2. **Best-effort traffic never rents hardware.** The default tier mask is paid
   only. `TIER_BACKGROUND` is stripped in `oscale_add_lane`, so a misconfigured
   tier list cannot turn a batch backlog into a bill. Text generation has no
   lane at all: the local model serves it, and LLM overflow is best-effort by
   policy.
3. **Local capacity wins.** While the scheduler has a free permit, renting is
   refused with `local-has-room` no matter how deep the paid queue looks.
4. **Pressure must be sustained**, past both a queue-depth and a worst-wait
   threshold.
5. **The rent must pay for itself.** An instance can serve `3600 /
   seconds_per_req` requests an hour; the value counted is the *lesser* of that
   capacity and the observed eligible backlog, so a huge backlog is not a blank
   cheque and a thin one is refused as `not-worth-it`. Value must clear
   `price_usd_hr × margin` (default 1.5x).
6. **Caps and hysteresis**: `MAX_INSTANCES`, a lane `MAX_USD_HR` ceiling, and a
   cooldown between actions. Hard ceilings are reported ahead of the cooldown so
   the refusal names the real constraint.
7. **Scale-to-zero, with a hard TTL.** An instance is released after
   `IDLE_S` idle, and unconditionally at `TTL_S` even while busy and under
   pressure — an instance that outlives its lifetime is the failure mode that
   bills forever. Releasing also happens on shutdown and on a disabled lane.

Pod lifecycle deliberately stays in app.nz, which already owns provisioning,
per-second billing, the idle reap, and the orphan reconciler. This controller
only decides *whether* overflow is worth paying for and asks app.nz to warm it
over loopback; it never talks to a GPU provider directly, and it holds no
provider endpoint (the C data plane has no TLS by design). Scaling down is
therefore "stop routing there and let the idle reap release the pod", which
cannot leak a running instance if this process dies.

```bash
OMNISERVE_NATIVE_SCALE_CONTROL_BASE=http://127.0.0.1:8787 \
OMNISERVE_NATIVE_SCALE_API_KEY=... \
OMNISERVE_NATIVE_SCALE_TTS_ENABLED=1 \
OMNISERVE_NATIVE_SCALE_TTS_TEMPLATE=appnz-tts \
OMNISERVE_NATIVE_SCALE_TTS_HARDWARE=gpu-rtx4090 \
OMNISERVE_NATIVE_SCALE_TTS_REVENUE_USD_PER_REQ=0.02 \
OMNISERVE_NATIVE_SCALE_TTS_SECONDS_PER_REQ=3 \
OMNISERVE_NATIVE_SCALE_TTS_MAX_USD_HR=0.34 \
./build/omniserve-native --port 8791
```

Per-lane knobs, all prefixed `OMNISERVE_NATIVE_SCALE_<LANE>_` where `<LANE>` is
`TTS`, `STT`, `IMAGE`, or `MULTIMODAL`: `ENABLED`, `TEMPLATE`, `HARDWARE`,
`TIERS` (default `paid`), `PRICE_USD_HR` (defaults to the published list price
for the hardware), `REVENUE_USD_PER_REQ`, `SECONDS_PER_REQ`, `MARGIN`,
`QUEUE_DEPTH`, `QUEUE_MS`, `MAX_INSTANCES`, `MAX_USD_HR`, `COOLDOWN_S`,
`IDLE_S`, and `TTL_S`. Controller-wide: `SCALE_CONTROL_BASE`, `SCALE_API_KEY`,
`SCALE_POLL_S`, `SCALE_TIMEOUT_MS`.

`/status.capacity` reports, per lane, instances, ready count, price, current
spend rate, cumulative spend and instance-seconds, scale-up/down counts, TTL
kills, and the reason behind the last decision — so a bill can always be traced
back to the decision that caused it.

The generic `ocapacity_overflow_endpoint()` path remains warm-only, so those
lanes still ship disabled. H3 is the first end-to-end cost-aware lane: app.nz
now exposes an authenticated synchronous Cog prediction seam, and OmniServe
relays `/v1/video/generations` to it. app.nz first uses local GPU headroom when
the model fits, selects RunPod serverless for sparse traffic, and promotes to a
pod only when a five-minute request window and the model's learned runtime show
that continuous billing is cheaper. The break-even is
`1 / (1.20 × seconds_per_request)`, with at least three observations before a
pod is rented and a 50% demotion threshold to prevent flapping. Pods retain the
existing idle reap and orphan reconciliation.

```bash
OMNISERVE_NATIVE_H3_UPSTREAM=http://127.0.0.1:8787/api/cogs/$H3_COG_ID \
OMNISERVE_NATIVE_H3_API_KEY=... \
OMNISERVE_NATIVE_H3_PATH=/predict-sync \
OMNISERVE_NATIVE_H3_TIERS=paid \
OMNISERVE_NATIVE_H3_PERMITS=0 \
OMNISERVE_NATIVE_H3_TIMEOUT_MS=1800000 \
./build/omniserve-native --port 8791
```

`H3_API_KEY` is an app.nz API key belonging to the Cog owner. OmniServe removes
the caller's `Authorization`/`X-API-Key` before adding this credential. H3
defaults to the paid tier only; a free or background call gets HTTP 402 before
app.nz is contacted. `H3_PERMITS=0` is intentional because app.nz owns local
VRAM admission and remote capacity for this lane. `/status.upstreams.h3`,
`/status.proxy_pool.h3`, and the app.nz `X-AppNZ-Execution-Tier` response header
make each placement observable.

## Standing remote endpoints

Distinct from the rented lanes above: no provisioning, no per-hour bill, just a
URL that is already serving. Two things send a request there instead of queueing
it locally.

1. **A saturated local lane.** With an overflow configured, admission becomes
   non-blocking (`osched_try_acquire_n`). Waiting out an admission timeout buys
   nothing the remote would not have already delivered. Without an overflow the
   blocking acquire is unchanged. Try-acquire also refuses while anyone is
   queued even if a slot is free, so a caller with somewhere else to go cannot
   jump ahead of one that has already paid the latency.
2. **A local backend that failed before writing anything.** Once bytes are on
   the wire it is too late — retrying would splice a second response onto a
   partial one — so the retry is gated on `response_started`. The local permit
   is released first, so the retry cannot hold the device it just failed to use.

Paid tier only unless `OMNISERVE_NATIVE_OVERFLOW_TIERS` says otherwise, for the
same reason the rented lanes are: free traffic must never be able to spend
money. `omniserve_overflow_total{cause="saturated"|"local_failed"}` separates the
two causes, because they call for opposite responses — the first says buy local
capacity, the second says fix a backend.

```bash
OMNISERVE_NATIVE_IMAGE_OVERFLOW_UPSTREAM=https://... \
OMNISERVE_NATIVE_STT_OVERFLOW_UPSTREAM=https://... \
OMNISERVE_NATIVE_TTS_OVERFLOW_UPSTREAM=https://... \
OMNISERVE_NATIVE_OVERFLOW_TIERS=paid
```

## VRAM brokering between co-tenants

Four processes hold VRAM on this box and each sizes its workload from
`cudaMemGetInfo`. That read misleads in two directions at once: a caching
allocator's retained blocks look used, so a tenant under-sizes against memory
that is in fact reusable; and it is stale the moment two tenants read it
concurrently, so both size a batch against the same free bytes and the second
OOMs. The observed symptom was the image server pinned to a live microbatch of 1
for 40k consecutive jobs.

The fix is arbitration, not more accurate measurement. A tenant asks for
headroom and is told a number it may rely on until the lease expires; granted
but not yet materialised memory is subtracted from what the next caller sees,
which is what `cudaMemGetInfo` structurally cannot do. Every lease carries a
TTL, evaluated on every call rather than on a timer, so a tenant that dies
between lease and release cannot strand headroom and a broker whose reclaim
depends on its own liveness cannot leak exactly when it is least able to say so.

The broker never allocates, frees, or touches device memory — it hands out
permission, and enforcement stays with the tenant that owns the allocation. A
broker that could evict another process's memory would need privileges no
inference server should hold.

Higher tiers may dip further into the keep-free floor, so interactive paid
traffic is not starved by background batch work already holding leases. Leasing
costs every other tenant headroom, so `/v1/gpu/lease`, `/v1/gpu/renew`, and
`/v1/gpu/release` sit on the same trust boundary as the paid tier: loopback
callers only, never forwarded. Long-running resident models renew the same lease
ID, avoiding a release/reacquire race and bounding stale reservations after a
crash. A denial is a normal `200` with `"granted": false` — the caller's
fallback path is exactly what "no headroom" should trigger, and a 5xx would make
a healthy broker look broken to every monitor watching it.

```bash
OMNISERVE_NATIVE_VRAM_BROKER=1
OMNISERVE_NATIVE_VRAM_KEEP_FREE_MB=1024     # floor for unbrokered scratch allocations
OMNISERVE_NATIVE_VRAM_LEASE_TTL_S=120
OMNISERVE_NATIVE_VRAM_RESERVE=netwrck:2048  # growth a co-tenant will take but has not yet
```

`_VRAM_RESERVE` declares memory a tenant will grow into. Memory it already holds
must not be listed: the driver's free figure has counted it once already.

Giving memory back is only cheap if reloading is cheap, so the same decision has
a host-side half. `OMNISERVE_NATIVE_RAM_PREFETCH_ENABLED` (default on) warms the
weights the broker may later ask this process to drop, defaulting to the models
it already loads. Warming is advisory `posix_fadvise(WILLNEED)`, never
`MAP_POPULATE`: the kernel stays free to drop the pages under real pressure,
which is correct, since a prefetch that can evict a running process's working
set has turned an optimisation into an outage. Between 64 MiB chunks the loop
rechecks `MemAvailable` against `OMNISERVE_NATIVE_RAM_KEEP_FREE_PCT` (default 5)
and stops rather than pushing the box toward swap.

`omniserve_vram_headroom_mb` is the number worth graphing, not free memory:
headroom is what decides whether a co-tenant can batch. A denial rate climbing
while headroom stays high means leases are being held, not that VRAM ran out.

## Routes

`GET /health` `GET /readyz` `GET /status` `GET /v1/models` `GET /docs` `GET /openapi.json`

- Text: `POST /v1/chat/completions`, `/v1/completions`, `/v1/engines/{engine}/completions`, `/api/v1/generate`, `/api/v1/generate-large`, `/api/v1/autocomplete`, and `/api/v1/summarization`
- Embeddings: `POST /api/v1/feature-extraction` (legacy flat vector) and `/v1/embeddings` (OpenAI scalar or batched shape)
- Image: `POST /v1/images/generations`
- Foreground image generation: `POST /v1/images/foreground-generations/jobs` combines text-to-image and BiRefNet matting in one queued stage
- Background removal: `POST /v1/images/background-removals` or `/api/v1/birefnet`
- TTS: `POST /v1/audio/speech` and `/api/v1/generate_speech`
- STT: `POST /v1/audio/transcriptions`, `/api/v1/audio/transcribe`, `/api/v1/audio-file-extraction`, and `/api/v1/audio-extraction`
- Multimodal: `POST /api/v1/image-caption`, `/api/v1/video-question`, `/api/v1/multimodal-generate`, and `/api/v1/voice-chat`
- Device memory: `GET /v1/gpu/vram` (broker state), `GET /v1/host/memory` (page-cache warming state), and `POST /v1/gpu/lease` / `POST /v1/gpu/renew` / `POST /v1/gpu/release` (loopback callers only)
- Animation: `POST /v1/animations/generations` proxies a configured NVIDIA ACE/Animation Graph adapter and is forcibly admitted as `background`, regardless of the caller's requested tier.
- 3D: `POST /v1/3d/generations` proxies a configured TRELLIS.2/Pixal3D adapter and is forcibly admitted as `background`, regardless of the caller's requested tier.

Foreground generation keeps the generated PIL image in memory through the
stage-1 BiRefNet pass; it does not WebP-encode and decode the image between
generation and matting. The final artifact still defaults to WebP.

Configure the 3D worker with `OMNISERVE_NATIVE_3D_UPSTREAM`, optionally rewrite its path with `OMNISERVE_NATIVE_3D_PATH`, and label it with `OMNISERVE_NATIVE_3D_MODEL`. The route consumes every scheduler permit and therefore starts only when the OmniServe GPU is otherwise idle. With `OMNISERVE_NATIVE_3D_SWAP_EMBEDDED_MODELS=1` (the default), the gateway unloads its embedded LLM and embedding model only after that exclusive background admission, runs the one-shot 3D worker, reloads both models, and then reopens interactive admission. The supplied worker performs a second NVML free-VRAM check before loading the 4B checkpoint, defaults to TRELLIS.2 at 512³ on a 32 GB RTX 5090, and can publish GLB/WebP outputs into R2 plus a searchable JSONL manifest.

The 3D worker cuts its input out first, through this gateway's BiRefNet route.
TRELLIS.2 will segment an opaque image itself, with a general-purpose salient
object model; a decontaminated BiRefNet cutout is better on two counts. The matte
is sharper on hair and thin structures, and the RGB under the soft edge is the
subject's own colour rather than the subject blended with its backdrop. The
second matters more than it sounds: that edge band gets baked into the generated
texture and then lit, so a green-screened figure comes back with a green rim no
retexturing removes. An image that already carries alpha is passed through
untouched - segmenting an existing cutout only erodes it. The pass is best
effort, so a cutout service that is down or slow never fails a 3D job that would
have worked without it; `OMNISERVE_3D_CUTOUT=0` disables it, and
`OMNISERVE_3D_CUTOUT_BASE`, `_PATH`, `_SECRET` and `_TIMEOUT` configure it.

Set `OMNISERVE_3D_PUBLIC_BASE` to the externally reachable gateway prefix used
for non-R2 assets (the packaged service uses
`http://127.0.0.1:8791/v1/3d` for same-host development). Do not derive public
asset URLs from the HTTP `Host` header.

The packaged 3D and BiRefNet units optionally load `/etc/omniserve-r2.env`.
Set the S3-compatible credentials/bucket variables used by `workers/object_store.py`
for image artifacts, and set `OMNISERVE_3D_R2_REMOTE` plus
`OMNISERVE_3D_R2_PUBLIC_BASE` for GLB publication. Successful 3D jobs append an
atomic JSONL entry to `OMNISERVE_3D_MANIFEST`; no entry is created for a failed
or dependency-blocked generation.

On a shared GPU, set `OMNISERVE_3D_GPU_COORDINATORS` to a comma-separated list
of peer service bases that implement `POST /admin/hold?seconds=N` and
`POST /admin/release`. The 3D adapter acquires every bounded hold before its
final free-VRAM check and releases them in reverse order on success or failure.
The packaged service coordinates with the local Z-Image worker on port 8100,
so an image cold-load cannot race TRELLIS.2 after background admission.
Coordinator calls wait up to 180 seconds by default, configurable with
`OMNISERVE_3D_GPU_COORDINATOR_TIMEOUT_S`, so an in-flight generation or model
load can finish before the hold is acquired.

For an RTX 5090, clone the official repository recursively and run the isolated
Blackwell installer. It uses current CUDA 12.8 wheels with the local 12.9
toolkit, builds custom operators for compute capability 12.0, and selects
xFormers because the repository's older pinned FlashAttention package predates
Blackwell:

```bash
git clone --branch main --recursive https://github.com/microsoft/TRELLIS.2.git /nvme0n1-disk/code/TRELLIS.2
CUDA_HOME=/usr/local/cuda-12.9 ./workers/install_trellis2_blackwell.sh
```

TRELLIS.2 also requires Meta's gated
[`facebook/dinov3-vitl16-pretrain-lvd1689m`](https://huggingface.co/facebook/dinov3-vitl16-pretrain-lvd1689m)
image encoder. Accept its terms with the deployment account, then prefetch both
repositories into the worker's shared cache:

```bash
HF_HOME=/nvme0n1-disk/models/huggingface hf download microsoft/TRELLIS.2-4B
HF_HOME=/nvme0n1-disk/models/huggingface hf download facebook/dinov3-vitl16-pretrain-lvd1689m
```

The worker reports `model_dependency_missing` without taking a GPU hold when
that gated encoder is absent. Configure `HF_HOME` on the large NVMe volume if
the default Hugging Face cache is small. The packaged worker also sets
`HF_HUB_DISABLE_XET=1`; on this host the standard resumable HTTP downloader is
materially faster and more stable than Xet.

Configure the animation adapter with `OMNISERVE_NATIVE_ANIMATION_UPSTREAM`, optionally rewrite its worker path with `OMNISERVE_NATIVE_ANIMATION_PATH`, and label it with `OMNISERVE_NATIVE_ANIMATION_MODEL`. The request carries the target rig family and bones plus `publish.r2`, `publish.searchable`, and `publish.collection`; the gateway also forces `X-Animation-Publish: r2-searchable`. The worker owns ACE command/animation-data conversion, BVH/GLB output, R2 upload, and atomic searchable-manifest updates. NVIDIA ACE currently provides Animation Graph, animation-data, gesture-command, and Audio2Face components rather than a general public text-to-full-body-motion model, so text planning belongs in that adapter instead of being represented as a native ACE model.

The public text-generator.io OpenAPI surface—feature extraction, summarization, speech generation, file/URL transcription, text generation, large generation, image captioning, and legacy engine completion—is explicitly routed. Native worker endpoints for image creation, inpainting, style transfer, captioning, TTS, and STT are also admitted through the same scheduler.

### Exact Music3 repeats

The RunPod Music3 adapter keeps a bounded cache of completed, quality-checked
WAV masters on its network volume. An identical normalized prompt, lyrics,
duration and seed returns the same WAV bytes without loading or invoking the
model; upload URLs are deliberately excluded, so the same master can be sent to
a new destination. The cache key includes the model, precision/serve settings,
continuity policy and a release namespace. Quality retries retain the selected
seed and attempt count in cache metadata, and responses expose
`exact_result_cache_hit`.

The deployment enables 16 entries / 4 GiB by default. Set
`MUSIC3_RESULT_CACHE=0` to disable it, adjust `MUSIC3_RESULT_CACHE_ENTRIES` and
`MUSIC3_RESULT_CACHE_MIB` to bound it, or change
`MUSIC3_RESULT_CACHE_NAMESPACE` whenever generation behavior changes outside
the versioned deployment. Fresh requests still use the full quality-gated
generation path.

Proxied responses are relayed byte-for-byte after incremental framing validation, so `stream:true` stays streaming and WAV/PNG/multipart payloads are not JSON-reencoded. Content-Length and chunked responses retain downstream keep-alive; close-delimited responses close safely. Request bodies grow lazily up to the public edge's 80 MiB limit. Embedded image generation returns PNG bytes.

## BiRefNet cutout worker

BiRefNet runs as a persistent isolated CUDA worker while the gateway keeps the
request hot path in C: weighted admission, connection reuse, incremental HTTP
framing validation, and transparent PNG relay happen without Python JSON or
image re-encoding in the gateway.

```bash
python -m venv .venv
.venv/bin/pip install -r workers/requirements-birefnet.txt
HF_HOME=/nvme0n1-disk/models/huggingface \
BIREFNET_MODEL=ZhengPeng7/BiRefNet \
.venv/bin/python workers/birefnet_worker.py --port 9094

OMNISERVE_NATIVE_BIREFNET_UPSTREAM=http://127.0.0.1:9094 \
OMNISERVE_NATIVE_BIREFNET_PERMITS=2 \
./build/omniserve-native --port 8791
```

The worker uses FP16, channels-last CUDA tensors, TF32 where applicable,
inference mode, and a persistent loaded model. `BIREFNET_TORCH_COMPILE=1` uses
the correctness-safe `default` Inductor mode, warms the production input graph
before readiness, compares it with eager output, and falls back to eager if
compilation or validation fails. Keep `BIREFNET_CUDNN_BENCHMARK=0`: plan search
roughly doubled the measured 1024px startup peak without improving steady
throughput. The persistent model holds a renewable broker lease and falls back
to CPU at startup when the shared card cannot safely grant it.

Keep the gateway at two BiRefNet permits on the shared RTX 5090 profile. The
worker serializes the model invocation, but preprocessing, CUDA matte work, and
WebP encoding can overlap around it. A 12-request production canary measured
2.98 images/s at one permit, 4.95 at two, and 7.29 at four; four also increased
mean latency from 335 ms to 509 ms. Two is the measured throughput/latency knee
and matches the worker's default two job threads without increasing model
residency.

`BIREFNET_WEBP_QUALITY=85` and `BIREFNET_WEBP_METHOD=4` are the balanced output
defaults. Fully transparent pixels always have their RGB set to black before
encoding, and libwebp is allowed to discard invisible RGB data; alpha remains
unchanged. Existing cached/old images remain valid; this applies to new
encodes.
On the tested human cutout method 4 encoded about 26x faster than method 6 with
identical alpha and about 4% more bytes. The video path similarly streams
bounded RGBA chunks through a small encoder queue while inference continues,
and blackens RGB wherever the RVM alpha is zero. The default 1024-pixel
inference size can be tuned with `BIREFNET_INPUT_SIZE`, though reducing it
changes fine-edge masks rather than being a free memory optimisation.

### Colour decontamination stays on the device

The alpha BiRefNet produces is only half a cutout. Under a semi-transparent edge
the RGB is still the *composite* - the subject blended with whatever it was
photographed against - so a naive cutout carries a green or blue rim onto its new
background. `src/omatte.c` and `cuda/omatte_cuda.cu` solve for the true
foreground colour (and the backdrop) per pixel; see `matte/README.md` for the
algorithm and its accuracy against pymatting.

The pass runs where the matte already is. `omatte_estimate_fb_cuda_device` takes
device pointers and torch's own stream, so nothing round-trips: the numpy entry
point moved ~28 MB per 1024x1024 cutout (alpha down, image and alpha up, F and B
down) around ~1 ms of solving. Measured on an RTX 5090 already saturated by
another tenant:

| path | 1024x1024x3 |
| --- | --- |
| pymatting (numba, CPU) | 886 ms |
| numpy entry point, before | 15.7 ms |
| numpy entry point, now | 9.2 ms |
| torch device pointers | 3.1 ms |

The device path agrees with the host path to 6.9e-07 max abs - the only
difference is that the seed colour is reduced on the GPU in double rather than in
host float32 row-major order, which is the more accurate of the two and seeds a
1x1 pyramid level. `matte/run_eval.sh` still reports the CUDA backend as
byte-identical to the CPU red-black order.

Device buffers are allocated once and reused (84 MiB for 1024x1024, grow-only;
`omatte_cuda_release_workspace()` gives it back). That is not micro-optimisation:
`cudaFree` synchronises the whole context, so a cutout releasing its pyramid used
to block until the segmentation model sharing that context had drained.

### Backdrops: return one, or replace it

The estimator solves for `B` jointly with `F` in the same 2x2 system at every
pixel of every level, so the backdrop is already computed - `return_background`
just stops throwing it away. It is a real solve, not `(I - aF)/(1-a)`, so it
stays in range where alpha approaches 1 and is usable as a style-transfer input.

```bash
# cutout + the backdrop that was removed + the subject on white
curl localhost:8791/v1/images/background-removals -d '{
  "image_url": "https://example.com/chair.webp",
  "return_background": true, "background": "#ffffff"}'
```

`background` takes `#rrggbb`, a colour name, `estimated`, or an image URL, and
compositing runs through the same C kernel. More than one image in the answer
means JSON instead of raw bytes.

`background_prompt` generates a replacement with the diffusion lane instead, and
is accepted **only** on `/jobs`: diffusion takes seconds to minutes and the
synchronous handler holds its gateway permit for the whole request, so allowing
it there would park a backdrop on an interactive slot. The worker calls back
through the gateway's `/v1/images/backgrounds`, which is pinned to the background
tier, rather than reaching the diffusion backend directly and bypassing admission
altogether. With `background_strength` above zero it goes through
`/v1/images/backgrounds/style` instead, style-transferring the *estimated*
backdrop so the replacement inherits the original's lighting.

Backdrop requests set `teleport: true`. Latent teleportation replays a cached
latent and resumes the sampler near the end instead of running every step, and
backdrops are the workload it fits best: nobody is waiting on them, prompts
repeat hard across a batch. The native exact-key path preserves the original
Euler sigma schedule and global step numbering; its baseline, split-prime, and
replay images must be pixel-identical in `image_teleport_bench.py`. Approximate
prompt matching is intentionally not implemented. Measured through the gateway,
a repeat backdrop prompt came back `exact_prompt_latent_replay`, resuming at
step 7 of 9. The native cache also retains that request's encoded single image;
the next identical request reports `exact_prompt_result_cache` and skips the
remaining denoise step, VAE decode, and image encode. The result cache uses the
same exact prompt, negative prompt, dimensions, seed, step count, guidance,
LoRA, and resume-step key as latent replay and is evicted with its latent.

### Generate a cutout in one stage

`/v1/images/foreground-generations/jobs` accepts a subject prompt and owns both
text-to-image generation and BiRefNet matting. It is queued because the worker
calls the gateway's background diffusion lane first; holding a synchronous
gateway permit across that callback would deadlock an all-slot image request.

```bash
curl localhost:8791/v1/images/foreground-generations/jobs -d '{
  "prompt": "full-body deckhand on a plain grey backdrop, no shadow",
  "width": 768, "height": 1024, "seed": 42, "output_format": "webp"
}'
curl localhost:8791/v1/images/foreground-generations/jobs/<job_id>
```

`tools/generate_vn_art.py` consumes an art-plan JSON file, uses this composite
route for sprites and `/v1/images/backgrounds` for scenes, validates alpha and
sprite placement, and writes the final PNG assets into a Ren'Py project.

In the text-generator.io deployment, nginx sends the public API to this C gateway. A managed CPU-only compatibility worker on port 9083 supplies TTS, STT, multimodal endpoints, and dynamic provider routing. Local-model callbacks carry `X-Omniserve-Internal: local` so provider routing cannot loop. OpenAI embeddings accept either one string or a batch of up to 256 strings.

Upstreams must use plain `http://` on loopback or a private service mesh; terminate TLS at the public edge. Route every GPU-heavy public path through this gateway so its admission decision covers the full response lifetime.

Auth mirrors text-generator.io: `secret`, `X-API-Key`, `X-Rapid-API-Key`, `Authorization: Bearer`, or `?secret=`; priority via `X-Omniserve-Tier: paid|sub|free|background`.

With `SLOTS=N`, text/audio calls consume their configured modality permits, diffusion defaults to all N permits, and background calls only start on an otherwise idle GPU. This prevents a diffusion launch from racing Gemma for the last VRAM while retaining controlled text/audio concurrency. `/status` exposes permits, used capacity, queue maxima, timeouts, per-tier counters, and upstream pool reuse/failures.

Admission hands slots to the front of the queue rather than announcing that one
is free. A releasing thread walks the ordered waiter list, grants permits to
whoever fits, and signals only those threads. The previous shape - one shared
condition variable, broadcast on every release, each woken thread scanning the
whole list to learn whether it was the head - was O(n^2) of contended work per
release with n-1 threads going straight back to sleep, and it got most expensive
exactly when the queue was longest. Head-of-line order is unchanged: the scan
stops at the first waiter that does not fit, so a cheap request still cannot
overtake an expensive one that is already queued.

Measured with 4 slots, mixed tiers, 500 acquire/release rounds per thread:

| queued threads | before | after |
| --- | --- | --- |
| 32 | 628 ms | 121 ms |
| 128 | 30.0 s | 0.73 s |

Neither version ever exceeded the slot cap. At 256 threads the old version did
not finish inside half an hour; the new one takes 21 s.

The production Gemma and CuteDSL image workers add a second residency handshake: the vLLM manager holds and unloads image admission throughout boot/wake, while an image cold-load sleeps vLLM and refuses to interrupt active text inference. The weighted gateway plus that two-sided handoff covers both request concurrency and model residency; either mechanism alone is insufficient on a 32 GB card.

text-generator.io playground-compatible request:

```bash
curl localhost:8791/api/v1/generate \
  -H 'Content-Type: application/json' \
  -d '{"text":"Hi I am bored so looking","number_of_results":1,"max_length":100,"max_sentences":1,"min_probability":0.7,"model":"best","enable_thinking":false}'
```

The embedded path honors `enable_thinking`, `min_probability`, `min_p`, `max_sentences`, and string/array stop sequences. Qwen no-thinking requests prefill the closed reasoning block, so reasoning tags do not leak into `generated_text` or consume output tokens. Repeated chat/system prefixes reuse the matching KV prefix; `usage.cached_prompt_tokens` reports the saving.

## Model conversion

The checked-in conversion workflow creates serving artifacts without mutating or duplicating the production Hugging Face checkpoints:

```bash
./scripts/convert_models.sh modernbert
./scripts/convert_models.sh gemma
./scripts/convert_models.sh qwen
# or: ./scripts/convert_models.sh all
```

Outputs default to `/nvme0n1-disk/models/omniserve-native`. Override source/output paths with `ONATIVE_MODERNBERT_SOURCE`, `ONATIVE_GEMMA_SOURCE`, `ONATIVE_QWEN_SOURCE`, and `ONATIVE_MODEL_DIR`.

To prepare the Gemma 4 31B MeroMero checkpoint from Hugging Face, use the
capacity-checked workflow below. It resumes interrupted downloads and refuses
to start when the source, temporary BF16 GGUF, quantized GGUF, and safety
margin cannot coexist on the target filesystem:

```bash
./scripts/prepare_gemma4_model.sh
# Optional: ONATIVE_GEMMA4_QUANT=IQ4_NL ./scripts/prepare_gemma4_model.sh
```

`IQ4_XS` is the default native llama.cpp target because it is the smallest of
the listed 4-bit choices. `NVFP4` is a ModelOpt/TensorRT-LLM checkpoint format,
not a normal `llama-quantize` target in this checkout; the script rejects it
instead of producing a mislabeled file. Set `OMNISERVE_NATIVE_NGL=auto` for
conservative full-offload detection: the gateway compares the GGUF size with
current free VRAM plus `OMNISERVE_NATIVE_NGL_AUTO_KEEP_FREE_MB` (2 GiB by
default). Explicit numeric NGL values remain the way to request partial
offload when the shared card cannot fit all weights.

For a shared 32 GiB RTX 5090 with roughly 13--14 GiB free, the tested
`G4-MEROMERO-V2-31B-IQ4_XS.gguf` profile is `NGL=20`: llama.cpp offloads 20 of
61 layers, uses about 5.8 GiB of VRAM, and leaves enough headroom for the
other tenants. The isolated chat canary was coherent and cut decode time by
about half versus CPU-only placement. Keep `NGL=auto` for the conservative
full-fit decision, or pass the measured numeric value when selecting this
partial profile.

The checked-in `systemd/omniserve-native-gemma4-iq4.conf` is an optional
machine-specific drop-in for that profile. It sets one context, `q8_0` KV,
adaptive batch geometry, and the tested partial offload without changing the
defaults used by older Gemma, CPU-only, or Z-Image hosts. Apply it only on the
target machine:

```bash
sudo install -D -m 0644 systemd/omniserve-native-gemma4-iq4.conf \
  /etc/systemd/system/omniserve-native.service.d/gemma4-iq4.conf
sudo systemctl daemon-reload
sudo systemctl restart omniserve-native
```

To return to the normal shared-image profile, remove that one drop-in, run
`sudo systemctl daemon-reload`, and restart the service. The LLM scheduler can
also evict and reload this model through the existing loopback admin hooks, so
other model configurations do not need to inherit the 31B settings.

The embedded LLM also supports lifecycle-safe, loopback-only replacement when
`OMNISERVE_NATIVE_LLM_SWAP_DIR` is configured. The swap operation waits for
active chats to finish, unloads the old model, loads the new allow-listed GGUF,
and restores the previous model if loading fails:

```bash
curl -X POST http://127.0.0.1:8791/admin/llm/swap \
  -H 'content-type: application/json' \
  -d '{"path":"/nvme0n1-disk/models/omniserve-native/G4-MEROMERO-V2-31B-IQ4_XS.gguf","ngl":"20","ctx":4096,"contexts":1}'
```

`/admin/llm/unload` and `/admin/llm/load` are available for the scheduler's
evict/reload cycle. The Python scheduler's proxy catalog can call those hooks
with `OMNISERVE_PROXY_PROXY_LLM_EVICT_URL` and
`OMNISERVE_PROXY_PROXY_LLM_LOAD_URL`.

The experimental Blackwell/NVFP4 route is capacity-checked separately:

```bash
git clone --depth 1 --filter=blob:none --sparse \
  https://github.com/NVIDIA/Model-Optimizer.git ../Model-Optimizer
git -C ../Model-Optimizer sparse-checkout set examples/hf_ptq modelopt_recipes
./scripts/prepare_gemma4_nvfp4.sh
```

That wrapper refuses an incomplete download and uses ModelOpt's low-memory,
sequential device-map PTQ settings. The current released TensorRT-LLM wheel
still needs a Gemma4-capable model registry; do not point production at the
export until its `trtllm-serve` load-and-generate probe passes.

| Capability | Artifact | Runtime |
|---|---|---|
| Gemma 4 roleplay text | `gemma-roleplay-v2-q8_0.gguf` (8,005,436,224 bytes) | embedded libllama |
| Gemma 4 MeroMero 31B text | `G4-MEROMERO-V2-31B-IQ4_XS.gguf` (16,862,233,024 bytes) | embedded libllama, partial NGL |
| Qwen 3.5 text | `qwen3.5-4b-text-q8_0.gguf` (4,482,403,104 bytes) | embedded libllama or llama.cpp worker |
| Qwen 3.5 vision | `mmproj-qwen3.5-4b-f16.gguf` (672,423,488 bytes) | llama.cpp `mtmd` worker behind `OMNISERVE_NATIVE_MULTIMODAL_UPSTREAM` |
| ModernBERT features (legacy contract) | `modernbert-base-q8_0.gguf` (160,208,000 bytes) | embedded libllama encoder, mean pooling |
| Retrieval-grade features | `gte-modernbert-base-q8_0.gguf` (160,208,576 bytes) | embedded libllama encoder, `EMBEDDING_POOLING=cls` |
| Diffusion | existing `.safetensors`/`.gguf` checkpoint | embedded stable-diffusion.cpp or image worker |
| STT/TTS | existing Parakeet/Whisper and Supertonic/Kokoro assets | audio worker behind `OMNISERVE_NATIVE_STT_UPSTREAM`/`OMNISERVE_NATIVE_TTS_UPSTREAM` |

ModernBERT conversion removes only the unused masked-language-model head. Audio models that are already ONNX/native-worker consumable and diffusion checkpoints already accepted by stable-diffusion.cpp do not benefit from being repackaged as text GGUFs. The gateway still owns authentication, weighted admission, timeouts, streaming, and connection pooling for those workers, so every public route passes through the same C scheduler.

## Measured local results

On this 72-thread RTX 5090 host, the native C load generator measured the loopback proxy at concurrency 32 improving from about 29k requests/s before pooling to 46–75k requests/s after resolved-once pooled relaying (host load causes run-to-run variance). Disabling the final pool in matched runs measured 15–18k requests/s. Persistent `/health` remained around 97–124k requests/s.

For a repeated 60-token prompt on the bundled Qwen3 0.6B Q8 GGUF, embedded inference fell from 82.9 ms uncached to 19.6 ms with 59 cached prompt tokens. The converted Gemma 4 artifact was verified CPU-only through the C gateway with the playground-shaped `min_probability=0.7` request: HTTP 200 and a non-empty one-result array. A same-prefix one-token request fell from 4.2 s cold-prefill to 0.39 s with prefix reuse on CPU.

The converted ModernBERT Q8 embedding matched the existing FP32 Python result at 0.99875 cosine similarity. Warm 8-thread feature extraction measured about 17 ms for a short input. The legacy route honors `num_features` (default 256); `/v1/embeddings` returns the full 768-dimensional mean-pooled vector unless `dimensions` is supplied.

A proxied request now costs two socket writes instead of four. The upstream
request head and body go out in one `sendmsg` rather than a `sendto` each, and
the response headers plus whatever body arrived in the same read — which for a
small proxied response is all of it — are relayed downstream in one write rather
than a header the client waits behind. Verified by tracing a single proxied
`POST`: `sendto(137) + sendto(43)` and `write(285) + write(61)` became
`sendmsg(180)` and `write(346)`, the same bytes in half the syscalls and half
the segments.

Throughput was not re-measured for that change: this host sits at load 90 of 72
cores serving production traffic, and back-to-back runs of the *unchanged*
binary varied between 22k and 15k requests/s — wider than the effect being
measured. The syscall counts are deterministic, and are what is claimed here.

Reproduce transport measurements with `./scripts/bench.sh 8791 100000 32`; it uses `build/onative_bench`, a persistent-socket C client.

## Quality bench

Transport speed is only half of a serving regression. `./scripts/quality_bench.sh 8791`
grades the live gateway on model behaviour: cold-prefill determinism,
prefix-cache agreement and speedup, `max_tokens`/stop-sequence/`max_sentences`
adherence, reasoning-tag leakage, a small graded task set, embedding
determinism, batch-vs-scalar equality, legacy `num_features` truncation,
paraphrase-vs-distractor separation, reference-vector drift, and image/audio
container validity when those workers are configured. Scores are compared
against `performance/quality-baseline.json` and the run exits non-zero on
regression, so CI can gate on it:

```bash
./scripts/quality_bench.sh 8791 --update-baseline   # record the current models
./scripts/quality_bench.sh 8791                     # gate a change
./scripts/quality_bench.sh 8791 --suite embedding   # one suite
./scripts/quality_bench.sh 8791 --suite llm --force-local  # bypass a compatibility proxy
```

Image cutovers have a deeper two-endpoint gate. It saves reference/candidate
PNGs and a contact sheet, checks the OpenAI envelope and dimensions, then gates
CPU CLIP image/prompt similarity, aesthetic-score drift, entropy, and latency:

```bash
python tools/image_parity_bench.py \
  --reference-base http://127.0.0.1:8791 \
  --candidate-base http://127.0.0.1:8792 \
  --reference-steps 9 --candidate-steps 4 \
  --monthly-gpu-cost 1000
```

When both checkpoints cannot coexist, capture the immutable reference first,
then compare after the handoff:

```bash
python tools/image_parity_bench.py --capture-reference \
  --reference-base http://127.0.0.1:8791
python tools/image_parity_bench.py --reference-run evals/image_reference_<timestamp> \
  --candidate-base http://127.0.0.1:8792
python tools/image_teleport_bench.py --base http://127.0.0.1:8792 --limit 3
```

The teleport gate is stricter than perceptual parity: unsplit baseline,
split-prime, and replay must have identical decoded RGB bytes, and replay must
report an exact cache hit. The prime must be a miss; use `--seed-offset` when
rerunning against a persistent cache. Unless explicitly overridden, exact
replay resumes at the last scheduled step (`steps-1`), the fastest exact point
in the RTX 5090 resume-step sweep. Latency is reported but exactness is the hard
gate.

The parity report prefers server-side `inference_time_ms` for accelerator cost
and latency ratios when both endpoints provide it, while retaining wall time to
show queue and transport overhead. The measured RTX 5090 step-count/cost
ablation is in `performance/zimage-cost-ablation-2026-08-30.md`.

Measured findings, including the embedding-artifact comparison and the KV
quantization result, are in `performance/quality.md`. Two of them matter for
callers: the shipped `modernbert-base` mean-pooled vectors rank paraphrases at
chance (swap to `gte-modernbert-base-q8_0` with `EMBEDDING_POOLING=cls`), and
temperature-0 output is not byte-reproducible across prefix-cache states.

## Fleet integration

For the pure-C data plane, point the public edge directly at this server. During migration, omniserve (Python) can remain an optional catalog/eviction control plane while all inference traffic is pointed here:

```bash
OMNISERVE_PROXY_PROXY_LLM=http://127.0.0.1:8791
OMNISERVE_PROXY_PROXY_IMAGE=http://127.0.0.1:8791
```

## Roadmap to theoretical-fastest on RTX 5090

1. Continuous token-level batching across the prefix-aware context pool (parallel contexts remove the old global generation mutex; cross-request decode iteration batching is the next step).
2. FP8/NVFP4 weights (Blackwell) — llama.cpp Q4_K/Q8 today; TensorRT-LLM backend as a second `obackend` impl for the big-model path.
3. Diffusion: sd.cpp `--diffusion-flash-attn`, TAESD preview, step-distilled checkpoints (turbo 4-step); port the fused kernels from `../cutedsl/cutezimage/csrc` (rms/silu-gate/qk-norm, already 5090-tuned) into a custom ggml op.
4. CUDA graph capture for the small-model path — cutedsl measured 21x on Chronos-2 from graph capture; same lever applies to short-seq LLM decode.
