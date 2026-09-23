# Frontier scheduler: local 5090 vs RunPod (2026-09-23)

Cost/latency/quality routing for GPU workloads that can run either on the shared
RTX 5090 or on RunPod serverless, plus one GPU-seconds/$ ledger for all of them.
Only quality-equivalent backends compete (same weights, same sampler settings or a
stricter cache threshold); nothing routes to a lossier variant.

## Pieces

| piece | where | what |
| --- | --- | --- |
| ledger | `frontier/ledger.py` | SQLite (WAL) `/nvme0n1-disk/data/omniserve-frontier/ledger.db`; per job: workload, backend, endpoint, gpu, tier, queue_ms, exec_ms, wall_ms, cold, est_usd, quality_tier, status, cache_hit. Writes go through a background thread (`record()` never blocks). |
| cost model | `estimate_usd` | RunPod: endpoint $/h × (exec + delay when delay > 3 s, i.e. cold start). Local: `FRONTIER_LOCAL_USD_PER_H` (0.10) × exec, plus `FRONTIER_OPPORTUNITY_USD_PER_H` (0.70, cheapest 24 GB pool) × exec when the job queued behind another. Cache hits cost 0 and are reported as `…:cache_hit`, never mixed into render latency. |
| reconcile | `Ledger.reconcile` | Daily RunPod billing (`rest.runpod.io/v1/billing/endpoints`) per endpoint-day; factor = billed / estimated, median of recent days scales remote $/job in the frontier. |
| frontier | `frontier/policy.py`, `frontier/seeds.json` | Candidates per workload (seeded from perf docs, replaced by ledger p50/p95/$ after 5 jobs in 7 days); quality filter (`quality >= reference - 0.01`); Pareto over (p50, $/job); per-tier policy and remote order. |
| routing file | `/nvme0n1-disk/data/omniserve-frontier/routing.json` | Written atomically by `python3 -m frontier cycle` (systemd `omniserve-frontier.timer`, every 10 min; reconciles once per 20 h). Hot-reloaded by the C gateway (mtime check ≤ 1/s) and by Python routers. |
| gateway | `omniserve-native-ra2overflow/src/ofrontier.c`, `handle_images` | On a busy image lane: expected wait = (queued ahead + running) × local p50; policy decides queue-locally vs overflow. Appends one JSONL line per job (local/queued/overflow/cache) to `gateway-ra2.jsonl`, with the listening port so experiment gateways cloned from the unit are ignored at ingest (`FRONTIER_GATEWAY_PORTS=8792`). Forwards `X-Omniserve-Tier` to the overflow. |
| RA2 overflow adapter | `omniserve-native-ra2overflow/tools/runpod_overflow.py` | Endpoint chosen from the tier's `remote_order` among `RUNPOD_RA2_ENDPOINTS`; concurrent jobs (threaded server, no shared lock); dead-primary breaker; strips inputs the RunPod worker rejects; ledger rows with RunPod delay/execution split; `GET /metrics`, `GET /ledger/summary`. |
| YuE | `workers/yue_worker.py` | Ledger rows (local, runpod, cache/in-flight share). With `YUE_FRONTIER_ROUTING=1`, a busy local lock is waited on when the policy says local finishes first; otherwise RunPod as before. |
| Pixal3D | `workers/remote_3d.py`, hook in `trellis2_worker.py` | Pixal3D requests go to RunPod `akgefm0nzzr4jo` when the local runtime is missing or free VRAM < `OMNISERVE_3D_MIN_FREE_MIB`; no local GPU lock held, at most `OMNISERVE_3D_REMOTE_MAX` (2) concurrent. |

## Policies

Busy local lane only (an idle lane always runs locally). `local_eta = wait + local_p50`,
`remote_eta = remote p50` (measured wall incl. cold starts).

| policy | rule | default tiers |
| --- | --- | --- |
| `local_only` | queue locally | free (since 2026-09-23 17:30 NZST, once netwrck/cutedsl tag paid traffic) |
| `background` | never preempts; waits for an idle lane; overflows only if `allow_overflow` (`FRONTIER_BACKGROUND_OVERFLOW=1`, default off) or a stated deadline would be missed locally but met remotely | background |
| `cheapest_within_deadline` | local if `local_eta <= deadline`, else remote if `remote_eta <= deadline`, else the faster | paid |
| `fastest` | remote if `remote_eta < local_eta` | sub, priority |
| `overflow_on_busy` | legacy: always overflow | anything unlisted, and the whole gateway when no routing file is loaded |

Deadline: workload default (ra2 45 s, yue 180 s) or `X-Omniserve-Deadline-Ms` from an
internal caller. If the local queue then times out (150 s admission), non-local-only
tiers still overflow.

### Tiers and admission

Admission order is paid > sub > free > background (`osched` keeps waiters sorted by
rank, FIFO within a rank). Background also needs the device otherwise idle, so it
never shares the lane with a running job. Starvation bound
(`OMNISERVE_NATIVE_BACKGROUND_MAX_WAIT_S`, prod 600 s): a background waiter older
than the bound is ranked as free, older than twice the bound as paid (behind those
already queued). Background has its own queue timeout
(`OMNISERVE_NATIVE_BACKGROUND_ADMISSION_TIMEOUT_S`, prod 1800 s; others 150 s); on
timeout it gets 503 unless overflow is allowed. Public callers cannot claim
background (only loopback, unproxied requests may).

Caller tags (deployed): netwrck `ra2Post` (generate, edit, FAL qwen proxy) and cutedsl
`proxyToRA2` send `X-Omniserve-Tier: paid`; netwrck radio YuE renders (untracked WIP, not yet in a prod build),
`cmd/generate_talking_avatars` (was the unknown tier `portrait`, i.e. free),
`cmd/generate_character_portraits` and `tools/generate_vn_art.py` send `background`.
The manifoldgen farm uses the 8100 Z-Image worker's own low-priority mode, not this
gateway. netwrck's user-facing YuE route calls RunPod directly (no gateway).

Revert free to the old spill behaviour: set
`FRONTIER_TIER_OVERRIDES={"free":{"policy":"cheapest_within_deadline"}}` in
`omniserve-frontier.service` and run it once (the gateway hot-reloads).

Live canary 2026-09-23 (`:8792`, 1024², uncached): plan paid, background (+1.5 s),
background (+3 s), paid (+4.5 s). The late paid request was admitted before both
waiting background jobs: paid 13.7 s / 19.0 s wall (queue 9.2 s), background 32.7 s /
41.0 s (queue 22.0 / 31.2 s), all local, no RunPod spend. End to end: a netwrck `POST /api/ra2-art-generator`
(prod `netwrckprod154`) landed as a `tier=paid` ledger row on `:8792` (28.7 s, local).
cutedsl's tag is covered by `server/ra2_test.go` and the deployed binary; no organic
cutedsl ra2 request arrived during the window.

## Measured frontier (2026-09-23)

Uncached renders only. Local is the shared 5090 under production load.

| workload | backend | p50 | p95 / cold | $/job | quality | on frontier |
| --- | --- | ---: | ---: | ---: | --- | --- |
| ra2 1024², 20 steps | local 5090 | 12.0 s (seed; canary 9.6-17.6 s, prod rows 10-19 s) | 16-23 s | $0.0003 (+$0.002 opportunity when queued) | reference (EasyCache 0.08) | yes |
| ra2 | RunPod `tlofa06vj7iab7` 4090/A5000/L4/3090 | 84 s (3 parallel canary: 60.5/84.3/145.5 s) | 145 s; single warm 34.5 s | $0.012-0.030 est, billing factor 2.4 today | equal (same GGUF, EasyCache 0.05) | no (dominated when idle; wins only when local wait > ~72 s) |
| ra2 | daisy 3090 via tunnel | 16 s (doc) | | ~$0.0005 | equal | unavailable (Cloudflare 530) |
| yue clip | local 5090 | not measured (needs 18 GiB free; 5.6 GiB free today) | | ~$0.0007 | equal | seed only |
| yue clip | RunPod `tmozxvnm9fuuud` 4090 | 74.7 s cold (300 tokens: delay 6.8 s + exec 66.5 s); 32.2 s warm (doc) | 250 s cold (doc) | $0.022 cold, $0.008 warm | equal | yes when local busy |
| pixal3d 1024 | local | runtime absent | | | | unavailable |
| pixal3d 1024 | RunPod `akgefm0nzzr4jo` 4090 | 287 s cold (delay 19.4 s + exec 266.7 s, predict 124 s) | | $0.087 cold | equal (same weights) | yes |
| qwen-edit 2511 512px | local | 73.5 s | 87 s first | ~$0.002 | reference | yes (no remote; see `qwen-edit-offload-plan.md`) |

Cache hits (reported separately): the ra2 repeat in canary B was not an exact-cache
hit (request had no `cache: true`, 15.1 s full render). YuE and gateway cache hits are
recorded as `local:cache_hit` rows at $0.

Consequence for ra2: remote is 3.5-8x slower than local, so the old "overflow on any
contention" sent a paid request that would have waited ≤12 s to a 60-145 s remote and
paid $0.01-0.03 for it. With the frontier, paid/free queue locally up to ~2-3 jobs deep
and only then spill.

## Canary results (live `:8792`, forced saturation, 1024², unique seeds)

1. Legacy policy, before the fix: both overflows returned 502 in 7-8 s. Cause: the
   RunPod `qwen_image` worker rejects any unknown input and gateway callers send
   `model: "ra2"` (`unknown Qwen Image inputs: model`). This was also behind the
   production overflow 502s at 05:32-06:07. Fixed in the adapter (`RUNPOD_INPUT_KEYS`
   allowlist mirroring `ALLOWED_INPUTS`).
2. Legacy policy, after the fix, 4 concurrent paid: 1 local 17.6 s, 3 overflowed in
   parallel across the 3 workers (workersMax raised 2 → 3): 60.5 / 84.3 / 145.5 s,
   all 200, ledger rows tier=paid with delay/exec split, `saturated` counter 3.
3. Frontier policy, 4 concurrent paid + repeat: all local, 16.4 / 30.3 / 45.8 / 55.1 s,
   `frontier_kept_local` 3, `saturated` 0, gateway rows `queued` with predicted waits
   12.3 / 24.6 / 36.9 s vs actual 16.1 / 30.0 / 45.5 s. Every request finished sooner
   than the fastest legacy overflow, at $0 RunPod spend.
4. YuE via worker (paid, 300 tokens): RunPod 74.7 s wall, ledger row recorded.
5. Pixal3D via 3D worker: RunPod 287 s wall, GLB uploaded, ledger row recorded.

## Knobs

| env | where | default | now |
| --- | --- | --- | --- |
| `OMNISERVE_NATIVE_FRONTIER_POLICY` | qwen gateway | unset = legacy | routing.json |
| `OMNISERVE_NATIVE_FRONTIER_LOG`, `_WORKLOAD` | qwen gateway | unset | gateway-ra2.jsonl, ra2 |
| `FRONTIER_LEDGER`, `FRONTIER_LEDGER_DB` | adapter, yue, 3D | 0 | 1 |
| `FRONTIER_ROUTING` | adapter | 0 | 1 (one endpoint today) |
| `PRIMARY_BREAKER_S` | adapter | 0 | 120 |
| `RUNPOD_RA2_ENDPOINTS`, `RUNPOD_INPUT_KEYS` | adapter | endpoint id, worker allowlist | default |
| `YUE_FRONTIER_ROUTING` | yue worker | 0 | 0 (local YuE unmeasured on the 5090) |
| `OMNISERVE_3D_PIXAL_REMOTE_ENDPOINT`, `_MODE` (fallback/always/off), `OMNISERVE_3D_REMOTE_MAX` | 3D worker | unset | akgefm0nzzr4jo, fallback, 2 |
| `FRONTIER_LOCAL_USD_PER_H`, `FRONTIER_OPPORTUNITY_USD_PER_H`, `FRONTIER_RATES` (JSON) | ledger | 0.10, 0.70, built-in | default |
| `FRONTIER_GATEWAY_PORTS` | ingest | 8792 | 8792 |

Drop-ins live in `systemd/frontier/` and are installed as
`omniserve-native-qwen.service.d/zz-frontier.conf`,
`omniserve-runpod-overflow.service.d/frontier.conf`,
`omniserve-yue-worker.service.d/frontier.conf`,
`omniserve-3d-worker.service.d/pixal-remote.conf` (key in `/etc/omniserve-3d-remote.env`, 0600);
timer `systemd/omniserve-frontier.{service,timer}`.

RunPod changes: `tlofa06vj7iab7` workersMax 2 → 3; `akgefm0nzzr4jo` flashboot off → on.

## Operating

```
cd /nvme0n1-disk/code/omniserve-native
python3 -m frontier table        # current frontier from seeds + ledger
python3 -m frontier summary      # 24 h per workload/backend jobs, errors, $, p50/p95, billing
python3 -m frontier cycle        # ingest + reconcile (if due) + rebuild routing.json
curl -s localhost:18797/metrics  # Prometheus view of the same ledger
curl -s localhost:8792/status | jq .overflow   # frontier on/off, kept_local, saturated
```

Billing factor caveat: RunPod bills idle tails and any job not sent through the
ledger (manual probes, app.nz cogs on the same endpoint), so factors > 1 are
expected; they scale remote $/job upward, which only makes the router more
conservative about spending.

## Rollback

- Gateway policy only: set `Environment=OMNISERVE_NATIVE_FRONTIER_POLICY=` in
  `/etc/systemd/system/omniserve-native-qwen.service.d/zz-frontier.conf`, daemon-reload,
  restart `omniserve-native-qwen` (or delete routing.json: the gateway falls back to
  legacy overflow within a second, no restart).
- Gateway binary: `build-qwen/omniserve-native.pre-frontier` is the previous binary;
  point `ExecStart` at it or copy it back, remove `zz-frontier.conf`, restart.
- Adapter: remove `omniserve-runpod-overflow.service.d/frontier.conf`, restart
  (keep the input allowlist; without it every RunPod overflow fails).
- YuE / 3D: remove their drop-ins, restart.
- `systemctl disable --now omniserve-frontier.timer`.
- Background admission: drop the two `OMNISERVE_NATIVE_BACKGROUND_*` lines from
  `zz-frontier.conf` and restart (priority order itself is unchanged osched behaviour).
- Caller tags: netwrck revert commit `85dbc601` and rebuild (prod is now
  `netwrckprod154`, built by another operator on top of it); cutedsl
  `/opt/cutedsl-site/server/cutedsl-server.pre-tier-20260923`.
- RunPod: `PATCH /v1/endpoints/tlofa06vj7iab7 {"workersMax":2}`; flashboot can stay on.

## Follow-ups

- Local YuE on the 5090 is unmeasured (never 18 GiB free); enable
  `YUE_FRONTIER_ROUTING=1` after one local render is in the ledger.
- The ra2 RunPod worker is 3x slower than local even warm (pool includes A5000/L4/3090
  and streams weights): pin the endpoint to 4090 or keep weights resident to make it a
  real frontier point.
- `8791` runs the ra2overflow `build-full-ra2` binary without the frontier code (its
  image overflow is disabled); rebuild only when Z-Image overflow is restored.
- Another agent's `qwen21_canary.py` launches gateways with the production unit env
  (including the frontier drop-in); ingest now filters by port, but those runs still
  append to the shared JSONL.
