# Local-first serving investigation — 2026-09-06

## Changes and verification

`oscale_decide` now subtracts existing active instance capacity from the hourly
demand estimate before valuing another rental. Warming instances count too;
released instances do not. Regression coverage includes thin marginal demand,
sufficient residual demand, cooldowns and hard caps. Full native core tests
passed in a fresh CPU-only CMake build. This does not alter inference math.

Example using the test fixture's configured values (not current provider quotes):
5 seconds/request gives 720 requests/hour/instance. At 200 expected requests,
the old gate could justify a second instance against the same demand. The new
gate refuses it; at 920 expected requests it can justify the extra 200.

No service binary/config was deployed, no service restarted, no cloud capacity
rented and no unload hooks called. Existing unrelated changes were preserved.
Production's generic native rental lanes are disabled, so this fix has no
realized production savings yet.

## Live production baseline

Host: `93.127.141.100`, RTX 5090, 32,607 MiB VRAM. Initial snapshot: 18,752 MiB
used, 74% GPU utilization, eleven compute processes. Native gateway admission
had one of eight slots used and no queued requests at the subsequent snapshot.
GPU utilization and scheduler permits measure different things: spare permits
are not proof that the device is idle.

Production embedding quality harness: 8/8 checks passed, semantic ranking
25/25, STS margin 0.2229, reference-vector cosine 1.0. This tests the existing
embedding route, not GPU placement or quality of other modalities.

The new stdlib-only `tools/forecast_serving_canary.py` ran against the existing
local CUDA Chronos worker on port 8101, with explicit cold-load permission.
It checks worker idleness, CUDA health, at least 4 GiB free and GPU utilization
below 85% before starting. These are snapshot guards, not a global reservation;
normal worker admission still applies. No concurrency stress was attempted.

| Measurement | Observed |
| --- | ---: |
| Cold first forecast | 9,245 ms |
| Four warm single requests | 48.6, 37.6, 33.7, 35.0 ms |
| Three four-series batches | 62.4, 43.2, 42.1 ms |
| Sequential time / median batch time | 3.58x |
| Largest single/batch absolute output delta | 0.0625 |
| Provisional parity threshold | 0.02 — FAIL |

Inputs were four synthetic 128-point series, 16-step horizon, three quantiles.
The parity check covers means and all quantiles. This short live test is noisy,
not a controlled before/after benchmark, and does not measure forecasting
accuracy. The threshold is a conservative canary choice, not an established
model accuracy budget. Do not infer degraded task accuracy from this delta alone.
No batching defaults were changed. The canary loads Chronos through its normal
request path; normal model lifecycle management retains control afterward.

App-site's isolated GPU router tests also passed with the Go race detector.

## Next gates before broader rollout

1. Investigate Chronos single/batch delta using equal-length and ragged contexts,
   held-out forecast MAE/quantile loss, and upstream-versus-CuteDSL comparison.
   The current batch implementation zero-pads ragged contexts; validate that
   against the upstream missing-value semantics before increasing batch use.
2. Capture per-model arrival timestamps, local/remote placement, queue delay,
   cold-load time, host-to-device transfer time, peak VRAM and execution time.
   GPU device-memory bandwidth is not host-to-device bandwidth. The 9.25-second
   cold request includes loading/setup, not just PCIe transfer.
3. Replay those traces for recency-decayed frequency residency. Score expected
   avoided reload latency per VRAM byte; reserve scratch and paid-request
   headroom. Prefetch only with idle admission, a broker lease, bounded host RAM
   and a cooldown; never evict active work. Existing native host page-cache
   warming is not a predictive cross-model GPU cache.
4. App-site `gpu_router.go` currently permits two in-flight jobs to trigger a
   pod independently of the utilization cost gate. Validate burst-only versus
   sustained remote demand, observed cold/idle charges and actual configured
   provider rates before changing it. The native README's older 1.20-factor
   description differs from the shared router's 1.4-factor commentary.
5. Native capacity observations currently use global tier counters and recent
   served rate, not per-model excess arrivals after local service. Marginal
   rental accounting fixes duplicate demand within a lane, not that estimator
   or cross-lane double counting. Keep generic lanes disabled until lane-level
   demand and end-to-end overflow routing are verified.
6. ManifoldGen already has explicit serverless drain/zero-worker reconciliation
   (see its scheduler-efficiency report). Preserve it when testing cutovers;
   verify empty queues and actual worker shutdown, not just desired minimums.

Require workload-specific quality baselines, p50/p95 latency, completed jobs per
GPU-second and measured provider dollars per successful job for any rollout.
No throughput or cost improvement across all models is claimed by this pass.

## Follow-up: batch diagnostics and singleton optimization

Read-only inspection established that the production site server is newer than
the local checkout: production already passes ragged contexts as a list for
NaN padding. The local zero-padding concern above does not describe the live
worker. Do not overwrite production with that older local server.

The extended HTTP canary (`--ragged`) tested context lengths 128/97/64/33 and
repeated every single request. Single repeats were identical. Each batch repeat
changed 7/256 values by exactly one BF16 step (max 0.0625); largest normalized
delta was 0.110 context standard deviations. Batch time was 41–44 ms versus
188 ms for four sequential singles, a noisy 4.47x ratio. The original 0.02
absolute gate remains unchanged and still fails. Diagnostic tests check BF16
step counting and reject malformed/nonfinite outputs.

A separate opt-in CuteChronos singleton-group-attention candidate removes
redundant Q/K projections for one-series requests without changing batch size.
Standalone eager BF16 A/B on the production 5090 measured 1.19–1.30x speedup
with exact equality across all output quantiles for three synthetic cases.
See `../cutedsl/cutechronos/SINGLETON_PERFORMANCE.md` from the OmniServe repo
root for measurements, reproduction and remaining rollout gates. No service
configuration or default was changed.
