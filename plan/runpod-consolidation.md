# RunPod cost consolidation plan (2026-10-08)

Read-only analysis. Live data: RunPod REST GET (endpoints, pods, volumes, billing) plus GraphQL `myself`/`gpuTypes` reads. Static data: repo scan (see `_notes_inventory.md`, `_notes_omniserve.md`). Machine-readable: `inventory.json`. Nothing on RunPod, prod or other repos was modified; no jobs run. Secrets were not recorded (note: the endpoint `template.env` returned by the REST API contains a plaintext HF token; `inventory.json` omits env. Rotate that token if the API output was ever logged elsewhere).

## 1. Findings

### 1.1 Real spend (billing API; history only starts 2026-09-07, days with no spend are absent)

| bucket | window | USD | run-rate/mo | share |
| --- | --- | ---: | ---: | ---: |
| serverless endpoints | 09-12..10-04 (23d) | 26.31 | ~34 | 13% |
| pods (all deleted, 0 running now) | 09-09..10-05 (27d) | 143.55 | ~160 | 69% |
| network volumes | 09-07..10-05 | 38.12 | 24.6 now (was 57) | 18% |
| total | ~4 weeks | 207.98 | ~219 | |

- Account is small and R&D-dominated. Endpoint idle burn is already near zero (all but 5 endpoints parked at workersMax=0, cost guard active). Endpoint consolidation alone cannot save more than a fraction of $34/mo at today's volume; the plan therefore ranks pod governance and storage first and treats pooling as a scale/latency play with explicit break-evens.
- Pods: 10-05 one pod billed 17.2h = $60.19 (~$3.50/h); 09-30 ~35 short pods $41.5 (bench sweep); 09-20 ~20 pods $14.9; 09-09..14 the idle errored pixal3d pod `8lawkbhewkm81g` $22.3 (63h at $0.35/h, terminated 09-14 per cost audit). Deleted pods: GPU/name unrecoverable via API, so attribution of the 10-05 pod is unknown. Cost guard only auto-terminates allowlisted `h3upscale-probe-*` names; everything else is inventory-only.
- Endpoint spend by endpoint (23d): qwen-mt-overflow $6.77 (5.83h, 4090/L40S, eff $1.16/h), h3-pinkcherry $4.53 (0.94h, $4.8/h), h3-swap $4.04, ltx $3.73 (1.17h, $3.19/h), pixal3d $2.74 (2.47h), yue2 $0.92, minimax-h3-shared $0.91, ra2-overflow $0.32, orbit $0.28, cog-qwen-image $0.24, others ~$0.7. Daily peaks: 09-29/30 $18.7 (H3 tests), 09-23 $2.41 (qwen-mt bench); otherwise <$1.1/day. Effective rates match list: 4090 $1.10-1.16/h, H100 $4.80/h.
- Volumes: 150GB `manifold-models` EU-NL-1 ($10.5/mo, LTX + PinkCherry) and 100GB high-performance `manifold-wan-blackwell-cache` US-CA-2 ($14/mo at ~$0.14/GB-mo) whose Wan B200 endpoints billed $0 in the window. Three other volumes (612GB) were archived to R2 on 09-23 (already saved ~$36.8/mo).
- Account health flag: `clientBalance = -$0.27`, spendLimit 80, and `GET api.runpod.ai/v2/{id}/health` returned HTTP 402 Payment Required for all 37 endpoints. REST management calls work. Verify balance/autopay now: if 402 also applies to job submission, every serverless lane (YuE music, ra2 overflow, pixal3d, H3) is down regardless of this plan.
- Live vs docs: 37 endpoints (docs said 35); max>0 only for qwen-mt-overflow (2), pixal3d (2), yue2 (1), cog-qwen-image-2.1 (2), wan-animate-fast-b200-v2 (1). `workersStandby` is a read-only artefact, not capacity.
- Per-job cold-start tax is where money actually leaks at low volume (billed worker time, list rates):
  - ra2 4090: 17s boot x $1.11/h = $0.005 (small).
  - YuE 4090: +39s first-run penalty + 178s queue/init = ~$0.012 billed + latency.
  - H3 swap H100: 240s cold = $0.32 vs job price ~$0.70 (5s clip at $0.14/s).
  - H3 orbit H100: 197s cold vs 64s warm: +$0.18 vs fixed price $0.45-0.50.
  - LTX no-volume cold 300-700s on $3.19/h = $0.27-0.62 vs warm clip $0.0075.
  - PinkCherry cold 10-28 min on H100 = $0.80-2.24 vs warm $0.04-0.06.
  So H3/LTX lanes lose 30-60% of job price to one cold start; ra2/YuE lose cents. Idle timeout 5s on H100 lanes means essentially every job is cold.

### 1.2 Inventory summary (full detail in inventory.json)

| lane | model | GPU now | VRAM (weights / peak) | cold -> warm | caller | live billed 23d |
| --- | --- | --- | --- | --- | --- | ---: |
| H3 normal / pinkcherry / swap / orbit | MiniMax H3 INT8 (+LoRA variants) | H100 80GB | 43GB / 43-60GB | 90-240s -> 39-108s | manifoldgen | ~$9.8 |
| LTX | LTX-2.3 fp8 | PRO 6000 Blackwell / H100 | 41.6GB / 24GB | 300-700s (no vol) -> 24s | manifoldgen adult lane | $3.73 |
| Wan-Animate std/fast/xfast | Wan-Animate-2 FP8 | Ada48 / H100 / B200 | - / 35.4GiB | 27s (flashboot) / 93s+418s queue -> 6.4s | manifoldgen | $0 |
| Music3 x3 | MiniMax Music3 FP8 | H200 pool (archived) | 38GB / ~45GB | ~60s | manifoldgen (zero traffic: YuE serves all tiers) | $0 |
| YuE2 | YuE2-3B bf16 | 4090 | ~7.5GB / 7.7GiB (18GiB admit) | 242s -> 25s | manifoldgen, netwrck, omniserve | $0.92 |
| ra2 + edit (qwen-mt) | Qwen Image 2.1 + Edit 2511 | 4090 -> L40S | 11GB / 21GB; edit TE 16.6GB | 39s -> 9s / 22s | netwrck, ebank, cutedsl via overflow | $6.77 |
| Pixal3D | Pixal3D | 4090 | ~12GB / 23GiB | 19s -> 266s job | omniserve 3D, app.nz | $2.74 |
| anima, tts, matting, splat, ltxfast, comfy, train | assorted | 3090/L40S/5090 | 3-24GB | n/a | app.nz dynamic endpoints | (not in recent billing) |
| owned: prod 5090 (32GB, ~30GB used, ~12 tenants), daisy 3090 (down since 09-23 Xid 79), local 3090 Ti | Gemma q8, Z-Image, ra2 streaming, birefnet, asr... | owned | see omniserve notes | Gemma reload 2.7s | netwrck et al. | $0.10/h effective |

Settings pattern: all endpoints workersMin=0, FlashBoot on (3 off), idle 5s (H3/LTX/orbit/swap), 20-30s (YuE, qwen-mt), exec timeouts 480s-14400s, REQUEST_COUNT:1 (H3/Wan) or QUEUE_DELAY 2-4s (images). Non-volume lanes pull weights from app static/HF at cold start.

### 1.3 omniserve-native capabilities relevant to packing (detail: _notes_omniserve.md)

- Two runtimes. C gateway (`src/main.c`): epoll HTTP, osched tiered admission (paid>sub>free>background), embedded llama.cpp + sd.cpp, VRAM broker leases (`src/ovram.c`), overflow/oscale/frontier routing. One model per kind per process; multi-model = multiple processes via `OMNISERVE_NATIVE_IMAGE_MODEL_UPSTREAMS` (<=16, boot-time env). Python manifest runtime (`runtime/handler.py` + `workloads/workloads.json`): the real multi-model host with LRU-for-space co-residency, profile fallback on OOM, broker lease; one job per worker; no idle timer; no metrics.
- Idle unload today: LLM only under pressure (`EVICT_LLM_IDLE_S` prod 600s, `main.c:1673`), guard judge auto-evict, YuE child idle exit (30s), BiRefNet video idle release (300s). Nothing timer-based for SD, embeddings, or Python workloads. SD ctx has no unload path at all (`backend_sd.c:252-261`).
- LoRA: runtime per-request LoRAs without reloading the base (`backend_sd.c:897-913`); turbo LoRA XOR request LoRAs on main; the unmerged `omniserve-native-lorastack` patch composes them. `omniserve-native-tiers` is an identical-commit duplicate; `omniserve-turbo` is a superseded snapshot. This LoRA-on-resident-base mechanism is exactly what lets "one base + many lane LoRAs" be one worker.
- Batching: SD_MAX_BATCH=1 (<=8 after canary), no cross-request LLM batching, one job per RunPod worker. No per-tenant quotas or accounting.
- Metrics gaps: no GPU util, no per-model residency, no busy-seconds; `/status.gpu_sched` has eviction counters but `/metrics` does not. Latent bug: `char body[8192]` in `handle_metrics` can be overrun (`main.c:697`).
- Cache assets already present: Z-Image result cache (1-2ms hit), YuE 2GiB/24h disk cache + in-flight dedupe, Music3 WAV cache, H3 8GiB result cache on volume. Not shared across lanes or with overflow workers (qwen-mt image has no volume so no cache).

## 2. Target architecture

```
clients (manifoldgen, netwrck/ebank, app.nz, openpaths)
        |
   gateway/router  (omniserve-native frontier + oscale; per-model queue; exact-key result cache; spend caps)
        |-- owned tier (steady load): prod 5090 [LLM idle-unload, Z-Image resident, ra2], daisy 3090 [restored]
        |-- RunPod pool V "video"   : 1 endpoint, base H3 INT8 + lane LoRAs (normal/orbit/swap/pinkcherry via task),
        |                              LTX co-resident; 96GB PRO 6000 Blackwell or H100; weights on 1 volume / cached-model mount
        |-- RunPod pool I "image-audio" : 1 endpoint, 48GB (L40S/6000Ada/A6000/A40): ra2+edit+anima+YuE+tts+matting,
        |                              LRU in handler.py + idle_unload_s; pixal3d separate 4090 endpoint (23GiB, long jobs)
        |-- RunPod dedicated (zero-traffic, parked): wan-animate B200 lanes, music3 -> archive/retire
        '-- scratch pods: only via launcher with TTL + tag; guard terminates anything untagged/older than budget
```

Design decisions:
1. Affinity. RunPod serverless gives no worker affinity. Get it by endpoint granularity (one endpoint per pool) plus FlashBoot; inside the pool, `task`/`workload` routing to the resident model (handler.py already LRU-swaps in 6-19s from page cache, cheaper than 40-250s cold). Router orders queue by model so consecutive same-model jobs hit the same worker.
2. Co-resident vs time-share (peak VRAM, GB): 96GB card: H3 43 + LTX 24 = 67 fits; on 80GB H100 time-share. 48GB: ra2 21 + YuE 7.7 + anima 12.8 + matting 3 = 44.5 fits co-resident (but handler serialises, so co-residency buys no concurrency, only swap avoidance); edit TE 16.6 pushes it to LRU swap; pixal3d 23 + ra2 21 = 44 tight, long 266s jobs block a pool, keep separate. 24GB cards: time-share only (edit evicts ra2, 18.9s).
3. Weights. Three options, per lane: (a) baked in image (qwen-mt 44GB; no DC lock, 5 min unbilled pull on fresh host); (b) network volume (DC lock; 5090 absent in EU-NL-1; $0.07-0.14/GB-mo); (c) RunPod cached-model mount for HF-hosted weights (YuE already uses `/runpod-volume/huggingface-cache/hub`; no volume rent, no DC pin). Prefer (c) for HF-hosted lanes (H3 mirror, LTX, ra2 GGUF at netwrck/ra2, Music3), (a) for small lanes, (b) only for private non-HF weights (Wan prequant, H3 LoRAs).
4. Pod vs serverless break-even = pod rate / serverless rate: 4090 community 0.34/1.10 = 31%, secure 0.74/1.10 = 67%; H100 secure 3.49/4.80 = 73%, community 2.69/4.80 = 56%; PRO 6000 2.09/3.50 = 60%. Nothing in the account is near these utilisations; steady load belongs on owned GPUs (effective $0.10/h) and pods only past break-even (app.nz router promote 0.70 / demote 0.40 already encodes it).
5. Right-size. H3 INT8 set "needs 32GB" (h3-cog README): 5090 serverless $1.58/h vs H100 $4.80/h; at 55-60% of H100 speed (assumption) cost per job falls ~40%. PRO 6000 Blackwell $3.50/h vs H100 $4.80/h at ~0.7-0.8x speed is roughly cost-neutral to -15%, but already hosts LTX. Needs a paid benchmark (not run).
6. Spot/community: serverless has no spot. Community pods only for scratch/bench: H100 $2.69 vs $3.49 secure (-23%), 4090 $0.34 vs $0.74 (-54%).
7. Cache. Gateway-level content-addressed exact-key cache (model, params, seed, LoRA ids) in R2 consulted before any overflow/RunPod dispatch; plus in-flight dedupe. Hit rate unmeasured (assume 3-10% for image/music).

## 3. Ranked proposals

Savings are monthly USD against run-rate 219/mo (endpoints 34, pods 160, volumes 24.6). "10x" column assumes endpoint traffic grows 10x with same lane mix. Estimates labelled; assumptions in section 6.

| # | proposal | saving now | saving at 10x | effort | risk |
| --- | --- | ---: | ---: | --- | --- |
| 0 | Fix account balance/402 (not a saving, outage prevention) | n/a | n/a | XS | none |
| 1 | Pod governance: launcher with mandatory TTL + tag, guard terminates all untagged pods > 3h, per-pod and daily pod cap | $65-95 (40-60% of 160; the 10-05 pod alone at 3h cap = -$50) | same | S | low |
| 2 | Storage: archive `fb96eh7ozc` HP volume to R2 (restore script exists from 09-23 archive); move HF-hosted lanes to cached-model mounts and retire `65aknc5k8g` when LTX/PinkCherry are off it | $12.5 now, up to $24.6 | same | S-M | low-med (lane restore time) |
| 3 | H3-family single pool: one image/endpoint, shared base + LoRAs on one weights source, task routing; idle 5s -> 60-120s on pool; retire PinkCherry (LTX is 6x cheaper per clip: $0.0075 vs $0.042-0.056) | $5-10 (H3 family ~$13/mo now) | $50-130 (cold tax 30-60% of job price removed ~ half) | M | med (H3 license/territory gating, dep pins, LoRA correctness) |
| 4 | Right-size H3/LTX to 5090 or PRO 6000 where benchmarks confirm | $3-6 | $30-80 (25-40% per job) | M (paid bench ~$15-30) | med (quality/speed, 5090 absent in EU-NL-1, CUDA13) |
| 5 | Pool I on 48GB with idle_unload_s + task routing; keep pixal3d separate | $3-8 | $30-70 | M | med (1.6x per-second rate vs 4090 pays only if warm hits rise; swap 6-19s) |
| 6 | Owned tier: LLM idle-unload (Gemma idle gaps >10min cover 91% of day), Z-Image resident (5.1s vs 15.5s streamed), restore daisy; fewer ra2/Z-Image overflows | $1-5 (overflow cost) | $10-40 | S-M | low |
| 7 | Cost guard v2 (below): pod/volume coverage, 402/balance alert, tighter global cap, overhead ratio | risk cut: phantom-worker exposure $5.7/h = $4.1k/mo worst case seen 08-16 | same | S | low |
| 8 | Exact-key result cache in gateway before overflow; shared across lanes | $0.5-2 | $5-20 (3-10% hit) | M | low |
| 9 | Hygiene: delete 18 archived/parked endpoints with deleted volumes, fix stale env refs, retire Music3 lanes (zero traffic) | $0 | avoids accidental scale-up | S | low |

Top-5 by value: 1, 2, 3, 4, 5 (6/7 are cheap enablers; 7 is the best risk-per-effort).

### Cost guard v2 (small changes to runpod_cost_guard.py; owner: manifoldgen pricing agent, do not edit from here)
- Pods: terminate any pod with no `ttl` tag or older than `RUNPOD_COST_GUARD_POD_MAX_HOURS` (default 3, exception list), not only `h3upscale-probe-*`; alert on GPU util 0% for >30min (pixal3d precedent: errored idle pod billed 5 days).
- Caps: global daily $120 vs observed max day $60 (pod) and $11.5 (endpoints); set global ~$45, pods $30, endpoints $25 (per-lane caps already exist) so a runaway is caught at ~1/3 of today's ceiling.
- Volumes: daily alert for volumes with no billed endpoint spend in 14d (catches `fb96eh7ozc` $14/mo).
- Balance: poll `myself.clientBalance`; alert below $10; alert on 402 from health.
- Metrics: per endpoint `billed_s / sum(execution_s)` (overhead ratio) from billing + job `executionTime`; alert when >3 for 7d (indicates cold/idle tax needing pool or idle tuning).
- Add `omniserve-qwen-mt-`, `cog-manifold-ltx`, `cog-manifold-h3-swap/orbit` prefixes explicitly (qwen-mt overflow is the top spender but alert-only/unlisted: `omniserve-ra2-` prefix does not match `omniserve-qwen-mt-overflow`).

## 4. Phased rollout (all steps reversible)

| phase | scope | gate | rollback |
| --- | --- | --- | --- |
| 0 (day 0) | Verify balance/402; rotate exposed HF token if needed; snapshot `inventory.json` as baseline | health API returns 200 | n/a |
| 1 (week 1) | Cost guard v2 in dry-run, then `--apply`; pod launcher TTL; archive HP volume (R2 copy verified by byte count, same procedure as 09-23) | dry-run shows only expected terminations | stop timer; restore volume from R2 (restore script) |
| 2 (week 1-2) | omniserve-native tasks 1-5 (section 5): metrics, LLM idle unload, Python idle sweep; deploy to owned boxes first, default off then on | `/metrics` residency matches `nvidia-smi`; reload p95 < 3s | env var off, unit drop-in revert |
| 3 (week 2-4) | Measure 14 days with metrics (section 7); choose idle timeouts from observed gap distribution; Pool I canary: add `yue2`, `anima`, `matting` to workloads manifest on a new endpoint at max=1, router sends 10% | warm-hit and p95 latency better than separate endpoints; $/job lower | router weight to 0; old endpoints kept parked (max=0) |
| 4 (week 4-8) | H3 pool: image with base + LoRAs, task routing; benchmark 5090 / PRO 6000 vs H100 on 20 jobs each (~$15-30 paid, approval needed); migrate lanes one at a time (orbit, swap, normal, retire PinkCherry) | quality parity (existing ltx-vs-pinkcherry style eval), cost/job -25% | per-lane env endpoint id back to old endpoint; keep old endpoint config JSON |
| 5 | Delete archived endpoints; cached-model mounts replace volumes; gateway result cache | no 404 on weights in cold test | re-create volume from R2 |

## 5. omniserve-native: smallest high-confidence changes (ordered, NOT implemented)

All default-off, independently shippable; refs in `_notes_omniserve.md` section 8.
1. Fix `/metrics` overflow: bump `char body[8192]` to 16384 and clamp `len` before `ohttp_respond` (`src/main.c:697`, end of `handle_metrics` ~855). Prereq for 2-3.
2. Residency/eviction metrics (pure additions): `omniserve_model_loaded{kind}` (llm/embedding/diffusion/judge via existing `*_ready()`), `omniserve_llm_idle_seconds`, `_llm_evictions_total`, `_llm_reloads_total`, `_llm_reload_failures_total`, `_llm_reload_ms_last` (struct `g_gs`, `main.c:1625`, currently only in `/status.gpu_sched`), per-tier `omniserve_admission_served_total`, complete `queue_ms_max` for sub/background (`main.c:728-729`).
3. GPU utilisation + busy-seconds: dlsym `nvmlDeviceGetUtilizationRates` next to `nvml_mem_info` (`src/osched.c:377-381`), export `omniserve_gpu_util_percent`; add `account_locked()` at the four `used_slots` change sites (`osched.c:190, 211, 224, 294`) -> `omniserve_admission_busy_seconds_total`, `_slot_seconds_total`.
4. LLM idle unload timer: thread beside `judge_admit_main` (`main.c:1773`, start near 4051), env `OMNISERVE_NATIVE_LLM_IDLE_UNLOAD_S`; factor eviction body `main.c:1696-1713` into `gs_llm_evict_locked`; lazy reload already exists (`llm_ensure_loaded`, `main.c:1725`). Also make `/admin/llm/unload` set `llm_evicted` so it reloads (`main.c:3242`), behind env.
5. Python runtime idle sweep: `idle_unload_s` + `prewarm` keys in `workloads/workloads.json`; daemon thread in `runtime/handler.py` calling `_release_name` under `_dispatch_lock` using `_loaded_modules[name]["last_used"]`; same for `qwen_mt._ctxs` (`qwen_mt.py:187`); `/metrics` text on `runtime/cog_pod.py` (resident names, required_mib, free_mib, last_used age, oom_retries). Log `resident_workloads`/billed seconds per job into the response so the router can compute overhead.
6. Learned footprints: record measured MiB after first load in `dispatch` (`handler.py:277-340`), persist to `/runpod-volume/omniserve/footprints.json`, use max(required_mib, learned).
7. Add pool workloads to the manifest (yue, anima, matting already; fold `qwen_mt.py` task routing into `workloads.json`; optional pixal3d wrapper) so Pool I is one image.
8. Merge the uncommitted `omniserve-native-lorastack` `SD_TURBO_STACK` patch with a cache-key test; delete `omniserve-native-tiers` (identical commit) and archive `omniserve-turbo` to cut confusion.
9. `OMNISERVE_NATIVE_LLM_MODEL_UPSTREAMS` table mirroring `IMAGE_MODEL_UPSTREAMS` (`main.c:4153-4190`) for sibling LLM processes.
10. SD ctx unload/lazy reload (`osd_shutdown`/`osd_reload`, separate "configured" vs "resident" flag since `osd_ready()` gates overflow at `main.c:3598`) only after confirming `free_sd_ctx` frees VRAM in the prod sd.cpp fork.

## 6. Assumptions and uncertainty

- Run-rate extrapolation uses lumpy 3-4 week data dominated by test activity; production traffic share unknown. Re-baseline after Phase 3.
- Savings at 10x scale linear in lane mix; "cold tax removed ~half" assumes pool idle 60-120s and bursty user sessions (P(next job within T) > T x rate / cold_cost ~ 0.3). Needs gap histogram from job logs.
- 5090 H3 speed at 55-60% of H100 is an assumption; PRO 6000 0.7-0.8x likewise. Serverless active-worker discount (~30% under flex, published by RunPod) is unverified and not used in numbers.
- Pod attribution for 10-05 ($60) and 09-30 sweeps is unknown; 50% avoidable is judgement from the guard's 3h scratch budget.
- Billing data excludes anything on non-RunPod accounts, app.nz customer-provisioned pods (btcmarketsbot/app.nz rentals are user-funded passthrough) may be inside the pod total; check by pod tag before enforcing termination.
- Not measured anywhere: Music3 cold start/VRAM, FlashBoot hit rate (set true everywhere), per-lane request counts.

## 7. What to measure (14 days, before Phase 4)

- Per endpoint daily: billed_s, jobs, sum execution_s, cold fraction (delayTime > 15s), overhead ratio, $/job, p50/p95 queue delay.
- Gap histogram between consecutive jobs per lane (decides idle timeout, pool membership).
- Per worker: model residency, swap count, swap seconds, OOM retries, VRAM high-water (from items 2, 5).
- Owned GPUs: NVML util, busy-seconds, LLM idle %, overflow count per cause (`omniserve_overflow_total{cause}`).
- Cache hit rate per lane; pod count/age/spend by tag; volume $ vs endpoint $ using each volume.
- Quality gates for any migration: existing evals (ltx-vs-pinkcherry style, qualitybench, SSIM>=0.985/PSNR>=37 floor used for ra2).
