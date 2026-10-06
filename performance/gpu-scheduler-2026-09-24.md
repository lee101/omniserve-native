# GPU scheduler on the shared 5090 (2026-09-24, branch `gpu-sched`)

One VRAM ledger for the box, owned by the `:8791` gateway's existing broker
(`src/ovram.c`). Tenants lease before a large allocation and release after;
the broker queues instead of refusing, ranks by tier, evicts idle residents
under pressure, and exposes everything at `GET :8791/v1/gpu/status`.

## Measured (before)

24 h access log (2026-09-23 10:30 to 09-24 10:30), `experiments/gpu-sched/access_mix.py`:

| path | req/day | p50 |
| --- | ---: | ---: |
| `/v1/images/generations` (Z-Image, plus ra2 relays) | 5800 | 14.3 s |
| embeddings + feature-extraction | 2140 | 25 ms |
| `/v1/guard/classify` | 827 | 0 ms |
| gemma chat/generate/autocomplete (model=gemma) | ~455 | 0.5 s |
| ra2 (Qwen 2.1, `:8792`) | ~40 | 10 s exec |

5xx in the same window: 334 image 503s answered in <5 ms (broker lease denied,
no queueing: bursts 21:2x 135, 06:51 65, 09:43-09:49 113), 9 image 500s (CUDA
OOM mid-denoise), 26 slow 503s. The bursts line up with a co-tenant grabbing
VRAM (canary holding 8 GB at 09:44; netwrck redeploy double-holding at 10:35).

VRAM per tenant (1 s NVML sampler, `vram_sample.sh`, ~45 min):

| tenant (unit) | idle MiB | peak MiB | notes |
| --- | ---: | ---: | --- |
| omniserve-native :8791 (gemma q8 NGL=99 + Z-Image Q4 streamed) | 6392 | 8400 | Z-Image growth per render <=1.7 GB (lease credit) |
| omniserve-native-qwen :8792 | 1166 | 11448 | +10.3 GB per 1024^2 render, ~7.7 GB at 768^2 |
| netwrck search (supervisor) | 954 | 5352 | two copies during redeploys |
| text-generator-tts-gpu | 2682 | 2682 | static (torch cap 1.5 + ORT 0.75), 0 OOMs in log |
| birefnet worker | 1076 | 3970 | holds a 3584 MB background lease, 30 min TTL, renewed |
| bitbankgo-chronos | 1058 | 1058 | |
| cutedsl-zimage :8100 | 498 | 498 | |
| box total | p50 19.0 GB | max 31.9 GB | of 32.6 GB |

Idle: gemma served 455 calls in 24.5 h; gaps >10 min cover 22.4 h (91 %).
Cold-reload cost from page cache: 2.7 s (restart log, 4.76 GiB CUDA buffer).
Z-Image sampling is PCIe-bound: 8 steps x 1.5 s/it streamed from CPU (Gen3 x8).

## What changed

- `ovram_lease_wait`: queue up to a per-tier budget
  (`OMNISERVE_NATIVE_VRAM_WAIT_MS_{PAID,SUB,FREE,BACKGROUND}`, 60/45/30/20 s)
  instead of an instant 503. A queued higher tier blocks lower tiers only while
  its need is coverable once live leases end and for at most `VRAM_BLOCK_MAX_S`
  (30 s in prod), so an unfittable paid job cannot starve everyone.
- Materialisation credit: leases carry the holder pid; bytes the holder has
  already allocated (NVML per-process) stop being charged twice.
- Lower-tier leases younger than `VRAM_JOB_LEASE_S` (180 s) bind every tier
  (in-flight jobs); older ones (standing reservations such as birefnet's) can be
  squeezed by higher tiers.
- Pressure hook (after `VRAM_PRESSURE_AFTER_MS`): evict the optional guard judge,
  then the LLM if idle >= `EVICT_LLM_IDLE_S` and the waiter tier <=
  `EVICT_LLM_MAX_TIER`; the LLM reloads lazily on its next request behind a
  lease (never onto CPU).
- `GUARD_JUDGE_AUTO=1`: ShieldGemma (~3.2 GB) is admitted only after 60 s of
  headroom >= need + `GUARD_JUDGE_MARGIN_MB` with nobody queued; evicted first.
- `VRAM_BROKER_URL`: a second gateway leases from `:8791` (`src/obroker.c`),
  fails open if `:8791` is unreachable. `VRAM_FORCE_TIERS` records the lease
  after the wait even if it does not fit (the holder runs as it did before
  brokering; everyone else queues behind it). `SD_LEASE_SCALE_PIXELS` sizes the
  lease as `lease_1024 * (0.45 + 0.55 * Mpx)`.
- `/v1/gpu/lease` accepts `wait_ms`, `pid`, `force`; `/v1/gpu/status` returns
  ledger (per-tier headroom, waiters, leases with charged MB, owners with idle
  time and waits, every GPU process by systemd unit) + scheduler state;
  `/status` gains `gpu_sched`.
- text-generator.io `tts_gpu_server.py`: on OOM, lease from the broker and retry once.

## Canaries (all guarded: abort on any prod 5xx)

| canary | result |
| --- | --- |
| broker-only `:8796` (no models, 0 VRAM): hold/queue/priority/timeout | paid waiter granted 3002 ms after release; background blocked while paid queued; free timeout after 1503 ms |
| Z-Image `:8797` leasing from `:8796` | 3/3 renders 200; queued 5.0 s behind a holder then rendered (was: instant 503); credit tracked growth |
| 0.6B LLM eviction/reload `:8796` | judge admitted after 70 s, evicted first under pressure (freed 3216 MB); LLM evicted (freed 1328 MB), reload 1127 ms, chats 200 |
| Qwen `:8798` leasing from prod `:8791`, v1 | 1st render 200 concurrent with Z-Image; 2nd: paid waiter could not fit by 13 MB and starved background Z-Image, 2 prod 503s -> abort -> feasibility/age-bounded blocking |
| v2 (forced lease) | a paid lease ignored an in-flight background Z-Image lease, Z-Image OOMed, 1 prod 500 -> abort -> young-lease rule |
| v3 (young-lease rule) | 3/3 renders 200 at 1024^2, 0 forced, 0 prod 5xx; Qwen queued avg 10 s behind in-flight Z-Image, Z-Image queued avg 5.4 s behind Qwen |

Unrelated hazard found: `ctest` on this box starts gateways that take ~3 GB of
VRAM each and write to the prod access log; run it with
`CUDA_VISIBLE_DEVICES= OMNISERVE_ACCESS_LOG=0`.

## Deployed

- `:8791`: `/etc/systemd/system/omniserve-native.service.d/zzz-gpu-sched.conf`
  (source `systemd/gpu-sched/`), binary `deploy-sched-<rev>/omniserve-native`
  (ra2-moe 85a5370 auth + gpu-sched). Z-Image lease stays 4096 MB
  (= `SD_MIN_FREE_MB`; a smaller lease would pass the queue and then fail the
  point check that protects the conditioner from aborting; credit returns the
  unused ~2.4 GB to others while it renders). LLM eviction for paid/sub
  waiters after 600 s idle, judge auto off.
- `:8792`: `omniserve-native-qwen.service.d/zzzz-gpu-sched.conf`, leases
  10240 MB (edit 11264) at 1024^2 scaled by pixels, forced after 20/20/25/30 s.

First 20 min after the final deploy (11:08-11:28 UTC): 0 VRAM-related 5xx on
`:8791`/`:8792` (3 unrelated `/v1/images/segmentations` 503s: no
IMAGE_EDITOR_UPSTREAM), 10 image requests queued for headroom and served
instead of the old instant 503, 0 prod wait timeouts, ra2 smoke render 200.

This drop-in overrides `ExecStart` from `overflow-daisy.conf`: a later binary
deploy must update `zzz-gpu-sched.conf` (or remove it), not only overflow-daisy.

Rollback: `sudo rm /etc/systemd/system/omniserve-native.service.d/zzz-gpu-sched.conf /etc/systemd/system/omniserve-native-qwen.service.d/zzzz-gpu-sched.conf && sudo systemctl daemon-reload && sudo systemctl restart omniserve-native omniserve-native-qwen`
(returns to deploy-auth-85a5370 and build-qwen 4e20d77).

## Next (not done)

1. `GUARD_JUDGE_AUTO=1` in prod once the judge's effect on guard verdicts is
   accepted (it changes moderation outcomes and adds latency to classify).
2. `EVICT_LLM_MAX_TIER=free` if paid/sub-only eviction proves too rare.
3. birefnet: send `pid` with its lease (credit) or lease per job; today it
   withholds 3.5 GB from background work around the clock.
4. Z-Image residency. Canary (`canary_b2.sh`, all params on GPU, leased from
   prod as background): sampling 0.53-0.58 s/it vs 1.5 s/it streamed (2.7x),
   ~7.5 s saved per render x 5.8k renders/day = ~12 h/day of image-lane time;
   cost: process peak 9.6 GB vs ~1-2 GB streamed growth (diffusion 3.9 GB +
   Qwen3-4B encoder 2.5 GB + compute). Pairs with evicting idle gemma (5.9 GB,
   idle 91 % of the day); try diffusion-only residency (encoder on CPU) first.
   The broker refused the canary's 2nd/3rd background leases once its resident
   weights ate the headroom, which is the intended behaviour.
5. Access log has no port: `:8791` and `:8792` (and test gateways) share
   `/var/log/omniserve/access.log`.
