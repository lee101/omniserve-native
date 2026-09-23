# Qwen Edit offload to RunPod: plan (2026-09-23)

No new image was built or deployed. This records the numbers that decide whether
Qwen Image Edit should move off the shared RTX 5090, and the gated steps to do it.

## What is live today

| lane | where | measured | source |
| --- | --- | ---: | --- |
| Qwen Image Edit 2511 Q4_K_M, 512px, 12 steps, CFG 2.5 | local 5090 canaries only, not a production route | 72.5-75.0 s warm, 87.2 s first | `qwen-edit-5090.md` |
| same, CPU encoder/VAE | local 5090 | 290.5-291.0 s | `qwen-edit-5090.md` |
| ra2 reference edit (`/v1/images/edits`, Qwen Image 2.1 Q4_K_M) | `:8792`, production | 27.6 s (20 steps, EasyCache 0.2); ~30 s end to end via netwrck | `../omniserve-native-ra2overflow/performance/qwen21-ra2-rollout-2026-09-22.md` |
| ra2 generation on RunPod `tlofa06vj7iab7` (4090/A5000/L4/3090 pool) | overflow | 33.4 s exec warm (1024², 20 steps); 101.7 s exec + 6.1 s delay cold | 2026-09-23 probe, `frontier-scheduler.md` |

The production edit traffic is ra2 edits. Those already overflow through the same
adapter as ra2 generations (`/v1/images/edits` is passed through, `image_base64`
and `strength` are accepted by the worker), so the frontier scheduler covers them
without a new endpoint.

## Cost and latency if 2511 moved to a 4090-class endpoint

- VRAM: DiT Q4_K_M (~12 GB) + Qwen2.5-VL-7B encoder + VAE fits a 24 GB card with
  the graph budget used locally; the shared 5090 has 10-12 GiB free and cannot hold
  it resident next to the other tenants, which is why local runs are 72-87 s.
- The ra2 remote runs ~3x slower than the local lane (33.4 s vs 11-12 s) because the
  pool mixes A5000/L4/3090 with 4090 and the worker streams weights. Expect 2511
  edits on that pool in the 60-150 s range until measured; this is an estimate,
  not a result.
- Cost at the observed pool rate ($0.744/h billed): 75 s ≈ $0.0155 per edit, plus
  a 30 s idle tail ≈ $0.006 per burst, plus cold start (~108 s ≈ $0.022) when the
  worker is not warm. Local marginal cost is ~$0.002 per edit (electricity at
  $0.10/h) but it blocks the shared GPU for 75 s.

## Recommendation

Do not build a 2511 image now. Qwen Edit 2511 has no production route and ra2 edits
already have local + RunPod overflow on the frontier. Revisit when 2511 gets a
product route or when ra2 edit quality is judged insufficient.

## Steps when it is needed (each is a gate)

1. Add a `qwen_edit_2511` profile to `workloads/qwen_image.py` in the existing
   `ghcr.io/lee101/omniserve-native:ra2` worker (same sd.cpp loader, reference-edit
   path, `qwen_image_zero_cond_t=true`), weights on a RunPod network volume so a
   cold worker does not download ~20 GB.
2. Parity: run the `qwen-edit-5090.md` corpus (two instructions, seed 90908, dense
   20 steps, PNG) on the worker and the local lane; require global SSIM >= 0.99 vs
   the local dense output and identical pixels on repeat. Only then mark the
   candidate `quality: 1.0`.
3. Create a separate endpoint (4090 only, flashboot on, workersMax 2, idle 30 s)
   rather than reusing `tlofa06vj7iab7`, so ra2 workers keep their warm weights.
4. Add a `qwen-edit` remote candidate to `frontier/seeds.json` with the measured
   p50/p95 and point the edit route's overflow at the adapter with
   `FRONTIER_WORKLOAD=qwen-edit`; leave `FRONTIER_ROUTING=0` until the ledger has
   at least 5 remote jobs, then enable.
5. Rollback: remove the edit overflow env line and restart the gateway; the
   endpoint can stay at workersMin 0 (no idle cost).
