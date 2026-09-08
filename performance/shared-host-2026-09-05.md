# Shared-host optimization: first pass

## Scope and observed state

Read-only production inspection; no deployments, restarts, evictions, model
loads, downloads, GPU benchmarks, paid requests, or Runpod provisioning.
Repositories resolved under `/nvme0n1-disk/code`, including `manifoldgen-site`.
CuteDSL-site, Netwrck and Manifoldgen have pre-existing worktree changes;
their application code was not modified in this pass.

At 05:39 UTC, one RTX 5090 reported 31.84 GiB total, 12.38 GiB free,
18.97 GiB used and 76% utilization. Initial free VRAM was roughly 14 GiB:
this is a variable shared budget, not an idle dedicated GPU.
The 72-logical-CPU host had load averages 26.43/27.37/22.19 and 137.68 GiB
available RAM. Swap was effectively full (80 GiB); memory PSI avg10 was
zero, so swap occupancy alone does not establish active thrashing.
NVMe free space was 30.50 GiB; avoid large conversion staging there.

Gateway-only quality checks: 4/4 passed on port 8791. `/status` reports
`gemma-roleplay-v2-q8_0`, CPU placement, q8_0 KV, flash attention enabled.
This does not measure roleplay quality, inference latency or upstream health.

## Candidate model and capacity

The existing `G4-MEROMERO-V2-31B-IQ4_XS.gguf` is 16,862,233,024 bytes
(15.704 GiB). It is a candidate inferred from repo documentation, not a
confirmed identification of the user's new model.
With an **unmeasured** 2 GiB runtime/KV allowance and 2 GiB reserve,
the snapshot gives 10.385 GiB candidate budget versus 17.704 GiB required:
the preflight correctly exits 2, short by 7.319 GiB.
A dedicated 24 GiB budget passes this simple estimate, but actual runtime
peak, prefill workspace and context/concurrency still need validation.

The README already documents a historical `NGL=20`, one-context partial
offload profile (~5.8 GiB GPU allocation). It was not rebenchmarked or deployed
here. The active gateway's smaller Q8 artifact is not the 31B artifact.
Do not increase offload or deploy NVFP4 based on nominal weight size alone.

## Implemented and measured

CuteDSL quantile interpolation no longer reads tensor scalars to determine
interpolation indices/fractions. Float32 metadata rounding and tensor
arithmetic are preserved. CPU reference/candidate microbenchmark:

| Measurement | Reference | Candidate |
|---|---:|---:|
| Median round mean, microseconds | 212.352 | 120.765 |
| Relative speed | 1.00x | 1.76x |

PyTorch 2.8.0+cu128, CPU only, one thread, FP32 `[8,5,64]`, three interpolated
quantiles, 20 warmups, seven alternating-order rounds of 200 calls.
Reference round means: 218.746, 210.654, 205.238, 212.352, 212.262, 214.129, 222.077.
Candidate: 118.198, 120.903, 116.987, 120.806, 123.028, 120.765, 120.027.
Exact parity checked. No GPU or whole-service speedup is claimed.

## Next experiment gates

1. Confirm exact roleplay checkpoint and whether 24 GB means weight size,
   dedicated-card capacity, or total runtime footprint.
2. Obtain a canary window/explicit residency budget before model loading.
   Preserve Netwrck search and active image/audio workers; do not kill idle-looking
   GPU processes. Reuse the existing cooperative image/text handoff.
3. Compare existing Q8 and candidate IQ4/NVFP4 only on supported runtimes.
   Record artifact hash, commit, driver, backend, NGL, KV type, context,
   concurrency, CPU threads, cold/warm state and host preflight per run.
4. Run fixed short/long and multi-turn roleplay prompts. Measure TTFT,
   decode tokens/s, p50/p95 end-to-end latency, failures, peak VRAM and RAM.
   Human-blind score persona retention, continuity, instruction following
   and repetition. The existing arithmetic quality suite alone is insufficient.
5. Test mixed chat/image/STT/search traffic on an isolated canary, then a small
   approved production canary. Reject quality failures and OOMs; require
   repeated A/B latency gains greater than observed run-to-run variation.
6. For CuteDSL-site, Netwrck/ebank and Manifoldgen, identify actual upstream
   routes before testing: port 8080 belongs to an unrelated voice service,
   Netwrck production is on 8124, and 8776 is development. Benchmark backend
   queue time separately from model time, then frontend LCP/INP and JS errors.
   No site speed improvements have been measured or deployed in this pass.
7. Runpod inventory is still uninspected. Use the relevant provider's read-only
   inventory before any launch; create only an approved bounded auto-stop job.
   Do not clean up existing unrelated pods or change their pool limits.

## Reproduce without model loads

```bash
python3 -m unittest tests/test_shared_host_preflight.py
python3 tools/shared_host_preflight.py --disk-path /nvme0n1-disk
python3 tools/shared_host_preflight.py --disk-path /nvme0n1-disk \
  --model /nvme0n1-disk/models/omniserve-native/G4-MEROMERO-V2-31B-IQ4_XS.gguf \
  --runtime-gib 2 --reserve-gib 2 --target-gib 24
python3 tools/quality_bench.py --suite gateway --timeout 5
```

Do not use `--update-baseline` to make a candidate's regression pass.
