# Z-Image cost and quality ablation — RTX 5090

At `$1000/month` and 730 hours/month, the host costs `$0.00038052/s`. Matching
`$0.005/MP` therefore requires no more than `13.14 inference-seconds/MP` at full
utilization. Queue time is reported separately because it affects request SLOs,
but not the number of GPU-seconds purchased per month.

| Variant | Size | Inference | Wall | Cost/image | Cost/MP | Target |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| new prompt, 4 steps | 1024² | 7.841 s | 22.686 s | $0.002984 | $0.002845 | pass |
| new prompt, 9 steps | 1024² | 15.594 s | 29.536 s | $0.005934 | $0.005659 | fail |
| exact final-step replay | 1024² | 3.013 s | 16.290 s | $0.001146 | $0.001093 | pass |
| new prompt, 4 steps | 512² | 4.406 s | 15.990 s | $0.001677 | $0.006396 | fail |

If the full host cost is charged to image generation, the 1024² four-step lane
must be productively occupied at least `56.9%` of the month to stay below the
target (`measured cost / $0.005`). At 100% occupancy it produces about `481.4
MP/hour`, earns `$2.41/hour`, and leaves `$1.04/hour` before network, storage,
support, and failed-request costs. Nine steps and 512² four-step generation do
not break even even at 100% occupancy. Exact replay needs only `21.9%` occupancy,
but repeat traffic is a separate, workload-dependent market.

The current production profile streams parameters from CPU, caps the native
backend at 2 GiB VRAM, and uses a tiled VAE because the 5090 is shared with live
LLM, search, image, and worker processes. It is not a clean dedicated-image
benchmark. A two-point resolution fit estimates break-even around `0.372 MP`
(roughly `610x610` square), but that estimate needs a quiet-host sweep.

The exact-replay result is byte-identical to the full nine-step result. Four
steps materially change pixels but retained the requested subjects and passed
the corruption/entropy checks in the three reviewed cases. At 1024² its global
SSIM against nine steps was `0.9525`; the two 512² cases measured `0.7148` and
`0.9906`. This is evidence of usable retention, not proof of equal aesthetic or
prompt quality; semantic scoring or blinded human review remains required.

The newly staged `exact_prompt_result_cache` is not included in this table. It
returns the already encoded byte-identical result and should remove the last
denoise, VAE, and encoding work, but it has not been loaded on the traffic-serving
process yet.
