# OmniServe Image Parity

- reference: `experiments/ra2-residency/cq_ref/image_reference_20261009T080243Z`
- candidate: `http://127.0.0.1:8794`
- passed: `True`
- semantic scorer: `True`
- median latency ratio (inference_time_ms): `0.617` (maximum `1.500`)

| Case | Size | Reference s | Candidate s | Candidate $/MP | CLIP image | Prompt Δ | Aesthetic Δ | Result |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `product_glass_robot` | 512x512 | 295.16 | 202.52 | 0.293798 | 0.899 | -0.014 | 0.050 | PASS |
| `portrait_hands` | 512x512 | 342.43 | 197.29 | 0.286276 | 0.994 | -0.001 | -0.044 | PASS |
| `typography_sign` | 512x512 | 322.00 | 207.23 | 0.300733 | 0.805 | 0.041 | 0.134 | PASS |
| `poster_text` | 512x512 | 430.19 | 212.84 | 0.308754 | 0.957 | 0.024 | 0.131 | PASS |
