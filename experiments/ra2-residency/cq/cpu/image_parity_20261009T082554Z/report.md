# OmniServe Image Parity

- reference: `experiments/ra2-residency/cq_ref/image_reference_20261009T080243Z`
- candidate: `http://127.0.0.1:8794`
- passed: `True`
- semantic scorer: `True`
- median latency ratio (inference_time_ms): `0.738` (maximum `1.500`)

| Case | Size | Reference s | Candidate s | Candidate $/MP | CLIP image | Prompt Δ | Aesthetic Δ | Result |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `product_glass_robot` | 512x512 | 295.16 | 303.05 | 0.439666 | 0.914 | -0.008 | 0.047 | PASS |
| `portrait_hands` | 512x512 | 342.43 | 277.22 | 0.402291 | 0.994 | 0.001 | -0.041 | PASS |
| `typography_sign` | 512x512 | 322.00 | 213.02 | 0.309144 | 0.728 | 0.018 | 0.013 | PASS |
| `poster_text` | 512x512 | 430.19 | 204.55 | 0.296751 | 0.974 | 0.029 | 0.097 | PASS |
