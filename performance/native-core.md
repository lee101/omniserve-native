# Native Profile Report

- Command: `./build-profile/onative_tests`
- Callgrind exit: `0` in `1.83s`
- Massif exit: `0` in `1.92s`
- Callgrind data: `/nvme0n1-disk/code/omniserve-native/performance/native-core.callgrind.out`
- Massif data: `/nvme0n1-disk/code/omniserve-native/performance/native-core.massif.out`

## CPU Hotspots

| Ir | % | Location | Binary |
| ---: | ---: | --- | --- |
| 158,313 | 24.40 | `tests/test_core.c:main` | `/nvme0n1-disk/code/omniserve-native/build-profile/onative_tests` |
| 105,146 | 16.21 | `src/ohttp.c:reactor_main` | `/nvme0n1-disk/code/omniserve-native/build-profile/onative_tests` |
| 101,767 | 15.69 | `src/ohttp.c:worker_main` | `/nvme0n1-disk/code/omniserve-native/build-profile/onative_tests` |
| 68,822 | 10.61 | `src/ojson.c:oj_parse` | `/nvme0n1-disk/code/omniserve-native/build-profile/onative_tests` |
| 64,410 | 9.93 | `src/ohttp.c:parse_request` | `/nvme0n1-disk/code/omniserve-native/build-profile/onative_tests` |
| 55,041 | 8.48 | `tests/test_core.c:echo_handler` | `/nvme0n1-disk/code/omniserve-native/build-profile/onative_tests` |
| 44,663 | 6.88 | `tests/test_core.c:test_http_server` | `/nvme0n1-disk/code/omniserve-native/build-profile/onative_tests` |
| 28,591 | 4.41 | `src/ojson.c:oj_obj_get` | `/nvme0n1-disk/code/omniserve-native/build-profile/onative_tests` |
| 19,454 | 3.00 | `src/oproxy.c:append_fmt` | `/nvme0n1-disk/code/omniserve-native/build-profile/onative_tests` |
| 16,640 | 2.56 | `src/oproxy.c:oproxy_relay` | `/nvme0n1-disk/code/omniserve-native/build-profile/onative_tests` |

## Heap Peak

- Peak total: `0.14 MB`
- Heap bytes: `0.14 MB`
- Extra heap bytes: `0.00 MB`
- Stack bytes: `0.00 MB`
- Peak snapshot: `32`

| Bytes | % Peak | Stack |
| ---: | ---: | --- |
| 65,536 | 45.88 | `http_roundtrip.constprop.0 (test_core.c:267) -> test_http_server (test_core.c:368) -> main (test_core.c:451)` |
| 65,536 | 45.88 | `oproxy_target_relay (oproxy.c:754) -> echo_handler (test_core.c:203) -> worker_main (ohttp.c:412) -> start_thread (pthread_create.c:447)` |
| 8,432 | 5.90 | `reactor_main (ohttp.c:496) -> start_thread (pthread_create.c:447) -> clone (clone.S:100)` |
| 3,176 | 2.22 | `unknown` |

## Source Hotspots

### `src/ohttp.c`

| Line | Ir | % | Code |
| ---: | ---: | ---: | --- |
| 1 | 127,373 | 19.63 | `#define _GNU_SOURCE` |

### `src/ojson.c`

| Line | Ir | % | Code |
| ---: | ---: | ---: | --- |
| 42 | 12,184 | 1.88 | `for (; p->pos < (int)len; p->pos++) {` |
| 53 | 10,834 | 1.67 | `if (c == '\\' && p->pos + 1 < (int)len) p->pos++;` |
| 107 | 9,194 | 1.42 | `for (int i = tok + 1; i < ntoks; i++) {` |
| 62 | 9,110 | 1.40 | `switch (c) {` |
| 109 | 8,492 | 1.31 | `if (toks[i].parent < depth_parent && toks[i].parent != -1) break;` |
| 43 | 7,896 | 1.22 | `char c = js[p->pos];` |
| 44 | 7,896 | 1.22 | `if (c == '"') {` |
| 108 | 6,720 | 1.04 | `if (toks[i].parent == depth_parent) return i;` |
| 60 | 4,216 | 0.65 | `for (; p.pos < (int)len; p.pos++) {` |
| 16 | 2,184 | 0.34 | `oj_tok *t = &toks[p->next_tok++];` |

### `src/osched.c`

- No source line attribution. Rebuild with debug info, e.g. `cmake --preset prof`.

## Notes

- CPU line attribution comes from `callgrind_annotate` source annotations.
- Exact per-line heap attribution is not available here; the heap section reports peak allocation stacks instead.
- Source annotations depend on debug info and may drop lines in stripped or highly optimized builds.
