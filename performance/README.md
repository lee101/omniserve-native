# Native performance profiles

`native-core.md` is a reproducible Callgrind and Massif report for the C HTTP,
proxy, JSON, and scheduler test workload. Its raw inputs are retained beside it.

The optimized, symbolized profiling build is intentionally separate from the
production binary:

```bash
cmake -S . -B build-profile \
  -DCMAKE_BUILD_TYPE=RelWithDebInfo \
  -DONATIVE_PROFILE=ON \
  -DONATIVE_NATIVE_ARCH=ON \
  -DONATIVE_LTO=OFF \
  -DWITH_LLAMA=OFF -DWITH_SD=OFF
cmake --build build-profile -j 8
ctest --test-dir build-profile --output-on-failure
```

Recreate the report with:

```bash
/home/administrator/code/dotfiles/tools/native-prof-report \
  --out performance/native-core.md \
  --prefix performance/native-core \
  --source-file src/ohttp.c \
  --source-file src/ojson.c \
  --source-file src/osched.c \
  -- ./build-profile/onative_tests
```

The profile's peak heap is about 0.14 MB, almost entirely two reusable 64 KiB
HTTP/proxy buffers. No per-request heap hotspot justified a production C change.
Loopback measurements on the running production service were:

| Path | Workload | Result |
| --- | --- | ---: |
| `/health` | 50,000 requests, concurrency 32 | 126,906 requests/s; p99 493 us |
| Legacy ModernBERT feature extraction | 200 requests, concurrency 1 | 27 requests/s; mean 36.6 ms |
| OpenAI ModernBERT embeddings | 200 requests, concurrency 1 | 27 requests/s; mean 36.5 ms |

ModernBERT latency is model inference dominated. The transport and test profiles
showed no allocation churn worth trading correctness or maintainability for, so
no additional profile-only code changes were made; the production restart was
limited to the API compatibility and routing deployment.
