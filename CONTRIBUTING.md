# Contributing

Thanks for improving OmniServe Native.

## Development

```bash
cmake -S . -B build -DWITH_LLAMA=OFF -DWITH_SD=OFF
cmake --build build -j
ctest --test-dir build --output-on-failure
```

Keep the gateway hot path in C. Model-specific Python processes belong behind
the existing HTTP worker boundary. New GPU workloads must participate in
weighted admission, expose health/capacity, and document residency behavior.

For background work, do not use `SIGSTOP` as a preemption mechanism: it retains
VRAM. Save a resumable checkpoint and exit within a bounded grace period.

Never add model weights, training audio, transcripts, manifests, credentials,
machine-local state, or build output. Add tests for routing, framing,
admission, consent gates, and failure fallback as applicable.

## Pull requests

- Explain the serving or cost impact.
- Include the commands used to validate the change.
- Call out new environment variables and external runtimes.
- Preserve third-party attribution and model licenses.

By submitting a contribution, you agree that it is licensed under Apache 2.0.
