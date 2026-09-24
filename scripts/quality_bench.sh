#!/usr/bin/env bash
# Model-quality gate. tools/bench_http.c covers transport throughput; this
# covers whether the served models still answer correctly after a KV-cache,
# sampler, template, pooling, or artifact change.
set -euo pipefail
PORT="${1:-${OMNISERVE_NATIVE_PORT:-8791}}"
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
shift || true

exec python3 "${SCRIPT_DIR}/../tools/quality_bench.py" --port "$PORT" \
    --json "${SCRIPT_DIR}/../performance/quality-last.json" "$@"
