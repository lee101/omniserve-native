#!/usr/bin/env bash
set -euo pipefail
PORT="${1:-8791}"
N="${2:-100000}"
C="${3:-32}"
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
BENCH="${ONATIVE_BENCH:-${SCRIPT_DIR}/../build/onative_bench}"

payload='{"messages":[{"role":"user","content":"hi"}],"max_tokens":8}'

if [[ ! -x "$BENCH" ]]; then
  echo "missing $BENCH; run: cmake --build build --target onative_bench" >&2
  exit 1
fi

echo "== /health persistent-socket throughput ($N requests, $C connections) =="
"$BENCH" "$PORT" "$N" "$C" /health

echo "== single chat latency =="
curl -s -o /dev/null -w 'chat %{http_code} %{time_total}s\n' "localhost:$PORT/v1/chat/completions" -d "$payload" || true

echo "== streaming first-token =="
curl -sN "localhost:$PORT/v1/chat/completions" -d '{"messages":[{"role":"user","content":"count to five"}],"max_tokens":32,"stream":true}' | head -3 || true
