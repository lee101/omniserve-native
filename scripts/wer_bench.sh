#!/usr/bin/env bash
# ASR accuracy gate. quality_bench.sh covers whether served models still answer;
# this covers whether a change to the ASR serving path moved word error rate.
#
#   scripts/wer_bench.sh corpus.jsonl                      # record/refresh
#   scripts/wer_bench.sh corpus.jsonl --baseline old.json  # gate a change
set -euo pipefail
MANIFEST="${1:?usage: wer_bench.sh <manifest.jsonl> [wer_bench.py args...]}"
shift
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PORT="${OMNISERVE_NATIVE_PORT:-8791}"

exec python3 "${SCRIPT_DIR}/../tools/wer_bench.py" \
    --manifest "$MANIFEST" \
    --url "http://127.0.0.1:${PORT}/v1/audio/transcriptions" \
    --out "${SCRIPT_DIR}/../performance/wer-last.json" "$@"
