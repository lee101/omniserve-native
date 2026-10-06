#!/usr/bin/env bash
# Forecast a time series through the ManifoldGen API (Chronos-2).
#
#   tools/forecast.sh --values 12,14,13,17,19,18,23
#   tools/forecast.sh --values-file history.csv --horizon 30 --quantiles 0.1,0.5,0.9
#
# --values-file takes one number per line, or a comma-separated single line.
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_manifold.sh"

usage() {
  sed -n '2,8p' "${BASH_SOURCE[0]}" | cut -c3-
  cat <<'EOF'

Options
      --values LIST      comma-separated history, oldest first
      --values-file F    read the history from a file instead
      --horizon N        steps to predict (default 12)
      --quantiles LIST   comma-separated quantile levels
      --json             print the raw API response
  -h, --help             this text
EOF
}

VALUES=""
VALUES_FILE=""
HORIZON=12
QUANTILES=""
AS_JSON=0

while (( $# )); do
  case "$1" in
    --values) VALUES="${2:?--values needs a list}"; shift 2 ;;
    --values-file) VALUES_FILE="${2:?--values-file needs a path}"; shift 2 ;;
    --horizon) HORIZON="${2:?--horizon needs a number}"; shift 2 ;;
    --quantiles) QUANTILES="${2:?--quantiles needs a list}"; shift 2 ;;
    --json) AS_JSON=1; shift ;;
    -h | --help) usage; exit 0 ;;
    -*) mg_die "unknown option: $1" ;;
    *) break ;;
  esac
done

if [[ -n "$VALUES_FILE" ]]; then
  VALUES="$(tr '\n,' '  ' <"$VALUES_FILE" | tr -s '[:space:]' ',')"
fi
[[ -n "$VALUES" ]] || { usage >&2; mg_die "--values or --values-file is required"; }

# The API takes a real array, so the two list options are built in python.
BODY="$(python3 - "$VALUES" "$HORIZON" "$QUANTILES" <<'PY'
import json, sys

raw, horizon, quantiles = sys.argv[1], int(sys.argv[2]), sys.argv[3]
values = [float(part) for part in raw.replace(",", " ").split()]
body = {"service": "forecast", "values": values, "prediction_length": horizon}
if quantiles:
    body["quantile_levels"] = [float(q) for q in quantiles.replace(",", " ").split()]
print(json.dumps(body))
PY
)"

mg_service "$BODY"

if ((AS_JSON)); then
  cat "$MG_BODY_FILE"
  exit 0
fi

mg_jget result