#!/usr/bin/env bash
# Call any ManifoldGen service by id, for the tools that have no wrapper yet.
#
#   tools/manifold-service.sh image prompt "a red fox" width 1024 height 1024
#   tools/manifold-service.sh music prompt "lofi beats" duration 60 --job audio
#   tools/manifold-service.sh --services
#
# Arguments are KEY VALUE pairs sent verbatim in the POST /api/service body.
# Numbers and booleans keep their JSON types; an empty value drops the key.
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_manifold.sh"

usage() {
  sed -n '2,10p' "${BASH_SOURCE[0]}" | cut -c3-
  cat <<'EOF'

Options
      --services        list the service ids the server accepts
      --job audio|video follow a queued job to completion (default: auto)
      --artifact PATH   response path to save; repeatable, first match wins
      --prefix NAME     output file prefix (default: the service name)
      --json            print the raw API response and save nothing
  -h, --help            this text
EOF
}

JOB=""
ARTIFACTS=()
PREFIX=""
AS_JSON=0
LIST=0
PAIRS=()

while (( $# )); do
  case "$1" in
    --services) LIST=1; shift ;;
    --job) JOB="${2:?--job needs audio or video}"; shift 2 ;;
    --artifact) ARTIFACTS+=("${2:?--artifact needs a response path}"); shift 2 ;;
    --prefix) PREFIX="${2:?--prefix needs a name}"; shift 2 ;;
    --json) AS_JSON=1; shift ;;
    -h | --help) usage; exit 0 ;;
    --) shift; break ;;
    -*) mg_die "unknown option: $1" ;;
    *) break ;;
  esac
done

if ((LIST)); then
  mg_request GET /api/pricing
  mg_require_ok
  python3 - "$MG_BODY_FILE" <<'PY'
import json, sys

for row in json.load(open(sys.argv[1])).get("pricing", []):
    print(f"{row.get('service', '-'):<26} {row.get('price_cute', 0):<9} {row.get('unit', '-')}")
PY
  exit 0
fi

SERVICE="${1:-}"
[[ -n "$SERVICE" ]] || { usage >&2; mg_die "a service id is required (see --services)"; }
shift
while (( $# )); do
  [[ $# -ge 2 ]] || mg_die "every field needs a value: $1"
  PAIRS+=("$1" "$2")
  shift 2
done

mg_service "$(mg_body service "$SERVICE" ${PAIRS+"${PAIRS[@]}"})"

# An accepted job names its own poll endpoint, so the kind is only a fallback.
if [[ -z "$JOB" && "$MG_STATUS" == 202 ]]; then
  case "$(mg_field result.status_url)" in
    /api/audio-jobs/*) JOB=audio ;;
    *) JOB=video ;;
  esac
fi
[[ -z "$JOB" ]] || mg_resolve_job "$JOB"

if ((AS_JSON)) || ((${#ARTIFACTS[@]} == 0)); then
  cat "$MG_BODY_FILE"
  exit 0
fi

path="$(mg_fetch "${PREFIX:-$SERVICE}.out" "${ARTIFACTS[@]}")"
printf '%s\n' "$path"