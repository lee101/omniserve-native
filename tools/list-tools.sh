#!/usr/bin/env bash
# Show the ManifoldGen server these CLIs talk to, the services it offers, and
# which of the local lanes are actually up.
#
#   tools/list-tools.sh          # server, lanes, service catalogue
#   tools/list-tools.sh --services
#   tools/list-tools.sh --lanes
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_manifold.sh"

usage() {
  sed -n '2,7p' "${BASH_SOURCE[0]}" | cut -c3-
  cat <<'EOF'

Options
      --services   only the service catalogue and its prices
      --lanes      only the upstream lane reachability
      --voice      only the voice models
  -h, --help       this text
EOF
}

WHAT=all
while (( $# )); do
  case "$1" in
    --services | --lanes | --voice) WHAT="${1#--}"; shift ;;
    -h | --help) usage; exit 0 ;;
    *) mg_die "unknown option: $1" ;;
  esac
done

mg_ensure_server

server() {
  mg_report "server:  $(mg_api)"
  mg_request GET /api/health
  mg_require_ok
  mg_report "health:  $(mg_get status)"
  mg_ensure_key
  mg_request GET /api/auth/session
  if [[ "$MG_STATUS" == 200 ]]; then
    mg_report "account: $(mg_field user.email wallet_address) credits=$(mg_field user.credits)"
  else
    mg_report "account: the API key was rejected ($(mg_field error))"
  fi
}

# Each lane is a separate process the ManifoldGen server proxies to; a refused
# connection here is the reason a tool will fail before it reaches a provider.
lanes() {
  printf '\n%-28s %s\n' LANE ADDRESS >&2
  while read -r name url; do
    [[ -n "$url" ]] || continue
    printf '%-28s %s\n' "$name" "$(curl -s -o /dev/null -m 2 -w '%{http_code}' "$url" || echo down)" >&2
  done <<EOF
ra2 ${RA2_BACKEND_URL:-http://127.0.0.1:8792}/health
omniserve ${OMNISERVE_NATIVE_URL:-http://127.0.0.1:8791}/health
zimage ${ZIMAGE_BACKEND_URL:-http://127.0.0.1:8100}/health
tts ${TTS_BACKEND_URL:-http://127.0.0.1:9083}/health
h3-cog ${H3_LOCAL_COG_URL:-http://127.0.0.1:18089}/health
EOF
}

services() {
  mg_request GET /api/pricing
  mg_require_ok
  printf '\n%-26s %-9s %s\n' SERVICE CREDITS UNIT >&2
  python3 - "$MG_BODY_FILE" <<'PY'
import json, sys

document = json.load(open(sys.argv[1]))
for row in document.get("pricing", []):
    print(f"{row.get('service', '-'):<26} {row.get('price_cute', 0):<9} {row.get('unit', '-')}")
PY
}

voices() {
  mg_request GET /api/voice/models
  mg_require_ok
  printf '\n' >&2
  python3 - "$MG_BODY_FILE" <<'PY'
import json, sys
for model in json.load(open(sys.argv[1])).get("models", []):
    print(f"{model['id']:<24} {', '.join(model.get('formats') or [])}")
    if model.get("voices"):
        print(f"{'':<24} voices: {', '.join(model['voices'])}")
PY
}

case "$WHAT" in
  services) services ;;
  lanes) lanes ;;
  voice) voices ;;
  all)
    server
    lanes
    services
    voices
    ;;
esac