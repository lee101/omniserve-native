#!/usr/bin/env bash
# Chat with the site's Gemma lane through the ManifoldGen API.
#
#   tools/chat.sh "explain EasyCache in one sentence"
#   echo "long prompt" | tools/chat.sh -
#
# A "-" prompt reads the message from stdin so a long prompt can be piped in.
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_manifold.sh"

usage() {
  sed -n '2,7p' "${BASH_SOURCE[0]}" | cut -c3-
  cat <<'EOF'

Options
      --system TEXT   system message to send first
      --max-tokens N  response cap (default 1024)
      --temperature F sampling temperature (default 0.7)
      --json          print the raw API response
  -h, --help          this text
EOF
}

SYSTEM=""
MAX_TOKENS=1024
TEMPERATURE=0.7
AS_JSON=0

while (( $# )); do
  case "$1" in
    --system) SYSTEM="${2:?--system needs text}"; shift 2 ;;
    --max-tokens) MAX_TOKENS="${2:?--max-tokens needs a number}"; shift 2 ;;
    --temperature) TEMPERATURE="${2:?--temperature needs a number}"; shift 2 ;;
    --json) AS_JSON=1; shift ;;
    -h | --help) usage; exit 0 ;;
    -*) mg_die "unknown option: $1" ;;
    *) break ;;
  esac
done

PROMPT="${*:-}"
if [[ "$PROMPT" == "-" ]]; then
  PROMPT="$(cat)"
elif [[ -z "$PROMPT" && ! -t 0 ]]; then
  PROMPT="$(cat)"
fi
[[ -n "$PROMPT" ]] || { usage >&2; mg_die "a prompt is required (or pipe one in)"; }

BODY="$(python3 - "$SYSTEM" "$PROMPT" "$MAX_TOKENS" "$TEMPERATURE" <<'PY'
import json, sys

system, prompt, max_tokens, temperature = sys.argv[1], sys.argv[2], int(sys.argv[3]), float(sys.argv[4])
messages = []
if system:
    messages.append({"role": "system", "content": system})
messages.append({"role": "user", "content": prompt})
print(json.dumps({
    "service": "gemma4",
    "messages": messages,
    "max_tokens": max_tokens,
    "temperature": temperature,
}))
PY
)"

mg_service "$BODY"

if ((AS_JSON)); then
  cat "$MG_BODY_FILE"
  exit 0
fi

python3 -c 'import json,sys; d=json.load(sys.stdin); r=d.get("result") or {}; \
print(r.get("response") or r.get("text") or r.get("content") or json.dumps(r, indent=2))' <"$MG_BODY_FILE"