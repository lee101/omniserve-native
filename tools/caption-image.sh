#!/usr/bin/env bash
# Caption an image through the ManifoldGen API.
#
#   tools/caption-image.sh photo.jpg
#   tools/caption-image.sh --url https://example.com/photo.jpg
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_manifold.sh"

usage() {
  sed -n '2,6p' "${BASH_SOURCE[0]}" | cut -c3-
  cat <<'EOF'

Options
      --url URL     use this public URL instead of uploading a file
      --json        print the raw API response
  -h, --help        this text
EOF
}

URL=""
AS_JSON=0

while (( $# )); do
  case "$1" in
    --url) URL="${2:?--url needs a URL}"; shift 2 ;;
    --json) AS_JSON=1; shift ;;
    -h | --help) usage; exit 0 ;;
    -*) mg_die "unknown option: $1" ;;
    *) break ;;
  esac
done

FILE="${1:-}"
if [[ -z "$URL" ]]; then
  [[ -n "$FILE" ]] || { usage >&2; mg_die "an image file or --url is required"; }
  FILE="$(mg_require_file "$FILE")"
  mg_report "uploading ${FILE##*/}"
  URL="$(mg_public_url "$FILE")"
fi

mg_service "$(mg_body service caption image_url "$URL")"

if ((AS_JSON)); then
  cat "$MG_BODY_FILE"
  exit 0
fi

python3 -c 'import json,sys; d=json.load(sys.stdin); r=d.get("result") or {}; \
print(r.get("caption") or r.get("text") or json.dumps(r, indent=2))' <"$MG_BODY_FILE"