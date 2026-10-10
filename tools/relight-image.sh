#!/usr/bin/env bash
# Relight an image through the ManifoldGen API (fal IC-Light v2).
#
#   tools/relight-image.sh --direction left portrait.jpg
#   tools/relight-image.sh --direction top --url https://example.com/room.jpg
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_manifold.sh"

usage() {
  sed -n '2,6p' "${BASH_SOURCE[0]}" | cut -c3-
  cat <<'EOF'

Options
      --direction D   left | right | top | bottom (default left)
      --prompt TEXT   optional relighting description
      --url URL       use this public URL instead of uploading a file
  -o, --out FILE      output path
      --json          print the raw API response
  -h, --help          this text
EOF
}

KIND=left
PROMPT=""
URL=""
OUT=""
AS_JSON=0

while (( $# )); do
  case "$1" in
    --direction) KIND="${2:?--direction needs a value}"; shift 2 ;;
    --prompt) PROMPT="${2:?--prompt needs text}"; shift 2 ;;
    --url) URL="${2:?--url needs a URL}"; shift 2 ;;
    -o | --out) OUT="${2:?--out needs a path}"; shift 2 ;;
    --json) AS_JSON=1; shift ;;
    -h | --help) usage; exit 0 ;;
    -*) mg_die "unknown option: $1" ;;
    *) break ;;
  esac
done

case "$KIND" in
  left | right | top | bottom) ;;
  *) mg_die "--direction must be left, right, top or bottom" ;;
esac

FILE="${1:-}"
if [[ -z "$URL" ]]; then
  [[ -n "$FILE" ]] || { usage >&2; mg_die "an image file or --url is required"; }
  FILE="$(mg_require_file "$FILE")"
  mg_report "uploading ${FILE##*/}"
  URL="$(mg_public_url "$FILE")"
fi

mg_service "$(mg_body service relight kind "$KIND" image_url "$URL" prompt "$PROMPT")"

if ((AS_JSON)); then
  cat "$MG_BODY_FILE"
  exit 0
fi

path="$(mg_fetch "${OUT:-$(mg_outfile relight jpg)}" \
  result.image_url \
  result.image_base64 \
  result.images.0.image_url \
  result.images.0.image_base64 \
  saved_image_url)"

printf 'direction=%s credits=%s\n' \
  "$KIND" "$(mg_field credits_used)" >&2
printf '%s\n' "$path"