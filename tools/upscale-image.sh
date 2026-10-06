#!/usr/bin/env bash
# Upscale an image 2x through the ManifoldGen API.
#
#   tools/upscale-image.sh photo.jpg
#   tools/upscale-image.sh --url https://example.com/photo.jpg
#
# The lane is fal's creative upscaler (fixed 2x scale, PNG out).
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_manifold.sh"

usage() {
  sed -n '2,7p' "${BASH_SOURCE[0]}" | cut -c3-
  cat <<'EOF'

Options
      --url URL      use this public URL instead of uploading a file
  -o, --out FILE     output path
      --json         print the raw API response
  -h, --help         this text
EOF
}

URL=""
OUT=""
AS_JSON=0

while (( $# )); do
  case "$1" in
    --url) URL="${2:?--url needs a URL}"; shift 2 ;;
    -o | --out) OUT="${2:?--out needs a path}"; shift 2 ;;
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

mg_service "$(mg_body service upscale-image image_url "$URL")"

if ((AS_JSON)); then
  cat "$MG_BODY_FILE"
  exit 0
fi

path="$(mg_fetch "${OUT:-$(mg_outfile upscale png)}" \
  result.image_url \
  result.image_base64 \
  result.images.0.image_base64 \
  result.images.0.image_url \
  saved_image_url)"

printf 'credits=%s\n' "$(mg_field credits_used)" >&2
printf '%s\n' "$path"