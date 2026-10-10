#!/usr/bin/env bash
# Remove an image background through the ManifoldGen API.
#
#   tools/remove-background.sh photo.jpg
#   tools/remove-background.sh --background '#ffffff' chair.webp
#   tools/remove-background.sh --url https://example.com/chair.webp
#
# The endpoints take a public http(s) URL and reject base64, so a local file is
# uploaded to the site's R2 bucket first. The native lane runs on omniserve and
# falls back to fal BiRefNet when it is not up.
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_manifold.sh"

usage() {
  sed -n '2,10p' "${BASH_SOURCE[0]}" | cut -c3-
  cat <<'EOF'

Options
      --background COLOR  also return the replaced backdrop; any CSS colour
      --url URL           use this public URL instead of uploading a file
  -o, --out FILE          output path
      --json              print the raw API response instead of the file path
  -h, --help              this text
EOF
}

BACKGROUND=""
URL=""
OUT=""
AS_JSON=0

while (( $# )); do
  case "$1" in
    --background) BACKGROUND="${2:?--background needs a colour}"; shift 2 ;;
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

if [[ -n "$BACKGROUND" ]]; then
  # The Image Editor route asks the native worker for the cutout and the
  # replaced backdrop in one pass, so there is only one call to make.
  mg_request POST /api/image-editor/background "$(mg_body image_url "$URL")"
  mg_require_ok
  target="${OUT:-$(mg_outfile cutout webp)}"
  path="$(mg_fetch "$target" result.cutout.data_url result.cutout.url image_url)"
  backdrop="$(mg_fetch "$(mg_outfile backdrop webp)" result.background.data_url result.background.url || true)"
  printf '%s\n' "$path"
  [[ -n "$backdrop" ]] && printf '%s\n' "$backdrop"
  exit 0
fi

mg_request POST /api/studio/remove-background "$(mg_body image_url "$URL")"
mg_require_ok

if ((AS_JSON)); then
  cat "$MG_BODY_FILE"
  exit 0
fi

target="${OUT:-$(mg_outfile cutout png)}"
path="$(mg_fetch "$target" image_url result.image_url result.image_base64)"
printf 'backend=%s credits=%s\n' \
  "$(mg_field backend)" \
  "$(mg_field credits_used)" >&2
printf '%s\n' "$path"