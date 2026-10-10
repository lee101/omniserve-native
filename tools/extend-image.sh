#!/usr/bin/env bash
# Outpaint an image through the ManifoldGen API.
#
#   tools/extend-image.sh --zoom 25 room.jpg
#   tools/extend-image.sh --left 0.2 --right 0.2 --top 0.1 photo.webp
#
# Every margin is a fraction of the source dimension, so --left 0.2 widens the
# frame by a fifth on that side.
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_manifold.sh"

usage() {
  sed -n '2,9p' "${BASH_SOURCE[0]}" | cut -c3-
  cat <<'EOF'

Options
      --left F      expand the left edge by this fraction
      --right F     expand the right edge
      --top F       expand the top edge
      --bottom F    expand the bottom edge
      --zoom F      zoom-out percentage, an alternative to the four margins
      --url URL     use this public URL instead of uploading a file
  -o, --out FILE    output path
      --json        print the raw API response
  -h, --help        this text
EOF
}

LEFT=""
RIGHT=""
TOP=""
BOTTOM=""
ZOOM=""
URL=""
OUT=""
AS_JSON=0

while (( $# )); do
  case "$1" in
    --left) LEFT="${2:?--left needs a fraction}"; shift 2 ;;
    --right) RIGHT="${2:?--right needs a fraction}"; shift 2 ;;
    --top) TOP="${2:?--top needs a fraction}"; shift 2 ;;
    --bottom) BOTTOM="${2:?--bottom needs a fraction}"; shift 2 ;;
    --zoom) ZOOM="${2:?--zoom needs a percentage}"; shift 2 ;;
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

[[ -n "$LEFT$RIGHT$TOP$BOTTOM$ZOOM" ]] || mg_die "set at least one of --left --right --top --bottom --zoom"

mg_service "$(mg_body \
  service extend-image \
  image_url "$URL" \
  expand_left "$LEFT" \
  expand_right "$RIGHT" \
  expand_top "$TOP" \
  expand_bottom "$BOTTOM" \
  zoom_out_percentage "$ZOOM")"

if ((AS_JSON)); then
  cat "$MG_BODY_FILE"
  exit 0
fi

path="$(mg_fetch "${OUT:-$(mg_outfile extend png)}" \
  result.image_url \
  result.image_base64 \
  result.images.0.image_base64 \
  result.images.0.image_url \
  saved_image_url)"

printf 'credits=%s\n' "$(mg_field credits_used)" >&2
printf '%s\n' "$path"