#!/usr/bin/env bash
# Edit an image through the ManifoldGen API.
#
#   tools/edit-image.sh --prompt "make it winter" summer.jpg
#   tools/edit-image.sh --prompt "a red bicycle" --mask mask.png photo.jpg
#
# Without a mask this is a RA2 (Qwen) reference edit: the source is sent to the
# RA2 lane with the instruction. With a mask it is the Image Editor's masked
# regeneration, where the mask luminance selects the region to repaint.
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_manifold.sh"

usage() {
  sed -n '2,11p' "${BASH_SOURCE[0]}" | cut -c3-
  cat <<'EOF'

Options
      --prompt TEXT   the edit instruction (required)
      --mask FILE     repaint only this region (white = repaint)
      --url URL       use this public URL instead of uploading the source
      --mask-url URL  public URL for the mask instead of uploading it
      --size WxH      output size, default 1024x1024
      --seed N        seed
  -o, --out FILE      output path
      --json          print the raw API response
  -h, --help          this text
EOF
}

PROMPT=""
MASK=""
URL=""
MASK_URL=""
SIZE=1024x1024
SEED=""
OUT=""
AS_JSON=0

while (( $# )); do
  case "$1" in
    --prompt) PROMPT="${2:?--prompt needs text}"; shift 2 ;;
    --mask) MASK="${2:?--mask needs a file}"; shift 2 ;;
    --url) URL="${2:?--url needs a URL}"; shift 2 ;;
    --mask-url) MASK_URL="${2:?--mask-url needs a URL}"; shift 2 ;;
    --size) SIZE="${2:?--size needs WxH}"; shift 2 ;;
    --seed) SEED="${2:?--seed needs a number}"; shift 2 ;;
    -o | --out) OUT="${2:?--out needs a path}"; shift 2 ;;
    --json) AS_JSON=1; shift ;;
    -h | --help) usage; exit 0 ;;
    -*) mg_die "unknown option: $1" ;;
    *) break ;;
  esac
done

[[ -n "$PROMPT" ]] || { usage >&2; mg_die "--prompt is required"; }
[[ "$SIZE" =~ ^[0-9]+x[0-9]+$ ]] || mg_die "--size must look like 1024x1024"

FILE="${1:-}"
if [[ -z "$URL" ]]; then
  [[ -n "$FILE" ]] || { usage >&2; mg_die "an image file or --url is required"; }
  FILE="$(mg_require_file "$FILE")"
  mg_report "uploading ${FILE##*/}"
  URL="$(mg_public_url "$FILE")"
fi

if [[ -n "$MASK" || -n "$MASK_URL" ]]; then
  if [[ -z "$MASK_URL" ]]; then
    mg_report "uploading ${MASK##*/}"
    MASK_URL="$(mg_public_url "$MASK")"
  fi
  mg_request POST /api/image-editor/edit "$(mg_body \
    image_url "$URL" mask_url "$MASK_URL" prompt "$PROMPT")"
  mg_require_ok
  CANDIDATES=(result.image_url result.image_base64 image_url)
else
  mg_service "$(mg_body \
    service image-edit \
    image_url "$URL" \
    prompt "$PROMPT" \
    width "${SIZE%x*}" \
    height "${SIZE#*x}" \
    seed "$SEED")"
  CANDIDATES=(result.image_base64 result.images.0.image_base64 result.image_url
    result.images.0.image_url saved_image_url)
fi

if ((AS_JSON)); then
  cat "$MG_BODY_FILE"
  exit 0
fi

path="$(mg_fetch "${OUT:-$(mg_outfile edit webp)}" "${CANDIDATES[@]}")"
printf 'engine=%s credits=%s\n' \
  "$(mg_field result.engine engine)" \
  "$(mg_field credits_used)" >&2
printf '%s\n' "$path"