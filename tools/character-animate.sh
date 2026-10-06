#!/usr/bin/env bash
# Drive a character image with a motion video through the ManifoldGen API.
#
#   tools/character-animate.sh --image pose.png --motion dance.mp4
#   tools/character-animate.sh --tier fast --image pose.png --motion dance.mp4
#
# The image supplies the identity, the video supplies the movement.
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_manifold.sh"

usage() {
  sed -n '2,8p' "${BASH_SOURCE[0]}" | cut -c3-
  cat <<'EOF'

Options
      --image FILE     character image to animate (required)
      --motion FILE    driving video (required)
      --tier NAME      standard | fast | xfast
      --url URL        public URL for the image instead of uploading
      --motion-url URL public URL for the video instead of uploading
  -o, --out FILE        output path
      --json            print the raw API response
  -h, --help            this text
EOF
}

IMAGE=""
MOTION=""
URL=""
MOTION_URL=""
TIER=""
OUT=""
AS_JSON=0

while (( $# )); do
  case "$1" in
    --image) IMAGE="${2:?--image needs a file}"; shift 2 ;;
    --motion) MOTION="${2:?--motion needs a file}"; shift 2 ;;
    --tier) TIER="${2:?--tier needs a name}"; shift 2 ;;
    --url) URL="${2:?--url needs a URL}"; shift 2 ;;
    --motion-url) MOTION_URL="${2:?--motion-url needs a URL}"; shift 2 ;;
    -o | --out) OUT="${2:?--out needs a path}"; shift 2 ;;
    --json) AS_JSON=1; shift ;;
    -h | --help) usage; exit 0 ;;
    -*) mg_die "unknown option: $1" ;;
    *) break ;;
  esac
done

[[ -n "$IMAGE" && -n "$MOTION" ]] || { usage >&2; mg_die "--image and --motion are both required"; }

if [[ -z "$URL" ]]; then
  mg_report "uploading ${IMAGE##*/}"
  URL="$(mg_public_url "$(mg_require_file "$IMAGE")")"
fi
if [[ -z "$MOTION_URL" ]]; then
  mg_report "uploading ${MOTION##*/}"
  MOTION_URL="$(mg_public_url "$(mg_require_file "$MOTION")")"
fi

mg_service "$(mg_body \
  service character_animation \
  image_url "$URL" \
  video_url "$MOTION_URL" \
  service_tier "$TIER")"

mg_resolve_job video

if ((AS_JSON)); then
  cat "$MG_BODY_FILE"
  exit 0
fi

path="$(mg_fetch "${OUT:-$(mg_outfile animate mp4)}" \
  job.result.video_url \
  video_url \
  result.video_url)"

printf 'duration=%ss credits=%s\n' \
  "$(mg_field job.result.duration_seconds duration_seconds)" \
  "$(mg_field job.result.charged_usd credits_used cost_usd)" >&2
printf '%s\n' "$path"