#!/usr/bin/env bash
# Remove a video background through the ManifoldGen API.
#
#   tools/remove-video-background.sh clip.mp4
#   tools/remove-video-background.sh --background '#ffffff' --keep-audio clip.mp4
#
# Source clips are capped at 30 seconds.
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_manifold.sh"

usage() {
  sed -n '2,7p' "${BASH_SOURCE[0]}" | cut -c3-
  cat <<'EOF'

Options
      --background COLOR   composite the subject onto this colour
      --keep-audio         keep the original audio track
      --max-quality        force the paid lane instead of the cheap one
      --url URL            use this public URL instead of uploading a file
  -o, --out FILE           output path
      --json               print the raw API response
  -h, --help               this text
EOF
}

BACKGROUND=""
KEEP_AUDIO=""
MAX_QUALITY=""
URL=""
OUT=""
AS_JSON=0

while (( $# )); do
  case "$1" in
    --background) BACKGROUND="${2:?--background needs a colour}"; shift 2 ;;
    --keep-audio) KEEP_AUDIO=true; shift ;;
    --max-quality) MAX_QUALITY=true; shift ;;
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
  [[ -n "$FILE" ]] || { usage >&2; mg_die "a video file or --url is required"; }
  FILE="$(mg_require_file "$FILE")"
  mg_report "uploading ${FILE##*/}"
  URL="$(mg_public_url "$FILE")"
fi

mg_service "$(mg_body \
  service video_background_removal \
  video_url "$URL" \
  background_color "$BACKGROUND" \
  preserve_audio "$KEEP_AUDIO" \
  max_quality "$MAX_QUALITY")"

mg_resolve_job video

if ((AS_JSON)); then
  cat "$MG_BODY_FILE"
  exit 0
fi

path="$(mg_fetch "${OUT:-$(mg_outfile cutout-video mp4)}" \
  job.result.video_url \
  video_url \
  result.video_url)"

printf 'duration=%ss credits=%s\n' \
  "$(mg_field job.result.duration_seconds duration_seconds)" \
  "$(mg_field job.result.charged_usd credits_used cost_usd)" >&2
printf '%s\n' "$path"