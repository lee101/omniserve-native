#!/usr/bin/env bash
# Swap the characters in a video through the ManifoldGen API.
#
#   tools/character-swap.sh --video source.mp4 --reference face.png
#   tools/character-swap.sh --kind recast --video cast.mp4 --reference a.png --reference b.png
#
# --kind picks the lane: exact (per-person faces), recast (rebuild the shot from
# reference photos) or lora (train-and-swap on the RunPod LoRA worker).
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_manifold.sh"

usage() {
  sed -n '2,10p' "${BASH_SOURCE[0]}" | cut -c3-
  cat <<'EOF'

Options
      --video FILE       source clip (required)
      --reference FILE   reference image; repeat for one per person (up to 4)
      --kind NAME        exact | recast | lora (default exact)
      --resolution RES   output resolution, e.g. 768P
      --prompt TEXT      optional scene description
      --url URL          public URL for the video instead of uploading
      --reference-url U  public URL for a reference instead of uploading
  -o, --out FILE         output path
      --json             print the raw API response
  -h, --help             this text
EOF
}

VIDEO=""
KIND=exact
RESOLUTION=""
PROMPT=""
URL=""
OUT=""
AS_JSON=0
REFERENCES=()
REFERENCE_URLS=()

while (( $# )); do
  case "$1" in
    --video) VIDEO="${2:?--video needs a file}"; shift 2 ;;
    --reference) REFERENCES+=("${2:?--reference needs a file}"); shift 2 ;;
    --kind) KIND="${2:?--kind needs a name}"; shift 2 ;;
    --resolution) RESOLUTION="${2:?--resolution needs a value}"; shift 2 ;;
    --prompt) PROMPT="${2:?--prompt needs text}"; shift 2 ;;
    --url) URL="${2:?--url needs a URL}"; shift 2 ;;
    --reference-url) REFERENCE_URLS+=("${2:?--reference-url needs a URL}"); shift 2 ;;
    -o | --out) OUT="${2:?--out needs a path}"; shift 2 ;;
    --json) AS_JSON=1; shift ;;
    -h | --help) usage; exit 0 ;;
    -*) mg_die "unknown option: $1" ;;
    *) break ;;
  esac
done

case "$KIND" in
  exact | recast | lora) ;;
  *) mg_die "--kind must be exact, recast or lora" ;;
esac

[[ -n "$VIDEO" ]] || { usage >&2; mg_die "--video is required"; }
[[ -n "$URL" ]] || { mg_report "uploading ${VIDEO##*/}"; URL="$(mg_public_url "$(mg_require_file "$VIDEO")")"; }

for reference in ${REFERENCES+"${REFERENCES[@]}"}; do
  mg_report "uploading ${reference##*/}"
  REFERENCE_URLS+=("$(mg_public_url "$(mg_require_file "$reference")")")
done

[[ ${#REFERENCE_URLS[@]} -gt 0 || "$KIND" != "recast" ]] || mg_die "--kind recast needs at least one --reference"

# reference_image_urls is a real array, so the body is assembled in python.
BODY="$(python3 - "$KIND" "$URL" "$RESOLUTION" "$PROMPT" ${REFERENCE_URLS+"${REFERENCE_URLS[@]}"} <<'PY'
import json, sys

kind, video_url, resolution, prompt, *references = sys.argv[1:]
body = {"service": "character_swap_video", "kind": kind, "video_url": video_url}
if resolution:
    body["resolution"] = resolution
if prompt:
    body["prompt"] = prompt
if references:
    body["reference_image_urls"] = references
print(json.dumps(body))
PY
)"

mg_service "$BODY"
mg_resolve_job video

if ((AS_JSON)); then
  cat "$MG_BODY_FILE"
  exit 0
fi

path="$(mg_fetch "${OUT:-$(mg_outfile swap mp4)}" \
  job.result.video_url \
  video_url \
  result.video_url)"

printf 'duration=%ss credits=%s\n' \
  "$(mg_field job.result.duration_seconds duration_seconds)" \
  "$(mg_field job.result.charged_usd credits_used cost_usd)" >&2
printf '%s\n' "$path"