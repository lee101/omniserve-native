#!/usr/bin/env bash
# Generate a video from reference images through the ManifoldGen API.
#
#   tools/reference-video.sh --image sketch.png "the camera pushes in slowly"
#   tools/reference-video.sh --model pro --resolution 1080p --image a.png --image b.png "a slow orbit"
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_manifold.sh"

usage() {
  sed -n '2,8p' "${BASH_SOURCE[0]}" | cut -c3-
  cat <<'EOF'

Options
      --image FILE       reference image; repeatable (up to 9)
      --image-url URL    reference image by URL; repeatable
      --video FILE       reference video; repeatable (up to 3)
      --audio FILE       reference audio; repeatable (up to 3)
      --model NAME       mini | pro (default mini)
      --resolution RES   480p | 720p | 1080p (default 720p)
      --duration N       seconds to generate
  -o, --out FILE         output path
      --json             print the raw API response
  -h, --help             this text
EOF
}

MODEL=mini
RESOLUTION=720p
DURATION=""
OUT=""
AS_JSON=0
IMAGES=()
IMAGE_URLS=()
VIDEOS=()
AUDIOS=()
AUDIO_URLS=()

while (( $# )); do
  case "$1" in
    --image) IMAGES+=("${2:?--image needs a file}"); shift 2 ;;
    --image-url) IMAGE_URLS+=("${2:?--image-url needs a URL}"); shift 2 ;;
    --video) VIDEOS+=("${2:?--video needs a file}"); shift 2 ;;
    --audio) AUDIOS+=("${2:?--audio needs a file}"); shift 2 ;;
    --model) MODEL="${2:?--model needs a name}"; shift 2 ;;
    --resolution) RESOLUTION="${2:?--resolution needs a value}"; shift 2 ;;
    --duration) DURATION="${2:?--duration needs a number}"; shift 2 ;;
    -o | --out) OUT="${2:?--out needs a path}"; shift 2 ;;
    --json) AS_JSON=1; shift ;;
    -h | --help) usage; exit 0 ;;
    -*) mg_die "unknown option: $1" ;;
    *) break ;;
  esac
done

PROMPT="${*:-}"
[[ -n "$PROMPT" ]] || { usage >&2; mg_die "a prompt is required"; }

# Local files are uploaded first; --image-url entries are already public.
upload_all() {
  local -n destination="$1"
  shift
  local item
  for item in "$@"; do
    mg_report "uploading ${item##*/}"
    destination+=("$(mg_public_url "$(mg_require_file "$item")")")
  done
}

upload_all IMAGE_URLS ${IMAGES+"${IMAGES[@]}"}
upload_all VIDEOS ${VIDEOS+"${VIDEOS[@]}"}
upload_all AUDIO_URLS ${AUDIOS+"${AUDIOS[@]}"}

# Each reference list crosses as a JSON array so the kinds stay separate
# instead of being re-guessed from the URL.
json_array() {
  python3 -c 'import json,sys;print(json.dumps(list(sys.argv[1:])))' ${1+"$@"}
}

BODY="$(python3 - "$MODEL" "$RESOLUTION" "$DURATION" "$PROMPT" \
  "$(json_array ${IMAGE_URLS+"${IMAGE_URLS[@]}"})" \
  "$(json_array ${VIDEOS+"${VIDEOS[@]}"})" \
  "$(json_array ${AUDIO_URLS[@]+"${AUDIO_URLS[@]}"})" <<'PY'
import json, sys

model, resolution, duration, prompt, images_json, videos_json, audio_json = sys.argv[1:]
images = json.loads(images_json)
videos = json.loads(videos_json)
audios = json.loads(audio_json)
body = {
    "service": "reference-video",
    "model": model,
    "resolution": resolution,
    "prompt": prompt,
    "reference_image_urls": images,
    "reference_video_urls": videos,
    "reference_audio_urls": audios,
}
if duration:
    body["duration"] = int(duration)
print(json.dumps(body))
PY
)"

mg_service "$BODY"
mg_resolve_job video

if ((AS_JSON)); then
  cat "$MG_BODY_FILE"
  exit 0
fi

path="$(mg_fetch "${OUT:-$(mg_outfile reference-video mp4)}" \
  job.result.video_url \
  video_url \
  result.video_url)"

printf 'credits=%s\n' "$(mg_field job.result.charged_usd credits_used)" >&2
printf '%s\n' "$path"