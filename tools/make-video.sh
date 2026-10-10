#!/usr/bin/env bash
# Generate a video clip through the ManifoldGen API (H3 lane).
#
#   tools/make-video.sh "a paper boat drifting down a rain gutter"
#   tools/make-video.sh --duration 10 --aspect 16:9 --first-frame still.png "it begins to move"
#
# The service queues a job, so the script follows it to completion.
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_manifold.sh"

usage() {
  sed -n '2,8p' "${BASH_SOURCE[0]}" | cut -c3-
  cat <<'EOF'

Options
      --duration N       seconds, 4 - 60
      --aspect RATIO     16:9 | 9:16 | 1:1 and friends
      --size TIER        preview | balanced | native
      --steps N          inference steps
      --first-frame IMG  animate this image (uploaded first)
      --audio            generate the sound track with the clip
      --format FMT       output container
      --model ID         provider model for the OpenPaths catalogue
      --profile NAME     auto | small | balanced | throughput
      --quant NAME       int8_convrot | w4a8
  -o, --out FILE         output path
      --json             print the raw API response
  -h, --help             this text
EOF
}

SERVICE=video
DURATION=""
ASPECT=""
SIZE=""
STEPS=""
FIRST_FRAME=""
MODEL=""
PROFILE=""
QUANT=""
FORMAT=""
WITH_AUDIO=""
OUT=""
AS_JSON=0

while (( $# )); do
  case "$1" in
    --duration) DURATION="${2:?--duration needs seconds}"; shift 2 ;;
    --aspect) ASPECT="${2:?--aspect needs a ratio}"; shift 2 ;;
    --size) SIZE="${2:?--size needs a tier}"; shift 2 ;;
    --steps) STEPS="${2:?--steps needs a number}"; shift 2 ;;
    --first-frame) FIRST_FRAME="${2:?--first-frame needs a file}"; shift 2 ;;
    --model) MODEL="${2:?--model needs an id}"; shift 2 ;;
    --profile) PROFILE="${2:?--profile needs a name}"; shift 2 ;;
    --quant) QUANT="${2:?--quant needs a name}"; shift 2 ;;
    --format) FORMAT="${2:?--format needs a name}"; shift 2 ;;
    --audio) WITH_AUDIO=true; shift ;;
    -o | --out) OUT="${2:?--out needs a path}"; shift 2 ;;
    --json) AS_JSON=1; shift ;;
    -h | --help) usage; exit 0 ;;
    -*) mg_die "unknown option: $1" ;;
    *) break ;;
  esac
done

PROMPT="${*:-}"
[[ -n "$PROMPT" ]] || { usage >&2; mg_die "a prompt is required"; }

FRAME=""
if [[ -n "$FIRST_FRAME" ]]; then
  mg_report "uploading ${FIRST_FRAME##*/}"
  FRAME="$(mg_public_url "$FIRST_FRAME")"
fi

mg_service "$(mg_body \
  service "$SERVICE" \
  prompt "$PROMPT" \
  model "$MODEL" \
  duration "$DURATION" \
  aspect_ratio "$ASPECT" \
  size "$SIZE" \
  num_steps "$STEPS" \
  first_frame "$FRAME" \
  include_audio "$WITH_AUDIO" \
  output_format "$FORMAT" \
  execution_profile "$PROFILE" \
  quant "$QUANT")"

mg_resolve_job video

if ((AS_JSON)); then
  cat "$MG_BODY_FILE"
  exit 0
fi

path="$(mg_fetch "${OUT:-$(mg_outfile video mp4)}" \
  job.result.video_url \
  video_url \
  job.result.video \
  result.video_url)"

printf 'duration=%ss credits=%s\n' \
  "$(mg_field job.result.duration_seconds duration_seconds)" \
  "$(mg_field job.result.charged_usd credits_used cost_usd)" >&2
printf '%s\n' "$path"