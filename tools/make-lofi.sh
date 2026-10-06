#!/usr/bin/env bash
# Render a looping lofi video through the ManifoldGen API.
#
#   tools/make-lofi.sh --character "a cat at a desk" --scene "a rainy neon street"
#   tools/make-lofi.sh --preset rain-window --palette violet --visualizer wave
#
# GET /api/lofi-loop/spec is the authoritative list of presets, palettes,
# visualizers and motion settings; --spec prints it.
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_manifold.sh"

usage() {
  sed -n '2,9p' "${BASH_SOURCE[0]}" | cut -c3-
  cat <<'EOF'

Options
      --character TEXT  who is on screen, e.g. "a cat at a desk"
      --scene TEXT      where the scene happens
      --preset NAME     preset from --spec
      --palette NAME    palette from --spec
      --visualizer NAME visualizer from --spec
      --motion NAME     motion setting from --spec
      --loop-seconds N  clip length
      --seam-seconds F  crossfade length for the loop
      --spec            print the server's lofi specification and exit
  -o, --out FILE        output path
      --json            print the raw API response
  -h, --help            this text
EOF
}

CHARACTER=""
SCENE=""
PRESET=""
PALETTE=""
VISUALIZER=""
MOTION=""
LOOP_SECONDS=""
SEAM_SECONDS=""
OUT=""
AS_JSON=0
SPEC=0

while (( $# )); do
  case "$1" in
    --character) CHARACTER="${2:?--character needs text}"; shift 2 ;;
    --scene) SCENE="${2:?--scene needs text}"; shift 2 ;;
    --preset) PRESET="${2:?--preset needs a name}"; shift 2 ;;
    --palette) PALETTE="${2:?--palette needs a name}"; shift 2 ;;
    --visualizer) VISUALIZER="${2:?--visualizer needs a name}"; shift 2 ;;
    --motion) MOTION="${2:?--motion needs a name}"; shift 2 ;;
    --loop-seconds) LOOP_SECONDS="${2:?--loop-seconds needs a number}"; shift 2 ;;
    --seam-seconds) SEAM_SECONDS="${2:?--seam-seconds needs a number}"; shift 2 ;;
    --spec) SPEC=1; shift ;;
    -o | --out) OUT="${2:?--out needs a path}"; shift 2 ;;
    --json) AS_JSON=1; shift ;;
    -h | --help) usage; exit 0 ;;
    *) mg_die "unknown option: $1" ;;
  esac
done

mg_ensure_server
if ((SPEC)); then
  mg_request GET /api/lofi-loop/spec
  mg_require_ok
  mg_jget spec
  exit 0
fi

[[ -n "$CHARACTER" || -n "$SCENE" ]] || { usage >&2; mg_die "--character or --scene is required"; }

mg_service "$(mg_body \
  service lofi_loop \
  character_prompt "$CHARACTER" \
  scene_prompt "$SCENE" \
  preset "$PRESET" \
  palette "$PALETTE" \
  visualizer "$VISUALIZER" \
  motion "$MOTION" \
  loop_seconds "$LOOP_SECONDS" \
  seam_seconds "$SEAM_SECONDS")"

mg_resolve_job video

if ((AS_JSON)); then
  cat "$MG_BODY_FILE"
  exit 0
fi

path="$(mg_fetch "${OUT:-$(mg_outfile lofi mp4)}" \
  job.result.video_url \
  video_url \
  result.video_url)"

printf 'duration=%ss credits=%s\n' \
  "$(mg_field job.result.duration_seconds duration_seconds)" \
  "$(mg_field job.result.charged_usd credits_used)" >&2
printf '%s\n' "$path"