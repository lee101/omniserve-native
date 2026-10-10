#!/usr/bin/env bash
# Generate a sound effect through the ManifoldGen API.
#
#   tools/make-sfx.sh "distant thunder rolling over a valley"
#   tools/make-sfx.sh --duration 12 "a rusty gate creaking open"
#
# SFX shares the audio lane with music, so the job is polled the same way.
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_manifold.sh"

usage() {
  sed -n '2,6p' "${BASH_SOURCE[0]}" | cut -c3-
  cat <<'EOF'

Options
      --duration N   seconds
      --format FMT   output container
  -o, --out FILE     output path
      --json         print the raw API response
  -h, --help         this text
EOF
}

DURATION=""
FORMAT=""
OUT=""
AS_JSON=0

while (( $# )); do
  case "$1" in
    --duration) DURATION="${2:?--duration needs seconds}"; shift 2 ;;
    --format) FORMAT="${2:?--format needs a name}"; shift 2 ;;
    -o | --out) OUT="${2:?--out needs a path}"; shift 2 ;;
    --json) AS_JSON=1; shift ;;
    -h | --help) usage; exit 0 ;;
    -*) mg_die "unknown option: $1" ;;
    *) break ;;
  esac
done

PROMPT="${*:-}"
[[ -n "$PROMPT" ]] || { usage >&2; mg_die "a prompt is required"; }

mg_service "$(mg_body service sfx prompt "$PROMPT" duration "$DURATION" output_format "$FORMAT")"
mg_resolve_job audio

if ((AS_JSON)); then
  cat "$MG_BODY_FILE"
  exit 0
fi

path="$(mg_fetch "${OUT:-$(mg_outfile sfx wav)}" \
  job.result.audio_url \
  audio_url \
  job.result.video_url \
  result.audio_url)"

printf 'duration=%ss credits=%s\n' \
  "$(mg_field job.result.duration_seconds duration_seconds)" \
  "$(mg_field job.result.charged_usd credits_used cost_usd)" >&2
printf '%s\n' "$path"