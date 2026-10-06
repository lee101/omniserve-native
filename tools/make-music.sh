#!/usr/bin/env bash
# Generate a music track through the ManifoldGen API.
#
#   tools/make-music.sh "warm lofi hip hop at 70 bpm, dusty drums"
#   tools/make-music.sh --duration 90 --lyrics-file chorus.txt "an indie anthem"
#
# The service is asynchronous when the Music3 lane is configured (202 plus a job
# to poll) and synchronous on the fal fallback, so both shapes are handled.
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_manifold.sh"

usage() {
  sed -n '2,9p' "${BASH_SOURCE[0]}" | cut -c3-
  cat <<'EOF'

Options
      --duration N     seconds, 30 - 300 (default 30)
      --lyrics TEXT    lyrics, optionally with [verse] / [chorus] section tags
      --lyrics-file F  read the lyrics from a file
      --instrumental   skip vocals
      --tier NAME      standard | fast | xfast
      --seed N         non-negative seed; 0 picks a random one
      --compose        ask the LLM composer for lyrics and a title first
  -o, --out FILE       output path
      --json           print the raw API response
  -h, --help           this text
EOF
}

DURATION=30
LYRICS=""
TIER=""
SEED=0
INSTRUMENTAL=0
COMPOSE=0
OUT=""
AS_JSON=0

while (( $# )); do
  case "$1" in
    --duration) DURATION="${2:?--duration needs seconds}"; shift 2 ;;
    --lyrics) LYRICS="${2:?--lyrics needs text}"; shift 2 ;;
    --lyrics-file) LYRICS="$(cat "${2:?--lyrics-file needs a path}")"; shift 2 ;;
    --instrumental) INSTRUMENTAL=1; shift ;;
    --tier) TIER="${2:?--tier needs a name}"; shift 2 ;;
    --seed) SEED="${2:?--seed needs a number}"; shift 2 ;;
    --compose) COMPOSE=1; shift ;;
    -o | --out) OUT="${2:?--out needs a path}"; shift 2 ;;
    --json) AS_JSON=1; shift ;;
    -h | --help) usage; exit 0 ;;
    -*) mg_die "unknown option: $1" ;;
    *) break ;;
  esac
done

PROMPT="${*:-}"
[[ -n "$PROMPT" ]] || { usage >&2; mg_die "a music prompt is required"; }
((INSTRUMENTAL)) && LYRICS="[instrumental]"

if ((COMPOSE)); then
  mg_request POST /api/music-compose "$(mg_body description "$PROMPT" duration "$DURATION" instrumental "$INSTRUMENTAL")"
  mg_require_ok
  TITLE="$(mg_get title 2>/dev/null || true)"
  COMPOSED="$(mg_get lyrics 2>/dev/null || true)"
  [[ -n "$COMPOSED" && "$COMPOSED" != "null" ]] || mg_die "the composer returned no lyrics"
  LYRICS="$COMPOSED"
  mg_report "composed: $TITLE"
fi

mg_service "$(mg_body \
  service music \
  prompt "$PROMPT" \
  lyrics "$LYRICS" \
  duration "$DURATION" \
  service_tier "$TIER" \
  seed "$SEED")"

mg_resolve_job audio

if ((AS_JSON)); then
  cat "$MG_BODY_FILE"
  exit 0
fi

# The async lane nests the payload under job.result; the fal fallback returns it
# at the top level.
path="$(mg_fetch "${OUT:-$(mg_outfile music wav)}" \
  job.result.audio_url \
  audio_url \
  job.result.video_url \
  result.audio_url)"

printf 'duration=%ss credits=%s\n' \
  "$(mg_field job.result.duration_seconds duration_seconds)" \
  "$(mg_field job.result.charged_usd credits_used cost_usd)" >&2
printf '%s\n' "$path"