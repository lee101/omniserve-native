#!/usr/bin/env bash
# Generate an Anima illustration through the ManifoldGen API.
#
#   tools/make-anima.sh "a girl with a fox mask in a snowy shrine"
#   tools/make-anima.sh --size 1024x1536 --steps 32 "a lighthouse at dusk"
#
# Anima is served by the native image lane, so GET /api/anima/status is checked
# first and the job is polled when the request is accepted.
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_manifold.sh"

usage() {
  sed -n '2,8p' "${BASH_SOURCE[0]}" | cut -c3-
  cat <<'EOF'

Options
  -s, --size WxH      pixel size, snapped to /64 (default 1024x1024)
      --steps N       10 - 50 (default 28)
      --guidance F    1 - 8 (default 4)
      --seed N        0 picks a random one
  -o, --out FILE      output path
      --json          print the raw API response
  -h, --help          this text
EOF
}

SIZE=1024x1024
STEPS=""
GUIDANCE=""
SEED=0
OUT=""
AS_JSON=0

while (( $# )); do
  case "$1" in
    -s | --size) SIZE="${2:?--size needs WxH}"; shift 2 ;;
    --steps) STEPS="${2:?--steps needs a number}"; shift 2 ;;
    --guidance) GUIDANCE="${2:?--guidance needs a number}"; shift 2 ;;
    --seed) SEED="${2:?--seed needs a number}"; shift 2 ;;
    -o | --out) OUT="${2:?--out needs a path}"; shift 2 ;;
    --json) AS_JSON=1; shift ;;
    -h | --help) usage; exit 0 ;;
    -*) mg_die "unknown option: $1" ;;
    *) break ;;
  esac
done

PROMPT="${*:-}"
[[ -n "$PROMPT" ]] || { usage >&2; mg_die "a prompt is required"; }
[[ "$SIZE" =~ ^[0-9]+x[0-9]+$ ]] || mg_die "--size must look like 1024x1024"

# The status route is unauthenticated and says whether the lane is loaded, which
# turns a slow queue into an immediate, useful failure.
mg_ensure_server
mg_request GET /api/anima/status
if [[ "$MG_STATUS" == 200 ]]; then
  mg_report "anima lane: $(mg_field status ready loaded)"
fi

mg_service "$(mg_body \
  service anima \
  prompt "$PROMPT" \
  width "${SIZE%x*}" \
  height "${SIZE#*x}" \
  num_steps "$STEPS" \
  guidance "$GUIDANCE" \
  seed "$SEED")"

mg_resolve_job video

if ((AS_JSON)); then
  cat "$MG_BODY_FILE"
  exit 0
fi

path="$(mg_fetch "${OUT:-$(mg_outfile anima webp)}" \
  job.result.image_url \
  result.image_url \
  image_url \
  saved_image_url)"

printf 'steps=%s credits=%s\n' \
  "$(mg_field job.result.steps steps)" \
  "$(mg_field job.result.charged_usd credits_used cost_usd)" >&2
printf '%s\n' "$path"