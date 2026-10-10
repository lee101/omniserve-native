#!/usr/bin/env bash
# Generate an image through the ManifoldGen API (RA2 = Qwen Image 2.1 by default).
#
#   tools/make-image.sh "a red fox in snow, photograph"
#   tools/make-image.sh --size 1536x1024 --count 2 --seed 7 "a lakeside cabin at dawn"
#   tools/make-image.sh --service gpt-image-2 "a pencil sketch of a heron"
#
# The server is started automatically when nothing is listening on MANIFOLDGEN_API.
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_manifold.sh"

usage() {
  sed -n '2,8p' "${BASH_SOURCE[0]}" | cut -c3-
  cat <<'EOF'

Options
  -b, --backend NAME   image lane: ra2 (default, Qwen), omniserve, images3, r1, auto
  -s, --size WxH       pixel size, default 1024x1024
  -n, --count N        images to generate, default 1
      --steps N        diffusion steps (priced higher from 20 up)
      --guidance F     guidance scale
      --hq             max-quality tier: 30-step base model instead of 6-step turbo (2x price)
      --direct         skip the API and render on the local RA2 lane (no credits, no gallery save)
      --retries N      retry transient API errors with backoff, default 3 (MG_RETRIES)
      --no-fallback    fail instead of rendering locally when the API stays unavailable
      --service NAME   api service id, default image (anima, gpt-image-2, openpaths-image)
      --model NAME     model id for services that take one
  -o, --out FILE       output path (single image only)
      --json           print the raw API response instead of file paths
  -h, --help           this text
EOF
}

BACKEND=ra2
SIZE=1024x1024
COUNT=1
STEPS=""
GUIDANCE=""
SEED=0
SERVICE=image
MODEL=""
OUT=""
AS_JSON=0
QUALITY=""
DIRECT=0
RETRIES="${MG_RETRIES:-3}"
FALLBACK=1

while (( $# )); do
  case "$1" in
    -b | --backend) BACKEND="${2:?--backend needs a value}"; shift 2 ;;
    -s | --size) SIZE="${2:?--size needs WxH}"; shift 2 ;;
    -n | --count) COUNT="${2:?--count needs a number}"; shift 2 ;;
    --steps) STEPS="${2:?--steps needs a number}"; shift 2 ;;
    --guidance) GUIDANCE="${2:?--guidance needs a number}"; shift 2 ;;
    --seed) SEED="${2:?--seed needs a number}"; shift 2 ;;
    --service) SERVICE="${2:?--service needs a name}"; shift 2 ;;
    --model) MODEL="${2:?--model needs a name}"; shift 2 ;;
    -o | --out) OUT="${2:?--out needs a path}"; shift 2 ;;
    --json) AS_JSON=1; shift ;;
    --hq) QUALITY=hq; shift ;;
    --direct) DIRECT=1; shift ;;
    --retries) RETRIES="${2:?--retries needs a number}"; shift 2 ;;
    --no-fallback) FALLBACK=0; shift ;;
    -h | --help) usage; exit 0 ;;
    -*) mg_die "unknown option: $1" ;;
    *) break ;;
  esac
done

PROMPT="${*:-}"
[[ -n "$PROMPT" ]] || { usage >&2; mg_die "a prompt is required"; }
[[ "$SIZE" =~ ^[0-9]+x[0-9]+$ ]] || mg_die "--size must look like 1024x1024"
[[ "$COUNT" =~ ^[0-9]+$ ]] && ((COUNT >= 1)) || mg_die "--count must be a positive integer"
[[ -z "$OUT" || "$COUNT" == 1 ]] || mg_die "--out takes a single image; --count is $COUNT"

WIDTH="${SIZE%x*}"
HEIGHT="${SIZE#*x}"

# One process does the request, the retry/backoff, the local fallback and the
# file writes; the old shell pipeline spent ~0.4 s in interpreter starts.
((DIRECT)) || mg_ensure_server
((DIRECT)) || mg_ensure_key

run() {
  local extra=()
  ((AS_JSON)) && extra+=(--json)
  ((DIRECT)) && extra+=(--direct)
  ((FALLBACK)) || extra+=(--no-fallback)
  python3 -S "$MG_LIB_DIR/mg_image.py" --api "$(mg_api)" --key "${MANIFOLDGEN_API_KEY:-}" \
    --service "$SERVICE" --backend "$BACKEND" --model "$MODEL" \
    --width "$WIDTH" --height "$HEIGHT" --count "$COUNT" --steps "${STEPS:-0}" \
    --guidance "${GUIDANCE:-0}" --seed "$SEED" --quality "$QUALITY" \
    --out "$OUT" --out-dir "${MG_OUT_DIR:-results}" --retries "$RETRIES" \
    "${extra[@]}" -- "$PROMPT"
}

rc=0
run || rc=$?
if ((rc == 77)); then
  mg_forget_key
  mg_ensure_key
  rc=0
  run || rc=$?
fi
exit "$rc"
