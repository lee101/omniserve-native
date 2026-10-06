#!/usr/bin/env bash
# Generate speech through the ManifoldGen API.
#
#   tools/make-voice.sh "The lake is calm tonight."
#   tools/make-voice.sh --model eleven-v3 --voice Rachel --mood happy "Read this aloud."
#   tools/make-voice.sh --list
#
# Default lane is the Voice Studio (POST /api/voice/generate, seven models).
# --local uses the site's own text-generator speech lane instead.
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_manifold.sh"

usage() {
  sed -n '2,10p' "${BASH_SOURCE[0]}" | cut -c3-
  cat <<'EOF'

Options
      --list             list the available models, voices and formats
  -m, --model ID         voice model; default is the first the server offers
      --voice NAME       voice id from the model
      --mood MOOD        angry | neutral | happy
      --speed F          0.5 - 2.0
      --pitch N          -12 - 12
      --volume F         0.1 - 2.0
      --format FMT       output format the model supports
      --sample-rate N    sample rate the model supports
      --language CODE    language hint (seed-speech, grok-voice)
  -n, --batch N          1 - 4 takes of the same text
      --local            use the local text-generator speech lane
  -o, --out FILE         output path (single take only)
      --json             print the raw API response
  -h, --help             this text
EOF
}

MODEL=""
VOICE=""
MOOD=""
SPEED=""
PITCH=""
VOLUME=""
FORMAT=""
SAMPLE_RATE=""
LANGUAGE=""
BATCH=""
LOCAL=0
OUT=""
AS_JSON=0
LIST=0

while (( $# )); do
  case "$1" in
    --list) LIST=1; shift ;;
    -m | --model) MODEL="${2:?--model needs an id}"; shift 2 ;;
    --voice) VOICE="${2:?--voice needs a name}"; shift 2 ;;
    --mood) MOOD="${2:?--mood needs a value}"; shift 2 ;;
    --speed) SPEED="${2:?--speed needs a number}"; shift 2 ;;
    --pitch) PITCH="${2:?--pitch needs a number}"; shift 2 ;;
    --volume) VOLUME="${2:?--volume needs a number}"; shift 2 ;;
    --format) FORMAT="${2:?--format needs a value}"; shift 2 ;;
    --sample-rate) SAMPLE_RATE="${2:?--sample-rate needs a number}"; shift 2 ;;
    --language) LANGUAGE="${2:?--language needs a code}"; shift 2 ;;
    -n | --batch) BATCH="${2:?--batch needs a number}"; shift 2 ;;
    --local) LOCAL=1; shift ;;
    -o | --out) OUT="${2:?--out needs a path}"; shift 2 ;;
    --json) AS_JSON=1; shift ;;
    -h | --help) usage; exit 0 ;;
    -*) mg_die "unknown option: $1" ;;
    *) break ;;
  esac
done

# /api/voice/models is the one unauthenticated route, so the catalogue can be
# read before a key is needed and the defaults follow the server's own list.
mg_ensure_server
mg_request GET /api/voice/models
mg_require_ok
# The next request overwrites the shared body file, so the catalogue is kept.
mg_stash_body CATALOGUE

if ((LIST)); then
  python3 - "$CATALOGUE" <<'PY'
import json, sys
for model in json.load(open(sys.argv[1])).get("models", []):
    print(f"{model['id']}  {model.get('name','')}")
    print(f"    voices: {', '.join(model.get('voices') or []) or '-'}")
    print(f"    formats: {', '.join(model.get('formats') or [])}"
          f"  rates: {', '.join(str(r) for r in model.get('sample_rates') or [])}")
PY
  exit 0
fi

TEXT="${*:-}"
[[ -n "$TEXT" ]] || { usage >&2; mg_die "some text to speak is required"; }

if ((LOCAL)); then
  mg_service "$(mg_body service speech text "$TEXT" voice "$VOICE" language "$LANGUAGE" speed "$SPEED")"
  ((AS_JSON)) && { cat "$MG_BODY_FILE"; exit 0; }
  path="$(mg_fetch "${OUT:-$(mg_outfile speech mp3)}" \
    result.audio_url result.audio_base64 audio_url)"
  printf '%s\n' "$path"
  exit 0
fi

# Default to the first catalogue entry so a bare invocation always names a model
# the server accepts.
if [[ -z "$MODEL" ]]; then
  MODEL="$(python3 - "$CATALOGUE" <<'PY'
import json, sys
models = json.load(open(sys.argv[1])).get("models", [])
print(models[0]["id"] if models else "")
PY
)"
  [[ -n "$MODEL" ]] || mg_die "the server offered no voice models"
fi

mg_request POST /api/voice/generate "$(mg_body \
  model "$MODEL" \
  text "$TEXT" \
  voice "$VOICE" \
  mood "$MOOD" \
  speed "$SPEED" \
  pitch "$PITCH" \
  volume "$VOLUME" \
  output_format "$FORMAT" \
  sample_rate "$SAMPLE_RATE" \
  language "$LANGUAGE" \
  batch_size "$BATCH")"
mg_require_ok

if ((AS_JSON)); then
  cat "$MG_BODY_FILE"
  exit 0
fi

COUNT="$(mg_count results)"
[[ -z "$OUT" || "$COUNT" == 1 ]] || mg_die "--out takes one file; the batch returned $COUNT"

written=()
for ((index = 0; index < COUNT; index++)); do
  target="${OUT:-$(mg_outfile voice mp3)}"
  written+=("$(mg_fetch "$target" "results.$index.audio_url" "results.$index.audio_base64")")
done

printf 'model=%s credits=%s\n' \
  "$(mg_field model)" \
  "$(mg_field credits_used)" >&2
printf '%s\n' "${written[@]}"