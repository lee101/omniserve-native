#!/usr/bin/env bash
# Does speculative decoding pay on this machine?
#
# Runs the same prompts twice against the same weights, once with speculation
# off and once on, and reports three things per arm: wall clock, the acceptance
# rate the model actually achieved, and how many model calls speculation
# avoided. Acceptance is a property of the model and the prompt; whether the
# saved calls turn into saved seconds is a property of the hardware, which is
# why both are printed rather than a single speedup number.
#
# Speculation only pays where decode is bandwidth-bound. On CPU the wider verify
# batch costs proportionally more arithmetic and the acceptance rate buys
# nothing, so a run with the model on CPU is expected to come out flat.
#
#   ./scripts/spec_bench.sh /path/to/model.gguf [ngl] [draft]
set -euo pipefail

MODEL="${1:?usage: spec_bench.sh MODEL.gguf [ngl] [draft]}"
NGL="${2:-999}"
DRAFT="${3:-4}"
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
BIN="${ONATIVE_BIN:-${SCRIPT_DIR}/../build-full/omniserve-native}"
WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT

[[ -x "$BIN" ]] || { echo "missing $BIN (set ONATIVE_BIN)" >&2; exit 1; }

PROMPTS=(
  'Repeat this list back exactly, twice: alpha bravo charlie delta echo foxtrot golf hotel'
  'Rewrite this sentence with no changes at all: The farmer counted seventeen sheep near the riverbank at dawn.'
  'Summarize the following in one sentence: A native C inference server schedules paid, subscriber, free and background traffic onto a single GPU.'
  'Write a short poem about the sea.'
)

arm() { # label port draft
  local label="$1" port="$2" draft="$3" pid started elapsed
  OMNISERVE_NATIVE_PORT="$port" \
  OMNISERVE_NATIVE_LLM_GGUF="$MODEL" \
  OMNISERVE_NATIVE_NGL="$NGL" \
  OMNISERVE_NATIVE_LLM_CONTEXTS=1 \
  OMNISERVE_NATIVE_SPEC_DRAFT="$draft" \
  OMNISERVE_NATIVE_VRAM_BROKER=0 \
  OMNISERVE_NATIVE_RAM_PREFETCH_ENABLED=0 \
  "$BIN" >"$WORK/$label.log" 2>&1 &
  pid=$!
  for _ in $(seq 300); do
    curl -s -m 2 -o /dev/null "localhost:$port/health" && break
    sleep 1
  done

  : > "$WORK/$label.txt"
  started=$(date +%s.%N)
  local tokens=0
  for p in "${PROMPTS[@]}"; do
    local body
    body=$(python3 -c 'import json,sys; print(json.dumps({"messages":[{"role":"user","content":sys.argv[1]}],"max_tokens":160,"temperature":0}))' "$p")
    curl -s -m 900 "localhost:$port/v1/chat/completions" \
         -H 'content-type: application/json' -d "$body" > "$WORK/resp.json"
    python3 - "$WORK/resp.json" "$WORK/$label.txt" <<'PY'
import json, sys
d = json.load(open(sys.argv[1]))
open(sys.argv[2], "a").write(json.dumps(d["choices"][0]["message"]["content"]) + "\n")
print(d.get("usage", {}).get("completion_tokens", 0))
PY
  done > "$WORK/$label.tokens"
  elapsed=$(python3 -c "import sys;print(f'{float(sys.argv[1])-float(sys.argv[2]):.2f}')" "$(date +%s.%N)" "$started")
  tokens=$(python3 -c 'import sys;print(sum(int(l) for l in open(sys.argv[1]) if l.strip()))' "$WORK/$label.tokens")

  curl -s "localhost:$port/status" > "$WORK/$label.status"
  kill "$pid" 2>/dev/null || true
  wait "$pid" 2>/dev/null || true

  python3 - "$WORK/$label.status" "$label" "$elapsed" "$tokens" <<'PY'
import json, sys
status, label, elapsed, tokens = sys.argv[1:5]
s = json.load(open(status)).get("speculation", {})
tps = float(tokens) / float(elapsed) if float(elapsed) else 0.0
print(f"{label:>10}: {elapsed}s for {tokens} tokens ({tps:.1f} tok/s)"
      f"  draft_max={s.get('draft_max')} acceptance={s.get('acceptance')}"
      f" drafted={s.get('drafted')} calls_saved={s.get('calls_saved')}")
PY
}

echo "model=$MODEL ngl=$NGL draft=$DRAFT"
arm off 18871 0
arm on  18872 "$DRAFT"

echo
if diff -q "$WORK/off.txt" "$WORK/on.txt" >/dev/null; then
  echo "greedy output: identical"
else
  echo "greedy output: diverges on $(python3 - "$WORK/off.txt" "$WORK/on.txt" <<'PY'
import json, sys
a = [json.loads(l) for l in open(sys.argv[1])]
b = [json.loads(l) for l in open(sys.argv[2])]
print(f"{sum(1 for x, y in zip(a, b) if x != y)} of {len(a)} prompts")
PY
)"
  echo "  Expected: a wider verify batch sums logits in a different order, so a"
  echo "  near-tied argmax can flip. Distribution is unchanged; token identity is"
  echo "  not guaranteed. See performance/quality.md."
fi
