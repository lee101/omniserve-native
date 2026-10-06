#!/usr/bin/env bash
# Sweep MoE expert placement x speculation for a GGUF on the shared 5090.
#
#   ./scripts/moe_bench.sh /path/to/model.gguf
#
# Placements: all experts on CPU, MOE_CPU_EXPERTS=24/16/8, auto; each with
# SPEC_DRAFT=0 and 4. Single-stream (contexts=1) plus 4 concurrent streams
# (contexts=4) where VRAM allows. CTX=4096, KV q8_0, 200-token completions.
#
# The chat endpoint is non-streaming, so TTFT is measured as the elapsed time
# of a max_tokens=1 probe on the same prompt; decode tok/s excludes it.
# A config that leaves <1.5 GiB free after load is aborted, not measured.
set -euo pipefail

MODEL="${1:?usage: moe_bench.sh MODEL.gguf}"
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
BIN="${ONATIVE_BIN:-${SCRIPT_DIR}/../build-dev/omniserve-native}"
PORT="${ONATIVE_PORT:-8799}"
WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"; kill "$SRV_PID" 2>/dev/null || true' EXIT
SRV_PID=""

[[ -x "$BIN" ]] || { echo "missing $BIN (set ONATIVE_BIN)" >&2; exit 1; }
[[ -f "$MODEL" ]] || { echo "missing $MODEL" >&2; exit 1; }

gpu_free_mb() {
  nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1 | tr -d ' '
}

start_server() { # moe spec contexts; result lines go to $WORK/start.out
  local moe="$1" spec="$2" contexts="$3"
  local free_before load_start load_s free_after
  : >"$WORK/start.out"
  if ss -tln 2>/dev/null | grep -q ":$PORT "; then
    echo "PORT_BUSY $PORT already bound" >"$WORK/start.out"
    return 3
  fi
  free_before=$(gpu_free_mb)
  load_start=$(date +%s.%N)
  OMNISERVE_NATIVE_PORT="$PORT" \
  OMNISERVE_NATIVE_LLM_GGUF="$MODEL" \
  OMNISERVE_NATIVE_NGL=999 \
  OMNISERVE_NATIVE_CTX=4096 \
  OMNISERVE_NATIVE_LLM_CONTEXTS="$contexts" \
  OMNISERVE_NATIVE_SLOTS="$contexts" \
  OMNISERVE_NATIVE_LLM_PERMITS=1 \
  OMNISERVE_NATIVE_KV_TYPE=q8_0 \
  OMNISERVE_NATIVE_BATCH=auto \
  OMNISERVE_NATIVE_UBATCH=auto \
  OMNISERVE_NATIVE_SPEC_DRAFT="$spec" \
  OMNISERVE_NATIVE_MOE_CPU_EXPERTS="$moe" \
  OMNISERVE_NATIVE_VRAM_BROKER=0 \
  OMNISERVE_NATIVE_RAM_PREFETCH_ENABLED=0 \
  OMNISERVE_NATIVE_LLM_SWAP_DIR="$(dirname "$MODEL")" \
  "$BIN" >"$WORK/server.log" 2>&1 &
  SRV_PID=$!
  local ok=0
  for _ in $(seq 240); do
    if ! kill -0 "$SRV_PID" 2>/dev/null; then break; fi
    if curl -s -m 2 -o /dev/null "localhost:$PORT/health"; then ok=1; break; fi
    sleep 2
  done
  load_s=$(python3 -c "print(f'{float(\"$(date +%s.%N)\")-float(\"$load_start\"):.1f}')")
  if [[ "$ok" != 1 ]]; then
    kill "$SRV_PID" 2>/dev/null || true
    wait "$SRV_PID" 2>/dev/null || true
    SRV_PID=""
    echo "LOAD_FAIL load_s=$load_s" >"$WORK/start.out"
    return 1
  fi
  if ! curl -s "localhost:$PORT/status" | python3 -c 'import json,sys; sys.exit(0 if json.load(sys.stdin)["llm"]["ready"] else 1)'; then
    echo "LOAD_FAIL load_s=$load_s (llm not ready)" >"$WORK/start.out"
    return 1
  fi
  lpids="$(ss -tlnp 2>/dev/null | grep ":$PORT " | grep -o 'pid=[0-9]*' | cut -d= -f2 | sort -u | tr '\n' ' ')"
  if [[ "$lpids" != "$SRV_PID " && "$lpids" != "$SRV_PID" ]]; then
    echo "LOAD_FAIL load_s=$load_s (port shared: $lpids)" >"$WORK/start.out"
    return 1
  fi
  free_after=$(gpu_free_mb)
  echo "LOADED load_s=$load_s gpu_mb=$((free_before - free_after)) free_mb=$free_after" >"$WORK/start.out"
  if [[ "$free_after" -lt 1536 ]]; then
    echo "ABORT free ${free_after}MiB < 1536MiB floor" >>"$WORK/start.out"
    return 2
  fi
  return 0
}

stop_server() {
  if [[ -n "$SRV_PID" ]]; then
    kill "$SRV_PID" 2>/dev/null || true
    wait "$SRV_PID" 2>/dev/null || true
    SRV_PID=""
  fi
  sleep 2
}

run_load() { # label n_parallel out_file
  local label="$1" npar="$2" out="$3"
  python3 - "$PORT" "$npar" "$out" <<'PY'
import concurrent.futures as cf
import json, subprocess, sys, time

port, npar, out = sys.argv[1], int(sys.argv[2]), sys.argv[3]
prompts = [
    "Tell the lost knight the full tale of how his sword came to the Thornwood, at least 180 words, stay in character.",
    "Describe your hut, your raven, and the price you will ask for your help, at least 180 words, stay in character.",
    "Tell the story of the wraith on the forest road: who it was and what it wants, at least 180 words, stay in character.",
    "Explain the three trials the knight must pass before you join his quest, at least 180 words, stay in character.",
]

def one(prompt, max_tokens, seed):
    body = json.dumps({"messages": [
        {"role": "system", "content": "You are Morgana, a sly witch of the Thornwood. Speak in first person with dry wit. Never break character."},
        {"role": "user", "content": prompt}],
        "max_tokens": max_tokens, "temperature": 0.7,
        "enable_thinking": False, "seed": seed})
    t0 = time.monotonic()
    p = subprocess.run(["curl", "-s", "-m", "900",
                        f"localhost:{port}/v1/chat/completions",
                        "-H", "content-type: application/json", "-d", body],
                       capture_output=True, text=True)
    dt = time.monotonic() - t0
    d = json.loads(p.stdout)
    n = d["usage"]["completion_tokens"]
    return dt, n

ttft, _ = one(prompts[0], 1, 11)
one(prompts[1], 200, 12)
t0 = time.monotonic()
with cf.ThreadPoolExecutor(max_workers=npar) as ex:
    futs = [ex.submit(one, prompts[i % len(prompts)], 200, 100 + i)
            for i in range(npar)]
    res = [f.result() for f in futs]
wall = time.monotonic() - t0
toks = sum(n for _, n in res)
gen = [dt for dt, _ in res]
json.dump({"ttft_s": round(ttft, 2),
           "wall_s": round(wall, 1),
           "tokens": toks,
           "agg_tps": round(toks / wall, 1) if wall else 0,
           "mean_stream_s": round(sum(gen) / len(gen), 1),
           "single_tps": round(toks / (wall - ttft), 1) if npar == 1 and wall > ttft else None},
          open(out, "w"))
print(open(out).read())
PY
}

spec_line() {
  curl -s "localhost:$PORT/status" | python3 -c '
import json, sys
s = json.load(sys.stdin).get("speculation", {})
print("acc=%s drafted=%s saved=%s" % (s.get("acceptance", 0), s.get("drafted", 0), s.get("calls_saved", 0)))'
}

place_line() {
  curl -s "localhost:$PORT/status" | python3 -c '
import json, sys
d = json.load(sys.stdin)
l = d["llm"]
print("place=%s gpu_exp=%s cpu_exp=%s est_gpu_mib=%s free_gib=%.1f" % (
    d.get("gpu", {}).get("placement", "?"), l.get("gpu_expert_layers", 0),
    l.get("cpu_expert_layers", 0), l.get("est_gpu_bytes", 0) // 1024 // 1024,
    d.get("vram_free_gib", -1)))'
}

echo "model=$MODEL ctx=4096 kv=q8_0"
printf '%-14s %-5s %-4s %-9s %-9s %-8s %-6s %-9s %-9s %-24s %s\n' \
  placement spec ctxs load_s gpu_MB ttft_s toks tps agg_tps spec place

for moe in ${MOE_SWEEP:-all 24 16 8 auto}; do
  for spec in ${SPEC_SWEEP:-0 4}; do
    for ctxs in ${CTX_SWEEP:-1 4}; do
      rc=0
      busy=0
      for _try in 1 2 3; do
        rc=0
        start_server "$moe" "$spec" "$ctxs" || rc=$?
        if [[ "$rc" != 3 ]]; then busy=0; break; fi
        busy=1
        sleep 120
      done
      if [[ "$busy" == 1 ]]; then
        printf '%-14s %-5s %-4s %-9s %-9s %-8s %-9s %-9s %-28s %s\n' \
          "$moe" "$spec" "$ctxs" "?" "?" "-" "-" "-" "-" "PORT_BUSY after 3 retries"
        stop_server
        continue
      fi
      line="$(cat "$WORK/start.out")"
      load_s="$(echo "$line" | grep -o 'load_s=[0-9.]*' | cut -d= -f2 || true)"
      gpu_mb="$(echo "$line" | grep -o 'gpu_mb=[0-9-]*' | cut -d= -f2 || true)"
      if [[ "$rc" != 0 ]]; then
        printf '%-14s %-5s %-4s %-9s %-9s %-8s %-6s %-9s %-9s %-24s %s\n' \
          "$moe" "$spec" "$ctxs" "${load_s:-?}" "${gpu_mb:-?}" \
          "-" "-" "-" "-" "-" "$(echo "$line" | tail -1)"
        stop_server
        continue
      fi
      if ! run_load "$moe/$spec/$ctxs" "$ctxs" "$WORK/res.json" >"$WORK/run.out" 2>&1; then
        printf '%-14s %-5s %-4s %-9s %-9s %-8s %-6s %-9s %-9s %-24s %s\n' \
          "$moe" "$spec" "$ctxs" "$load_s" "$gpu_mb" \
          "-" "-" "-" "-" "-" "RUN_FAIL $(tail -c 200 "$WORK/run.out")"
        stop_server
        continue
      fi
      lpids2="$(ss -tlnp 2>/dev/null | grep ":$PORT " | grep -o 'pid=[0-9]*' | cut -d= -f2 | sort -u | tr '\n' ' ')"
      shared=""
      if [[ "$lpids2" != "$SRV_PID " && "$lpids2" != "$SRV_PID" ]]; then shared=" SHARED-PORT:$lpids2"; fi
      m=$(cat "$WORK/res.json")
      toks=$(echo "$m" | python3 -c 'import json,sys; print(json.load(sys.stdin)["tokens"])')
      ttft=$(echo "$m" | python3 -c 'import json,sys; print(json.load(sys.stdin)["ttft_s"])')
      tps=$(echo "$m" | python3 -c 'import json,sys; v=json.load(sys.stdin)["single_tps"]; print(v if v else "-")')
      agg=$(echo "$m" | python3 -c 'import json,sys; print(json.load(sys.stdin)["agg_tps"])')
      printf '%-14s %-5s %-4s %-9s %-9s %-8s %-6s %-9s %-9s %-24s %s\n' \
        "$moe" "$spec" "$ctxs" "$load_s" "$gpu_mb" \
        "$ttft" "$toks" "$tps" "$agg" "$(spec_line || echo 'spec=?')" "$(place_line || echo 'place=?')$shared"
      stop_server
    done
  done
done
