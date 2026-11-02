#!/usr/bin/env bash
# Single-image smoke test for the MOSS-Image lane: build the gateway if needed, start
# qualitybench/serve_moss.sh, render one image, write it under moss-out/, stop the lane.
# Reuses a lane already answering on PORT.
# Env: LANE=sfw|fast|hq|nsfw, MOSS_DIR, MOSS_BUILD, MOSS_SD_LIB, PORT, OUT, W/H/STEPS/SEED,
#      WAIT_S, KEEP=1 (leave the lane running).
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
PORT=${PORT:-8794}
OUT=${OUT:-$ROOT/moss-out}
LANE=${LANE:-sfw}
W=${W:-1024}; H=${H:-1024}; STEPS=${STEPS:-4}; SEED=${SEED:-3}
BUILD=${MOSS_BUILD:-/vfast/data/code/omniserve-native-moss-build}
SD_LIB=${MOSS_SD_LIB:-/vfast/data/code/sdcpp-moss/build/bin/libstable-diffusion.so}
mkdir -p "$OUT"

if [ ! -x "$BUILD/omniserve-native" ]; then
  [ -f "$SD_LIB" ] || { echo "missing MOSS-capable libstable-diffusion.so at $SD_LIB (set MOSS_SD_LIB)" >&2; exit 1; }
  SD_DIR=$(cd "$(dirname "$SD_LIB")/../.." && pwd)
  cmake -S "$ROOT" -B "$BUILD" -G Ninja -DWITH_SD=ON -DSD_DIR="$SD_DIR" -DCMAKE_BUILD_TYPE=Release >"$OUT/configure.log"
  cmake --build "$BUILD" -j"$(nproc)" --target omniserve-native >"$OUT/build.log"
fi

health() { curl -fsS -m 2 "http://127.0.0.1:$PORT/status" >/dev/null 2>&1; }
PID=
if health; then
  echo "reusing lane on :$PORT" >&2
else
  LANE=$LANE PORT=$PORT bash "$ROOT/qualitybench/serve_moss.sh" >"$OUT/server.log" 2>&1 &
  PID=$!
  [ "${KEEP:-0}" = 1 ] || trap '[ -n "$PID" ] && kill $PID 2>/dev/null; wait $PID 2>/dev/null || true' EXIT
  for _ in $(seq "${WAIT_S:-900}"); do
    health && break
    kill -0 "$PID" 2>/dev/null || { tail -20 "$OUT/server.log" >&2; exit 1; }
    sleep 2
  done
  health || { tail -20 "$OUT/server.log" >&2; exit 1; }
fi

[ $# -gt 0 ] || set -- "a red fox sitting in snow, photograph"
i=0
for p in "$@"; do
  python3 - "$PORT" "$OUT/$(printf %02d "$i").webp" "$p" "$W" "$H" "$STEPS" "$((SEED + i))" <<'PY'
import base64, json, sys, time, urllib.request
port, path, prompt, w, h, steps, seed = sys.argv[1:]
body = json.dumps({"prompt": prompt, "width": int(w), "height": int(h), "steps": int(steps),
                   "seed": int(seed), "guidance_scale": 1.0, "output_format": "webp"}).encode()
req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/images/generations", body,
                             {"Content-Type": "application/json"})
t = time.time()
d = json.load(urllib.request.urlopen(req, timeout=1800))
open(path, "wb").write(base64.b64decode(d["data"][0]["b64_json"]))
print(f"{path} {time.time() - t:.1f}s {d['data'][0].get('inference_time_ms', '?')}ms {d.get('model', '')}")
PY
  i=$((i + 1))
done
