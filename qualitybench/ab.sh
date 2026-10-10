#!/usr/bin/env bash
# usage: [LIB=libstable-diffusion.so] [PORT=8793] ab.sh TAG [OMNISERVE_NATIVE_SD_X=Y ...]
# Starts a private lane with the prod config, renders the 4 bench prompts at seed 0, logs to out/ab.log, stops it.
cd "$(dirname "$0")/.."
tag=$1; shift
port=${PORT:-8793}
export RA2_PORT=$port OMNISERVE_NATIVE_SD_LOG=1
[ -n "$LIB" ] && export RA2_SD_LIB=$LIB
for kv in "$@"; do export "$kv"; done
mkdir -p qualitybench/out
tools/start-ra2-lane.sh --foreground >"qualitybench/out/$tag.server.log" 2>&1 &
pid=$!
until curl -fsS -m 3 "http://127.0.0.1:$port/health" >/dev/null 2>&1; do kill -0 $pid 2>/dev/null || { echo "$tag server died" >&2; exit 1; }; sleep 1; done
curl -s -m 600 -X POST "http://127.0.0.1:$port/v1/images/generations" -H 'Content-Type: application/json' -d '{"prompt":"warm","size":"512x512"}' -o /dev/null
ONLY=${ONLY:-fox2,trio,duo,quad} uv run --with pillow python qualitybench/run.py --url "http://127.0.0.1:$port/v1/images/generations" --tag "$tag" --seed 0 --steps 30 --extra "${EXTRA:-{\}}" --only "${ONLY:-fox2,trio,duo,quad}" 2>&1 | rg "wall=|HTTP" | tee -a qualitybench/out/ab.log
kill $pid; wait $pid 2>/dev/null
