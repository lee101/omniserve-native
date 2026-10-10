#!/bin/bash
# usage: run.sh NAME ARMS CASES [bench args...]
N=$1 A=$2 C=$3; shift 3
cd /nvme0n1-disk/code/omniserve-native-residency
exec python3 /nvme0n1-disk/code/monitoring/run_agent.py flock /nvme0n1-disk/tmp/gpu-experiments.lock \
  uv run -q --with requests --with pillow --with numpy python tools/ra2_residency_bench.py \
  --arms experiments/ra2-residency/$A --cases "$C" --out experiments/ra2-residency/$N "$@"
