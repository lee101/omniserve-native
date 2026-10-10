#!/usr/bin/env bash
# usage: sweep.sh "name|ENV=V ENV2=V2" ... ; runs each variant on :8793 against all prompts
cd "$(dirname "$0")/.."
export PORT=8793
for spec in "$@"; do
  name="${spec%%|*}"; envs="${spec#*|}"
  (
    set -a; source /vfast/data/code/qwen-image-2.1-bench/omniserve_qwen.env; set +a
    unset OMNISERVE_NATIVE_SD_PARAMS_BACKEND
    export OMNISERVE_NATIVE_PORT=8793
    for kv in $envs; do export "OMNISERVE_NATIVE_SD_$kv"; done
    exec ./build-qwen/omniserve-native --port 8793 >qualitybench/out/srv_$name.log 2>&1
  ) &
  pid=$!
  until curl -s -m 2 localhost:8793/status >/dev/null; do sleep 2; done
  uv run --with pillow python qualitybench/run.py --url http://127.0.0.1:8793/v1/images/generations --tag "$name" --seed "${SEED:-0}" --steps "${STEPS:-20}" ${ONLY:+--only $ONLY}
  kill $pid; wait $pid 2>/dev/null; sleep 3
done
