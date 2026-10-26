#!/usr/bin/env bash
set -uo pipefail
tag="$1"
push="${2:-}"
min_free_gb="${MIN_FREE_GB:-12}"
cd "$(dirname "${BASH_SOURCE[0]}")"
nice -n 15 ionice -c3 docker build --network host -t "$tag" . > build.log 2>&1 &
build=$!
while kill -0 "$build" 2>/dev/null; do
  free=$(df -BG --output=avail /nvme0n1-disk | tail -1 | tr -dc 0-9)
  if [ "$free" -lt "$min_free_gb" ]; then echo "DISK GUARD: only ${free}G free, stopping build"; kill "$build"; wait "$build" 2>/dev/null; docker builder prune -f >/dev/null 2>&1; exit 3; fi
  sleep 5
done
wait "$build" || { echo BUILD FAILED; tail -n 30 build.log; exit 1; }
echo BUILD OK
[ "$push" = "--push" ] && { docker push "$tag" 2>&1 | tail -n 8; echo PUSH DONE; }
docker image inspect "$tag" --format '{{.Id}} {{.Size}}'
echo ALL DONE
