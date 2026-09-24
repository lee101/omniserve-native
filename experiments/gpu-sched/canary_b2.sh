#!/bin/bash
# Z-Image canary on :8797 leasing from the broker canary :8796. Aborts on any prod 5xx.
set -u
D=$(cd "$(dirname "$0")/../.." && pwd); LOG=${LOG:-/tmp/canaryB.log}; N=${N:-3}; SIZE=${SIZE:-512}
p5(){ curl -s localhost:8791/metrics | awk -F' ' '/responses_total\{class="5xx"\}/{a=$2} END{print a+0}'; }
q5(){ curl -s localhost:8792/metrics | awk -F' ' '/responses_total\{class="5xx"\}/{a=$2} END{print a+0}'; }
B0=$(p5); Q0=$(q5)
env -i PATH=/usr/bin:/bin HOME=$HOME OMNISERVE_NATIVE_BIND=127.0.0.1 OMNISERVE_ACCESS_LOG=0 \
 OMNISERVE_NATIVE_VRAM_BROKER_URL=${BROKER:-http://127.0.0.1:8796} OMNISERVE_NATIVE_VRAM_OWNER=canary-zimage \
 OMNISERVE_NATIVE_SD_LEASE_MB=${LEASE:-4096} OMNISERVE_NATIVE_SD_MIN_FREE_MB=4096 OMNISERVE_NATIVE_VRAM_WAIT_MS_BACKGROUND=20000 \
 OMNISERVE_NATIVE_RAM_PREFETCH_ENABLED=0 OMNISERVE_NATIVE_GUARD=0 OMNISERVE_NATIVE_GUARD_JUDGE=0 OMNISERVE_NATIVE_SLOTS=1 OMNISERVE_NATIVE_IMAGE_PERMITS=1 \
 OMNISERVE_NATIVE_SD_DIFFUSION_MODEL=/nvme0n1-disk/models/omniserve-native/zimage-q4/z_image_turbo-Q4_K.gguf \
 OMNISERVE_NATIVE_SD_VAE=/nvme0n1-disk/models/omniserve-native/zimage-q4/split_files/vae/ae.safetensors \
 OMNISERVE_NATIVE_SD_LLM=/nvme0n1-disk/models/omniserve-native/zimage-q4/Qwen3-4B-Instruct-2507-Q4_K_M.gguf \
 OMNISERVE_NATIVE_SD_STREAM_LAYERS=${STREAM:-1} OMNISERVE_NATIVE_SD_MMAP=1 OMNISERVE_NATIVE_SD_EAGER_LOAD=0 OMNISERVE_NATIVE_SD_PARAMS_BACKEND=${PARAMS-cpu} \
 OMNISERVE_NATIVE_SD_MAX_VRAM=${MAXV-2} OMNISERVE_NATIVE_SD_ZERO_GUIDANCE=1.0 OMNISERVE_NATIVE_SD_VAE_TILING=1 OMNISERVE_NATIVE_SD_VAE_TILE_X=64 OMNISERVE_NATIVE_SD_VAE_TILE_Y=64 \
 OMNISERVE_NATIVE_IMAGE_PREFER_EMBEDDED=1 OMNISERVE_NATIVE_SD_IMAGE_FORMAT=webp \
 setsid "$D/build-sched/omniserve-native" --port 8797 > "$LOG" 2>&1 < /dev/null &
CP=$!
trap 'kill $CP 2>/dev/null' EXIT
for i in $(seq 60); do curl -fs -o /dev/null localhost:8797/readyz && break; sleep 1; done
for i in $(seq $N); do
  if [ -n "${HOLD_S:-}" ]; then
    H=$(curl -s localhost:8796/v1/gpu/status | python3 -c "import json,sys;print(json.load(sys.stdin)['ledger']['headroom_mb']['paid'])")
    HID=$(curl -s -XPOST localhost:8796/v1/gpu/lease -d "{\"owner\":\"holder\",\"mb\":$((H-2000)),\"tier\":\"paid\",\"ttl_s\":60}" | python3 -c "import json,sys;print(json.load(sys.stdin)['lease_id'])")
    ( sleep "$HOLD_S"; curl -s -XPOST localhost:8796/v1/gpu/release -d "{\"lease_id\":\"$HID\"}" >/dev/null ) &
    ( for k in 1 2 3 4 5 6 7 8 9 10 11 12; do sleep 1.5; curl -s localhost:8796/v1/gpu/status | python3 -c "import json,sys;d=json.load(sys.stdin)['ledger'];print('   t+',$k*1.5,'leases',[(l['owner'],l['mb'],l['charged_mb']) for l in d['leases']],'waiting',d['waiting'])"; done ) &
  fi
  t=$(date +%s.%N)
  code=$(curl -s -o /tmp/canaryB.$i.json -w '%{http_code}' -XPOST localhost:8797/v1/images/generations -H 'X-Omniserve-Tier: background' -H 'Content-Type: application/json' \
    -d "{\"prompt\":\"a lighthouse on a cliff at dusk, canary $i $RANDOM\",\"size\":\"${SIZE}x${SIZE}\",\"seed\":$RANDOM}")
  echo "render $i http=$code wall=$(echo "$(date +%s.%N) - $t" | bc)s prod5xx=$(( $(p5) - B0 )) qwen5xx=$(( $(q5) - Q0 ))"
  curl -s localhost:8796/v1/gpu/status | python3 -c "import json,sys;d=json.load(sys.stdin)['ledger'];print('  broker',d['counters'],[ (o['owner'],o['waits'],o['wait_ms_max']) for o in d['owners']])"
  [ $(( $(p5) - B0 + $(q5) - Q0 )) -gt 0 ] && { echo "ABORT: prod 5xx rose"; break; }
done
