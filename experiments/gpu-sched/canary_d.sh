#!/bin/bash
# Qwen ra2 canary on :8798 (sched binary) leasing its peak from the prod :8791 broker at paid tier.
# Prod Z-Image queues behind it instead of OOMing. Aborts on any prod 5xx.
set -u
D=$(cd "$(dirname "$0")/../.." && pwd); ENVF=${ENVF:?qwen env file}; LOG=${LOG:-/tmp/canaryD.log}; N=${N:-2}; SIZE=${SIZE:-768}
p5(){ curl -s localhost:$1/metrics | awk '/responses_total\{class="5xx"\}/{a=$2} END{print a+0}'; }
A0=$(p5 8791); Q0=$(p5 8792)
env -i PATH=/usr/bin:/bin HOME=$HOME $(cat "$ENVF") OMNISERVE_ACCESS_LOG=0 OMNISERVE_NATIVE_GUARD=0 OMNISERVE_NATIVE_GUARD_JUDGE=0 \
 OMNISERVE_NATIVE_RAM_PREFETCH_ENABLED=0 OMNISERVE_NATIVE_VRAM_BROKER_URL=http://127.0.0.1:8791 OMNISERVE_NATIVE_VRAM_OWNER=canary-qwen \
 OMNISERVE_NATIVE_SD_LEASE_MB=10240 OMNISERVE_NATIVE_SD_EDIT_LEASE_MB=11264 OMNISERVE_NATIVE_SD_LEASE_SCALE_PIXELS=1 OMNISERVE_NATIVE_VRAM_WAIT_MS_PAID=20000 OMNISERVE_NATIVE_VRAM_FORCE_TIERS=paid,sub,free,background \
 setsid "$D/deploy-sched-${REV:?}/omniserve-native-qwen" --port 8798 > "$LOG" 2>&1 < /dev/null &
trap 'P=$(ss -ltnp | rg ":8798 " | rg -o "pid=[0-9]+" | cut -d= -f2); [ -n "$P" ] && kill $P' EXIT
for i in $(seq 180); do curl -fs -o /dev/null localhost:8798/readyz && break; sleep 1; done
echo "ready after ${i}s"
for i in $(seq $N); do
  t=$(date +%s.%N)
  ( for k in $(seq 20); do sleep 2; curl -s localhost:8791/v1/gpu/status | python3 -c "import json,sys;d=json.load(sys.stdin)['ledger'];print('   +%ds'%($k*2),'free',d['device_free_mb'],'leases',[(l['owner'],l['tier'],l['mb'],l['charged_mb']) for l in d['leases']],'waiting',{k:v for k,v in d['waiting'].items() if v})"; done ) &
  W=$!
  code=$(curl -s -o /dev/null -w '%{http_code}' -XPOST localhost:8798/v1/images/generations -H 'X-Omniserve-Tier: paid' -H 'Content-Type: application/json' \
    -d "{\"prompt\":\"a red fox in snowy birch forest, canary $i $RANDOM\",\"size\":\"${SIZE}x${SIZE}\",\"seed\":$RANDOM}")
  kill $W 2>/dev/null
  echo "render $i http=$code wall=$(echo "$(date +%s.%N) - $t" | bc)s prod8791_5xx=$(( $(p5 8791) - A0 )) prod8792_5xx=$(( $(p5 8792) - Q0 ))"
  [ $(( $(p5 8791) - A0 + $(p5 8792) - Q0 )) -gt 0 ] && { echo "ABORT: prod 5xx rose"; break; }
done
rg -i "gpu-sched|out of memory|failed" "$LOG" | head
