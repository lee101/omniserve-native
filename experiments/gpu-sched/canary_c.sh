#!/bin/bash
# Eviction/reload + judge-admission canary on :8796 with a 0.6B LLM. Aborts on prod 5xx.
set -u
D=$(cd "$(dirname "$0")/../.." && pwd); LOG=${LOG:-/tmp/canaryC.log}; B=http://127.0.0.1:8796
p5(){ curl -s localhost:8791/metrics | awk '/responses_total\{class="5xx"\}/{a=$2} END{print a+0}'; }
B0=$(p5)
env -i PATH=/usr/bin:/bin HOME=$HOME OMNISERVE_NATIVE_BIND=127.0.0.1 OMNISERVE_ACCESS_LOG=0 OMNISERVE_NATIVE_VRAM_BROKER=1 \
 OMNISERVE_NATIVE_VRAM_KEEP_FREE_MB=1024 OMNISERVE_NATIVE_RAM_PREFETCH_ENABLED=0 OMNISERVE_NATIVE_GUARD=0 \
 OMNISERVE_NATIVE_LLM_GGUF=/nvme0n1-disk/models/qwen3-0.6b-q8.gguf OMNISERVE_NATIVE_NGL=99 OMNISERVE_NATIVE_CTX=2048 \
 OMNISERVE_NATIVE_LLM_CONTEXTS=1 OMNISERVE_NATIVE_SLOTS=2 \
 OMNISERVE_NATIVE_EVICT_LLM_IDLE_S=5 OMNISERVE_NATIVE_EVICT_LLM_MAX_TIER=background OMNISERVE_NATIVE_VRAM_PRESSURE_AFTER_MS=1000 \
 OMNISERVE_NATIVE_GUARD_JUDGE=${JUDGE:-0} OMNISERVE_NATIVE_GUARD_JUDGE_AUTO=${JUDGE:-0} OMNISERVE_NATIVE_GUARD_JUDGE_MARGIN_MB=4096 \
 setsid "$D/build-sched/omniserve-native" --port 8796 > "$LOG" 2>&1 < /dev/null &
trap 'P=$(ss -ltnp | rg ":8796 " | rg -o "pid=[0-9]+" | cut -d= -f2); [ -n "$P" ] && kill $P' EXIT
for i in $(seq 60); do curl -fs -o /dev/null $B/readyz && break; sleep 1; done
chat(){ curl -s -o /dev/null -w "http=%{http_code} t=%{time_total}s" -XPOST $B/v1/chat/completions -H 'Content-Type: application/json' -d '{"messages":[{"role":"user","content":"Say hi."}],"max_tokens":8}'; }
st(){ curl -s $B/v1/gpu/status | python3 -c "import json,sys;d=json.load(sys.stdin);s=d['sched'];print('  llm',s['llm'],'judge',s['judge'],'counters',d['ledger']['counters'])"; }
echo "chat1 $(chat)"; st
if [ "${JUDGE:-0}" = 1 ]; then for i in $(seq 12); do sleep 10; curl -s $B/v1/gpu/status | rg -q '"judge":\{"auto":true,"loaded":true' && break; done; echo "judge admitted after ~$((i*10))s"; st; fi
sleep 6
H=$(curl -s $B/v1/gpu/status | python3 -c "import json,sys;print(json.load(sys.stdin)['ledger']['headroom_mb']['paid'])")
HID=$(curl -s -XPOST $B/v1/gpu/lease -d "{\"owner\":\"holder\",\"mb\":$((H+200)),\"tier\":\"paid\",\"ttl_s\":60}" | python3 -c "import json,sys;print(json.load(sys.stdin)['lease_id'])")
echo "pressure lease: $(curl -s -XPOST $B/v1/gpu/lease -d '{"owner":"needs-700","mb":700,"min_mb":700,"tier":"paid","wait_ms":15000}')"
curl -s -XPOST $B/v1/gpu/release -d "{\"lease_id\":\"$HID\"}" >/dev/null
st
echo "chat2 (reload) $(chat)"; st
echo "chat3 $(chat)"
rg "gpu-sched" "$LOG"
echo "prod5xx delta=$(( $(p5) - B0 ))"
