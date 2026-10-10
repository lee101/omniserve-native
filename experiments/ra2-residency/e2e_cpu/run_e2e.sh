#!/bin/bash
set -u
B=/nvme0n1-disk/code/omniserve-native-residency/build-res/omniserve-native
M=/nvme0n1-disk/models/omniserve-native/zimage-q4
P=8794
if ss -ltnH | grep -q ":$P "; then echo "port busy"; exit 1; fi
env -i PATH=/usr/bin:/bin HOME=$HOME CUDA_VISIBLE_DEVICES= OMNISERVE_ACCESS_LOG=0 OMNISERVE_NATIVE_GUARD=0 OMNISERVE_NATIVE_GUARD_JUDGE=0 \
 OMNISERVE_NATIVE_RAM_PREFETCH_ENABLED=0 OMNISERVE_NATIVE_BIND=127.0.0.1 OMNISERVE_NATIVE_PORT=$P \
 OMNISERVE_NATIVE_SD_LIB=/nvme0n1-disk/models/omniserve-native/qwen-image-2.1/lib/libstable-diffusion-9bc7b35-release.so \
 OMNISERVE_NATIVE_SD_DIFFUSION_MODEL=$M/z_image_turbo-Q4_K.gguf OMNISERVE_NATIVE_SD_VAE=$M/split_files/vae/ae.safetensors \
 OMNISERVE_NATIVE_SD_LLM=$M/Qwen3-4B-Instruct-2507-Q4_K_M.gguf OMNISERVE_NATIVE_SD_PARAMS_BACKEND='*=cpu' \
 OMNISERVE_NATIVE_SD_MIN_FREE_MB=0 OMNISERVE_NATIVE_SD_LEASE_MB=0 OMNISERVE_NATIVE_IMAGE_PREFER_EMBEDDED=1 \
 OMNISERVE_NATIVE_SD_RESIDENCY=demand OMNISERVE_NATIVE_SD_RESIDENT_MB=4000 OMNISERVE_NATIVE_SD_STREAM_MB=100 \
 OMNISERVE_NATIVE_SD_RESIDENT_SAVED_MS=1000 OMNISERVE_NATIVE_SD_RESIDENCY_TICK_MS=500 OMNISERVE_NATIVE_SD_RESIDENT_LEASE_MB=0 \
 OMNISERVE_NATIVE_RESIDENCY_IDLE_RELEASE_S=90 OMNISERVE_NATIVE_RESIDENCY_MIN_HOLD_S=5 \
 OMNISERVE_NATIVE_VRAM_BROKER_URL=http://127.0.0.1:$P OMNISERVE_NATIVE_VRAM_OWNER=cpu-e2e \
 $B --port $P > server.log 2>&1 &
S=$!
trap 'kill $S 2>/dev/null; wait $S 2>/dev/null' EXIT
for i in $(seq 1 120); do curl -sf -m2 127.0.0.1:$P/status >/dev/null && break; sleep 1; done
st() { curl -s 127.0.0.1:$P/v1/gpu/status | python3 -c "import json,sys;r=json.load(sys.stdin)['residency'];c=r['client'];print('$1',json.dumps({'sd':c['sd'],'reason':c['reason'],'claims':[(x['owner'],x['admitted'],x['resident'],x['demand']) for x in r['broker']['claims']]}))"; }
gen() { curl -s -m600 -o /dev/null -w "gen %{http_code} %{time_total}s\n" -H 'Content-Type: application/json' 127.0.0.1:$P/v1/images/generations -d '{"prompt":"a red cube","width":128,"height":128,"steps":1,"seed":7,"cache":false}'; }
sleep 2; st idle
gen; sleep 3
for i in $(seq 1 60); do curl -s 127.0.0.1:$P/v1/gpu/status | grep -q '"switches":1' && break; sleep 2; done
st after_promote
gen; st after_gen_resident
F=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits)
for k in 1 2 3; do curl -s 127.0.0.1:$P/v1/gpu/residency -d "{\"owner\":\"sim-big\",\"pid\":1,\"resident_mb\":$F,\"stream_mb\":0,\"saved_ms\":1000000,\"requests\":5,\"resident\":false}"; echo; sleep 3; done
for i in $(seq 1 60); do curl -s 127.0.0.1:$P/v1/gpu/status | grep -q '"switches":2' && break; sleep 2; done
st after_idle_demote
gen; st final
grep -E "residency ->|promotion|demand residency" server.log
