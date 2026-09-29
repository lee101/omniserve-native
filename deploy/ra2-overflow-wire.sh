#!/usr/bin/env bash
# Wire the prod ra2 instance (:8792) to its app.nz overflow cog. Run ON PROD as administrator after
# ghcr.io/lee101/omniserve-native:ra2 is pushed and app.nz is deployed with the predict-sync seam.
#   bash deploy/ra2-overflow-wire.sh cog      # create/lookup the qwen-image-2.1 cog, print its id
#   bash deploy/ra2-overflow-wire.sh env <id> # append overflow env to /etc/omniserve-qwen.env and restart the unit
#   bash deploy/ra2-overflow-wire.sh status
set -euo pipefail
STEP=${1:-status}
APPNZ=${APPNZ_BASE:-http://127.0.0.1:8787}
KEYFILE=/etc/omniserve-ra2-appnz.key   # app.nz API key (papers_api row named omniserve-ra2-overflow), 0600
key() { sudo cat "$KEYFILE" 2>/dev/null | tr -d '\n'; }
case $STEP in
  cog)
    K=$(key); [ -n "$K" ] || { echo "no app.nz API key found in the main gateway env" >&2; exit 1; }
    curl -sS -X POST "$APPNZ/api/cogs/run" -H "Authorization: Bearer $K" -H 'Content-Type: application/json' \
      -d '{"template":"qwen-image-2.1","input":{"prompt":"a red fox in snow, overflow smoke","width":768,"height":768,"steps":8}}' \
      | python3 -c 'import sys,json; d=json.load(sys.stdin); print(json.dumps({k:d.get(k) for k in ("model","prediction","error") if k in d})[:600]); m=d.get("model") or {}; print("COG_ID", m.get("id",""))';;
  env)
    ID=${2:?cog id}; K=$(key)
    sudo sed -i '/^OMNISERVE_NATIVE_IMAGE_OVERFLOW_/d;/^OMNISERVE_NATIVE_OVERFLOW_TIERS/d' /etc/omniserve-qwen.env
    printf 'OMNISERVE_NATIVE_IMAGE_OVERFLOW_UPSTREAM=%s/api/cogs/%s\nOMNISERVE_NATIVE_IMAGE_OVERFLOW_PATH=/predict-sync\nOMNISERVE_NATIVE_IMAGE_OVERFLOW_API_KEY=%s\nOMNISERVE_NATIVE_IMAGE_OVERFLOW_TIMEOUT_MS=600000\nOMNISERVE_NATIVE_OVERFLOW_TIERS=paid\n' "$APPNZ" "$ID" "$K" | sudo tee -a /etc/omniserve-qwen.env > /dev/null
    sudo systemctl restart omniserve-native-qwen; for i in $(seq 1 90); do curl -s -m 2 localhost:8792/status >/dev/null && break; sleep 2; done
    curl -s localhost:8792/status | python3 -c 'import sys,json;d=json.load(sys.stdin);print("overflow:",d.get("overflow"),"upstreams:",{k:v for k,v in (d.get("upstreams") or {}).items() if "overflow" in k or "image" in k})';;
  status) curl -s localhost:8792/status | python3 -c 'import sys,json;d=json.load(sys.stdin);print("overflow:",d.get("overflow"))';;
  *) echo "usage: $0 cog|env <id>|status"; exit 2;;
esac
