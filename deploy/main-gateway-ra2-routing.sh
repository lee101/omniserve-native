#!/usr/bin/env bash
# Point the main prod gateway (omniserve-native.service, :8791) at the ra2 sibling instance so
# "model":"ra2" image requests share the same admission/permits, and swap in the ra2-overflow
# binary built against the prod stable-diffusion.cpp fork. Run ON PROD as administrator:
#   bash deploy/main-gateway-ra2-routing.sh install   # drop-in + binary swap, no restart
#   bash deploy/main-gateway-ra2-routing.sh restart   # restart + health check + rollback on failure
set -euo pipefail
STEP=${1:-install}
BIN=/nvme0n1-disk/code/omniserve-native-ra2overflow/build-full-ra2/omniserve-native
UNIT=omniserve-native.service
DROPIN=/etc/systemd/system/$UNIT.d/ra2-routing.conf
S=$(sudo grep -h OMNISERVE_NATIVE_SECRET /etc/omniserve-qwen.env | sed 's/.*=//')
CUR=$(systemctl show $UNIT -p ExecStart --value | grep -oE 'path=[^ ;]+' | head -1 | cut -d= -f2)
case $STEP in
  install)
    sudo mkdir -p "$(dirname "$DROPIN")"
    sudo tee "$DROPIN" > /dev/null <<CONF
[Service]
Environment=OMNISERVE_NATIVE_IMAGE_MODEL_UPSTREAMS=ra2=http://127.0.0.1:8792,qwen-image-2.1=http://127.0.0.1:8792,qwen=http://127.0.0.1:8792
Environment=OMNISERVE_NATIVE_IMAGE_MODEL_UPSTREAM_SECRET=$S
ExecStart=
ExecStart=$BIN --port 8791
CONF
    sudo chmod 600 "$DROPIN"; sudo systemctl daemon-reload
    echo "installed drop-in; current binary: $CUR -> $BIN (takes effect on restart)";;
  restart)
    echo "previous ExecStart binary: $CUR"
    sudo systemctl restart $UNIT; ok=0
    for i in $(seq 1 180); do curl -s -m 2 -o /dev/null -w '%{http_code}' localhost:8791/status | grep -q 200 && { ok=1; break; }; sleep 1; done
    if [ $ok -ne 1 ]; then echo "!! gateway unhealthy, rolling back drop-in"; sudo rm -f "$DROPIN"; sudo systemctl daemon-reload; sudo systemctl restart $UNIT; exit 1; fi
    curl -s localhost:8791/status | python3 -c 'import sys,json;d=json.load(sys.stdin);print("diffusion",d.get("diffusion"),"llm",d.get("llm",{}).get("ready"),"model_upstreams",d.get("image_model_upstreams"))'
    curl -s -m 200 localhost:8791/v1/images/generations -d '{"model":"ra2","prompt":"a red apple","width":512,"height":512,"steps":8,"seed":1}' | head -c 120; echo;;
  *) echo "usage: $0 install|restart"; exit 2;;
esac
