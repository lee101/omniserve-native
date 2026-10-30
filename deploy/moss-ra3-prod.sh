#!/usr/bin/env bash
# Stand up the ra3 lanes (MOSS-Image, internal) on a GPU host next to the other omniserve instances.
# Idempotent; run ON the target host as the service user:
#   bash deploy/moss-ra3-prod.sh [check|build|install|start|smoke|all]
#   LANES="fast hq" (default) | LANES=fast
# fast = :8793 (4-step merged, ~3.5 s/1024px). hq = :8794 (30-step CFG 3 + EasyCache; also serves /v1/images/edits task=clean|colorize).
# Netwrck reaches them via RA3_LOCAL_IMAGE_URL / RA3_LOCAL_HQ_IMAGE_URL / RA3_LOCAL_IMAGE_SECRET (docs/moss-ra3-lanes.md).
# The lane recipe lives in qualitybench/serve_moss.sh; the units only call it, so recipe changes need no unit edits.
set -euo pipefail
STEP=${1:-all}
LANES=${LANES:-"fast hq"}
OMNI=${OMNI:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
SDCPP=${SDCPP:-$OMNI/../sdcpp-moss}   # needs the moss-i2i-reference branch (github.com/lee101/stable-diffusion.cpp-opt)
MODELS=${MOSS_DIR:-/vfast/data/models/moss}
CUDA_ARCH=${CUDA_ARCH:-native}
ENVFILE=/etc/omniserve-ra3.env
port() { [ "$1" = hq ] && echo 8794 || echo 8793; }

check() {
  local miss=0
  for f in sft4/MOSS-Image-SFT-4step-v2.0-merged-Q4_0.gguf sft/MOSS-Image-SFT-Q4_K.gguf te/Qwen3.5-9B-Q8_0.gguf \
           vae/flux2-vae.safetensors tok/tokenizer.json embed/modelint8_512dim.safetensors embed/tokenizer.json \
           lora/moss_i2i_sanitize_v3.safetensors lora/moss_colorize_3k_v4.safetensors lora/moss-image-pixelart-16x16.safetensors; do
    [ -e "$MODELS/$f" ] || { echo "missing $MODELS/$f" >&2; miss=1; }
  done
  [ "$miss" = 0 ] && echo "models ok ($MODELS)"
  return $miss
}

build() {
  git -C "$SDCPP" submodule update --init -q
  cmake -S "$SDCPP" -B "$SDCPP/build" -DSD_CUDA=ON -DSD_BUILD_SHARED_LIBS=ON -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=$CUDA_ARCH > "$SDCPP/build_configure.log"
  cmake --build "$SDCPP/build" -j "$(nproc)" --target stable-diffusion > "$SDCPP/build.log"
  cmake -S "$OMNI" -B "$OMNI/build-moss" -DWITH_SD=ON -DSD_DIR="$SDCPP" -DCMAKE_BUILD_TYPE=Release > "$OMNI/build-moss-configure.log"
  cmake --build "$OMNI/build-moss" -j "$(nproc)" --target omniserve-native > "$OMNI/build-moss.log"
  ls -la "$SDCPP/build/bin/libstable-diffusion.so" "$OMNI/build-moss/omniserve-native"
}

install() {
  check
  for lane in $LANES; do
    sudo tee /etc/systemd/system/omniserve-native-ra3-$lane.service > /dev/null <<EOF
[Unit]
Description=omniserve-native ra3 $lane lane (port $(port $lane))
After=network.target nvidia-persistenced.service
[Service]
User=$(id -un)
WorkingDirectory=$OMNI
EnvironmentFile=-$ENVFILE
Environment=LANE=$lane PORT=$(port $lane) MOSS_DIR=$MODELS MOSS_BUILD=$OMNI/build-moss MOSS_SD_LIB=$SDCPP/build/bin/libstable-diffusion.so
Environment=OMNISERVE_NATIVE_SD_WEBP_QUALITY=85
ExecStart=/usr/bin/env bash $OMNI/qualitybench/serve_moss.sh
Restart=always
RestartSec=5
[Install]
WantedBy=multi-user.target
EOF
    sudo systemctl enable omniserve-native-ra3-$lane.service
  done
  if [ ! -f "$ENVFILE" ]; then
    # shared secret: reuse the existing omniserve one so netwrck keeps a single value
    S=$(sudo grep -h OMNISERVE_NATIVE_SECRET /etc/omniserve-qwen.env /etc/omniserve-h3.env 2>/dev/null | head -1 || true)
    echo "${S:-OMNISERVE_NATIVE_SECRET=change-me}" | sudo tee "$ENVFILE" > /dev/null
    sudo chmod 600 "$ENVFILE"
  fi
  sudo systemctl daemon-reload
}

start() {
  for lane in $LANES; do sudo systemctl restart omniserve-native-ra3-$lane.service; done
  sleep 30
  for lane in $LANES; do systemctl --no-pager status omniserve-native-ra3-$lane.service | head -4; done
}

smoke() {
  SECRET=$(sudo grep -h OMNISERVE_NATIVE_SECRET "$ENVFILE" 2>/dev/null | sed 's/.*=//' || true)
  for lane in $LANES; do
    echo "== $lane :$(port $lane)"
    curl -s "localhost:$(port $lane)/status" | head -c 300; echo
    curl -s "localhost:$(port $lane)/v1/images/generations" -H "X-API-Key: $SECRET" \
      -d '{"prompt":"a lighthouse at dusk, painterly","width":1024,"height":1024,"seed":3,"output_format":"webp"}' \
      | python3 -I -c 'import sys,json; d=json.load(sys.stdin); print({k:v for k,v in d.items() if k!="data"}, d["data"][0].get("inference_time_ms"), "ms")'
  done
  echo "netwrck env: RA3_LOCAL_IMAGE_URL=http://127.0.0.1:8793/v1/images/generations RA3_LOCAL_HQ_IMAGE_URL=http://127.0.0.1:8794/v1/images/generations RA3_LOCAL_IMAGE_SECRET=<secret in $ENVFILE>"
}

case $STEP in
  check) check;; build) build;; install) install;; start) start;; smoke) smoke;;
  all) check; build; install; start; smoke;;
  *) echo "unknown step $STEP"; exit 1;;
esac
