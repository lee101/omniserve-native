#!/usr/bin/env bash
# Stand up the Qwen Image 2.1 ("ra2") omniserve-native instance on prod next to the
# existing zimage instance. Idempotent; run ON PROD as administrator:
#   bash deploy/qwen-ra2-prod.sh [build|models|install|start|smoke|all]
# Nothing here touches the zimage service (omniserve-native.service, port 8791).
set -euo pipefail
STEP=${1:-all}
CODE=/nvme0n1-disk/code
SD_MASTER=$CODE/stable-diffusion.cpp-master          # upstream master worktree (Qwen 2.1 + cache modes)
OMNI=${OMNI:-$CODE/omniserve-native-qwen}                 # git worktree of branch qwen-ra2, separate from the zimage checkout
MODELS=/nvme0n1-disk/models/omniserve-native/qwen-image-2.1
PORT=${RA2_PORT:-8792}
CUDA_ARCH=${CUDA_ARCH:-120}                            # RTX 5090
HF=${HF:-/nvme0n1-disk/code/omniserve-native/.venv/bin/hf}
NVCC=${NVCC:-/usr/local/cuda-12.9/bin/nvcc}

build() {
  if [ ! -d "$SD_MASTER" ]; then
    git -C "$CODE/stable-diffusion.cpp" fetch -q origin
    git -C "$CODE/stable-diffusion.cpp" worktree add "$SD_MASTER" origin/master
  else
    git -C "$SD_MASTER" fetch -q origin && git -C "$SD_MASTER" checkout -q --detach origin/master
  fi
  git -C "$SD_MASTER" submodule update --init -q
  cmake -S "$SD_MASTER" -B "$SD_MASTER/build" -DSD_CUDA=ON -DSD_BUILD_SHARED_LIBS=ON -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_CUDA_ARCHITECTURES=$CUDA_ARCH -DCMAKE_CUDA_COMPILER=$NVCC -DGGML_CUDA_FA=ON > "$SD_MASTER/build_configure.log"
  cmake --build "$SD_MASTER/build" -j "$(nproc)" --target stable-diffusion sd-cli > "$SD_MASTER/build.log"
  cmake -S "$OMNI" -B "$OMNI/build-qwen" -DWITH_SD=ON -DSD_DIR="$SD_MASTER" -DCMAKE_BUILD_TYPE=Release > "$OMNI/build-qwen-configure.log"
  cmake --build "$OMNI/build-qwen" -j "$(nproc)" --target omniserve-native > "$OMNI/build-qwen.log"
  ls -la "$SD_MASTER/build/bin/libstable-diffusion.so" "$OMNI/build-qwen/omniserve-native"
}

models() {
  mkdir -p "$MODELS"
  FREE_GB=$(df -BG --output=avail /nvme0n1-disk | tail -1 | tr -dc 0-9)
  if [ "$FREE_GB" -lt 14 ]; then echo "need >=14 GB free on /nvme0n1-disk for the Qwen 2.1 models, have ${FREE_GB}G; free space first" >&2; exit 1; fi
  $HF download abenzerps/Qwen-Image-2.1-Uncensored-GGUF qwen-image-2.1-Q4_K_M.gguf vae/qwen_image_2.1_vae_bf16.safetensors --local-dir "$MODELS"
  $HF download Qwen/Qwen3-VL-8B-Instruct-GGUF Qwen3VL-8B-Instruct-Q4_K_M.gguf mmproj-Qwen3VL-8B-Instruct-F16.gguf --local-dir "$MODELS"
  ls -la "$MODELS" "$MODELS/vae"
}

install() {
  sudo tee /etc/systemd/system/omniserve-native-qwen.service > /dev/null <<EOF
[Unit]
Description=omniserve-native Qwen Image 2.1 (ra2) gateway (port $PORT)
After=network.target nvidia-persistenced.service
[Service]
User=administrator
WorkingDirectory=$OMNI
EnvironmentFile=-/etc/omniserve-qwen.env
Environment=OMNISERVE_NATIVE_BIND=127.0.0.1
Environment=OMNISERVE_NATIVE_PORT=$PORT
Environment=OMNISERVE_NATIVE_SLOTS=1
Environment=OMNISERVE_NATIVE_IMAGE_PERMITS=1
Environment=OMNISERVE_NATIVE_IMAGE_PREFER_EMBEDDED=1
Environment=OMNISERVE_NATIVE_SD_LIB=$SD_MASTER/build/bin/libstable-diffusion.so
Environment=OMNISERVE_NATIVE_SD_DIFFUSION_MODEL=$MODELS/qwen-image-2.1-Q4_K_M.gguf
Environment=OMNISERVE_NATIVE_SD_LLM=$MODELS/Qwen3VL-8B-Instruct-Q4_K_M.gguf
Environment=OMNISERVE_NATIVE_SD_LLM_VISION=$MODELS/mmproj-Qwen3VL-8B-Instruct-F16.gguf
Environment=OMNISERVE_NATIVE_SD_VAE=$MODELS/vae/qwen_image_2.1_vae_bf16.safetensors
Environment=OMNISERVE_NATIVE_SD_REFERENCE_EDIT=1
Environment=OMNISERVE_NATIVE_SD_EAGER_LOAD=1
Environment=OMNISERVE_NATIVE_SD_PARAMS_BACKEND=te=cpu
Environment=OMNISERVE_NATIVE_SD_MIN_FREE_MB=256
Environment=OMNISERVE_NATIVE_SD_VAE_TILING=1
Environment=OMNISERVE_NATIVE_SD_VAE_TILE_X=32
Environment=OMNISERVE_NATIVE_SD_VAE_TILE_Y=32
Environment=OMNISERVE_NATIVE_SD_CACHE_MODE=easycache
Environment=OMNISERVE_NATIVE_SD_EASYCACHE_THRESHOLD=0.2
Environment=OMNISERVE_NATIVE_SD_ZERO_GUIDANCE=1.0
Environment=OMNISERVE_NATIVE_SD_WEBP_QUALITY=88
ExecStart=$OMNI/build-qwen/omniserve-native --port $PORT
Restart=always
RestartSec=5
[Install]
WantedBy=multi-user.target
EOF
  # secret shared with the zimage instance so netwrck/cutedsl can reuse their config
  if [ ! -f /etc/omniserve-qwen.env ]; then
    SECRET=$(sudo grep -h OMNISERVE_NATIVE_SECRET /etc/omniserve-h3.env /etc/systemd/system/omniserve-native.service /etc/systemd/system/omniserve-native.service.d/*.conf 2>/dev/null | head -1 | sed 's/^Environment=//' || true)
    echo "${SECRET:-OMNISERVE_NATIVE_SECRET=change-me}" | sudo tee /etc/omniserve-qwen.env > /dev/null
    sudo chmod 600 /etc/omniserve-qwen.env
  fi
  sudo systemctl daemon-reload
  sudo systemctl enable omniserve-native-qwen.service
}

start() { sudo systemctl restart omniserve-native-qwen.service; sleep 20; systemctl --no-pager status omniserve-native-qwen.service | head -8; }

smoke() {
  SECRET=$(sudo grep -h OMNISERVE_NATIVE_SECRET /etc/omniserve-qwen.env 2>/dev/null | sed 's/.*=//' || true)
  curl -s "localhost:$PORT/status" | head -c 400; echo
  time curl -s "localhost:$PORT/v1/images/generations" -H "X-API-Key: $SECRET" \
    -d '{"prompt":"editorial portrait of a smiling woman, golden hour, 85mm","width":1024,"height":1024,"steps":20,"seed":3,"output_format":"webp"}' \
    | python3 -c 'import sys,json,base64; d=json.load(sys.stdin); open("/tmp/ra2_smoke.webp","wb").write(base64.b64decode(d["data"][0]["b64_json"])); print({k:v for k,v in d.items() if k!="data"}, d["data"][0]["inference_time_ms"], "ms")'
}

case $STEP in
  build) build;; models) models;; install) install;; start) start;; smoke) smoke;;
  all) build; models; install; start; smoke;;
  *) echo "unknown step $STEP"; exit 1;;
esac
