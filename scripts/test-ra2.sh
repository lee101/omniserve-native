#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
SD=${SD_MASTER:-$HOME/code/stable-diffusion.cpp-master}
[ -f "$SD/build/bin/libstable-diffusion.so" ] || [ ! -d /vfast/data/code/stable-diffusion.cpp-master ] || SD=/vfast/data/code/stable-diffusion.cpp-master
MODELS=${RA2_MODELS:-$HOME/models/omniserve-native/qwen-image-2.1}
OUT=${RA2_OUT:-$ROOT/ra2-out}
PORT=${RA2_PORT:-8792}
ARCH=${CUDA_ARCH:-$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1 | tr -d .)}
NVCC=${NVCC:-$(command -v nvcc || echo /usr/local/cuda/bin/nvcc)}
DIT=${RA2_DIT:-$MODELS/qwen-image-2.1-Q4_K_M.gguf}
LLM=${RA2_LLM:-$MODELS/Qwen3VL-8B-Instruct-Q4_K_M.gguf}
MMPROJ=${RA2_MMPROJ:-$MODELS/mmproj-Qwen3VL-8B-Instruct-F16.gguf}
VAE=${RA2_VAE:-$MODELS/vae/qwen_image_2.1_vae_bf16.safetensors}
FREE_MB=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)
FREE_RAM_MB=$(awk '/MemAvailable/{print int($2/1024)}' /proc/meminfo)
if [ -z "${RA2_PARAMS_BACKEND+x}" ]; then
  if [ "$FREE_MB" -ge 10240 ]; then BACKEND=te=cpu; else BACKEND='*=cpu'; fi
else BACKEND=$RA2_PARAMS_BACKEND; fi
NEED_RAM_MB=$([ "$BACKEND" = 'te=cpu' ] && echo 7000 || echo 12000)
echo "vram_free=${FREE_MB}MB ram_avail=${FREE_RAM_MB}MB backend=$BACKEND" >&2
[ "$FREE_RAM_MB" -ge "$NEED_RAM_MB" ] || echo "warn: available RAM ${FREE_RAM_MB}MB < ~${NEED_RAM_MB}MB needed, expect swapping" >&2
W=${RA2_W:-1024}; H=${RA2_H:-1024}; STEPS=${RA2_STEPS:-20}; SEED=${RA2_SEED:-3}
BUILD=$ROOT/build-qwen
SECRET=$(head -c 12 /dev/urandom | od -An -tx1 | tr -d ' \n')
LOG=$OUT/server.log
mkdir -p "$OUT"

build() {
  [ -f "$SD/build/bin/libstable-diffusion.so" ] || {
    [ -d "$SD/.git" ] || [ -f "$SD/.git" ] || git clone -q https://github.com/leejet/stable-diffusion.cpp "$SD"
    git -C "$SD" submodule update --init -q
  }
  [ -f "$SD/build/bin/libstable-diffusion.so" ] || {
    cmake -S "$SD" -B "$SD/build" -G Ninja -DSD_CUDA=ON -DSD_BUILD_SHARED_LIBS=ON -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_CUDA_ARCHITECTURES="$ARCH" -DCMAKE_CUDA_COMPILER="$NVCC" -DGGML_CUDA_FA=ON >"$OUT/sd-configure.log"
    cmake --build "$SD/build" -j"$(nproc)" --target stable-diffusion >"$OUT/sd-build.log"
  }
  cmake -S "$ROOT" -B "$BUILD" -G Ninja -DWITH_SD=ON -DSD_DIR="$SD" -DCMAKE_BUILD_TYPE=Release >"$OUT/omni-configure.log"
  cmake --build "$BUILD" -j"$(nproc)" --target omniserve-native >"$OUT/omni-build.log"
}

models() {
  mkdir -p "$MODELS/vae"
  [ -e "$DIT" ] || { hf download netwrck/ra2 ra2-dit-q4_k_m.gguf --local-dir "$MODELS"; ln -sf ra2-dit-q4_k_m.gguf "$DIT"; }
  [ -e "$VAE" ] || hf download abenzerps/Qwen-Image-2.1-Uncensored-GGUF vae/qwen_image_2.1_vae_bf16.safetensors --local-dir "$MODELS"
  [ -e "$LLM" ] && [ -e "$MMPROJ" ] || hf download Qwen/Qwen3-VL-8B-Instruct-GGUF Qwen3VL-8B-Instruct-Q4_K_M.gguf mmproj-Qwen3VL-8B-Instruct-F16.gguf --local-dir "$MODELS"
}

start() {
  curl -fs -m 2 "127.0.0.1:$PORT/status" >/dev/null 2>&1 && { echo "port $PORT busy" >&2; exit 1; }
  env OMNISERVE_NATIVE_BIND=127.0.0.1 OMNISERVE_NATIVE_PORT="$PORT" OMNISERVE_NATIVE_SECRET="$SECRET" \
    OMNISERVE_NATIVE_SLOTS=1 OMNISERVE_NATIVE_IMAGE_PERMITS=1 OMNISERVE_NATIVE_IMAGE_PREFER_EMBEDDED=1 \
    OMNISERVE_NATIVE_SD_LIB="$SD/build/bin/libstable-diffusion.so" \
    OMNISERVE_NATIVE_SD_DIFFUSION_MODEL="$DIT" \
    OMNISERVE_NATIVE_SD_LLM="$LLM" \
    OMNISERVE_NATIVE_SD_LLM_VISION="$MMPROJ" \
    OMNISERVE_NATIVE_SD_VAE="$VAE" \
    OMNISERVE_NATIVE_SD_REFERENCE_EDIT=1 OMNISERVE_NATIVE_SD_EAGER_LOAD=1 \
    OMNISERVE_NATIVE_SD_PARAMS_BACKEND="$BACKEND" OMNISERVE_NATIVE_SD_MIN_FREE_MB=256 \
    OMNISERVE_NATIVE_SD_VAE_TILING=1 OMNISERVE_NATIVE_SD_VAE_TILE_X=32 OMNISERVE_NATIVE_SD_VAE_TILE_Y=32 \
    OMNISERVE_NATIVE_SD_CACHE_MODE=easycache OMNISERVE_NATIVE_SD_EASYCACHE_THRESHOLD=${RA2_EASYCACHE:-0.05} \
    OMNISERVE_NATIVE_SD_ZERO_GUIDANCE=1.0 OMNISERVE_NATIVE_SD_WEBP_QUALITY=90 \
    "$BUILD/omniserve-native" --port "$PORT" >"$LOG" 2>&1 &
  PID=$!
  trap 'kill $PID 2>/dev/null; wait $PID 2>/dev/null || true' EXIT
  for _ in $(seq 240); do
    curl -fs -m 2 "127.0.0.1:$PORT/status" >/dev/null 2>&1 && return
    kill -0 $PID 2>/dev/null || { tail -20 "$LOG" >&2; exit 1; }
    sleep 2
  done
  tail -20 "$LOG" >&2; exit 1
}

gen() {
  local i=$1 prompt=$2
  python3 - "$PORT" "$SECRET" "$OUT/$(printf %02d "$i").webp" "$prompt" "$W" "$H" "$STEPS" "$((SEED + i))" <<'PY'
import sys, json, base64, time, urllib.request
port, secret, path, prompt, w, h, steps, seed = sys.argv[1:]
body = json.dumps({"prompt": prompt, "width": int(w), "height": int(h), "steps": int(steps), "seed": int(seed), "output_format": "webp"}).encode()
req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/images/generations", body, {"X-API-Key": secret, "Content-Type": "application/json"})
t = time.time()
d = json.load(urllib.request.urlopen(req, timeout=1800))
open(path, "wb").write(base64.b64decode(d["data"][0]["b64_json"]))
print(f"{path} {time.time() - t:.1f}s {d['data'][0].get('inference_time_ms', '?')}ms")
PY
}

[ "${1:-}" = "--setup" ] && { build; models; exit 0; }
if [ $# -eq 0 ]; then
  set -- "editorial portrait of a smiling woman, golden hour, 85mm" \
         "anime girl with silver hair on a rooftop at dusk, detailed illustration" \
         "a cabin in a snowstorm, cinematic, volumetric light" \
         "a red fox sitting in snow, photograph"
fi
build; models; start
i=0
for p in "$@"; do gen "$i" "$p"; i=$((i + 1)); done
echo "done: $OUT"
