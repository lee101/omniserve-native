#!/usr/bin/env bash
# Start the local RA2 image lane (Qwen Image 2.1) that tools/make-image.sh renders
# on. ManifoldGen tries this lane first and falls through to the others only when
# it is refused, so with nothing listening every image request ends in
# "service temporarily unavailable".
#
#   tools/start-ra2-lane.sh              # start in the background, wait for health
#   tools/start-ra2-lane.sh --foreground # run in the foreground (supervisor, systemd)
#   tools/start-ra2-lane.sh --status     # report whether it is up
#
# Weights are pinned to the ones staged on this box; override any of them with
# the matching variable below. It loads about 11 GB (4.6 GB DiT, 5 GB text
# encoder, 1.2 GB vision tower, VAE), which is why the script refuses to start a
# second copy: a shared box that is already near its memory limit will have the
# first one OOM-killed instead.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BIN="$REPO/build-qwen/omniserve-native"

PORT="${RA2_PORT:-8792}"
# The prod-parity build (latent replay API, VAE OOM retry) is preferred: with its OOM retry the VAE can decode
# untiled, 1.3 s instead of 3.7 s per 1024^2 image, output within PSNR 56 of the tiled decode. The older
# master build has no retry, so it keeps tiling on.
SD_LIB_PROD=/vfast/data/code/sdcpp-qwen-prefixkv/build-86/bin/libstable-diffusion.so
SD_LIB_MASTER=/vfast/data/code/stable-diffusion.cpp-master/build/bin/libstable-diffusion.so
if [[ -z "${RA2_SD_LIB:-}" && -e "$SD_LIB_PROD" ]]; then
  SD_LIB="$SD_LIB_PROD"
  TILING_DEFAULT=0
else
  SD_LIB="${RA2_SD_LIB:-$SD_LIB_MASTER}"
  TILING_DEFAULT=1
fi
# /vfast is a ZFS pool on a spinning disk (about 20 MB/s on a cold read, minutes for the 11 GB
# set); a copy on the NVMe root loads at GB/s. Seed it once with:
#   mkdir -p ~/models-fast/qwen-image-2.1/vae && cp <the four files> there (see FAST below).
FAST="${RA2_MODELS_FAST:-$HOME/models-fast/qwen-image-2.1}"
fast() { [[ -e "$FAST/$1" ]] && printf '%s' "$FAST/$1" || printf '%s' "$2"; }
DIT="${RA2_DIFFUSION_MODEL:-$(fast qwen-image-2.1-Q4_K_M.gguf /vfast/data/models/qwen-image-2.1-uncensored/qwen-image-2.1-Q4_K_M.gguf)}"
VAE="${RA2_VAE:-$(fast vae/qwen_image_2.1_vae_bf16.safetensors /vfast/data/models/qwen-image-2.1/vae/qwen_image_2.1_vae_bf16.safetensors)}"
TEXT_ENCODER="${RA2_LLM:-$(fast Qwen3VL-8B-Instruct-Q4_K_M.gguf /vfast/data/models/qwen3vl-8b-gguf/Qwen3VL-8B-Instruct-Q4_K_M.gguf)}"
VISION="${RA2_LLM_VISION:-$(fast mmproj-Qwen3VL-8B-Instruct-F16.gguf /vfast/data/models/qwen3vl-8b-gguf/mmproj-Qwen3VL-8B-Instruct-F16.gguf)}"
TURBO_LORA="${RA2_TURBO_LORA:-$(fast Qwen-Image-2.1-viggle-turbo-v0.3-6step-lora-r256.safetensors /vfast/data/models/qwen-image-2.1-turbo/Qwen-Image-2.1-viggle-turbo-v0.3-6step-lora-r256.safetensors)}"
ANIME_LORA="${RA2_ANIME_LORA:-$(fast fusal-qwen-image-2.1-stack-exact-r80.safetensors "")}"
LOG="${RA2_LOG:-/tmp/ra2-lane.log}"

# te=cpu is deliberately not set: with the text encoder on the CPU a 512x512
# nine-step render takes 254 s and ManifoldGen's backend client gives up at
# 180 s. Everything resident on the GPU renders the same image in about 15 s.
export OMNISERVE_NATIVE_SD_LIB="$SD_LIB"
export OMNISERVE_NATIVE_SD_DIFFUSION_MODEL="$DIT"
export OMNISERVE_NATIVE_SD_VAE="$VAE"
export OMNISERVE_NATIVE_SD_LLM="$TEXT_ENCODER"
export OMNISERVE_NATIVE_SD_LLM_VISION="$VISION"
export OMNISERVE_NATIVE_SD_CACHE_MODE="${RA2_CACHE_MODE:-easycache}"
export OMNISERVE_NATIVE_SD_EASYCACHE_THRESHOLD="${RA2_EASYCACHE:-0.15}"
# qualitybench/README.md: 30 steps at 0.15 beats 20 steps at 0.05 for the same time; the notch removes the 2 px VAE lattice.
export OMNISERVE_NATIVE_SD_MIN_STEPS="${RA2_MIN_STEPS:-30}"
# Distilled 6-step tier (what prod serves by default). RA2_TURBO=1 makes it the default for text-to-image and
# {"turbo":false} selects the 30-step base; off, every request takes the base path.
if [[ "${RA2_TURBO:-0}" == 1 && -e "$TURBO_LORA" ]]; then
  export OMNISERVE_NATIVE_SD_DEFAULT_LORA="$TURBO_LORA"
  export OMNISERVE_NATIVE_SD_TURBO_NODES="${RA2_TURBO_NODES:-1.0,0.9375,0.875,0.75,0.5,0.25}"
fi
# anime + NSFW prompts: Fusal LoRA on the 30-step base path (trigger "fusal style." is prepended by the server; turbo is skipped whenever a LoRA is attached)
[[ -n "$ANIME_LORA" ]] && export OMNISERVE_NATIVE_ANIME_NSFW_LORA_PATH="$ANIME_LORA"
export OMNISERVE_NATIVE_SD_WARMUP="${RA2_WARMUP:-1}"
export OMNISERVE_NATIVE_SD_NOTCH="${RA2_NOTCH:-1}"
export OMNISERVE_NATIVE_SD_ZERO_GUIDANCE="${RA2_ZERO_GUIDANCE:-1.0}"
export OMNISERVE_NATIVE_SD_VAE_TILING="${RA2_VAE_TILING:-$TILING_DEFAULT}"
export OMNISERVE_NATIVE_SD_MIN_FREE_MB="${RA2_MIN_FREE_MB:-256}"

health() { curl -fsS -m 3 "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; }

# /health answers as soon as the listener binds, and /status already reports
# diffusion.ready while the first renders still fail, because the text encoder
# and VAE are not resident yet. Only a real generation proves the lane works, so
# readiness is a 256x256 render rather than a probe.
usable() {
  health || return 1
  curl -fsS -m "${RA2_WARMUP_TIMEOUT:-240}" -X POST "http://127.0.0.1:$PORT/v1/images/generations" \
    -H 'Content-Type: application/json' \
    -d '{"prompt":"warmup","size":"256x256","n":1}' >/dev/null 2>&1
}

case "${1:-start}" in
  --status | status)
    if health; then
      echo "ra2 lane is up on :$PORT"
      curl -s "http://127.0.0.1:$PORT/status"
      exit 0
    fi
    echo "ra2 lane is down on :$PORT"
    exit 1
    ;;
  --foreground | foreground)
    exec "$BIN" --port "$PORT"
    ;;
  --help | -h)
    sed -n '2,17p' "${BASH_SOURCE[0]}" | cut -c3-
    exit 0
    ;;
  start) ;;
  *) echo "usage: $0 [start|--foreground|--status]" >&2; exit 2 ;;
esac

usable && { echo "ra2 lane already usable on :$PORT"; exit 0; }

for file in "$BIN" "$SD_LIB" "$DIT" "$VAE" "$TEXT_ENCODER" "$VISION"; do
  [[ -e "$file" ]] || { echo "missing: $file" >&2; exit 1; }
done

# Two copies would double the VRAM and host memory footprint; the second one to
# load is what the OOM killer takes, so check before spending the load.
if ss -ltn 2>/dev/null | grep -q ":$PORT "; then
  echo "port $PORT is already bound by something that is not answering /health" >&2
  exit 1
fi

echo "starting ra2 lane on :$PORT (log: $LOG)" >&2
nohup "$BIN" --port "$PORT" >>"$LOG" 2>&1 &
lane_pid=$!

echo "waiting for the first render to succeed" >&2
deadline=$((SECONDS + ${RA2_READY_TIMEOUT:-420}))
while ((SECONDS < deadline)); do
  if usable; then
    echo "ra2 lane ready on :$PORT (pid $lane_pid)"
    exit 0
  fi
  if ! kill -0 "$lane_pid" 2>/dev/null; then
    tail -20 "$LOG" >&2
    echo "ra2 lane exited during startup" >&2
    exit 1
  fi
  sleep 3
done

tail -20 "$LOG" >&2
echo "ra2 lane never produced a render on :$PORT" >&2
exit 1