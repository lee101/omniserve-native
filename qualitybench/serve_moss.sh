#!/usr/bin/env bash
# MOSS-Image lane server for qualitybench (MOSS needs the MOSS-capable upstream sd.cpp build).
# LANE=fast: 4-step + keyword style routing (deploy/moss-lora-routes.json), ~3.5-4.5 s. LANE=hq: dense 30-step CFG 3 +
# EasyCache + routing with negatives, ~27-35 s. (Routing is internal; no model/style names leave the server.)
# LANE=sfw (default): 4-step merged, ~3.4 s/1024px. LANE=nsfw: same + NSFW LoRA @1.0 on every request, ~4.4 s.
# Env: MODEL=<dit gguf> (default 4-step merged), SDENV="K=V ..." adds OMNISERVE_NATIVE_SD_K=V, PORT.
cd "$(dirname "$0")/.."
M=${MOSS_DIR:-/vfast/data/models/moss}
BUILD=${MOSS_BUILD:-/vfast/data/code/omniserve-native-moss-build}
export OMNISERVE_NATIVE_BIND=127.0.0.1 OMNISERVE_NATIVE_SLOTS=1 OMNISERVE_NATIVE_IMAGE_PERMITS=1
export OMNISERVE_NATIVE_IMAGE_PREFER_EMBEDDED=1
export OMNISERVE_NATIVE_GUARD=0 OMNISERVE_NATIVE_GUARD_JUDGE=0
export OMNISERVE_NATIVE_SD_LIB=${MOSS_SD_LIB:-/vfast/data/code/sdcpp-moss/build/bin/libstable-diffusion.so}
export OMNISERVE_NATIVE_SD_DIFFUSION_MODEL=${MODEL:-$M/sft4/MOSS-Image-SFT-4step-v2.0-merged-Q4_0.gguf}
export OMNISERVE_NATIVE_SD_LLM=${TE:-$M/te/Qwen3.5-9B-Q8_0.gguf}
export OMNISERVE_NATIVE_SD_VAE=$M/vae/flux2-vae.safetensors
export OMNISERVE_NATIVE_SD_TOKENIZER=$M/tok/tokenizer.json
export OMNISERVE_NATIVE_SD_ZERO_GUIDANCE=1.0 OMNISERVE_NATIVE_SD_MIN_STEPS=0
export OMNISERVE_NATIVE_SD_EAGER_LOAD=1 OMNISERVE_NATIVE_SD_MIN_FREE_MB=256
export OMNISERVE_NATIVE_SD_WARMUP=${WARMUP:-0}
if [ "${LANE:-sfw}" = fast ] || [ "$LANE" = hq ]; then
  export OMNISERVE_NATIVE_LORA_AUTO_ROUTE=1 OMNISERVE_NATIVE_LORA_ROUTES=$PWD/deploy/moss-lora-routes.json
fi
if [ "${LANE:-sfw}" = fast ] || [ "$LANE" = hq ]; then export OMNISERVE_NATIVE_SD_PUBLIC_NAME=${PUBLIC_NAME:-ra3}; fi
if [ "$LANE" = fast ]; then
  export OMNISERVE_NATIVE_SD_LANE_STEPS=4 OMNISERVE_NATIVE_SD_LANE_GUIDANCE=1.0
elif [ "$LANE" = hq ]; then
  export OMNISERVE_NATIVE_SD_DIFFUSION_MODEL=${MODEL:-$M/sft/MOSS-Image-SFT-Q4_K.gguf}
  export OMNISERVE_NATIVE_SD_LANE_STEPS=30 OMNISERVE_NATIVE_SD_LANE_GUIDANCE=3.0
  export OMNISERVE_NATIVE_SD_CACHE_MODE=easycache OMNISERVE_NATIVE_SD_EASYCACHE_THRESHOLD=0.15
fi
if [ "${LANE:-sfw}" = nsfw ]; then
  export OMNISERVE_NATIVE_SD_DEFAULT_LORA=$M/lora/MOSS-Image-NSFW-Uncensored-LoRA-F16.gguf
  export OMNISERVE_NATIVE_SD_DEFAULT_LORA_SCALE=${LORA_SCALE:-1.0} OMNISERVE_NATIVE_SD_DEFAULT_LORA_ALWAYS=1
fi
[ -n "$SDENV" ] && for kv in $SDENV; do export "OMNISERVE_NATIVE_SD_$kv"; done
exec "$BUILD/omniserve-native" --port "${PORT:-8794}"
