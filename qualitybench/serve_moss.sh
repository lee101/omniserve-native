#!/usr/bin/env bash
# MOSS-Image lane server for qualitybench (MOSS needs the MOSS-capable upstream sd.cpp build).
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
[ -n "$SDENV" ] && for kv in $SDENV; do export "OMNISERVE_NATIVE_SD_$kv"; done
exec "$BUILD/omniserve-native" --port "${PORT:-8794}"
