#!/bin/bash
cd /nvme0n1-disk/code/omniserve-native-residency
AESTHETIC_DEVICE=cpu /nvme0n1-disk/code/cutedsl/.venv/bin/python tools/image_parity_bench.py \
  --reference-base "$CANARY_BASE" --candidate-base "$CANARY_BASE" --corpus performance/image-parity-corpus-qwen.json \
  --reference-extra '{"turbo":false}' --output-dir "$ARM_DIR" --reference-secret-env NONE_SECRET --candidate-secret-env NONE_SECRET
