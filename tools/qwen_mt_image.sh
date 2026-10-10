#!/usr/bin/env bash
# Assemble the qwen-mt serverless image without pushing weights from this host (uplink ~1 MB/s):
# weights were baked in-datacenter by tools/qwen_mt_layer_builder.py into :qwen-mt-weights-20260923
# (base :ra2 + 7 sha256-pinned weight layers under /models); this adds the small code layer + runtime env.
# usage: CRANE=/path/to/crane tools/qwen_mt_image.sh qwen-mt-YYYYMMDD
set -euo pipefail
TAG=${1:?tag}
CRANE=${CRANE:-crane}
REPO=ghcr.io/lee101/omniserve-native
ROOT=$(cd "$(dirname "$0")/.." && pwd)
WORK=$(mktemp -d)
trap 'rm -rf "$WORK"' EXIT
mkdir -p "$WORK/opt/omniserve/workloads"
cp "$ROOT"/workloads/{__init__,qwen_image,qwen_mt}.py "$WORK/opt/omniserve/workloads/"
tar -C "$WORK" --owner=0 --group=0 --numeric-owner -cf "$WORK/code.tar" opt
"$CRANE" append -b "$REPO:qwen-mt-weights-20260923" -f "$WORK/code.tar" -t "$REPO:$TAG-tmp"
"$CRANE" mutate "$REPO:$TAG-tmp" -t "$REPO:$TAG" \
  --entrypoint python,-u,/opt/omniserve/workloads/qwen_mt.py \
  --env MT_MODELS_DIR=/models --env TMPDIR=/tmp --env HF_HOME=/tmp/hf \
  --env RA2_WEBP_QUALITY=88 --env RA2_FLOW_SHIFT=3.0 \
  --env LD_LIBRARY_PATH=/opt/stable-diffusion.cpp/build/bin:/usr/local/lib/python3.12/site-packages/nvidia/nccl/lib:/usr/lib/x86_64-linux-gnu:/usr/local/nvidia/lib64:/usr/local/nvidia/bin
"$CRANE" digest "$REPO:$TAG"
