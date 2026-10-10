#!/bin/bash
# analyze.sh DATA_DIR TAG [fit args] : CPU-only label -> split -> features -> fit
set -e
D=$(cd "$(dirname "$0")" && pwd); W=/nvme0n1-disk/tmp/early-exit; P=$W/venv/bin/python
DATA=$1; TAG=$2; shift 2
export CUDA_VISIBLE_DEVICES=
cd $W
nice $P $D/label.py $DATA --out labels-$TAG.npz --device cpu 2>&1 | grep rows
$P $D/make_split.py $DATA --out split-$TAG.json
nice $P $D/features.py $DATA --split split-$TAG.json --out feats-$TAG.npz
$P $D/fit.py --feats feats-$TAG.npz --labels labels-$TAG.npz --split split-$TAG.json --out fit-$TAG "$@"
