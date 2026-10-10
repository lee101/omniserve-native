#!/bin/bash
cd /nvme0n1-disk/code/omniserve-native-residency
X=experiments/ra2-residency
PB="env AESTHETIC_DEVICE=cpu /nvme0n1-disk/code/cutedsl/.venv/bin/python tools/image_parity_bench.py --corpus $X/corpus_cpu512.json --reference-extra {\"turbo\":false} --reference-secret-env NONE_SECRET --candidate-secret-env NONE_SECRET --timeout 3600"
if [ "$ARM" = cpu ]; then
  $PB --reference-base "$CANARY_BASE" --capture-reference --output-dir $X/cq_ref || exit 1
fi
REF=$(ls -d $X/cq_ref/image_reference_* | tail -1)
$PB --candidate-base "$CANARY_BASE" --reference-run "$REF" --output-dir "$ARM_DIR"
