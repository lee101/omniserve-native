#!/usr/bin/env bash
# usage: variants.sh "tag|steps|extra-json" ...   (against :8792)
cd "$(dirname "$0")/.."
for spec in "$@"; do
  IFS='|' read -r tag steps extra <<<"$spec"
  uv run --with pillow python qualitybench/run.py --tag "$tag" --seed "${SEED:-0}" --steps "$steps" --extra "${extra:-{\}}" ${ONLY:+--only $ONLY}
done
