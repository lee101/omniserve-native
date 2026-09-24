#!/bin/sh
# per-process VRAM sampler: ts,pid,mib lines + ts,TOTAL,used
out=${1:-vram.csv}; iv=${2:-1}
while :; do
  t=$(date +%s)
  nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader,nounits 2>/dev/null | sed "s/^/$t,/;s/ //g" >> "$out"
  nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits 2>/dev/null | sed "s/^/$t,TOTAL,/;s/ //g" >> "$out"
  sleep "$iv"
done
