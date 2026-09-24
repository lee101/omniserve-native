#!/bin/sh
# per-process VRAM sampler: ts,pid,mib,unit(comm) lines + ts,TOTAL,used,util
out=${1:-vram.csv}; iv=${2:-1}
while :; do
  t=$(date +%s)
  nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader,nounits 2>/dev/null | while IFS=', ' read -r pid mb; do
    u=$(sed -n '1s#.*/##p' /proc/$pid/cgroup 2>/dev/null); c=$(cat /proc/$pid/comm 2>/dev/null)
    echo "$t,$pid,$mb,$u($c)"
  done >> "$out"
  nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits 2>/dev/null | sed "s/^/$t,TOTAL,/;s/ //g" >> "$out"
  sleep "$iv"
done
