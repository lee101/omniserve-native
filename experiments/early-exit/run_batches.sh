#!/bin/bash
# run_batches.sh JOBS OUT LOG [MAX_RUNS] : repeated bounded capture windows, yielding the GPU lock between them
JOBS=$1; OUT=$2; LOG=$3; MAX=${4:-100}
D=$(cd "$(dirname "$0")" && pwd)
for i in $(seq 1 $MAX); do
  [ -f "$OUT/STOP" ] && { echo "STOP file" >> $LOG; break; }
  NEED_MIB=$(( ${PEAK_MIB:-4096} + 4096 )) /nvme0n1-disk/tmp/early-exit/gpu_run.sh $LOG /nvme0n1-disk/code/cutedsl/.venv/bin/python $D/capture.py --jobs $JOBS --out $OUT --minutes ${MINUTES:-15} --peak-mib ${PEAK_MIB:-4096} --max-bg-5xx ${MAXBG:-10} ${EXTRA:+--extra-env "$EXTRA"}
  tail -3 $LOG | grep -q "ABORT" && sleep 900
  tail -2 $LOG | grep -q "nothing to do" && break
  sleep ${GAP:-120}
done
echo "batches done $(date -u +%T)" >> $LOG
