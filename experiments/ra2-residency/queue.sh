#!/bin/bash
# Waits outside the flock for free VRAM >= NEED, then runs one bench. usage: queue.sh NEED_MIB NAME ARMS CASES [args]
NEED=$1; shift
D=/nvme0n1-disk/code/omniserve-native-residency/experiments/ra2-residency
end=$((SECONDS+${QWAIT_S:-21600}))
while [ $SECONDS -lt $end ]; do
  f=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits)
  [ "$f" -ge "$NEED" ] && exec $D/run.sh "$@" --margin-mib 5300 --floor-mib 4300 --free-wait-s 120
  sleep 15
done
echo "SKIP $2: free never reached $NEED MiB"
