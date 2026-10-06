#!/bin/sh
# 60 s prod watch: 5xx on :8791/:8792, broker waits/denials, image queue stats.
while :; do
  a=$(curl -s -m 5 localhost:8791/metrics | awk '/responses_total\{class="(2xx|5xx)"\}/{printf "%s ", $2}')
  b=$(curl -s -m 5 localhost:8792/metrics | awk '/responses_total\{class="(2xx|5xx)"\}/{printf "%s ", $2}')
  s=$(curl -s -m 5 localhost:8791/v1/gpu/status | python3 -c "import json,sys;d=json.load(sys.stdin);c=d['ledger']['counters'];s=d['sched'];print('free=%d grants=%d denials=%d waits=%d timeouts=%d img_waits=%d img_denials=%d llm_evict=%d reload=%d' % (d['ledger']['device_free_mb'],c['grants'],c['denials'],c['waits'],c['wait_timeouts'],s['image_waits'],s['image_wait_denials'],s['llm']['evictions'],s['llm']['reloads']))" 2>/dev/null)
  echo "$(date +%H:%M:%S) 8791[2xx 5xx]=$a 8792=$b $s"
  sleep 60
done
