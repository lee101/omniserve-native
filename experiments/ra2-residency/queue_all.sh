#!/bin/bash
cd /nvme0n1-disk/code/omniserve-native-residency/experiments/ra2-residency
./queue.sh 15300 e1_prod arms_prod.json '[{"name":"t2i","kind":"t2i","steps":20},{"name":"hq","kind":"t2i","steps":30,"body":{"turbo":false}},{"name":"edit","kind":"edit","steps":20}]' --per-case 2 > e1.log 2>&1
./queue.sh 15300 e3_turbo arms_prod.json '[]' --exec /nvme0n1-disk/code/omniserve-native-residency/experiments/ra2-residency/e3_exec.sh > e3.log 2>&1
./queue.sh 15300 e2_cache arms_cache.json '[{"name":"edit","kind":"edit","steps":20},{"name":"t2i20","kind":"t2i","steps":20,"body":{"turbo":false}}]' --per-case 2 > e2.log 2>&1
./queue.sh 21000 e4_res arms_res.json '[{"name":"t2i","kind":"t2i","steps":20},{"name":"edit","kind":"edit","steps":20},{"name":"t2iback","kind":"t2i","steps":20}]' --per-case 2 > e4.log 2>&1
