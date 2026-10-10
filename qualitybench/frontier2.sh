#!/usr/bin/env bash
cd "$(dirname "$0")/.."
export ONLY=${ONLY:-fox2,trio,duo,quad} SEED=${SEED:-0}
run() { qualitybench/variants.sh "$@" 2>&1 | rg "wall=|HTTP" >> qualitybench/out/frontier.log; }
sed -i '/^DONE$/d' qualitybench/out/frontier.log
run 'ec15_30|30|{"cache_threshold":0.15}'
run 'ec20_30|30|{"cache_threshold":0.2}'
run 'ec30_30|30|{"cache_threshold":0.3}'
run 'ec10_30_x02|30|{"cache_threshold":0.1,"extra_sample_args":"exit_tol=0.02"}'
run 'ec10_30_x01|30|{"cache_threshold":0.1,"extra_sample_args":"exit_tol=0.01"}'
run 'ec10_30_x005|30|{"cache_threshold":0.1,"extra_sample_args":"exit_tol=0.005"}'
run 'ec10_30_beta|30|{"cache_threshold":0.1,"scheduler":"beta"}'
run 'd30_x01|30|{"cache_threshold":0,"extra_sample_args":"exit_tol=0.01"}'
echo DONE >> qualitybench/out/frontier.log
