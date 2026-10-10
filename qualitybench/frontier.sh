#!/usr/bin/env bash
# runs the config grid; timing lines go to out/frontier.log
cd "$(dirname "$0")/.."
export ONLY=${ONLY:-fox2,trio,duo,quad} SEED=${SEED:-0}
run() { qualitybench/variants.sh "$@" 2>&1 | rg "wall=|HTTP" >> qualitybench/out/frontier.log; }
run 'ref_d30|30|{"cache_threshold":0}'
run 'ec05_30|30|{}'
run 'ec05_24|24|{}'
run 'ec05_20|20|{}'
run 'ec05_16|16|{}'
run 'd20|20|{"cache_threshold":0}'
run 'ec08_30|30|{"cache_threshold":0.08}'
run 'ec10_30|30|{"cache_threshold":0.1}'
run 'ec05_20_sh2|20|{"flow_shift":2.0}'
run 'ec05_20_sh45|20|{"flow_shift":4.5}'
run 'ec05_20_sh6|20|{"flow_shift":6.0}'
run 'ec05_20_resms|20|{"sampler":"res_multistep"}'
run 'ec05_20_dpm2m|20|{"sampler":"dpm++2m"}'
run 'ec05_20_ipndm|20|{"sampler":"ipndm"}'
run 'ec05_20_ersde|20|{"sampler":"er_sde"}'
run 'ec05_16_resms|16|{"sampler":"res_multistep"}'
run 'ec05_20_beta|20|{"scheduler":"beta"}'
run 'ec05_20_sgm|20|{"scheduler":"sgm_uniform"}'
echo DONE >> qualitybench/out/frontier.log
