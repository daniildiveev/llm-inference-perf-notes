#!/bin/bash
# Resolve Phase 2 bs=256 discrepancy (C +40% wall vs +11% decode GPU time):
# same prompt shape as the decode sweep (in 128) but output 1 -> prefill-only time at bs=256 / 128.
source /workspace/fusion-study/scripts/cfg.sh
cd /workspace/fusion-study; mkdir -p raw/p2split logs/p2split
for bs in 256 128; do for cfg in A C; do
  cfg_args $cfg
  env $CFG_ENV vllm bench latency --model $MODEL --input-len 128 --output-len 1 --batch-size $bs \
    --num-iters-warmup 5 --num-iters 20 --output-json raw/p2split/${cfg}_bs${bs}_out1.json "${CFG_ARGS[@]}" \
    > logs/p2split/${cfg}_bs${bs}.log 2>&1
  echo "$cfg bs=$bs $(grep -a 'Avg latency' logs/p2split/${cfg}_bs${bs}.log)"
done; done
echo ALL_DONE
