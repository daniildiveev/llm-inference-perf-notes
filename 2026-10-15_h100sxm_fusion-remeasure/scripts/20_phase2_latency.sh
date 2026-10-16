#!/bin/bash
# Phase 2: wall-clock latency. 3 repeats x 9 batch sizes x 3 configs, config order rotated per repeat;
# then one prefill-heavy run (in 2048 / out 1) per config at bs 1, 8, 32.
# Resumable: skips cells whose JSON already exists. Raw JSON -> raw/p2/, logs -> logs/p2/.
set -u
source /workspace/fusion-study/scripts/cfg.sh
cd /workspace/fusion-study
mkdir -p raw/p2 logs/p2
ROT=("A B C" "B C A" "C A B")
run() {  # cfg bs in out warm iters outfile
  local cfg=$1 bs=$2 in=$3 out=$4 warm=$5 iters=$6 f=$7
  [ -s raw/p2/$f.json ] && { echo "skip $f"; return; }
  cfg_args $cfg
  echo "$(date +%T) $f $(nvidia-smi --query-gpu=temperature.gpu,clocks.sm,power.draw --format=csv,noheader)"
  env $CFG_ENV vllm bench latency --model $MODEL --input-len $in --output-len $out \
    --batch-size $bs --num-iters-warmup $warm --num-iters $iters \
    --output-json raw/p2/$f.json "${CFG_ARGS[@]}" > logs/p2/$f.log 2>&1
  echo "  exit=$? $(grep -a 'Avg latency' logs/p2/$f.log)"
}
for rep in 0 1 2; do
  for bs in 1 8 16 24 32 48 64 128 256; do
    for cfg in ${ROT[$rep]}; do run $cfg $bs 128 256 5 20 ${cfg}_bs${bs}_r${rep}; done
  done
done
for bs in 1 8 32; do
  for cfg in A B C; do run $cfg $bs 2048 1 5 20 prefill_${cfg}_bs${bs}; done
done
echo ALL_DONE
