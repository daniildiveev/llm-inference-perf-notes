#!/bin/bash
# Phase 1/3: profile steady-state decode steps per config & batch size.
# Usage: 10_phase1_profile.sh <tag> "<cfgs>" "<batch sizes>"
# Traces -> traces/<tag>/<cfg>_bs<bs>/ ; logs -> logs/<tag>_<cfg>_bs<bs>.log
set -u
source /workspace/fusion-study/scripts/cfg.sh
TAG=$1; CFGS=$2; BSS=$3
cd /workspace/fusion-study
for cfg in $CFGS; do
  for bs in $BSS; do
    cfg_args $cfg
    out=/workspace/fusion-study/traces/$TAG/${cfg}_bs${bs}
    rm -rf $out; mkdir -p $out
    # prefill = iteration 0; skip 8 engine steps, then record 3 decode steps
    PCFG="{\"profiler\":\"torch\",\"torch_profiler_dir\":\"$out\",\"delay_iterations\":${DELAY:-8},\"max_iterations\":3,\"torch_profiler_with_stack\":false,\"ignore_frontend\":true}"
    echo "=== $cfg bs=$bs $(date +%T)"
    env $CFG_ENV vllm bench latency --model $MODEL --input-len 128 --output-len ${OUTLEN:-32} \
      --batch-size $bs --num-iters-warmup 3 --num-iters 1 --profile \
      --profiler-config "$PCFG" "${CFG_ARGS[@]}" > logs/${TAG}_${cfg}_bs${bs}.log 2>&1
    echo "exit=$? $(date +%T)"
  done
done
echo ALL_DONE
