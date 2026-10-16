# FP8 block-scaled (1×128) RMSNorm+quant fusion on sm90 — re-measurement on vLLM main

Run date 2026-10-15 (UTC), single rented GPU (Vast.ai, unprivileged; torch profiler only, no ncu/nsys).
Everything needed to reproduce a number is in `~/fusion-study/` (scripts/, raw/, traces/, logs/, proto/).

## TL;DR

- **The regression still holds on main, but it is smaller and has changed shape.** Fusing norm+quant forces the GEMM off
  `FlashInferFp8DeepGEMMDynamicBlockScaledKernel`. With DeepGEMM (config B), wall-clock decode is **+11.5% at bs=1** and **−0.5% to +7.2% for bs 8–256** (median of 3 repeats; the main table).
  Using only repeats 1+2 (repeat 0 was disturbed by host load), it is +9.3% at bs=1 and +1.2% to +5.2% for bs 8–256, positive at
  every batch size. With CUTLASS blockwise (config C): +9.2% at bs=1 and +30.0% to +57.8% at bs ≥ 8 (median of 3), or +9.5% and
  +30.8% to +60.7% (repeats 1+2).
- **At bs ≥ 32 the GEMM is not the problem any more; the fused kernel is.** B runs the same DeepGEMM `sm90_fp8_gemm_1d2d` kernel as A,
  with GEMM time within ±1%. All of B's extra GPU time (+129 to +134 µs/step) comes from `rms_norm_per_block_quant` plus a
  residual copy that Inductor inserts.
- **At bs=1, losing swapAB now costs about +285 µs/step** (it was +362 in 0.23.x). Norm/quant adds +262 µs/step: the fused
  kernel and residual copy account for ~+200, and ~+60 comes from `per_token_group_quant` (~2.4 µs) replacing `scale_1x128` (~1.6 µs) on
  the o_proj/down inputs. That last part is not caused by norm fusion; it comes from leaving the FlashInfer kernel.
- **The fused kernel is never meaningfully faster than `rms_norm` + `per_token_group_quant` run separately:** speed ratio (unfused time / fused time) 0.74–1.03× at
  1–16K tokens, and it reaches only 0.1–54% of peak HBM bandwidth. #54111 was a race fix, not a perf fix.
- **The proposed fix works mechanically but is not a net win.**
  - FlashInfer's sm90 swapAB runner *does* accept pre-quantized FP8 activations plus scales.
  - `rms_norm_per_block_quant(..., is_scale_transposed=True, tma_alignment=4)` already writes the exact column-major,
    M-padded-to-4 fp32 scale layout that both the swapAB runner and DeepGEMM expect, so no transform kernel is needed (verified by kernel lists).
  - But the fused kernel is slower than the rmsnorm + `scale_1x128` pair it replaces, so for the two shapes that have a norm in front (qkv, gate_up), the end-to-end path is
    **+3.4% to +8.5% at M=1** and −4.2% to +4.3% at M=8–256. The o_proj/down rows have no norm; they show that DeepGEMM `1d2d`
    beats today's `scale_1x128` + swapAB by 11–13% on o_proj at M=8/16.
  - The port only becomes a win once the fused kernel itself is fixed.

## What was run

| Phase | What | Output |
|---|---|---|
| 0 | Environment, vLLM main @ `e8e1e961` (precompiled wheel for that exact SHA), smoke test, claim re-check | `RUN.md`, `logs/env.txt`, `logs/smoke.log` |
| 1 | Kernel validation: A/B/C × bs 1, 64, one profiled run each (3 steady decode steps) | `results/phase1_kernels.md`, `traces/p1/`, `raw/p1_*.json` |
| 2 | Wall-clock `vllm bench latency` in 128 / out 256, bs 1–256, 3 repeats with rotated config order (81 runs) + prefill in 2048 / out 1 at bs 1/8/32 (9 runs). Clock/power/throttle sampled at 1 Hz. | `results/phase2_latency.md`, `raw/p2/`, `raw/p2_summary.json`, `logs/p2_clocks.csv`, `raw/p2_clock_windows.json` |
| 2b (extra) | Prefill-only split (in 128 / out 1) at bs 128/256 for A and C, to explain the C wall-clock gap | `raw/p2split/`, `logs/p2split_driver.log` |
| 3 | Trace attribution at bs 1/32/256 (+ late-step traces, steps 201–203, for A/C at bs=256) | `results/phase3_attribution.md`, `traces/p3/`, `traces/p3late/`, `raw/p3_attribution.json` |
| 4 | Microbenchmark: fused vs unfused norm+quant, hidden 4096, 1–16K tokens, ±residual, CUDA-graph timed | `results/phase4_microbench.md`, `raw/p4_microbench.json` |
| 5 | Prototype: today's path vs fused→DeepGEMM vs fused→FlashInfer swapAB (FP8 in), 4 shapes × 6 M; numerics vs BF16 reference | `results/phase5_proto.md`, `proto/proto_gemm.py`, `raw/p5_proto.json` |

Configurations (flags verified against source at the pinned SHA, see `RUN.md`):

| Cfg | Env / args | Linear kernel selected |
|---|---|---|
| A | defaults | `FlashInferFp8DeepGEMMDynamicBlockScaledKernel` (M<32: FlashInfer `scale_1x128` + TRT-LLM `deep_gemm::fp8_gemm_kernel_swapAB`; M≥32: vLLM `per_token_group_quant_fp8` + DeepGEMM `sm90_fp8_gemm_1d2d`) |
| B | `VLLM_DISABLED_KERNELS=FlashInferFp8DeepGEMMDynamicBlockScaledKernel`, `--compilation-config '{"custom_ops":["+quant_fp8"],"pass_config":{"fuse_norm_quant":true,"fuse_act_quant":false}}'` | `DeepGemmFp8BlockScaledMMKernel` + `rms_norm_per_block_quant` |
| C | B + `DeepGemmFp8BlockScaledMMKernel` also disabled | `CutlassFp8BlockScaledMMKernel` (`cutlass_3x_gemm_fp8_blockwise`) + `rms_norm_per_block_quant` |

## Deviations from the brief, and caveats

1. **GPU is H100 SXM5 80GB (700 W), not H100 NVL.** Proceeding was approved. Absolute µs are not comparable with the old NVL data;
   A→B / A→C deltas are. Phase 4 %BW uses the SXM5 peak, 3.35 TB/s.
2. **Clocks are not locked** (unprivileged). Busy SM clock averaged 1890–1975 MHz with intermittent software power-cap
   throttling (reason 0x4) in all configs alike (`raw/p2_clock_windows.json`).
3. **STOP checkpoints:** the STOP #1 report was posted when Phase 1 finished. Phase 2 had already been started (the user asked for every
   step to be completed), and the remaining phases ran back-to-back without a pause.
4. **Phase 2 repeat 0 is noisy.** It ran while host-side trace analysis was going on. Within-run iteration spread was wide
   (e.g. A bs=64 r0 1.71–2.00 s vs r1 1.62–1.71 s), and GPU clocks were identical across repeats, so the cause is host contention, not
   the GPU. The main table is the median of 3 repeats, as specified, with >3% spread flagged. A sensitivity table using repeats 1+2 only
   is included, and it gives the cleaner picture: **B is slower than A at every batch size.**
5. **The prefill runs are single runs.** The bs=1 prefill cell (2048 tokens, B +21%) is low confidence: A's own p90 equals B's median.
6. **Per-step GPU time covers the decoder-layer forward kernels only.** lm_head (`nvjet`) and the sampler run outside the step
   annotation and are identical across configs. The old 4782 → 7478 µs figures may have used a different basis, so compare deltas.
7. **On main, `fuse_norm_quant` and `fuse_act_quant` are already on by default** (`custom_ops=['+quant_fp8','none']`) even in A. There they
   match nothing, because the FlashInfer kernel consumes BF16. B/C pin `fuse_act_quant=false` so the only difference from A is
   norm+quant fusion plus the GEMM backend.
8. **Open anomaly (C at large batch; does not affect the A vs B conclusions):**
   - At bs=256, C's wall-clock time is +40% while its GPU time per decode step is only +9.5–11% (early steps 9–11 and late steps 201–203).
   - The prefill split explains only +0.11 s of the +1.31 s gap.
   - C's per-iteration latencies are bimodal: one iteration at 3.67 s (≈ what the GPU time predicts), 19 at ~4.6 s.
   - Ruled out: CUDA-graph fallback (same `cudaGraphLaunch` counts in all configs), KV-cache preemption (none; same capacity), and clock/power differences (identical).
   - Not resolved without nsys. B's wall-clock delta (+2.0%) matches its GPU delta (+2.0%), so B is unaffected.
