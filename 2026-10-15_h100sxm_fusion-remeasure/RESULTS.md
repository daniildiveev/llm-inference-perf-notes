# FP8 block-scaled (1×128) RMSNorm+quant fusion on sm90 — re-measurement on vLLM main

> **Revision 2026-10-16** (no new GPU runs; text and derived numbers only):
> Phase 5 "best prop vs today" column recomputed with round-half-up (the old column truncated; ≤0.1 pp changes).
> The o_proj/down "1d2d beats swapAB at M=8/16" result is marked as unsettled (L2-warm isolated benchmark).
> The bs ≥ 32 quant_norm delta is split per site (fused kernel vs residual copy) instead of assuming half the quant launches sit at norm sites.
> The 0.23.x "swapAB at bs=256" label is confirmed wrong: the M<32 dispatch is identical in the v0.23.0 source.
> New section "Upstream state as of 2026-10-16". Recommendation rewritten.

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
  residual copy that Inductor inserts: ≈ +71 µs fused kernel / +56 µs copy at bs=32 and ≈ +51 / +80 µs at bs=256
  (split method in Phase 3). The copy is 43–60% of the regression at bs ≥ 32; open PR #60481 removes it.
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
    **+3.4% to +8.5% at M=1** and −4.2% to +4.3% at M=8–256.
  - The o_proj/down rows have no norm in front. In them DeepGEMM `1d2d` beats today's `scale_1x128` + swapAB by 11–13% on
    o_proj at M=8/16, but this is **not settled**: each shape runs in isolation with warm L2, and the gain shrinks as the weight
    stops fitting in the 50 MB L2 (o_proj 16.8 MB −11 to −13%, qkv 25 MB −3 to −4%, down 50 MB −3%, gate_up 100 MB +2 to +3%).
    In-model at M=1, `1d2d` is 6–15% slower per launch than swapAB (Phase 3). Needs an L2-flushed rerun before anyone acts on it.
  - The port becomes a win only with a faster producer kernel. Upper bound if the producer cost no more than today's
    rmsnorm alone and the residual copy were gone: −3.1% GPU time per step at bs=1, −4.0% at bs=32, −2.6% at bs=256.

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
   is included, and it gives the cleaner picture: **B is slower than A at every batch size.** The primary evidence for B vs A
   is GPU time per step (Phase 3), which host load does not affect; at bs ≥ 8 the wall-clock B deltas are within the repeat spread.
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

# 1. Environment (RUN.md)

## RUN.md — environment and pinned state

### Hardware (DEVIATION from brief)
- GPU: **NVIDIA H100 80GB HBM3 (SXM5, 700 W cap, 132 SMs)** — NOT H100 NVL as the brief expected. User approved
  proceeding (2026-10-15). Absolute µs are not comparable with the earlier 0.23.x NVL numbers; A→B / A→C deltas are.
- HBM3 peak used for % of bandwidth: 3.35 TB/s (H100 SXM5 datasheet).
- Driver 610.57.04 (CUDA UMD 13.3). nvcc 13.2 (V13.2.86). Image env CUDA_VERSION=13.2.1.
- Clocks: NOT locked (unprivileged). Max SM 1980 MHz, mem 2619 MHz. Idle 345 MHz; observed 1980 MHz under load.
  Temp/SM clock/power logged before every Phase 2 run (logs/p2_driver.log).
- No other GPU processes (nvidia-smi --query-compute-apps empty) at start.
- Host: 192 cores, 1 TB RAM, 534 GB overlay disk. $WORK=/workspace (no separate volume; / is the largest).

### Software
- vLLM main @ `e8e1e96196725f097eae2394cbc458b1f5a679b8` (2026-10-15, "[ROCm][DSv4] AITER MegaMoE ... (#61262)"),
  version 0.32.0rc1.dev218+ge8e1e9619. `VLLM_USE_PRECOMPILED=1`; nightly wheel exists for this **exact** SHA
  (https://wheels.vllm.ai/e8e1e961.../vllm/metadata.json -> 200). No source build.
- torch 2.14.0+cu130 (vLLM-pinned, not reinstalled), triton 3.8.0.
- flashinfer-python 0.7.1 (vLLM pin `requirements/cuda.txt`). No flashinfer-cubin/jit-cache: FlashInfer JIT-compiled
  `fp8_blockscale_gemm_90` and `sampling` on first run (smoke test) and cached in ~/.cache/flashinfer.
- DeepGEMM 2.8.1 — **vendored** as `vllm.third_party.deep_gemm` inside the vLLM wheel (pinned in
  cmake/external_projects/deepgemm.cmake, ref 6ce1c5ec of vllm-project/DeepGEMM). `import deep_gemm` fails (not installed
  separately) by design; vllm/utils/deep_gemm.py falls back to the vendored copy. Did not run tools/install_deepgemm.sh
  because an external install would shadow the vendored one.
- Model: Qwen/Qwen3-8B-FP8 (HF_HOME=/workspace/hf).
- Smoke test (in 32 / out 32 / bs 1): avg 0.168 s. Startup log: `Selected FlashInferFp8DeepGEMMDynamicBlockScaledKernel`.

### Flags used (checked against source at the pinned SHA)
| Purpose | Flag | Source |
|---|---|---|
| Force GEMM backend | `VLLM_DISABLED_KERNELS=<class names, comma-sep>` (matches `kernel.__name__`) | vllm/envs.py:1203, kernels/linear/__init__.py:639 |
| Fusion | `--compilation-config '{"custom_ops":["+quant_fp8"],"pass_config":{"fuse_norm_quant":true,"fuse_act_quant":false}}'` | vllm/config/compilation.py:128 (`PassConfig.fuse_norm_quant`) |
| Profiling | `--profile --profiler-config '{"profiler":"torch","torch_profiler_dir":...,"delay_iterations":8,"max_iterations":3,"torch_profiler_with_stack":false,"ignore_frontend":true}'` | vllm/config/profiler.py. **`VLLM_TORCH_PROFILER_DIR` no longer exists.** |

Configurations:
- **A** default: no flags. Selected kernel `FlashInferFp8DeepGEMMDynamicBlockScaledKernel`.
- **B** `VLLM_DISABLED_KERNELS=FlashInferFp8DeepGEMMDynamicBlockScaledKernel` + fusion flags -> `DeepGemmFp8BlockScaledMMKernel`.
- **C** `VLLM_DISABLED_KERNELS=FlashInferFp8DeepGEMMDynamicBlockScaledKernel,DeepGemmFp8BlockScaledMMKernel` + fusion flags
  -> `CutlassFp8BlockScaledMMKernel`.

Note: on main, defaults already resolve to `custom_ops=['+quant_fp8','none']`, `fuse_norm_quant=True`,
`fuse_act_quant=True` (log: "Enabled custom fusions: norm_quant, act_quant") — even in A. In A they match nothing because
the FlashInfer kernel consumes BF16. B/C pin `fuse_act_quant=false` so B/C differ from A only in norm+quant fusion and GEMM
backend (silu_mul+quant stays unfused, as it would in a norm+quant-only model-code port).

### Re-check of earlier claims on current main
| Claim (0.23.x) | Status at e8e1e961 |
|---|---|
| Default sm90 block-FP8 kernel is `FlashInferFp8DeepGEMMDynamicBlockScaledKernel` | **Holds** (first in `_POSSIBLE_FP8_BLOCK_KERNELS[CUDA]`; selected in log) |
| `apply_input_quant = False`, BF16 in, quant inside | **Holds** (scaled_mm/flashinfer.py:170) |
| M < 32 -> FlashInfer swapAB, M ≥ 32 -> DeepGEMM | **Holds** (`torch.cond(input.shape[0] < 32, ...)`). **Nuance:** the M ≥ 32 branch does NOT use TRT-LLM `scale_1x128`; it calls vLLM `per_token_group_quant_fp8(column_major_scales=True, use_ue8m0=...)` then `deep_gemm.fp8_gemm_nt`. Only M < 32 uses `scale_1x128` + `deep_gemm::fp8_gemm_kernel_swapAB` (TRT-LLM DeepGEMM port inside FlashInfer). |
| Fusion test skips DeepGEMM on sm90 when FlashInfer blockscale is available | **Holds** (tests/compile/fusions_e2e/test_tp1_quant.py:86) |
| `rms_norm_per_block_quant`: 3 grid passes, VEC_SIZE=4, no `__launch_bounds__` | **Holds.** #54111 (5bbc58c0, 2026-08-27) is a race fix: smem `volatile` max-reduction replaced with warp-shuffle `warpReduceMax`. No structural change; VEC_SIZE=4; grid=num_tokens, block=min(hidden,512 or 256). |
| Related change | #61343 (0fae39a0) sped up `per_token_group_quant_8bit` (reciprocal multiply) — the unfused baseline got faster. |
| Fusion patterns | `RMSNormQuantFusionPass` registers group(1x128/1x64) × col-major × e8m0 × tma-aligned variants; fused op takes `is_scale_transposed`. B is buildable. |

# 2. Phase 1 — kernels

## Phase 1 — observed kernels per configuration (3 steady decode steps, input 128, profiled steps 9–11)

Source: traces/p1/<cfg>_bs<bs>/, summaries raw/p1_<cfg>_bs<bs>.json (scripts/summarize_trace.py).
µs = GPU time per decode step (mean of 3 steps), forward-pass kernels only (lm_head + sampler excluded: 30 kernels
outside the step annotation, identical across configs).

| Cfg | bs | Selected linear kernel | GEMM kernel (µs/launch: qkv / o / gate_up / down) | Norm / quant kernels (launches per step × µs/launch) | GPU µs/step |
|---|---|---|---|---|---|
| A | 1 | FlashInferFp8DeepGEMMDynamicBlockScaled | `deep_gemm::fp8_gemm_kernel_swapAB` (FlashInfer/TRT-LLM) 10.84 / 8.79 / 34.94 / 21.16 | `scale_1x128` 144×1.59; triton `fused_add_rms_norm` 72×1.9 | 3862 |
| A | 64 | same | `deep_gemm::sm90_fp8_gemm_1d2d_impl` 11.16 / 8.79 / 36.68 / 21.95 | `per_token_group_quant_8bit` 144×2.75; triton `fused_add_rms_norm` 72×2.2 | 4962 |
| B | 1 | DeepGemmFp8BlockScaledMM | `deep_gemm::sm90_fp8_gemm_1d2d_impl` 12.13 / 10.12 / 36.65 / 24.33 | **`rms_norm_per_block_quant`** 72×5.7; `per_token_group_quant_8bit` 72×2.3; triton silu_mul 36×1.5; triton_poi_fused_0 36×1.2 | 4367 |
| B | 64 | same | `sm90_fp8_gemm_1d2d_impl` 11.35 / 8.85 / 37.49 / 22.08 | **`rms_norm_per_block_quant`** 72×5.7; `per_token_group_quant_8bit` 72×2.96; triton silu_mul 36×2.2; triton_poi_fused_0 36×1.6 | 5146 |
| C | 1 | CutlassFp8BlockScaledMM | `cutlass_3x_gemm_fp8_blockwise` 13.27 / 10.04 / 37.79 / 21.93 | **`rms_norm_per_block_quant`** 72×5.66; `per_token_group_quant_8bit` 72×2.1 | 4326 |
| C | 64 | same | `cutlass_3x_gemm_fp8_blockwise` 16.41 / 15.37 / 44.53 / 36.95 | **`rms_norm_per_block_quant`** 72×5.5; `per_token_group_quant_8bit` 72×2.7 | 6278 |

Checks:
- A never runs `rms_norm_per_block_quant` (nothing to match; FlashInfer kernel consumes BF16). ✔
- A switches `fp8_gemm_kernel_swapAB` (bs 1) → `sm90_fp8_gemm_1d2d_impl` (bs 64) at M=32 as documented. ✔
- B: fusion and DeepGEMM coexist — B **is buildable**. 71 fused launches with residual + 1 without (first layer) per step. ✔
- C: `cutlass_3x_gemm_fp8_blockwise` + fused norm. ✔
- 144 FP8 GEMM launches/step in every trace = 36 layers × 4 linears. ✔
- `deep_gemm::fp8_gemm_kernel_swapAB` appears only in A at M < 32. Neither B nor C can reach it.
- `triton_poi_fused_0` (36/step, 1.2–1.6 µs, B/C only) = Inductor copy of the graph-input residual before the in-place fused op (load→store; generated code in ~/.cache/vllm/torch_compile_cache). Counted as quant_norm ("residual_copy").

First-look deltas (one trace each, Phase 3 repeats this at bs 1/32/256):
- bs=1: B +505 µs/step (+13%): GEMM +270, quant/norm +262. C +463 µs (+12%): GEMM +263, quant/norm +241.
- bs=64: B +185 µs (+3.7%): GEMM +43 (same kernel), quant/norm +149. C +1317 µs (+26.5%): GEMM +1248.

# 3. Phase 2 — latency

## Phase 2 — wall-clock latency (`vllm bench latency`)

Decode: input 128, output 256, warmup 5, iters 20, 3 repeats (config order rotated per repeat).
median / p90 in seconds per batch (median over repeats of per-repeat median / p90).
Δ = median vs A. ⚠ = spread across repeats > 3%.

| bs | A median | A p90 | B median | B p90 | B Δ | C median | C p90 | C Δ | spread A/B/C |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 1.1834 | 1.1990 | 1.3198 | 1.3333 | +11.5% | 1.2927 | 1.3049 | +9.2% | 6.4% ⚠ / 7.7% ⚠ / 3.2% ⚠ |
| 8 | 1.2743 | 1.3185 | 1.3422 | 1.3644 | +5.3% | 1.6573 | 1.6654 | +30.0% | 1.0% / 4.6% ⚠ / 0.4% |
| 16 | 1.3053 | 1.3197 | 1.3992 | 1.4495 | +7.2% | 1.7454 | 1.7540 | +33.7% | 5.3% ⚠ / 2.4% / 2.3% |
| 24 | 1.4353 | 1.5036 | 1.4701 | 1.4948 | +2.4% | 1.8745 | 1.8949 | +30.6% | 8.1% ⚠ / 3.4% ⚠ / 1.7% |
| 32 | 1.4956 | 1.5650 | 1.5075 | 1.5316 | +0.8% | 2.0535 | 2.0944 | +37.3% | 8.9% ⚠ / 0.6% / 2.6% |
| 48 | 1.5810 | 1.5955 | 1.6466 | 1.6624 | +4.1% | 2.3428 | 2.4013 | +48.2% | 2.4% / 2.9% / 0.9% |
| 64 | 1.7195 | 1.7370 | 1.7672 | 1.8211 | +2.8% | 2.5966 | 2.6612 | +51.0% | 10.3% ⚠ / 3.1% ⚠ / 4.5% ⚠ |
| 128 | 2.2682 | 2.3194 | 2.2579 | 2.3086 | -0.5% | 3.5793 | 3.6227 | +57.8% | 4.9% ⚠ / 6.6% ⚠ / 5.1% ⚠ |
| 256 | 3.3287 | 3.4036 | 3.3746 | 3.4540 | +1.4% | 4.6129 | 4.7111 | +38.6% | 14.3% ⚠ / 16.0% ⚠ / 1.0% |

Prefill-heavy: input 2048, output 1, warmup 5, iters 20, one run per cell (no spread).

| bs | A median | A p90 | B median | B p90 | B Δ | C median | C p90 | C Δ | — |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 0.0437 | 0.0534 | 0.0531 | 0.0639 | +21.4% | 0.0514 | 0.0634 | +17.5% | — |
| 8 | 0.3060 | 0.3242 | 0.3063 | 0.3160 | +0.1% | 0.3431 | 0.3540 | +12.1% | — |
| 32 | 1.1060 | 1.1119 | 1.1213 | 1.1322 | +1.4% | 1.2882 | 1.2978 | +16.5% | — |

### Sensitivity: repeats 1+2 only (repeat 0 overlapped host-side analysis; see RESULTS.md)

| bs | A median | B median | B Δ | C median | C Δ | spread (r1 vs r2) A/B/C |
|---|---|---|---|---|---|---|
| 1 | 1.1780 | 1.2870 | +9.3% | 1.2895 | +9.5% | 0.9% / 5.1% ⚠ / 0.5% |
| 8 | 1.2681 | 1.3170 | +3.9% | 1.6585 | +30.8% | 1.0% / 3.8% ⚠ / 0.1% |
| 16 | 1.3219 | 1.3904 | +5.2% | 1.7450 | +32.0% | 2.5% / 1.3% / 0.1% |
| 24 | 1.4243 | 1.4564 | +2.2% | 1.8846 | +32.3% | 1.5% / 1.9% / 1.1% |
| 32 | 1.4641 | 1.5059 | +2.9% | 2.0570 | +40.5% | 4.3% ⚠ / 0.6% / 2.5% |
| 48 | 1.5845 | 1.6623 | +4.9% | 2.3477 | +48.2% | 0.5% / 1.9% / 0.9% |
| 64 | 1.6804 | 1.7624 | +4.9% | 2.5661 | +52.7% | 4.6% ⚠ / 0.5% / 2.4% |
| 128 | 2.2173 | 2.2432 | +1.2% | 3.5630 | +60.7% | 5.0% ⚠ / 1.3% / 0.9% |
| 256 | 3.3029 | 3.3681 | +2.0% | 4.6134 | +39.7% | 1.6% / 0.4% / 1.0% |

# 4. Phase 3 — attribution

## Phase 3 — trace attribution (GPU µs per decode step, forward-pass kernels; lm_head + sampler excluded)

gemm = FP8 linear GEMMs; quant_norm = rmsnorm, activation quant (scale_1x128 / per_token_group_quant), fused
rms_norm_per_block_quant, and the B/C-only residual copy inductor inserts before the in-place fused op;
other = attention, qk-norm+rope, silu_mul, misc.

| bs | cfg | total | gemm | quant_norm | other | Δtotal vs A | Δgemm | Δquant_norm | Δother |
|---|---|---|---|---|---|---|---|---|---|
| 1 | A | 3839 | 2720 | 375 | 744 | — | — | — | — |
| 1 | B | 4397 | 3005 | 637 | 755 | +558 (+14.5%) | +285 | +262 | +11 |
| 1 | C | 4307 | 2986 | 600 | 721 | +468 (+12.2%) | +265 | +225 | -22 |
| 32 | A | 4504 | 2860 | 523 | 1122 | — | — | — | — |
| 32 | B | 4656 | 2881 | 652 | 1123 | +151 (+3.4%) | +21 | +129 | +1 |
| 32 | C | 5770 | 4055 | 615 | 1100 | +1265 (+28.1%) | +1195 | +92 | -22 |
| 256 | A | 8540 | 3586 | 850 | 4104 | — | — | — | — |
| 256 | B | 8707 | 3587 | 983 | 4137 | +167 (+2.0%) | +1 | +134 | +33 |
| 256 | C | 9512 | 4415 | 979 | 4119 | +972 (+11.4%) | +828 | +129 | +15 |

### GEMM per-launch averages (µs/launch; kernel)

| bs | cfg | qkv 4096→6144 | o 4096→4096 | gate_up 4096→24576 | down 12288→4096 | kernel |
|---|---|---|---|---|---|---|
| 1 | A | 10.81 | 8.80 | 34.69 | 21.26 | `deep_gemm::fp8_gemm_kernel_swapAB<N=6144,K=4096>` |
| 1 | B | 12.17 | 10.15 | 36.73 | 24.43 | `void deep_gemm::sm90_fp8_gemm_1d2d_impl<` |
| 1 | C | 13.30 | 10.00 | 37.67 | 21.96 | `void cutlass::device_kernel<vllm::cutlass_3x_gemm_fp8_blockwise<cutlas` |
| 32 | A | 11.86 | 8.90 | 36.92 | 21.76 | `void deep_gemm::sm90_fp8_gemm_1d2d_impl<` |
| 32 | B | 11.61 | 8.99 | 37.69 | 21.72 | `void deep_gemm::sm90_fp8_gemm_1d2d_impl<` |
| 32 | C | 16.49 | 15.24 | 44.07 | 36.85 | `void cutlass::device_kernel<vllm::cutlass_3x_gemm_fp8_blockwise<cutlas` |
| 256 | A | 13.63 | 11.21 | 45.10 | 29.68 | `void deep_gemm::sm90_fp8_gemm_1d2d_impl<` |
| 256 | B | 13.26 | 11.26 | 45.41 | 29.71 | `void deep_gemm::sm90_fp8_gemm_1d2d_impl<` |
| 256 | C | 16.54 | 16.23 | 49.04 | 40.81 | `void cutlass::device_kernel<vllm::cutlass_3x_gemm_fp8_blockwise<cutlas` |

### quant_norm breakdown (µs/step)

| bs | cfg | category µs/step |
|---|---|---|
| 1 | A | {'rmsnorm': 140, 'quant': 235} |
| 1 | B | {'norm_fused_quant': 418, 'quant': 172, 'residual_copy': 45, 'rmsnorm': 2} |
| 1 | C | {'norm_fused_quant': 403, 'quant': 151, 'residual_copy': 44, 'rmsnorm': 2} |
| 32 | A | {'rmsnorm': 152, 'quant': 371} |
| 32 | B | {'norm_fused_quant': 405, 'quant': 189, 'residual_copy': 56, 'rmsnorm': 2} |
| 32 | C | {'norm_fused_quant': 388, 'quant': 171, 'residual_copy': 54, 'rmsnorm': 2} |
| 256 | A | {'rmsnorm': 251, 'quant': 598} |
| 256 | B | {'norm_fused_quant': 521, 'quant': 379, 'residual_copy': 80, 'rmsnorm': 3} |
| 256 | C | {'norm_fused_quant': 523, 'quant': 370, 'residual_copy': 83, 'rmsnorm': 3} |

Per-site split of B's quant_norm delta (added 2026-10-16). Norm sites = qkv and gate_up inputs (both K=4096);
non-norm sites = o_proj (K=4096) and down (K=12288) inputs. At M ≥ 32, A and B quantize the o_proj/down inputs with the same
`per_token_group_quant_8bit` call, so A's norm-site quant = A quant − B quant. At M=1, A uses `scale_1x128` (~1.6 µs, roughly
size-independent at M=1) for all four sites, so its norm-site share is taken as half.

| bs | A norm sites (rmsnorm + quant) | B norm sites (fused + copy + rmsnorm) | Δ norm sites | of which copy | Δ o_proj/down quant |
|---|---|---|---|---|---|
| 1 | 140 + ~117 = ~257 | 418 + 45 + 2 = 465 | ~+208 | +45 | ~+55 (`per_token_group_quant` 2.4 µs vs `scale_1x128` 1.6 µs) |
| 32 | 152 + 182 = 334 | 405 + 56 + 2 = 463 | +129 | +56 | ≈0 |
| 256 | 251 + 219 = 470 | 521 + 80 + 3 = 604 | +134 | +80 | ≈0 |

So the fused kernel alone is +71 µs/step at bs=32 and +51 µs/step at bs=256 versus Triton rmsnorm + `per_token_group_quant`;
the Inductor residual copy adds 56 and 80 µs/step.

### Late decode steps (201–203), bs=256 (traces/p3late)

| cfg | total | gemm | quant_norm | other | Δtotal vs A |
|---|---|---|---|---|---|
| A | 10112 | 3586 | 848 | 5679 | — |
| C | 11070 | 4395 | 978 | 5698 | +958 (+9.5%) |

C wall-clock at bs=256 is +40% (Phase 2) despite +9.5–11.4% GPU time per step; see anomaly in RESULTS.md.

# 5. Phase 4 — microbenchmark

## Phase 4 — `rms_norm_per_block_quant` vs `rms_norm` + `per_token_group_quant` (group 128)

GPU NVIDIA H100 80GB HBM3, hidden 4096, BF16 in, FP8 e4m3 out, fp32 column-major TMA-aligned scales, no UE8M0.
CUDA-graph timed (scripts/40_microbench.py). Unfused = vLLM `_C.rms_norm` (or `_C.fused_add_rms_norm` with residual)
+ `per_token_group_quant_fp8`. Note: in-model config A uses an Inductor Triton rmsnorm instead of `_C.rms_norm`.
%BW = ideal fused bytes / time / 3.35 TB/s (same byte count for both columns).
max rel diff = max |dequant(fused) - dequant(unfused)| / max |dequant(unfused)| (a max-element metric). 5-6% is one
e4m3 ulp (3 mantissa bits) on a single element: the unfused path rounds the normalized row to BF16 before quantizing,
the fused kernel quantizes from fp32, so isolated elements land on adjacent FP8 codes. Correctness evidence vs a BF16
reference is in Phase 5 (rel. Frobenius error 2.5e-2, fused <= unfused).

| tokens | residual | fused µs | unfused µs | fused speedup (unfused/fused) | fused %BW | unfused %BW | max rel diff |
|---|---|---|---|---|---|---|---|
| 1 | no | 4.72 | 4.05 | 0.86× | 0.1% | 0.2% | 6.3e-02 |
| 1 | yes | 5.06 | 4.79 | 0.95× | 0.2% | 0.2% | 5.1e-02 |
| 16 | no | 4.88 | 4.69 | 0.96× | 1.3% | 1.3% | 5.9e-02 |
| 16 | yes | 5.07 | 4.95 | 0.98× | 2.8% | 2.8% | 6.2e-02 |
| 64 | no | 4.97 | 4.86 | 0.98× | 4.8% | 4.9% | 5.4e-02 |
| 64 | yes | 5.17 | 5.15 | 1.00× | 10.7% | 10.7% | 5.1e-02 |
| 256 | no | 6.40 | 6.22 | 0.97× | 14.9% | 15.3% | 5.5e-02 |
| 256 | yes | 6.90 | 6.53 | 0.95× | 31.9% | 33.7% | 5.6e-02 |
| 1024 | no | 16.75 | 12.35 | 0.74× | 22.7% | 30.7% | 4.7e-02 |
| 1024 | yes | 18.97 | 13.95 | 0.74× | 46.4% | 63.1% | 5.9e-02 |
| 4096 | no | 60.78 | 49.93 | 0.82× | 25.0% | 30.4% | 5.0e-02 |
| 4096 | yes | 71.57 | 71.09 | 0.99× | 49.2% | 49.5% | 6.2e-02 |
| 16384 | no | 222.44 | 181.35 | 0.82× | 27.3% | 33.5% | 5.8e-02 |
| 16384 | yes | 260.94 | 269.67 | 1.03× | 54.0% | 52.2% | 6.2e-02 |

# 6. Phase 5 — prototype

## Phase 5 — can the GEMM consume the fused norm+quant output directly? (proto/proto_gemm.py)

µs = one CUDA-graph replay of the whole path (norm/quant + GEMM). qkv / gate_up include the residual-add RMSNorm
(today: `_C.fused_add_rms_norm` → BF16 → FlashInfer/DeepGEMM op with internal quant; proposed: fused
`rms_norm_per_block_quant`). o_proj / down have no norm in front: proposed = standalone `per_token_group_quant`.
prop_dg = vLLM `deep_gemm.fp8_gemm_nt`; prop_fi = FlashInfer `fp8_blockscale_gemm_sm90` with FP8 input + scales
(swapAB for M<32). Transform cost = 0: the fused op writes the required layout (see kernel lists).
rel err = ‖y − y_ref‖_F / ‖y_ref‖_F against BF16-activation × dequantized-FP8-weight reference.
Baseline caveat: 'today' uses vLLM _C.fused_add_rms_norm (~2.5-3 us here); in-model config A uses an Inductor Triton
add+rmsnorm (~1.9 us), so in-model the fused kernel's deficit is slightly larger than shown.
Only qkv and gate_up have a norm in front; the o_proj/down rows compare quant+GEMM backends (swapAB vs 1d2d), not fusion.

| shape | M | today µs | prop_dg µs | prop_fi µs | best prop vs today | rel err today | rel err dg | rel err fi | max abs today / dg / fi |
|---|---|---|---|---|---|---|---|---|---|
| qkv 4096→6144 | 1 | 13.81 | 15.15 | 14.98 | +8.5% | 2.52e-02 | 2.49e-02 | 2.49e-02 | 1.02e-01 / 9.57e-02 / 9.57e-02 |
| qkv 4096→6144 | 8 | 13.62 | 13.05 | 14.29 | -4.2% | 2.56e-02 | 2.54e-02 | 2.54e-02 | 1.35e-01 / 1.11e-01 / 1.11e-01 |
| qkv 4096→6144 | 16 | 13.50 | 13.12 | 14.08 | -2.8% | 2.57e-02 | 2.55e-02 | 2.55e-02 | 1.25e-01 / 1.24e-01 / 1.24e-01 |
| qkv 4096→6144 | 32 | 12.96 | 13.24 | 17.32 | +2.2% | 2.70e-02 | 2.55e-02 | 2.55e-02 | 1.26e-01 / 1.20e-01 / 1.20e-01 |
| qkv 4096→6144 | 64 | 12.04 | 12.57 | 12.52 | +4.0% | 2.69e-02 | 2.54e-02 | 2.54e-02 | 1.26e-01 / 1.20e-01 / 1.20e-01 |
| qkv 4096→6144 | 256 | 18.99 | 19.08 | 20.96 | +0.5% | 2.68e-02 | 2.54e-02 | 2.54e-02 | 1.40e-01 / 1.28e-01 / 1.28e-01 |
| o_proj 4096→4096 | 1 | 10.23 | 10.86 | 10.98 | +6.2% | 2.64e-02 | 2.64e-02 | 2.64e-02 | 9.07e-02 / 9.07e-02 / 9.07e-02 |
| o_proj 4096→4096 | 8 | 9.88 | 8.62 | 10.63 | -12.8% | 2.55e-02 | 2.55e-02 | 2.55e-02 | 1.12e-01 / 1.12e-01 / 1.12e-01 |
| o_proj 4096→4096 | 16 | 9.73 | 8.67 | 10.54 | -10.9% | 2.58e-02 | 2.58e-02 | 2.58e-02 | 1.12e-01 / 1.12e-01 / 1.12e-01 |
| o_proj 4096→4096 | 32 | 9.04 | 8.84 | 13.29 | -2.2% | 2.66e-02 | 2.59e-02 | 2.59e-02 | 1.33e-01 / 1.22e-01 / 1.22e-01 |
| o_proj 4096→4096 | 64 | 8.38 | 7.97 | 9.00 | -4.9% | 2.67e-02 | 2.57e-02 | 2.57e-02 | 1.30e-01 / 1.19e-01 / 1.19e-01 |
| o_proj 4096→4096 | 256 | 12.26 | 12.08 | 14.29 | -1.5% | 2.67e-02 | 2.57e-02 | 2.57e-02 | 1.40e-01 / 1.25e-01 / 1.25e-01 |
| gate_up 4096→24576 | 1 | 39.99 | 43.07 | 41.34 | +3.4% | 2.56e-02 | 2.54e-02 | 2.54e-02 | 1.13e-01 / 1.06e-01 / 1.06e-01 |
| gate_up 4096→24576 | 8 | 40.96 | 43.64 | 42.10 | +2.8% | 2.58e-02 | 2.56e-02 | 2.56e-02 | 1.28e-01 / 1.18e-01 / 1.18e-01 |
| gate_up 4096→24576 | 16 | 41.44 | 43.55 | 42.45 | +2.4% | 2.57e-02 | 2.55e-02 | 2.55e-02 | 1.18e-01 / 1.28e-01 / 1.28e-01 |
| gate_up 4096→24576 | 32 | 43.61 | 44.01 | 43.41 | -0.5% | 2.71e-02 | 2.55e-02 | 2.55e-02 | 1.29e-01 / 1.29e-01 / 1.29e-01 |
| gate_up 4096→24576 | 64 | 44.24 | 44.66 | 44.35 | +0.2% | 2.69e-02 | 2.55e-02 | 2.55e-02 | 1.47e-01 / 1.34e-01 / 1.34e-01 |
| gate_up 4096→24576 | 256 | 57.55 | 60.03 | 67.26 | +4.3% | 2.69e-02 | 2.55e-02 | 2.55e-02 | 1.52e-01 / 1.37e-01 / 1.37e-01 |
| down 12288→4096 | 1 | 23.97 | 27.50 | 24.71 | +3.1% | 2.61e-02 | 2.61e-02 | 2.61e-02 | 8.99e-02 / 9.17e-02 / 9.17e-02 |
| down 12288→4096 | 8 | 23.79 | 23.19 | 24.81 | -2.5% | 2.56e-02 | 2.56e-02 | 2.56e-02 | 1.13e-01 / 1.13e-01 / 1.13e-01 |
| down 12288→4096 | 16 | 24.09 | 23.37 | 25.14 | -3.0% | 2.56e-02 | 2.56e-02 | 2.56e-02 | 1.07e-01 / 1.05e-01 / 1.05e-01 |
| down 12288→4096 | 32 | 23.21 | 23.15 | 32.15 | -0.3% | 2.68e-02 | 2.58e-02 | 2.58e-02 | 1.28e-01 / 1.08e-01 / 1.08e-01 |
| down 12288→4096 | 64 | 23.85 | 24.41 | 25.61 | +2.3% | 2.67e-02 | 2.58e-02 | 2.58e-02 | 1.26e-01 / 1.11e-01 / 1.11e-01 |
| down 12288→4096 | 256 | 35.71 | 35.23 | 39.46 | -1.3% | 2.67e-02 | 2.58e-02 | 2.58e-02 | 1.45e-01 / 1.20e-01 / 1.20e-01 |

### Kernels launched per path (one call, M=1 and M=64, qkv)

- M=1 today: `std::enable_if<(((8)>(0)))&&vllm::_typeConvert<c10::BFloat16>::exists,`, `void tensorrt_llm::kernels::fp8_blockscale_gemm::scale_1x128_kernel<__`, `void deep_gemm::fp8_gemm_kernel_swapAB<6144u, 4096u, 128u, 16u, 128u, `
- M=1 prop_dg: `void vllm::rms_norm_per_block_quant_kernel<c10::BFloat16, c10::Float8_`, `void deep_gemm::sm90_fp8_gemm_1d2d_impl<(cute::UMMA::Major)0, 0u, 6144`
- M=1 prop_fi: `void vllm::rms_norm_per_block_quant_kernel<c10::BFloat16, c10::Float8_`, `void deep_gemm::fp8_gemm_kernel_swapAB<6144u, 4096u, 128u, 16u, 128u, `
- M=64 today: `std::enable_if<(((8)>(0)))&&vllm::_typeConvert<c10::BFloat16>::exists,`, `void per_token_group_quant_8bit_register_kernel<c10::BFloat16, c10::Fl`, `void deep_gemm::sm90_fp8_gemm_1d2d_impl<(cute::UMMA::Major)0, 0u, 6144`
- M=64 prop_dg: `void vllm::rms_norm_per_block_quant_kernel<c10::BFloat16, c10::Float8_`, `void deep_gemm::sm90_fp8_gemm_1d2d_impl<(cute::UMMA::Major)0, 0u, 6144`
- M=64 prop_fi: `void vllm::rms_norm_per_block_quant_kernel<c10::BFloat16, c10::Float8_`, `void deep_gemm::fp8_gemm_kernel<6144u, 4096u, 64u, 48u, 128u, 1u, 6u, `

# 7. Conclusions

## Conclusions

### Old (0.23.x, H100 NVL) vs now (main e8e1e961, H100 SXM5)

| Claim | 0.23.x | Now | Verdict |
|---|---|---|---|
| Default sm90 kernel takes BF16, quantizes internally; fusion has nothing to match | yes | yes (`apply_input_quant=False`; e2e fusion test still skips DeepGEMM when FlashInfer blockscale exists) | holds |
| M<32 swapAB / M≥32 DeepGEMM | yes | yes. But M≥32 quantizes with vLLM's `per_token_group_quant_fp8` (column-major, UE8M0), not TRT-LLM `scale_1x128` | holds, nuance |
| bs=1 decode regression when fusing | ≈+13% (GEMM +362 µs, fused +186 µs) | B +11.5% wall (median of 3; +9.3% repeats 1+2); GPU +558 µs/step = GEMM +285 + quant/norm +262. quant/norm: B 637 = fused 418 + `per_token_group_quant` on o_proj/down inputs 172 + residual copy 45 + 2, vs A 375 = Triton rmsnorm 140 + `scale_1x128` ×144 235. So ~+60 µs of it is `per_token_group_quant` (2.4 µs) vs `scale_1x128` (1.6 µs) on non-norm GEMM inputs, not fusion. C +9.2% wall | holds |
| bs=256 fused + CUTLASS | GPU +56% (CUTLASS 33.6 vs 19.9 µs/launch) | C GPU +11.4% (CUTLASS 16.5/16.2/49.0/40.8 vs DeepGEMM 13.6/11.2/45.1/29.7 µs/launch qkv/o/gate_up/down), wall +39 to +40% (see anomaly) | holds, smaller GPU gap |
| bs=256 fused + DeepGEMM (missing cell) | — | B wall +1.4 to +2.0%, GPU +2.0%: GEMM +1 µs, quant/norm +134 µs | **new: small regression, entirely from norm/quant** |
| `rms_norm_per_block_quant` slower than rms_norm + quant | 1.08–1.36× | standalone 0.74–1.03× speed of unfused (i.e. up to 1.35× slower); in-model 5.7 µs/launch vs ~2 µs Triton rmsnorm | holds. #54111 only fixed a race (warp-shuffle max); structure unchanged (3 passes, VEC_SIZE=4, 1 CTA/token, no `__launch_bounds__`) |

Note: the old "DeepGEMM swapAB at bs=256" label was wrong, not a dispatch change. The v0.23.0 source has the same
`torch.cond(input.shape[0] < 32, ...)` split (FlashInfer swapAB below 32, DeepGEMM `fp8_gemm_nt` at 32 and above) and the same
vendored DeepGEMM fallback, so at bs=256 arm A ran a DeepGEMM kernel there too. The 0.23.x "CUTLASS 33.6 vs swapAB 19.9 µs"
comparison is CUTLASS vs DeepGEMM.

### Phase 5: swapAB answer

**Yes, the M<32 FlashInfer swapAB entry point accepts pre-quantized FP8 input plus scales:**
- `flashinfer.gemm.fp8_blockscale_gemm_sm90(input_fp8, weight_fp8, input_scale_fp32, weight_scale)` calls the TRT-LLM runner's `gemm(__nv_fp8_e4m3*, ...)` overload.
- That overload calls the same `fp8_gemm_run` → `gemm_dispatch` (M<32 → `runGemmSwapAB`) that the BF16 path reaches after `scale_1x128_kernel`.
- The scale layout it expects is `scales[k_block * align4(M) + m]`, fp32 dequant scale (amax/448), no UE8M0.
- `rms_norm_per_block_quant(is_scale_transposed=True, tma_alignment=4)` writes exactly that layout, so the transform cost is zero.
- Kernel lists confirm: proposed path at M=1 = `rms_norm_per_block_quant_kernel` → `deep_gemm::fp8_gemm_kernel_swapAB`, with nothing in between.
- vLLM does not expose this today: `FlashInferFp8BlockScaledMMKernel.apply_block_scaled_mm` always passes BF16. Using it needs a
  kernel class with `apply_input_quant=True` that forwards `(q, s)` to `fp8_blockscale_gemm_sm90`.
- The bundled deep_gemm 2.8.1 has no sm90 swapAB kernel (only `1d1d`/`1d2d`), so the swapAB path must go through FlashInfer.

**But it does not make the port a win:**
- With fused + FlashInfer FP8-in, M=1 is +3.4% (gate_up) to +8.5% (qkv) slower than today's path. The GEMM is the same kernel; the fused
  norm kernel (~5 µs) costs more than `fused_add_rms_norm` + `scale_1x128` (~3.5 µs).
- At M=8–16, fused + DeepGEMM `1d2d` is 2.5–12.8% faster than today on qkv/o_proj/down in this isolated, L2-warm benchmark.
  The gain tracks weight size versus L2, so it does not yet show that swapAB is the slower small-M kernel in the model.
- At M ≥ 32, all paths are within ±4%.
- Numerics are equivalent: rel. Frobenius error vs BF16 reference is 2.5–2.7e-2 for all paths. The fused paths are slightly lower
  (2.55e-2 vs 2.69e-2 at M≥32) because they keep exact fp32 scales, whereas today's M≥32 path rounds activation scales to UE8M0.
  The fused op has no UE8M0 option, so fused + DeepGEMM silently runs with non-UE8M0 activation scales while
  `VLLM_USE_DEEP_GEMM_E8M0=1`. That is numerically fine on sm90 (fp32 SF), but it is a semantic mismatch to note in the port.

### Five bullets

1. **Changed since 0.23.x:** fusion passes on by default; profiler is configured via `--profiler-config`; DeepGEMM bundled (2.8.1); the M≥32 branch uses vLLM's UE8M0 `per_token_group_quant`; #54111 is a race fix only (kernel structure unchanged); DeepGEMM `1d2d` at M=1 is now only 6–15% slower per launch than swapAB.
2. **The regression still holds:** B +11.5% at bs=1 and −0.5% to +7.2% at bs 8–256 (median of 3; repeats 1+2: +1.2% to +9.3%, positive everywhere); C +9% to +58%; prefill (2048 tok) B ≈ A at bs 8/32, C +12% to +17%.
3. **At bs ≥ 32 the GEMM is identical (±1%); the whole cost is norm/quant (+130 µs/step):** a fused kernel slower than Triton rmsnorm + group quant, plus an Inductor residual copy. At bs=1, losing swapAB adds +285 µs.
4. **The proposed fix is mechanically viable with zero transform cost** (FlashInfer swapAB takes FP8 + scales; `tma_alignment=4` gives the exact layout; numerics equal or better), **but it is not a net win:** +3.4% to +8.5% at M=1 for qkv/gate_up, ±4% elsewhere.
5. **Recommendation (revised 2026-10-16, see "Upstream state"):** the producer side of the manual port, including 1×128, is in
   open PR #61073, which writes row-major scales and calls the C++ `rms_norm_per_block_quant`. Nothing consumes a 1×128
   `QuantizedActivation` yet. What is missing: (a) a block-FP8 consumer (`Fp8BlockScaledMMLinearKernel.input_quant_key()`,
   `expose_input_quant_key` in `fp8.py`, and an FP8-in branch in the sm90 FlashInfer/DeepGEMM class); (b) a scale-layout
   convention, where fp32, column-major, M padded to 4 is zero-transform for both sm90 GEMMs; (c) a producer faster than
   ~3.5 µs at M ≤ 256: candidates are FlashInfer `fused_add_rmsnorm_fp8_block_quant` (#4480, already in FlashInfer ≥ 0.7.0),
   Helion (#36895), or a register-resident C++ path like #61753. Keep the unfused sm90 default until (c) is measured.

# 8. Upstream state as of 2026-10-16 (vLLM main 1a5b1870, FlashInfer v0.7.2)

Read from source and open PR heads. Nothing here was measured.

**Where `rms_norm_per_block_quant` runs by default (block-FP8 dense linears, `fuse_norm_quant` on by default):**

| Arch | Selected block-FP8 linear kernel | Does the norm+quant pass match? |
|---|---|---|
| sm90 | `FlashInferFp8DeepGEMMDynamicBlockScaledKernel` (BF16 in, `apply_input_quant=False`) | No: nothing to match |
| sm100 | `DeepGemmFp8BlockScaledMMKernel`; `QuantFP8` takes the packed-UE8M0 `per_token_group_quant_fp8_packed_for_deepgemm` path | No: e2e test skips it ("DeepGEMM & quant matching not currently supported") |
| sm120 | `CutlassFp8BlockScaledMMKernel` (CUTLASS block FP8 is enabled for capability ≥ 100) | Yes: col-major, non-UE8M0 `per_token_group_quant` → fused kernel runs (selection logic only, not checked on hardware) |
| sm89 (RTX 4090) | `MarlinFP8ScaledMMLinearKernel` (weight-only; DeepGEMM and CUTLASS block FP8 need sm90+) | No: BF16 activations |

So on the default path the fused C++ kernel is reachable only on sm120. A local 4090 run of Qwen3-8B-FP8 exercises Marlin W8A16
and cannot show an end-to-end effect of norm+quant fusion; it is still fine for kernel microbenchmarks.

**Manual fusion (RFC #42770 / #43224 / #43500):**
- The consumer-side `QuantizedActivation` contract is on main for per-tensor/per-token FP8 (`ScaledMMLinearKernel`), MXFP8 and NVFP4.
  No `Fp8BlockScaledMMLinearKernel` implements `input_quant_key()`, and `Fp8LinearMethod` (`fp8.py`) never calls
  `expose_input_quant_key`. The 1×128 act+quant producer already on main (`_silu_and_mul_fp8_dynamic_128`) therefore has no consumer
  for `fp8` block checkpoints such as Qwen3-8B-FP8.
- `QuantizedActivation`'s docstring leaves scale layout to "the consumer kernel's convention", with a TODO (mgoin) to encode it.
  The existing 1×128 act-quant producer writes row-major scales (`is_scale_transposed=False`).
- PR #61073 (edesoiza, open since 10-11, no approvals yet) adds `maybe_fused_norm_quant` with 1×128 and 1×64 producers via the C++
  `rms_norm_per_block_quant` using row-major scales, plus Llama call sites. It adds no block-FP8 consumer, and its benchmarks cover
  per-tensor/per-token only. Qwen2/Qwen3 decoder layers call neither `maybe_fused_*` helper.
- PR #45364 (HumphreySun98) now covers static per-tensor only and defers 1×128 because "its consumer interface differs".

**Related open PRs:** #60481 (ilyas-mallah) defunctionalizes `rms_norm_per_block_quant`, which removes the residual copy measured here.
#61753 (Samridh) adds a register-resident path for the per-token sibling `rms_norm_dynamic_per_token_quant`, only when there are
≥ 2 tokens per SM (prefill sizes); it does not touch the per-block kernel. #61641 (mgoin) adds paired FP8 conversion in
`norm_and_quant`, shared with the per-block kernel.

**FlashInfer #4480** (`flashinfer.norm.fused_add_rmsnorm_fp8_block_quant`) is in v0.7.0+ (so also in the 0.7.1 used here; vLLM now
pins 0.7.2). It writes fp32 scales of shape `(H/128, round_up(M,4))`, i.e. the same column-major TMA layout as
`rms_norm_per_block_quant(is_scale_transposed=True, tma_alignment=4)`. It has an add+residual variant only, and it also writes a
bf16 `normed_out`. Its published H200 numbers start at M=256 (10.2 µs at H=4096, not graph-timed), so there is no decode-size data.
v0.7.2 includes #6322 (reads device-produced scales after the PDL wait in norm/quant kernels), so benchmark on 0.7.2.

**Helion #36895** (`vllm.kernels.helion.ops.rms_norm_per_block_quant`) has the same signature as the C++ op, H100 configs exist, and it
is opt-in (`pip install vllm[helion]`).

# Reproduce

```
bash scripts/00_setup.sh; bash scripts/00_download.sh; bash scripts/01_env.sh
bash scripts/10_phase1_profile.sh p1 'A B C' '1 64'           # Phase 1
bash scripts/20_phase2_latency.sh && python scripts/21_aggregate_latency.py && python scripts/22_sensitivity.py
bash scripts/23_prefill_split.sh                               # Phase 2b
bash scripts/10_phase1_profile.sh p3 'A B C' '1 32 256' && python scripts/30_attribution.py p3 1,32,256
DELAY=200 OUTLEN=256 bash scripts/10_phase1_profile.sh p3late 'A C' 256
python scripts/40_microbench.py raw/p4_microbench.json; python proto/proto_gemm.py raw/p5_proto.json
python scripts/50_phase4_5_tables.py; bash scripts/60_assemble.sh
```
