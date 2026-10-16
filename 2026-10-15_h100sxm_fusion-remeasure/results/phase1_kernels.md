# Phase 1 — observed kernels per configuration (3 steady decode steps, input 128, profiled steps 9–11)

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
