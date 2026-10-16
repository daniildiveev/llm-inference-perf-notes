# Phase 3 — trace attribution (GPU µs per decode step, forward-pass kernels; lm_head + sampler excluded)

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

## GEMM per-launch averages (µs/launch; kernel)

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

## quant_norm breakdown (µs/step)

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

## Late decode steps (201–203), bs=256 (traces/p3late)

| cfg | total | gemm | quant_norm | other | Δtotal vs A |
|---|---|---|---|---|---|
| A | 10112 | 3586 | 848 | 5679 | — |
| C | 11070 | 4395 | 978 | 5698 | +958 (+9.5%) |

C wall-clock at bs=256 is +40% (Phase 2) despite +9.5–11.4% GPU time per step; see anomaly in RESULTS.md.
