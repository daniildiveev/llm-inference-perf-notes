# Phase 5 — can the GEMM consume the fused norm+quant output directly? (proto/proto_gemm.py)

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
| qkv 4096→6144 | 1 | 13.81 | 15.15 | 14.98 | +8.4% | 2.52e-02 | 2.49e-02 | 2.49e-02 | 1.02e-01 / 9.57e-02 / 9.57e-02 |
| qkv 4096→6144 | 8 | 13.62 | 13.05 | 14.29 | -4.2% | 2.56e-02 | 2.54e-02 | 2.54e-02 | 1.35e-01 / 1.11e-01 / 1.11e-01 |
| qkv 4096→6144 | 16 | 13.50 | 13.12 | 14.08 | -2.8% | 2.57e-02 | 2.55e-02 | 2.55e-02 | 1.25e-01 / 1.24e-01 / 1.24e-01 |
| qkv 4096→6144 | 32 | 12.96 | 13.24 | 17.32 | +2.2% | 2.70e-02 | 2.55e-02 | 2.55e-02 | 1.26e-01 / 1.20e-01 / 1.20e-01 |
| qkv 4096→6144 | 64 | 12.04 | 12.57 | 12.52 | +3.9% | 2.69e-02 | 2.54e-02 | 2.54e-02 | 1.26e-01 / 1.20e-01 / 1.20e-01 |
| qkv 4096→6144 | 256 | 18.99 | 19.08 | 20.96 | +0.5% | 2.68e-02 | 2.54e-02 | 2.54e-02 | 1.40e-01 / 1.28e-01 / 1.28e-01 |
| o_proj 4096→4096 | 1 | 10.23 | 10.86 | 10.98 | +6.2% | 2.64e-02 | 2.64e-02 | 2.64e-02 | 9.07e-02 / 9.07e-02 / 9.07e-02 |
| o_proj 4096→4096 | 8 | 9.88 | 8.62 | 10.63 | -12.7% | 2.55e-02 | 2.55e-02 | 2.55e-02 | 1.12e-01 / 1.12e-01 / 1.12e-01 |
| o_proj 4096→4096 | 16 | 9.73 | 8.67 | 10.54 | -11.0% | 2.58e-02 | 2.58e-02 | 2.58e-02 | 1.12e-01 / 1.12e-01 / 1.12e-01 |
| o_proj 4096→4096 | 32 | 9.04 | 8.84 | 13.29 | -2.2% | 2.66e-02 | 2.59e-02 | 2.59e-02 | 1.33e-01 / 1.22e-01 / 1.22e-01 |
| o_proj 4096→4096 | 64 | 8.38 | 7.97 | 9.00 | -4.9% | 2.67e-02 | 2.57e-02 | 2.57e-02 | 1.30e-01 / 1.19e-01 / 1.19e-01 |
| o_proj 4096→4096 | 256 | 12.26 | 12.08 | 14.29 | -1.5% | 2.67e-02 | 2.57e-02 | 2.57e-02 | 1.40e-01 / 1.25e-01 / 1.25e-01 |
| gate_up 4096→24576 | 1 | 39.99 | 43.07 | 41.34 | +3.4% | 2.56e-02 | 2.54e-02 | 2.54e-02 | 1.13e-01 / 1.06e-01 / 1.06e-01 |
| gate_up 4096→24576 | 8 | 40.96 | 43.64 | 42.10 | +2.8% | 2.58e-02 | 2.56e-02 | 2.56e-02 | 1.28e-01 / 1.18e-01 / 1.18e-01 |
| gate_up 4096→24576 | 16 | 41.44 | 43.55 | 42.45 | +2.4% | 2.57e-02 | 2.55e-02 | 2.55e-02 | 1.18e-01 / 1.28e-01 / 1.28e-01 |
| gate_up 4096→24576 | 32 | 43.61 | 44.01 | 43.41 | -0.4% | 2.71e-02 | 2.55e-02 | 2.55e-02 | 1.29e-01 / 1.29e-01 / 1.29e-01 |
| gate_up 4096→24576 | 64 | 44.24 | 44.66 | 44.35 | +0.2% | 2.69e-02 | 2.55e-02 | 2.55e-02 | 1.47e-01 / 1.34e-01 / 1.34e-01 |
| gate_up 4096→24576 | 256 | 57.55 | 60.03 | 67.26 | +4.3% | 2.69e-02 | 2.55e-02 | 2.55e-02 | 1.52e-01 / 1.37e-01 / 1.37e-01 |
| down 12288→4096 | 1 | 23.97 | 27.50 | 24.71 | +3.1% | 2.61e-02 | 2.61e-02 | 2.61e-02 | 8.99e-02 / 9.17e-02 / 9.17e-02 |
| down 12288→4096 | 8 | 23.79 | 23.19 | 24.81 | -2.5% | 2.56e-02 | 2.56e-02 | 2.56e-02 | 1.13e-01 / 1.13e-01 / 1.13e-01 |
| down 12288→4096 | 16 | 24.09 | 23.37 | 25.14 | -3.0% | 2.56e-02 | 2.56e-02 | 2.56e-02 | 1.07e-01 / 1.05e-01 / 1.05e-01 |
| down 12288→4096 | 32 | 23.21 | 23.15 | 32.15 | -0.2% | 2.68e-02 | 2.58e-02 | 2.58e-02 | 1.28e-01 / 1.08e-01 / 1.08e-01 |
| down 12288→4096 | 64 | 23.85 | 24.41 | 25.61 | +2.3% | 2.67e-02 | 2.58e-02 | 2.58e-02 | 1.26e-01 / 1.11e-01 / 1.11e-01 |
| down 12288→4096 | 256 | 35.71 | 35.23 | 39.46 | -1.3% | 2.67e-02 | 2.58e-02 | 2.58e-02 | 1.45e-01 / 1.20e-01 / 1.20e-01 |

## Kernels launched per path (one call, M=1 and M=64, qkv)

- M=1 today: `std::enable_if<(((8)>(0)))&&vllm::_typeConvert<c10::BFloat16>::exists,`, `void tensorrt_llm::kernels::fp8_blockscale_gemm::scale_1x128_kernel<__`, `void deep_gemm::fp8_gemm_kernel_swapAB<6144u, 4096u, 128u, 16u, 128u, `
- M=1 prop_dg: `void vllm::rms_norm_per_block_quant_kernel<c10::BFloat16, c10::Float8_`, `void deep_gemm::sm90_fp8_gemm_1d2d_impl<(cute::UMMA::Major)0, 0u, 6144`
- M=1 prop_fi: `void vllm::rms_norm_per_block_quant_kernel<c10::BFloat16, c10::Float8_`, `void deep_gemm::fp8_gemm_kernel_swapAB<6144u, 4096u, 128u, 16u, 128u, `
- M=64 today: `std::enable_if<(((8)>(0)))&&vllm::_typeConvert<c10::BFloat16>::exists,`, `void per_token_group_quant_8bit_register_kernel<c10::BFloat16, c10::Fl`, `void deep_gemm::sm90_fp8_gemm_1d2d_impl<(cute::UMMA::Major)0, 0u, 6144`
- M=64 prop_dg: `void vllm::rms_norm_per_block_quant_kernel<c10::BFloat16, c10::Float8_`, `void deep_gemm::sm90_fp8_gemm_1d2d_impl<(cute::UMMA::Major)0, 0u, 6144`
- M=64 prop_fi: `void vllm::rms_norm_per_block_quant_kernel<c10::BFloat16, c10::Float8_`, `void deep_gemm::fp8_gemm_kernel<6144u, 4096u, 64u, 48u, 128u, 1u, 6u, `
