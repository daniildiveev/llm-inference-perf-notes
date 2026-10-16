# Conclusions

## Old (0.23.x, H100 NVL) vs now (main e8e1e961, H100 SXM5)

| Claim | 0.23.x | Now | Verdict |
|---|---|---|---|
| Default sm90 kernel takes BF16, quantizes internally; fusion has nothing to match | yes | yes (`apply_input_quant=False`; e2e fusion test still skips DeepGEMM when FlashInfer blockscale exists) | holds |
| M<32 swapAB / M≥32 DeepGEMM | yes | yes. But M≥32 quantizes with vLLM's `per_token_group_quant_fp8` (column-major, UE8M0), not TRT-LLM `scale_1x128` | holds, nuance |
| bs=1 decode regression when fusing | ≈+13% (GEMM +362 µs, fused +186 µs) | B +11.5% wall (median of 3; +9.3% repeats 1+2); GPU +558 µs/step = GEMM +285 + quant/norm +262. quant/norm: B 637 = fused 418 + `per_token_group_quant` on o_proj/down inputs 172 + residual copy 45 + 2, vs A 375 = Triton rmsnorm 140 + `scale_1x128` ×144 235. So ~+60 µs of it is `per_token_group_quant` (2.4 µs) vs `scale_1x128` (1.6 µs) on non-norm GEMM inputs, not fusion. C +9.2% wall | holds |
| bs=256 fused + CUTLASS | GPU +56% (CUTLASS 33.6 vs 19.9 µs/launch) | C GPU +11.4% (CUTLASS 16.5/16.2/49.0/40.8 vs DeepGEMM 13.6/11.2/45.1/29.7 µs/launch qkv/o/gate_up/down), wall +39 to +40% (see anomaly) | holds, smaller GPU gap |
| bs=256 fused + DeepGEMM (missing cell) | — | B wall +1.4 to +2.0%, GPU +2.0%: GEMM +1 µs, quant/norm +134 µs | **new: small regression, entirely from norm/quant** |
| `rms_norm_per_block_quant` slower than rms_norm + quant | 1.08–1.36× | standalone 0.74–1.03× speed of unfused (i.e. up to 1.35× slower); in-model 5.7 µs/launch vs ~2 µs Triton rmsnorm | holds. #54111 only fixed a race (warp-shuffle max); structure unchanged (3 passes, VEC_SIZE=4, 1 CTA/token, no `__launch_bounds__`) |

Note: the old "DeepGEMM swapAB at bs=256" label cannot be right on main. swapAB is only used for M<32. At bs=256, A runs
`deep_gemm::sm90_fp8_gemm_1d2d_impl`.

## Phase 5: swapAB answer

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
- At M=8–16, fused + DeepGEMM `1d2d` is 3–13% faster than today on qkv/o_proj/down, i.e. swapAB is not always the faster small-M
  kernel.
- At M ≥ 32, all paths are within ±4%.
- Numerics are equivalent: rel. Frobenius error vs BF16 reference is 2.5–2.7e-2 for all paths. The fused paths are slightly lower
  (2.55e-2 vs 2.69e-2 at M≥32) because they keep exact fp32 scales, whereas today's M≥32 path rounds activation scales to UE8M0.
  The fused op has no UE8M0 option, so fused + DeepGEMM silently runs with non-UE8M0 activation scales while
  `VLLM_USE_DEEP_GEMM_E8M0=1`. That is numerically fine on sm90 (fp32 SF), but it is a semantic mismatch to note in the port.

## Five bullets

1. **Changed since 0.23.x:** fusion passes on by default; profiler is configured via `--profiler-config`; DeepGEMM bundled (2.8.1); the M≥32 branch uses vLLM's UE8M0 `per_token_group_quant`; #54111 is a race fix only (kernel structure unchanged); DeepGEMM `1d2d` at M=1 is now only 5–15% slower per launch than swapAB.
2. **The regression still holds:** B +11.5% at bs=1 and −0.5% to +7.2% at bs 8–256 (median of 3; repeats 1+2: +1.2% to +9.3%, positive everywhere); C +9% to +58%; prefill (2048 tok) B ≈ A at bs 8/32, C +12% to +17%.
3. **At bs ≥ 32 the GEMM is identical (±1%); the whole cost is norm/quant (+130 µs/step):** a fused kernel slower than Triton rmsnorm + group quant, plus an Inductor residual copy. At bs=1, losing swapAB adds +285 µs.
4. **The proposed fix is mechanically viable with zero transform cost** (FlashInfer swapAB takes FP8 + scales; `tma_alignment=4` gives the exact layout; numerics equal or better), **but it is not a net win:** +3.4% to +8.5% at M=1 for qkv/gate_up, ±4% elsewhere.
5. **Recommendation:** fix `rms_norm_per_block_quant` first (it is latency-bound at 0.1–32% of HBM peak for ≤256 tokens, 5–7 µs vs ~1.9 µs Triton rmsnorm), then route its output via an `apply_input_quant=True` kernel to `fp8_blockscale_gemm_sm90` (M<32) and `fp8_gemm_nt` (M≥32); keep the unfused default until then.
