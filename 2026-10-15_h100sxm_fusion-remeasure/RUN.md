# RUN.md — environment and pinned state

## Hardware (DEVIATION from brief)
- GPU: **NVIDIA H100 80GB HBM3 (SXM5, 700 W cap, 132 SMs)** — NOT H100 NVL as the brief expected. User approved
  proceeding (2026-10-15). Absolute µs are not comparable with the earlier 0.23.x NVL numbers; A→B / A→C deltas are.
- HBM3 peak used for % of bandwidth: 3.35 TB/s (H100 SXM5 datasheet).
- Driver 610.57.04 (CUDA UMD 13.3). nvcc 13.2 (V13.2.86). Image env CUDA_VERSION=13.2.1.
- Clocks: NOT locked (unprivileged). Max SM 1980 MHz, mem 2619 MHz. Idle 345 MHz; observed 1980 MHz under load.
  Temp/SM clock/power logged before every Phase 2 run (logs/p2_driver.log).
- No other GPU processes (nvidia-smi --query-compute-apps empty) at start.
- Host: 192 cores, 1 TB RAM, 534 GB overlay disk. $WORK=/workspace (no separate volume; / is the largest).

## Software
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

## Flags used (checked against source at the pinned SHA)
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

## Re-check of earlier claims on current main
| Claim (0.23.x) | Status at e8e1e961 |
|---|---|
| Default sm90 block-FP8 kernel is `FlashInferFp8DeepGEMMDynamicBlockScaledKernel` | **Holds** (first in `_POSSIBLE_FP8_BLOCK_KERNELS[CUDA]`; selected in log) |
| `apply_input_quant = False`, BF16 in, quant inside | **Holds** (scaled_mm/flashinfer.py:170) |
| M < 32 -> FlashInfer swapAB, M ≥ 32 -> DeepGEMM | **Holds** (`torch.cond(input.shape[0] < 32, ...)`). **Nuance:** the M ≥ 32 branch does NOT use TRT-LLM `scale_1x128`; it calls vLLM `per_token_group_quant_fp8(column_major_scales=True, use_ue8m0=...)` then `deep_gemm.fp8_gemm_nt`. Only M < 32 uses `scale_1x128` + `deep_gemm::fp8_gemm_kernel_swapAB` (TRT-LLM DeepGEMM port inside FlashInfer). |
| Fusion test skips DeepGEMM on sm90 when FlashInfer blockscale is available | **Holds** (tests/compile/fusions_e2e/test_tp1_quant.py:86) |
| `rms_norm_per_block_quant`: 3 grid passes, VEC_SIZE=4, no `__launch_bounds__` | **Holds.** #54111 (5bbc58c0, 2026-08-27) is a race fix: smem `volatile` max-reduction replaced with warp-shuffle `warpReduceMax`. No structural change; VEC_SIZE=4; grid=num_tokens, block=min(hidden,512 or 256). |
| Related change | #61343 (0fae39a0) sped up `per_token_group_quant_8bit` (reciprocal multiply) — the unfused baseline got faster. |
| Fusion patterns | `RMSNormQuantFusionPass` registers group(1x128/1x64) × col-major × e8m0 × tma-aligned variants; fused op takes `is_scale_transposed`. B is buildable. |
