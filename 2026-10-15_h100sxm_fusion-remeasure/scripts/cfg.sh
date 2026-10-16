# Source me. Usage: cfg_args <A|B|C>  -> sets CFG_ENV (env assignments) and CFG_ARGS (CLI args)
# Flags verified against vLLM e8e1e961:
#   VLLM_DISABLED_KERNELS: comma list of kernel class __name__ (vllm/envs.py, kernels/linear/__init__.py)
#   PassConfig.fuse_norm_quant (vllm/config/compilation.py); quant_fp8 custom op must be on for group-quant
#   matching (tests/compile/fusions_e2e/test_tp1_quant.py skips "-quant_fp8" for group quant).
#   fuse_act_quant pinned false so B/C differ from A only by norm+quant fusion + GEMM backend.
source /workspace/.venv/bin/activate
export HF_HOME=/workspace/hf
MODEL=Qwen/Qwen3-8B-FP8
FUSED_CC='{"custom_ops":["+quant_fp8"],"pass_config":{"fuse_norm_quant":true,"fuse_act_quant":false}}'
cfg_args() {
  case "$1" in
    A) CFG_ENV="VLLM_DISABLED_KERNELS="; CFG_ARGS=() ;;
    B) CFG_ENV="VLLM_DISABLED_KERNELS=FlashInferFp8DeepGEMMDynamicBlockScaledKernel"
       CFG_ARGS=(--compilation-config "$FUSED_CC") ;;
    C) CFG_ENV="VLLM_DISABLED_KERNELS=FlashInferFp8DeepGEMMDynamicBlockScaledKernel,DeepGemmFp8BlockScaledMMKernel"
       CFG_ARGS=(--compilation-config "$FUSED_CC") ;;
    *) echo "bad cfg $1"; return 1 ;;
  esac
}
