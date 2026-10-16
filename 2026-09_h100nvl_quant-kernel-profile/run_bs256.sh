#!/bin/bash
# Arm C @ bs=256 (fused rms_norm_per_block_quant path) + Arm A @ bs=256 (separate
# quant kernels) -> measures the fused-kernel per-launch cost and the threshold
# it must beat, both on the same box.
set -u
export HOME=/root
export UV_CACHE_DIR=/dev/shm/uvc
export HF_HOME=/dev/shm/hf
export PATH=/root/.local/bin:$PATH
# uv unpacks wheels into TMPDIR before moving them into the venv. Left on the
# 16G overlay that double-counts every package and hits ENOSPC on torch.
export TMPDIR=/dev/shm/tmp
# Inductor/Triton JIT their launchers into TMPDIR and dlopen them, so those two
# caches must stay on the exec-capable overlay even though TMPDIR does not.
export TORCHINDUCTOR_CACHE_DIR=/root/cache/inductor
export TRITON_CACHE_DIR=/root/cache/triton
mkdir -p /dev/shm/uvc /dev/shm/hf /dev/shm/traces /dev/shm/tmp /root/cache

st() { echo "$1" > /root/stage; }

# venv must live on the overlay, not /dev/shm: tmpfs here is mounted noexec, so
# torch's .so files fail to dlopen ("failed to map segment from shared object").
# Only data (uv cache, HF weights, traces) goes on tmpfs.
if [ ! -x /root/venv/bin/python ]; then
  st INSTALL_UV
  command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
  st VENV
  uv venv --python 3.12 /root/venv
fi
. /root/venv/bin/activate

if ! python -c "import vllm" 2>/dev/null; then
  st PIP
  uv pip install "vllm==0.23.0" 2>&1 | tail -5
fi
st VERIFY
python -c "import vllm;print('VLLM',vllm.__version__)" > /root/ver.txt 2>&1
grep -q VLLM /root/ver.txt || { st SETUP_FAILED; exit 1; }

M=Qwen/Qwen3-8B-FP8
COMMON="--model $M --input-len 1024 --output-len 16 --batch-size 256 \
  --num-iters-warmup 1 --num-iters 1 --profile \
  --profiler-config.profiler=torch \
  --profiler-config.torch_profiler_with_stack=false"

# Arm C: FlashInfer + DeepGemm disabled -> Cutlass blockwise, quant fused into
# the norm/act epilogue (rms_norm_per_block_quant_kernel).
if [ ! -d /dev/shm/traces/C256 ]; then
  st ARM_C
  VLLM_DISABLED_KERNELS=FlashInferFp8DeepGEMMDynamicBlockScaledKernel,DeepGemmFp8BlockScaledMMKernel \
    vllm bench latency $COMMON \
    --profiler-config.torch_profiler_dir=/dev/shm/traces/C256 > /root/logC256.txt 2>&1
fi

# Arm A: nothing disabled -> FlashInfer swapAB, separate norm + scale_1x128.
# Gives the measured threshold instead of an extrapolated one.
if [ ! -d /dev/shm/traces/A256 ]; then
  st ARM_A
  vllm bench latency $COMMON \
    --profiler-config.torch_profiler_dir=/dev/shm/traces/A256 > /root/logA256.txt 2>&1
fi

st ANALYZE
{
  echo "### version:";           cat /root/ver.txt
  echo "### selected kernel C:"; grep -hE "Selected .* for " /root/logC256.txt | tail -2
  echo "### selected kernel A:"; grep -hE "Selected .* for " /root/logA256.txt | tail -2
  echo "### latency C:";         grep -hiE "Avg latency" /root/logC256.txt
  echo "### latency A:";         grep -hiE "Avg latency" /root/logA256.txt
  echo; echo "===== ARM C bs=256 ====="; python /root/an2.py /dev/shm/traces/C256/*.json* 2>&1
  echo; echo "===== ARM A bs=256 ====="; python /root/an2.py /dev/shm/traces/A256/*.json* 2>&1
} > /root/RESULTS256.txt 2>&1
st DONE
