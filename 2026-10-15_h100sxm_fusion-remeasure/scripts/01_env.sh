#!/bin/bash
# Collect environment facts for RUN.md (raw dump -> logs/env.txt)
source /workspace/.venv/bin/activate
cd /tmp
{
echo "## date"; date -u
echo "## vllm sha"; cat ~/fusion-study/VLLM_SHA
echo "## versions"
python - <<'EOF' 2>/dev/null
import torch, triton, flashinfer, vllm
from vllm.utils.deep_gemm import _import_deep_gemm
dg = _import_deep_gemm()
print("vllm", vllm.__version__)
print("torch", torch.__version__, "cuda(torch)", torch.version.cuda)
print("triton", triton.__version__)
print("flashinfer", flashinfer.__version__)
print("deep_gemm", getattr(dg, "__version__", "?"), dg.__name__, dg.__file__)
print("gpu", torch.cuda.get_device_name(0), "cc", torch.cuda.get_device_capability(0),
      "SMs", torch.cuda.get_device_properties(0).multi_processor_count)
EOF
echo "## nvcc"; nvcc --version | tail -2
echo "## nvidia-smi"; nvidia-smi --query-gpu=name,driver_version,memory.total,power.limit,clocks.max.sm,clocks.max.mem --format=csv
nvidia-smi -q -d CLOCK
echo "## gpu processes"; nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv
echo "## cpu"; nproc; lscpu | grep 'Model name'
} > ~/fusion-study/logs/env.txt 2>&1
cat ~/fusion-study/logs/env.txt | head -30
