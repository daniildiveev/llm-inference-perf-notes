#!/bin/bash
# Phase 0.2-0.3: venv + vLLM main (precompiled).
set -euxo pipefail
export WORK=/workspace HF_HOME=/workspace/hf
cd $WORK
[ -d .venv ] || uv venv --python 3.12 .venv
source .venv/bin/activate
grep -q "$WORK/.venv/bin/activate" ~/.bashrc || echo "source $WORK/.venv/bin/activate" >> ~/.bashrc
[ -d vllm ] || git clone https://github.com/vllm-project/vllm.git vllm
cd vllm
git rev-parse HEAD | tee ~/fusion-study/VLLM_SHA
git log -1 --format='%H %cd %s' | tee ~/fusion-study/logs/vllm_commit.txt
VLLM_USE_PRECOMPILED=1 uv pip install -e . 2>&1 | tail -60
python -c "import vllm, torch; print('vllm', vllm.__version__, 'torch', torch.__version__, 'cuda', torch.version.cuda)"
echo SETUP_DONE
