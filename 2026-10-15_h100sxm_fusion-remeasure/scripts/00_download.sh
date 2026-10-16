#!/bin/bash
set -euxo pipefail
export HF_HOME=/workspace/hf
uvx --from huggingface_hub hf download Qwen/Qwen3-8B-FP8
echo DOWNLOAD_DONE
