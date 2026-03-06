#!/bin/bash
# vLLM Environment Configuration for DGX Spark
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/.vllm/bin/activate"
export TORCH_CUDA_ARCH_LIST=12.1a
export VLLM_USE_FLASHINFER_MXFP4_MOE=1
CUDA_PATH=$(ls -d /usr/local/cuda* 2>/dev/null | head -1)
export TRITON_PTXAS_PATH="$CUDA_PATH/bin/ptxas"
export PATH="$CUDA_PATH/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_PATH/lib64:$LD_LIBRARY_PATH"
# Cache tiktoken encodings to avoid re-downloading
export TIKTOKEN_CACHE_DIR="$SCRIPT_DIR/.tiktoken_cache"
mkdir -p "$TIKTOKEN_CACHE_DIR"
echo "=== vLLM Environment Active ==="
echo "Virtual env: $VIRTUAL_ENV"
echo "CUDA arch: $TORCH_CUDA_ARCH_LIST"
echo "Python: $(which python)"
echo "==============================="
