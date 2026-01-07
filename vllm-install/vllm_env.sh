#!/bin/bash
# vLLM Environment Configuration for DGX Spark
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/.vllm/bin/activate"
# Use 12.0f for Blackwell GB10 - matches CMakeLists.txt logic for CUDA >= 13.0
export TORCH_CUDA_ARCH_LIST=12.0f
export VLLM_USE_FLASHINFER_MXFP4_MOE=1
CUDA_PATH=$(ls -d /usr/local/cuda* 2>/dev/null | head -1)
export TRITON_PTXAS_PATH="$CUDA_PATH/bin/ptxas"
export PATH="$CUDA_PATH/bin:$PATH"
# Add PyTorch libraries to LD_LIBRARY_PATH
TORCH_LIB="$SCRIPT_DIR/.vllm/lib/python3.12/site-packages/torch/lib"
export LD_LIBRARY_PATH="$TORCH_LIB:$CUDA_PATH/lib64:$LD_LIBRARY_PATH"
# Cache tiktoken encodings to avoid re-downloading
export TIKTOKEN_CACHE_DIR="$SCRIPT_DIR/.tiktoken_cache"
mkdir -p "$TIKTOKEN_CACHE_DIR"
echo "=== vLLM Environment Active ==="
echo "Virtual env: $VIRTUAL_ENV"
echo "CUDA arch: $TORCH_CUDA_ARCH_LIST"
echo "Python: $(which python)"
echo "==============================="
