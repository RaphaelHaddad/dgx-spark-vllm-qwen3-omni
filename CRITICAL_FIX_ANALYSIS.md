# Critical Blackwell GB10 Fixes for vLLM

## Overview

Three critical fixes are required for vLLM on Blackwell GB10 (sm_121a) GPUs with CUDA 13.0+:

1. **CMakeLists.txt SM120 Support** - Add missing architecture
2. **vLLM Commit Version** - Use commit with Blackwell/Triton fixes
3. **Triton Version Pinning** - Use tested working commit

## Fix 1: CMakeLists.txt SM120 Support

### Root Cause

vLLM v0.11.1rc3 CMakeLists.txt has **incomplete architecture support** for Blackwell GB10 (sm_121a) MOE kernels when using CUDA 13.0+.

## The Problem

For CUDA 13.0+, the code uses these branches:
- **Line 490**: Regular MOE kernels
- **Line 671**: Grouped MM MOE kernels

Original v0.11.1rc3:
```cmake
# Line 490
cuda_archs_loose_intersection(SCALED_MM_ARCHS "10.0f;11.0f" "${CUDA_ARCHS}")

# Line 671
cuda_archs_loose_intersection(SCALED_MM_ARCHS "10.0f;11.0f" "${CUDA_ARCHS}")
```

**BOTH lines are missing `12.0f` (SM120) support!**

## The Fix

Both lines need `12.0f` added:
```cmake
# Line 490
cuda_archs_loose_intersection(SCALED_MM_ARCHS "10.0f;11.0f;12.0f" "${CUDA_ARCHS}")

# Line 671
cuda_archs_loose_intersection(SCALED_MM_ARCHS "10.0f;11.0f;12.0f" "${CUDA_ARCHS}")
```

## Error Symptoms

Without this fix:
```
ImportError: undefined symbol: _Z20cutlass_moe_mm_sm100RN2at6TensorERKS0_S3_S3_S3_S3_S3_S3_S3_S3_bb
```

The MOE kernels for SM100/SM120 aren't compiled, causing import failures.

## Why install.sh Works

The sed command on line 323:
```bash
sed -i 's/cuda_archs_loose_intersection(SCALED_MM_ARCHS "10.0f;11.0f"/cuda_archs_loose_intersection(SCALED_MM_ARCHS "10.0f;11.0f;12.0f"/' CMakeLists.txt
```

This replaces **ALL** occurrences, fixing both lines 490 and 671 in one command.

## Verified Solution

Tested on NVIDIA DGX Spark with Blackwell GB10, CUDA 13.0:
- [OK] Line 490 fixed: `"10.0f;11.0f;12.0f"`
- [OK] Line 671 fixed: `"10.0f;11.0f;12.0f"`
- [OK] vLLM imports successfully
- [OK] No cutlass_moe_mm_sm100 symbol errors
- [OK] Build time: ~19 minutes

## Fix 2: vLLM Commit Version

### Issue

vLLM tag `v0.11.1rc3` lacks critical Triton/PyTorch Inductor fixes for Blackwell.

### Solution

Use commit `66a168a197ba214a5b70a74fa2e713c9eeb3251a` (6 commits ahead of v0.11.1rc3):
- Contains Triton JIT compilation fixes
- Includes PyTorch Inductor optimizations for Blackwell
- Adds proper backend registration handling

### Installation

```bash
cd vllm
git checkout 66a168a197ba214a5b70a74fa2e713c9eeb3251a
git submodule update --init --recursive
```

## Fix 3: Triton Version Pinning

### Issue

Latest Triton main branch (as of late October 2025) has intermittent JITFunction compilation issues with PyTorch Inductor on Blackwell.

### Solution

Pin to tested working commit: `4caa0328bf8df64896dd5f6fb9df41b0eb2e750a` (October 25, 2025)
- Verified stable with Blackwell GB10
- Passes all compilation tests
- No JITFunction.constexprs errors

### Installation

```bash
cd triton
git checkout 4caa0328bf8df64896dd5f6fb9df41b0eb2e750a
git submodule update --init --recursive
python -m pip install --no-build-isolation -v .
```

## Complete Verified Configuration

| Component | Version/Commit | Notes |
|-----------|---------------|-------|
| **vLLM** | `66a168a197ba214a5b70a74fa2e713c9eeb3251a` | 6 commits ahead of v0.11.1rc3 |
| **Triton** | `4caa0328bf8df64896dd5f6fb9df41b0eb2e750a` | October 25, 2025 |
| **PyTorch** | `2.9.0+cu130` | From vLLM requirements |
| **CUDA** | `13.0` (V13.0.88) | System CUDA |
| **Python** | `3.12.3` | |

## Testing

Verified working with:
```bash
python -c "from vllm import LLM, SamplingParams; \
llm = LLM(model='Qwen/Qwen2.5-0.5B-Instruct', max_model_len=512); \
print(llm.generate(['Hello'], SamplingParams(max_tokens=20)))"
```

**All tests pass**: Import, compilation, CUDA graphs, and text generation all work correctly.
