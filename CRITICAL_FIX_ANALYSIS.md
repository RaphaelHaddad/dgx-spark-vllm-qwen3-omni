# Critical CMakeLists.txt Fix Analysis

## Root Cause

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
- ✅ Line 490 fixed: `"10.0f;11.0f;12.0f"`
- ✅ Line 671 fixed: `"10.0f;11.0f;12.0f"`
- ✅ vLLM imports successfully
- ✅ No cutlass_moe_mm_sm100 symbol errors
- ✅ Build time: ~19 minutes
