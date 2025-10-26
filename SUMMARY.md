# Repository Summary

## Overview

This repository provides a **production-ready, one-command installation** of vLLM for NVIDIA DGX Spark systems with Blackwell GB10 GPUs (sm_121 architecture).

## What's Included

### Core Files

1. **install.sh** (500+ lines)
   - Fully automated installation script
   - Pre-flight system checks
   - 8-step installation pipeline
   - Post-installation testing
   - Command-line argument support

2. **README.md** (300+ lines)
   - Quick start guide
   - System requirements
   - Usage examples
   - Critical fixes documentation
   - Troubleshooting guide

3. **CLUSTER.md** (400+ lines)
   - Multi-node setup instructions
   - Ray cluster configuration
   - Tensor/pipeline parallelism
   - Performance tuning
   - Load balancing examples

4. **requirements.txt**
   - Complete dependency list
   - PyTorch 2.9.0+cu130
   - All required packages

### Helper Scripts (scripts/)

- **vllm-serve.sh** - Start vLLM server with configurable model/port
- **vllm-stop.sh** - Gracefully stop server
- **vllm-status.sh** - Check server status and logs

### Examples (examples/)

- **basic_inference.py** - Simple Python API usage
- **api_client.py** - OpenAI-compatible REST API client
- **README.md** - Usage instructions and API examples

### Configuration

- **.gitignore** - Excludes build artifacts, venvs, logs
- **LICENSE** - MIT license

## Technical Specifications

### Target Platform
- **Hardware:** NVIDIA DGX Spark with GB10 GPU
- **Architecture:** Blackwell sm_121 (compute capability 12.1)
- **OS:** Ubuntu 22.04+ ARM64
- **CUDA:** 13.0+ (driver 580.95.05+)

### Software Stack
- **Python:** 3.12.3
- **PyTorch:** 2.9.0+cu130
- **Triton:** 3.5.0+git (from main branch)
- **vLLM:** 0.11.1rc4+
- **Package Manager:** uv (fast Python package installer)

### Critical Fixes Applied

1. **CMakeLists.txt (line 671)**
   - Added `12.0f` to SCALED_MM_ARCHS for SM100 MOE kernels
   - Enables Blackwell GPU compilation

2. **pyproject.toml**
   - Changed `license = "Apache-2.0"` to `license = {text = "Apache-2.0"}`
   - Removed deprecated `license-files` field
   - Compatible with setuptools 77.0+

3. **Triton Build**
   - Must use main branch (not release 3.5.0)
   - Non-editable install to avoid setuptools bug
   - Custom PTXAS path for CUDA integration

### Environment Variables

```bash
TORCH_CUDA_ARCH_LIST=12.1a               # Blackwell architecture
VLLM_USE_FLASHINFER_MXFP4_MOE=1         # Enable FlashInfer optimization
TRITON_PTXAS_PATH=/usr/local/cuda/bin/ptxas  # CUDA PTX assembler
```

## Installation Overview

The `install.sh` script performs these steps:

1. **Pre-flight Checks**
   - Verify ARM64 architecture
   - Check NVIDIA GPU (GB10)
   - Validate CUDA 13.0+
   - Ensure 50GB+ disk space

2. **Install uv Package Manager**
   - Fast Python package installer
   - Required for efficient dependency resolution

3. **Create Virtual Environment**
   - Python 3.12 virtual environment
   - Isolated from system packages

4. **Install PyTorch**
   - PyTorch 2.9.0 with CUDA 13.0 bindings
   - Verify CUDA availability

5. **Build Triton**
   - Clone from GitHub main branch
   - Build with Blackwell support
   - Non-editable install

6. **Install Dependencies**
   - xgrammar, setuptools-scm
   - apache-tvm-ffi (prerelease)
   - Build tools

7. **Clone and Fix vLLM**
   - Clone v0.11.1rc3
   - Apply CMakeLists.txt fix
   - Apply pyproject.toml fix
   - Configure use_existing_torch

8. **Build vLLM**
   - 15-20 minute compilation
   - All CUDA kernels for Blackwell
   - Editable install for development

9. **Create Helper Scripts**
   - Environment activation script
   - Server management scripts
   - Logging configuration

10. **Post-Installation Tests**
    - Import vLLM
    - Check CUDA availability
    - Verify GPU detection

## Quick Start

```bash
# One-command installation
curl -fsSL https://raw.githubusercontent.com/eelbaz/dgx-spark-vllm-setup/main/install.sh | bash

# Or clone and run
git clone https://github.com/eelbaz/dgx-spark-vllm-setup.git
cd dgx-spark-vllm-setup
./install.sh

# Activate environment
source ~/development/dgx/vllm_env.sh

# Start server
cd ~/development/dgx
./vllm-serve.sh

# Test API
curl http://localhost:8000/v1/models
```

## Repository Structure

```
dgx-spark-vllm-setup/
├── README.md              # Main documentation
├── CLUSTER.md             # Multi-node setup guide
├── SUMMARY.md             # This file
├── LICENSE                # MIT license
├── .gitignore             # Git ignore rules
├── install.sh             # Main installation script
├── requirements.txt       # Python dependencies
├── scripts/
│   ├── vllm-serve.sh      # Start vLLM server
│   ├── vllm-stop.sh       # Stop server
│   └── vllm-status.sh     # Check status
└── examples/
    ├── README.md          # Examples documentation
    ├── basic_inference.py # Python API example
    └── api_client.py      # REST API example
```

## Known Issues & Workarounds

### Triton Editable Build Fails
**Error:** `TypeError: can only concatenate str (not 'NoneType') to str`  
**Workaround:** Use non-editable install (`uv pip install --no-build-isolation .`)

### PyTorch CUDA Capability Warning
**Warning:** GPU capability 12.1 vs PyTorch max 12.0  
**Status:** Harmless - PyTorch 2.9.0+cu130 works correctly with GB10

### apache-tvm-ffi Prerelease
**Error:** `No solution found when resolving dependencies`  
**Fix:** Use `--prerelease=allow` flag with uv pip install

## Testing Status

- ✅ Single-node installation on spark-alpha.local
- ✅ Single-node installation on spark-omega.local  
- ✅ vLLM server startup and API functionality
- ✅ Model inference (Qwen/Qwen2.5-0.5B-Instruct)
- 🔄 Multi-node cluster mode (documented, not yet tested)

## Future Enhancements

- [ ] Add cluster mode testing results
- [ ] Include performance benchmarks
- [ ] Add Dockerfile for containerized deployment
- [ ] Create Ansible playbook for multi-node automation
- [ ] Add monitoring and logging setup (Prometheus/Grafana)
- [ ] Include model quantization examples (AWQ, GPTQ)

## Contributing

Contributions welcome! Please open issues or pull requests on GitHub.

## Community & Support

- **GitHub Issues:** Report bugs and feature requests
- **NVIDIA Forum:** [DGX Spark vLLM Discussion](https://forums.developer.nvidia.com/t/run-vllm-in-spark/348862)
- **vLLM Docs:** [Official Documentation](https://docs.vllm.ai/)

## License

MIT License - See LICENSE file for details.

## Acknowledgments

Developed and tested on NVIDIA DGX Spark systems. Special thanks to:
- vLLM project team
- Triton compiler team
- NVIDIA DGX Spark community
- Claude Code (AI assistant) for documentation automation

---

**Version:** 1.0.0  
**Last Updated:** 2025-10-26  
**Tested On:** DGX Spark with GB10, CUDA 13.0, Ubuntu 22.04 ARM64
