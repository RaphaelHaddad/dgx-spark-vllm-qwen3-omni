# vLLM Setup for NVIDIA DGX Spark - Qwen3-Omni Edition

**Fork of [eelbaz/dgx-spark-vllm-setup](https://github.com/eelbaz/dgx-spark-vllm-setup)** with additional modifications to support **Qwen3-Omni-30B** with full **multimodal capabilities** (text, audio, image).

## What's Different from the Original

This fork extends the excellent work from the original repository with the following additions:

### 🎯 Qwen3-Omni Support

- **Correct audio tokens**: Fixed `<|audio_start|><|audio_pad|><|audio_end|>` (Qwen3-Omni format) instead of `<|audio_bos|><|AUDIO|><|audio_eos|>` (Qwen2-Audio format)
- **Audio format support**: M4A/MP3/AAC via pydub+ffmpeg, WAV/FLAC/OGG via soundfile
- **Image support**: Full PIL-based image processing
- **Dependencies**: Automatic installation of `vllm[audio]` (librosa), pydub, ffmpeg

### 🚀 Enhanced Server Management

- **Multimodal server**: `server-qwen3-omni-multimodal.py` with OpenAI-compatible API
- **Text-only server**: `server-qwen3-omni.py` for faster text-only inference
- **Mode switching**: `./vllm-serve.sh [text|multimodal]` to select server mode
- **Improved process management**: `vllm-stop.sh` now kills all processes including EngineCore subprocesses
- **GPU verification**: Server launch scripts verify GPU is free before starting
- **Progress display**: Visual feedback during model loading (shards, percentage, time)

### 🧪 Testing & Validation

- **Audio validation**: `test-decode-audio.py` tests audio decoding without loading the full model
- **API tests**: `test-api-audio.py`, `test-api-chat.sh`, `test-api-text.sh`
- **Interactive testing**: `test-interactive.sh` for manual API testing

## Credits

**Original repository**: [eelbaz/dgx-spark-vllm-setup](https://github.com/eelbaz/dgx-spark-vllm-setup)

All core vLLM installation logic, Blackwell architecture fixes, and system setup come from the original work. This fork only adds Qwen3-Omni specific configurations and multimodal support.

---

## Quick Start

### Installation

The original installation script works as-is:

```bash
git clone https://github.com/YOUR_USERNAME/dgx-spark-vllm-setup.git
cd dgx-spark-vllm-setup
./install.sh
```

**Installation time:** ~20-30 minutes (mostly compilation)

### Additional Setup for Multimodal Support

After installation, install audio dependencies:

```bash
cd vllm-install
source vllm_env.sh
pip install 'vllm[audio]' pydub
sudo apt-get install -y ffmpeg  # If not already installed
```

## System Requirements

- **Hardware:** NVIDIA DGX Spark with GB10 GPU (Blackwell sm_121)
- **OS:** Ubuntu 22.04+ (tested on Linux 6.11.0 ARM64)
- **CUDA:** 13.0 or later (driver 580.95.05+)
- **Disk Space:** ~50GB free + ~60GB for Qwen3-Omni-30B model
- **RAM:** 8GB+ recommended during build

## Usage

All commands assume you're in the `vllm-install` directory.

### Activate Environment

```bash
cd vllm-install
source vllm_env.sh
```

### Start Qwen3-Omni Server

**Text-only mode** (faster, lower memory):

```bash
./vllm-serve.sh text
```

**Multimodal mode** (supports audio + image + text):

```bash
./vllm-serve.sh multimodal
```

The server will:
1. Verify GPU is free (kill zombies if needed)
2. Show loading progress (shards, percentage, time)
3. Start on `http://localhost:8000`

**Default model:** `Qwen/Qwen3-Omni-30B-A3B-Instruct`

### Check Server Status

```bash
./vllm-status.sh
```

### Stop Server

```bash
./vllm-stop.sh  # Kills all processes including EngineCore
```

### Restart Server

```bash
./vllm-restart.sh  # Stop + Start in one command
```

## API Examples

### Text Chat (OpenAI-compatible)

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen3-Omni-30B-A3B-Instruct",
    "messages": [
      {"role": "user", "content": "Hello, how are you?"}
    ],
    "max_tokens": 100,
    "temperature": 0.7
  }'
```

### Audio Analysis (Multimodal mode)

```bash
python test-api-audio.py /path/to/audio.m4a
```

Or via curl:

```bash
# Encode audio to base64
AUDIO_B64=$(base64 -w 0 audio.m4a)

curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen3-Omni-30B-A3B-Instruct",
    "messages": [
      {
        "role": "user",
        "content": [
          {
            "type": "audio_url",
            "audio_url": {"url": "data:audio/m4a;base64,'$AUDIO_B64'"}
          },
          {
            "type": "text",
            "text": "What is in this audio?"
          }
        ]
      }
    ],
    "max_tokens": 200
  }'
```

### Streaming Responses

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen3-Omni-30B-A3B-Instruct",
    "messages": [{"role": "user", "content": "Tell me a story"}],
    "stream": true
  }'
```

## What Gets Installed

Installed to `./vllm-install`:

- **Python 3.12** virtual environment at `.vllm/`
- **PyTorch 2.9.0+cu130** with full CUDA 13.0 support
- **Triton 3.5.0+git** from main branch (Blackwell support)
- **vLLM 0.11.1rc4+** with Blackwell-specific patches
- **FlashInfer 0.4.1** for optimized attention
- **Additional dependencies**: librosa, pydub, soundfile, PIL
- **Helper scripts** for server management
- **Qwen3-Omni servers**: text-only and multimodal variants

## Architecture & Fixes

This fork inherits all critical fixes from the original repository:

### From Original Repository

1. **CMakeLists.txt SM100/SM120 MOE Kernel Fix**: Added `12.0f` to SCALED_MM_ARCHS
2. **pyproject.toml License Field Format**: Converted to structured format
3. **GPT-OSS Triton MOE Kernels**: Updated deprecated Triton routing API
4. **Triton Main Branch**: Build from latest with Blackwell fixes
5. **Environment Configuration**: `TORCH_CUDA_ARCH_LIST=12.0f` workaround

### Additional Qwen3-Omni Fixes

6. **Audio Token Format**: Use Qwen3-Omni tokens (`<|audio_start|>` not `<|audio_bos|>`)
7. **Multimodal Prompt Construction**: Proper token insertion for audio/image data
8. **Audio Decoding Pipeline**: soundfile → pydub fallback → ffmpeg conversion
9. **Process Management**: Kill all EngineCore subprocesses, not just parent
10. **GPU Verification**: Ensure GPU is free before starting server

## File Structure

```
dgx-spark-vllm-setup/
├── install.sh                           # Original installation script
├── vllm-install/
│   ├── vllm_env.sh                      # Environment activation
│   ├── vllm-serve.sh                    # Start server (text|multimodal)
│   ├── vllm-stop.sh                     # Stop all processes
│   ├── vllm-restart.sh                  # Restart server
│   ├── vllm-status.sh                   # Check server status
│   ├── server-qwen3-omni.py             # Text-only server
│   ├── server-qwen3-omni-multimodal.py  # Multimodal server
│   ├── test-decode-audio.py             # Audio validation (fast)
│   ├── test-api-audio.py                # Audio API test
│   ├── test-api-chat.sh                 # Chat API test
│   ├── test-api-text.sh                 # Text completion test
│   └── test-interactive.sh              # Interactive testing
└── FUNCTIONAL_OHO_README.md             # Detailed technical guide
```

## Troubleshooting

### GPU Memory Issues

If you see "CUDA error: out of memory":

```bash
# Check for zombie processes
nvidia-smi --query-compute-apps=pid,used_memory --format=csv

# Kill all vLLM processes
./vllm-stop.sh

# Or manually
pkill -9 -f "python.*server-qwen3"
pkill -9 -f "EngineCore"
```

### Audio Not Working

1. **Verify ffmpeg is installed**:
   ```bash
   which ffmpeg ffprobe
   ```

2. **Test audio decoding**:
   ```bash
   python test-decode-audio.py /path/to/audio.m4a
   ```

3. **Check librosa installation**:
   ```bash
   python -c "import librosa; print(librosa.__version__)"
   ```

### Server Won't Start

1. **Check GPU is free**:
   ```bash
   nvidia-smi
   ```

2. **Verify environment**:
   ```bash
   source vllm_env.sh
   echo $TORCH_CUDA_ARCH_LIST  # Should be "12.0f"
   ```

3. **Check logs**:
   ```bash
   tail -100 vllm-server-multimodal.log  # or vllm-server.log
   ```

### Model Download Issues

Qwen3-Omni-30B is ~60GB. Ensure:
- Stable internet connection
- ~60GB free space in `~/.cache/huggingface`
- Hugging Face token if model is gated (usually not needed)

## Performance

**Hardware**: NVIDIA DGX Spark GB10 (120GB VRAM)

**Text-only mode**:
- Model load time: ~6-7 minutes (15 shards)
- Memory usage: ~105GB
- Inference speed: ~40-50 tokens/sec

**Multimodal mode**:
- Model load time: ~6-7 minutes
- Memory usage: ~105GB (same as text)
- Audio processing: ~2-3 seconds for 12-second M4A file
- Supports: 10 audios + 10 images per request

## Known Limitations

1. **Image support**: Implemented but not extensively tested
2. **Video support**: Not implemented (Qwen3-Omni doesn't support video natively)
3. **Multiple audios**: Works but inference time increases linearly
4. **Streaming with multimodal**: Text streams, but audio processing is blocking

## Contributing

This fork is maintained separately from the original. For:
- **vLLM/Blackwell issues**: Report to [eelbaz/dgx-spark-vllm-setup](https://github.com/eelbaz/dgx-spark-vllm-setup)
- **Qwen3-Omni/multimodal issues**: Report to this repository

## License

Same as original repository: Apache-2.0

## Acknowledgments

- **Original work**: [eelbaz/dgx-spark-vllm-setup](https://github.com/eelbaz/dgx-spark-vllm-setup) - Essential foundation for vLLM on Blackwell
- **vLLM team**: For the amazing inference engine
- **Qwen team**: For the Qwen3-Omni model
- **NVIDIA**: For Blackwell architecture and DGX Spark

---

**Status**: ✅ Functional - Tested on January 7, 2026

For detailed technical information, see [FUNCTIONAL_OHO_README.md](FUNCTIONAL_OHO_README.md).
