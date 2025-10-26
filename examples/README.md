# vLLM Examples for DGX Spark

This directory contains example scripts demonstrating various ways to use vLLM on DGX Spark systems.

## Prerequisites

Ensure vLLM is installed and the environment is activated:

```bash
# Assuming vllm-install is in your home directory
source ~/vllm-install/vllm_env.sh
```

## Examples

### 1. Basic Inference (`basic_inference.py`)

Simple text generation using the vLLM Python API.

**Usage:**
```bash
python basic_inference.py
```

**What it demonstrates:**
- Loading a model with vLLM
- Configuring sampling parameters
- Generating multiple completions
- Batch processing

### 2. API Client (`api_client.py`)

Using vLLM's OpenAI-compatible REST API.

**Prerequisites:**
Start the vLLM server first:
```bash
cd ~/vllm-install
./vllm-serve.sh
```

**Usage:**
```bash
python api_client.py
```

**What it demonstrates:**
- Listing available models
- Simple text completion
- Chat completion
- Streaming responses
- HTTP API interaction

### 3. Batch Processing (`batch_processing.py`)

Efficient processing of large batches of prompts.

**Usage:**
```bash
python batch_processing.py
```

**What it demonstrates:**
- High-throughput batch inference
- Dynamic batching
- Memory-efficient processing
- Performance monitoring

## Customization

### Change Model

Edit the model name in any example:

```python
llm = LLM(
    model="meta-llama/Llama-3.1-8B-Instruct",  # Change this
    trust_remote_code=True,
    gpu_memory_utilization=0.9
)
```

### Adjust Sampling Parameters

Modify `SamplingParams` for different generation behavior:

```python
sampling_params = SamplingParams(
    temperature=0.7,      # Lower = more deterministic (0.0-1.0)
    top_p=0.95,          # Nucleus sampling threshold
    max_tokens=100,      # Maximum tokens to generate
    top_k=50,            # Top-k sampling
    repetition_penalty=1.1  # Penalize repetition
)
```

### GPU Memory Management

Adjust memory utilization:

```python
llm = LLM(
    model="...",
    gpu_memory_utilization=0.9,  # Use 90% of GPU memory (0.0-1.0)
    max_model_len=2048           # Maximum sequence length
)
```

## API Server Examples

### cURL Examples

**List models:**
```bash
curl http://localhost:8000/v1/models
```

**Simple completion:**
```bash
curl http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen2.5-0.5B-Instruct",
    "prompt": "The meaning of life is",
    "max_tokens": 50,
    "temperature": 0.7
  }'
```

**Chat completion:**
```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen2.5-0.5B-Instruct",
    "messages": [
      {"role": "system", "content": "You are a helpful assistant."},
      {"role": "user", "content": "What is DGX Spark?"}
    ],
    "max_tokens": 100,
    "temperature": 0.7
  }'
```

**Streaming completion:**
```bash
curl http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen2.5-0.5B-Instruct",
    "prompt": "Write a story about",
    "max_tokens": 100,
    "stream": true
  }'
```

## Tested Models

These models work well on DGX Spark GB10:

- `Qwen/Qwen2.5-0.5B-Instruct` (small, fast)
- `Qwen/Qwen2.5-7B-Instruct` (balanced)
- `meta-llama/Llama-3.1-8B-Instruct` (high quality)
- `meta-llama/Llama-3.1-70B-Instruct` (requires tensor parallelism)

## Performance Tips

1. **Use GPU memory efficiently:**
   - Set `gpu_memory_utilization=0.95` for maximum throughput
   - Lower for models close to GPU memory limit

2. **Batch processing:**
   - Process multiple prompts together
   - vLLM automatically optimizes batch sizes

3. **Quantization:**
   - For larger models, use quantization:
   ```python
   llm = LLM(model="...", quantization="awq")
   ```

4. **Tensor parallelism:**
   - For models > 20GB, use multiple GPUs:
   ```python
   llm = LLM(model="...", tensor_parallel_size=2)
   ```

## Troubleshooting

### Out of Memory

Reduce `max_model_len` or `gpu_memory_utilization`:

```python
llm = LLM(
    model="...",
    gpu_memory_utilization=0.8,
    max_model_len=2048
)
```

### Slow Generation

Check if model is loaded correctly:

```python
python -c "import vllm; print(vllm.__version__)"
nvidia-smi  # Check GPU utilization
```

### Connection Refused (API)

Ensure server is running:

```bash
cd ~/vllm-install
./vllm-status.sh
```

## More Resources

- [vLLM Documentation](https://docs.vllm.ai/)
- [OpenAI API Compatibility](https://docs.vllm.ai/en/latest/serving/openai_compatible_server.html)
- [Main README](../README.md)
- [Cluster Setup](../CLUSTER.md)
