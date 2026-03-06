# Qwen3-Omni Benchmark

Simple benchmarking suite for Qwen3-Omni multimodal model.

## Quick Start

```bash
# Quick test (multimodal)
./run_benchmark.sh

# Full benchmark (all tests)
./run_benchmark.sh --full

# Process video in chunks
./run_benchmark.sh --chunks 10
```

## Options

| Option | Description |
|--------|-------------|
| `--quick` | Multimodal test only (default) |
| `--full` | Run all tests (image, audio, multimodal) |
| `--chunks [N]` | Process video in N-second chunks |
| `--help` | Show help |

## Examples

```bash
# Quick multimodal test
./run_benchmark.sh

# Full benchmark suite
./run_benchmark.sh --full

# 10-second chunks
./run_benchmark.sh --chunks 10

# 30-second chunks
./run_benchmark.sh --chunks 30
```

## Results

Results saved to `results/benchmark_*.json`

```json
{
  "test_name": "Multimodal (Images + Audio + Text)",
  "metrics": {
    "inference_time": 17.22,
    "tokens_generated": 333,
    "tokens_per_sec": 19.34
  }
}
```

## Performance

| Test | Time | Tokens/sec |
|------|------|------------|
| Image Only | 11.49s | 17.23 |
| Multimodal | 17.22s | 19.34 |
| Chunked (avg) | 18.91s | 16.34 |

## Cleanup

```bash
# Remove test results
rm -rf results/*.json

# Remove extracted data
rm -rf data/extracted/*
```

## Configuration

Edit `benchmark_config.sh` to change:
- API URL and model name
- Number of frames
- Max tokens
- Test prompts
