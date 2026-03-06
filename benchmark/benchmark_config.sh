#!/usr/bin/env bash
# Qwen3-Omni Benchmark Configuration
# Source this file to load configuration variables

# Script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BENCHMARK_DIR="${SCRIPT_DIR}"

# API Configuration
export VLLM_API_URL="${VLLM_API_URL:-http://localhost:8000/v1}"
export VLLM_API_KEY="${VLLM_API_KEY:-qwen3-omni-api-key}"
export MODEL_NAME="${MODEL_NAME:-qwen3-omni}"

# Paths
export VIDEO_PATH="${BENCHMARK_DIR}/data/videos/test.mp4"
export EXTRACTED_DIR="${BENCHMARK_DIR}/data/extracted"
export RESULTS_DIR="${BENCHMARK_DIR}/results"
export AUDIO_OUTPUTS_DIR="${RESULTS_DIR}/audio_outputs"

# Benchmark Settings
export GENERATE_AUDIO_OUTPUT="${GENERATE_AUDIO_OUTPUT:-false}"
export NUM_FRAMES="${NUM_FRAMES:-5}"
export MAX_TOKENS="${MAX_TOKENS:-512}"
export TEMPERATURE="${TEMPERATURE:-0.7}"
export CLIP_DURATION="${CLIP_DURATION:-0}"  # 0 = full video, >0 = chunk duration in seconds
export NUM_CHUNKS="${NUM_CHUNKS:-0}"  # 0 = auto-calculate based on clip duration

# Text prompts (from reference benchmark)
export MULTIMODAL_PROMPT="Please analyze this video content. I'm providing you with 5 frames from the video and the audio track. Describe what you observe in both the visual and audio elements. What is happening in this video?"
export IMAGE_PROMPT="Describe what you see in these 5 images from a video."
export AUDIO_PROMPT="Describe what you hear in this audio clip."
export TEXT_PROMPT="Hello! Can you tell me about your capabilities as an audio-visual-language model?"

# Performance thresholds (for validation)
export MAX_INFERENCE_TIME="${MAX_INFERENCE_TIME:-30}"  # seconds
export MIN_TOKENS_PER_SEC="${MIN_TOKENS_PER_SEC:-5}"   # tokens/sec

# Colors for output
export RED='\033[0;31m'
export GREEN='\033[0;32m'
export YELLOW='\033[1;33m'
export BLUE='\033[0;34m'
export NC='\033[0m' # No Color

# Logging
export LOG_LEVEL="${LOG_LEVEL:-INFO}"  # DEBUG, INFO, WARN, ERROR

# Helper functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $*"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $*"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $*"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $*"
}

# Check if server is running
check_server() {
    if ! curl -s --connect-timeout 5 "${VLLM_API_URL}/models" > /dev/null 2>&1; then
        log_error "vLLM server is not responding at ${VLLM_API_URL}"
        return 1
    fi
    return 0
}

# Create results directory if it doesn't exist
ensure_results_dir() {
    mkdir -p "${RESULTS_DIR}"
    mkdir -p "${AUDIO_OUTPUTS_DIR}"
}

# Get current timestamp for result files
get_timestamp() {
    date '+%Y%m%d_%H%M%S'
}

# Export functions
export -f log_info log_success log_warn log_error check_server ensure_results_dir get_timestamp
