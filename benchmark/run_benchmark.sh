#!/usr/bin/env bash
set -euo pipefail

# Load configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/benchmark_config.sh"

# Default values
RUN_FULL="${RUN_FULL:-false}"
GENERATE_AUDIO="${GENERATE_AUDIO:-false}"
USE_CHUNKS="${USE_CHUNKS:-false}"
CLIP_DURATION="${CLIP_DURATION:-0}"

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --full)
            RUN_FULL=true
            shift
            ;;
        --quick)
            RUN_FULL=false
            shift
            ;;
        --audio-output)
            GENERATE_AUDIO=true
            shift
            ;;
        --no-audio-output)
            GENERATE_AUDIO=false
            shift
            ;;
        --chunks)
            USE_CHUNKS=true
            CLIP_DURATION="${2:-5}"
            shift 2
            ;;
        --clip-duration)
            CLIP_DURATION="${2:-5}"
            USE_CHUNKS=true
            shift 2
            ;;
        --help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --full              Run full benchmark suite (all tests)"
            echo "  --quick             Run quick benchmark (multimodal only)"
            echo "  --audio-output      Enable audio output generation"
            echo "  --no-audio-output   Disable audio output (default)"
            echo "  --chunks [DURATION] Run benchmark on video chunks (default: 5s per chunk)"
            echo "  --clip-duration N   Same as --chunks N"
            echo "  --help              Show this help message"
            echo ""
            echo "Examples:"
            echo "  $0 --quick                                    # Quick test on full video"
            echo "  $0 --chunks 5                                 # 5-second chunks"
            echo "  $0 --full --chunks 10                         # Full suite with 10s chunks"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Export for child scripts
export GENERATE_AUDIO_OUTPUT="${GENERATE_AUDIO}"

log_info "=========================================="
log_info "Qwen3-Omni Benchmark"
log_info "=========================================="
log_info "Mode: $([[ "${GENERATE_AUDIO}" == "true" ]] && echo "With Audio Output" || echo "Text Output Only")"
log_info "Scope: $([[ "${RUN_FULL}" == "true" ]] && echo "Full Suite" || echo "Quick Test")"
if [[ "${USE_CHUNKS}" == "true" ]]; then
    log_info "Chunks: Yes (${CLIP_DURATION}s per chunk)"
else
    log_info "Chunks: No (full video)"
fi
log_info "API: ${VLLM_API_URL}"
log_info "Model: ${MODEL_NAME}"
log_info "=========================================="

# Check server
log_info "Checking server status..."
if ! check_server; then
    log_error "Server is not running. Please start the server first."
    exit 1
fi
log_success "Server is running!"

# Prepare data
if [[ "${USE_CHUNKS}" == "true" ]]; then
    log_info "Splitting video into ${CLIP_DURATION}s chunks..."
    "${SCRIPT_DIR}/scripts/split_video_into_chunks.sh" "${VIDEO_PATH}" "${CLIP_DURATION}"
    log_success "Video chunks ready!"
else
    log_info "Preparing test data..."
    "${SCRIPT_DIR}/scripts/prepare_video_data.sh" > /dev/null
    log_success "Test data ready!"
fi

# Initialize results summary
RESULTS_SUMMARY="${RESULTS_DIR}/summary_$(get_timestamp).txt"
echo "Benchmark Summary - $(date)" > "${RESULTS_SUMMARY}"
echo "Mode: $([[ "${GENERATE_AUDIO}" == "true" ]] && echo "Audio Output" || echo "Text Output")" >> "${RESULTS_SUMMARY}"
echo "==========================================" >> "${RESULTS_SUMMARY}"

# Run tests
if [[ "${USE_CHUNKS}" == "true" ]]; then
    log_info "Running benchmark on video chunks..."
    "${SCRIPT_DIR}/tests/test_multimodal_chunks.sh" 2>&1 | tee -a "${RESULTS_SUMMARY}"
elif [[ "${RUN_FULL}" == "true" ]]; then
    log_info "Running full benchmark suite..."
    echo "" >> "${RESULTS_SUMMARY}"

    # Baseline tests
    log_info "Running baseline tests..."
    for test in test_text_only test_audio_only test_image_only; do
        if [[ -f "${SCRIPT_DIR}/tests/${test}.sh" ]]; then
            log_info "  - Running ${test}..."
            "${SCRIPT_DIR}/tests/${test}.sh" 2>&1 | tee -a "${RESULTS_SUMMARY}"
        fi
    done

    # Primary multimodal test
    log_info "Running primary multimodal test..."
    "${SCRIPT_DIR}/tests/test_multimodal.sh" 2>&1 | tee -a "${RESULTS_SUMMARY}"
else
    log_info "Running quick benchmark (multimodal only)..."
    "${SCRIPT_DIR}/tests/test_multimodal.sh" 2>&1 | tee -a "${RESULTS_SUMMARY}"
fi

log_success "=========================================="
log_success "Benchmark Complete!"
log_success "=========================================="
log_info "Results saved to: ${RESULTS_DIR}"
log_info "Summary: ${RESULTS_SUMMARY}"

# Show summary
log_info ""
log_info "Performance Summary:"
cat "${RESULTS_SUMMARY}"
