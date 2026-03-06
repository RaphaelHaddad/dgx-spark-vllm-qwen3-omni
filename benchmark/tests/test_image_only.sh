#!/usr/bin/env bash
set -euo pipefail

# Load configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../benchmark_config.sh"

# Test name
TEST_NAME="Image Only (No Audio)"
TEST_ID="image_only"

log_info "=========================================="
log_info "Test: ${TEST_NAME}"
log_info "=========================================="

# Check server
if ! check_server; then
    log_error "Server check failed"
    exit 1
fi

# Ensure data is prepared
if [[ ! -f "${EXTRACTED_DIR}/frame_1.png" ]]; then
    log_warn "Test data not found. Running preparation..."
    "${BENCHMARK_DIR}/scripts/prepare_video_data.sh"
fi

# Ensure results directory exists
ensure_results_dir

# Start timing
START_TIME=$(date +%s.%3N)

# Build image-only content array
log_info "Building image-only request..."
log_info "  - ${NUM_FRAMES} frames from video"

CONTENT_JSON="["

# Add images
for i in $(seq 1 ${NUM_FRAMES}); do
    if [[ -f "${EXTRACTED_DIR}/frame_${i}.png" ]]; then
        BASE64_IMG=$(base64 -w 0 "${EXTRACTED_DIR}/frame_${i}.png")
        if [[ "${CONTENT_JSON}" != "[" ]]; then
            CONTENT_JSON+=","
        fi
        CONTENT_JSON+="{\"type\":\"image_url\",\"image_url\":{\"url\":\"data:image/png;base64,${BASE64_IMG}\"}}"
        log_info "  - Added frame_${i}.png"
    fi
done

# Add text prompt
if [[ "${CONTENT_JSON}" != "[" ]]; then
    CONTENT_JSON+=","
fi
CONTENT_JSON+="{\"type\":\"text\",\"text\":\"${IMAGE_PROMPT}\"}"
CONTENT_JSON+="]"

# Create temporary request file
TEMP_REQUEST=$(mktemp)
cat > "${TEMP_REQUEST}" <<EOF
{
  "model": "${MODEL_NAME}",
  "messages": [
    {
      "role": "user",
      "content": ${CONTENT_JSON}
    }
  ],
  "max_tokens": ${MAX_TOKENS},
  "temperature": ${TEMPERATURE}
}
EOF

# Make API request
log_info "Sending request to API..."
INFERENCE_START=$(date +%s.%3N)

RESPONSE=$(curl -s --max-time 120 -X POST "${VLLM_API_URL}/chat/completions" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer ${VLLM_API_KEY}" \
  --data "@${TEMP_REQUEST}")

INFERENCE_END=$(date +%s.%3N)
END_TIME=$(date +%s.%3N)

# Clean up temp file
rm -f "${TEMP_REQUEST}"

# Calculate times
INPUT_PROCESSING_TIME=$(echo "${INFERENCE_START} - ${START_TIME}" | bc)
INFERENCE_TIME=$(echo "${INFERENCE_END} - ${INFERENCE_START}" | bc)
TOTAL_TIME=$(echo "${END_TIME} - ${START_TIME}" | bc)

# Parse response
RESPONSE_CONTENT=$(echo "${RESPONSE}" | jq -r '.choices[0].message.content // .error.message // .error // "ERROR: No response"')

# Check for errors
if [[ "${RESPONSE_CONTENT}" == ERROR* ]] || echo "${RESPONSE}" | jq -e '.error' > /dev/null 2>&1; then
    log_error "Request failed!"
    echo "${RESPONSE}" | jq . 2>/dev/null || echo "${RESPONSE}"
    exit 1
fi

# Get token usage if available
TOKENS_USED=$(echo "${RESPONSE}" | jq -r '.usage.total_tokens // "N/A"')
COMPLETION_TOKENS=$(echo "${RESPONSE}" | jq -r '.usage.completion_tokens // "N/A"')

# Calculate tokens per second
if [[ "${COMPLETION_TOKENS}" != "N/A" ]] && [[ "${COMPLETION_TOKENS}" != "null" ]]; then
    TOKENS_PER_SEC=$(echo "scale=2; ${COMPLETION_TOKENS} / ${INFERENCE_TIME}" | bc)
else
    TOKENS_PER_SEC="N/A"
fi

# Output results
echo ""
log_success "=========================================="
log_success "Test: ${TEST_NAME}"
log_success "=========================================="
log_info "Input Processing Time: ${INPUT_PROCESSING_TIME}s"
log_info "Inference Time: ${INFERENCE_TIME}s"
log_info "Total Time: ${TOTAL_TIME}s"
log_info "Tokens Generated: ${COMPLETION_TOKENS}"
log_info "Tokens/sec: ${TOKENS_PER_SEC}"
log_success "Status: SUCCESS"
log_success "=========================================="
echo ""
log_info "Response:"
echo "${RESPONSE_CONTENT}"
echo ""

# Save results to JSON
RESULTS_FILE="${RESULTS_DIR}/benchmark_$(get_timestamp)_${TEST_ID}.json"
cat > "${RESULTS_FILE}" <<EOF
{
  "test_name": "${TEST_NAME}",
  "test_id": "${TEST_ID}",
  "timestamp": "$(date -Iseconds)",
  "video_path": "${VIDEO_PATH}",
  "num_frames": ${NUM_FRAMES},
  "mode": "text_output",
  "modality": "image_only",
  "metrics": {
    "input_processing_time": ${INPUT_PROCESSING_TIME},
    "inference_time": ${INFERENCE_TIME},
    "total_time": ${TOTAL_TIME},
    "tokens_generated": ${COMPLETION_TOKENS},
    "tokens_per_sec": ${TOKENS_PER_SEC},
    "total_tokens": ${TOKENS_USED}
  },
  "response": "${RESPONSE_CONTENT}"
}
EOF

log_success "Results saved to: ${RESULTS_FILE}"
