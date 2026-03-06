#!/usr/bin/env bash
set -euo pipefail

# Load configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../benchmark_config.sh"

# Test name
TEST_NAME="Multimodal Chunks (Images + Audio + Text)"
TEST_ID="multimodal_chunks"

log_info "=========================================="
log_info "Test: ${TEST_NAME}"
log_info "=========================================="

# Check server
if ! check_server; then
    log_error "Server check failed"
    exit 1
fi

# Check if chunks are prepared
CHUNKS_DIR="${EXTRACTED_DIR}/chunks"
if [[ ! -d "${CHUNKS_DIR}" ]] || [[ -z "$(ls -A ${CHUNKS_DIR}/chunk_* 2>/dev/null)" ]]; then
    log_error "No video chunks found. Please run split_video_into_chunks.sh first."
    log_info "Usage: ./scripts/split_video_into_chunks.sh <video> <clip_duration>"
    exit 1
fi

# Count chunks
NUM_CHUNKS=$(ls -1d "${CHUNKS_DIR}"/chunk_* 2>/dev/null | wc -l)
log_info "Found ${NUM_CHUNKS} video chunks"

# Ensure results directory exists
ensure_results_dir

# Initialize results array
declare -a CHUNK_RESULTS
declare -a CHUNK_TIMES

# Process each chunk
for CHUNK_NUM in $(seq 1 ${NUM_CHUNKS}); do
    CHUNK_DIR="${CHUNKS_DIR}/chunk_${CHUNK_NUM}"

    if [[ ! -d "${CHUNK_DIR}" ]]; then
        log_warn "Chunk ${CHUNK_NUM} not found, skipping..."
        continue
    fi

    log_info "=========================================="
    log_info "Processing Chunk ${CHUNK_NUM}/${NUM_CHUNKS}"
    log_info "=========================================="

    # Start timing
    START_TIME=$(date +%s.%3N)

    # Build multimodal content array
    log_info "Building multimodal request..."
    CONTENT_JSON="["

    # Add images
    FRAME_COUNT=0
    for frame in "${CHUNK_DIR}"/frame_*.png; do
        if [[ -f "${frame}" ]]; then
            BASE64_IMG=$(base64 -w 0 "${frame}")
            if [[ "${CONTENT_JSON}" != "[" ]]; then
                CONTENT_JSON+=","
            fi
            CONTENT_JSON+="{\"type\":\"image_url\",\"image_url\":{\"url\":\"data:image/png;base64,${BASE64_IMG}\"}}"
            FRAME_COUNT=$((FRAME_COUNT + 1))
        fi
    done

    # Add audio
    if [[ -f "${CHUNK_DIR}/audio.wav" ]]; then
        BASE64_AUDIO=$(base64 -w 0 "${CHUNK_DIR}/audio.wav")
        if [[ "${CONTENT_JSON}" != "[" ]]; then
            CONTENT_JSON+=","
        fi
        CONTENT_JSON+="{\"type\":\"audio_url\",\"audio_url\":{\"url\":\"data:audio/wav;base64,${BASE64_AUDIO}\"}}"
        log_info "  - Added audio.wav"
    fi

    # Add text prompt
    if [[ "${CONTENT_JSON}" != "[" ]]; then
        CONTENT_JSON+=","
    fi
    CONTENT_JSON+="{\"type\":\"text\",\"text\":\"${MULTIMODAL_PROMPT}\"}"
    CONTENT_JSON+="]"

    log_info "  - Added ${FRAME_COUNT} frames"

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
        log_error "Request failed for chunk ${CHUNK_NUM}!"
        echo "${RESPONSE}" | jq . 2>/dev/null || echo "${RESPONSE}"
        continue
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

    # Store results
    CHUNK_RESULTS+=("Chunk ${CHUNK_NUM}: ${COMPLETION_TOKENS} tokens, ${TOKENS_PER_SEC} tokens/sec, ${INFERENCE_TIME}s inference")
    CHUNK_TIMES+=("${INFERENCE_TIME}")

    # Output chunk results
    echo ""
    log_success "Chunk ${CHUNK_NUM} Results:"
    log_info "  - Input Processing: ${INPUT_PROCESSING_TIME}s"
    log_info "  - Inference Time: ${INFERENCE_TIME}s"
    log_info "  - Total Time: ${TOTAL_TIME}s"
    log_info "  - Tokens Generated: ${COMPLETION_TOKENS}"
    log_info "  - Tokens/sec: ${TOKENS_PER_SEC}"
    log_success "  - Status: SUCCESS"
    echo ""
done

# Calculate aggregate statistics
TOTAL_INFERENCE_TIME=0
for time in "${CHUNK_TIMES[@]}"; do
    TOTAL_INFERENCE_TIME=$(echo "${TOTAL_INFERENCE_TIME} + ${time}" | bc)
done

AVG_INFERENCE_TIME=$(echo "scale=3; ${TOTAL_INFERENCE_TIME} / ${#CHUNK_TIMES[@]}" | bc)

# Output final summary
echo ""
log_success "=========================================="
log_success "All Chunks Complete!"
log_success "=========================================="
log_info "Total Chunks Processed: ${#CHUNK_TIMES[@]}"
log_info "Total Inference Time: ${TOTAL_INFERENCE_TIME}s"
log_info "Average Inference Time: ${AVG_INFERENCE_TIME}s"
log_success "=========================================="
echo ""

# Save results to JSON
RESULTS_FILE="${RESULTS_DIR}/benchmark_$(get_timestamp)_${TEST_ID}.json"
cat > "${RESULTS_FILE}" <<EOF
{
  "test_name": "${TEST_NAME}",
  "test_id": "${TEST_ID}",
  "timestamp": "$(date -Iseconds)",
  "video_path": "${VIDEO_PATH}",
  "num_chunks": ${NUM_CHUNKS},
  "num_frames_per_chunk": ${NUM_FRAMES},
  "mode": "text_output",
  "metrics": {
    "total_chunks_processed": ${#CHUNK_TIMES[@]},
    "total_inference_time": ${TOTAL_INFERENCE_TIME},
    "avg_inference_time": ${AVG_INFERENCE_TIME}
  },
  "chunks": [
EOF

# Add chunk results to JSON
FIRST=true
for i in "${!CHUNK_RESULTS[@]}"; do
    if [[ "${FIRST}" == "true" ]]; then
        FIRST=false
    else
        echo "," >> "${RESULTS_FILE}"
    fi
    echo -n "    {\"result\": \"${CHUNK_RESULTS[$i]}\"}" >> "${RESULTS_FILE}"
done

echo "" >> "${RESULTS_FILE}"
echo "  ]" >> "${RESULTS_FILE}"
echo "}" >> "${RESULTS_FILE}"

log_success "Results saved to: ${RESULTS_FILE}"
