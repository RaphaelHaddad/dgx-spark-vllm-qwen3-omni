#!/usr/bin/env bash
set -euo pipefail

# Load configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../benchmark_config.sh"

log_info "=========================================="
log_info "Video Data Preparation"
log_info "=========================================="
log_info "Video: ${VIDEO_PATH}"
log_info "Output: ${EXTRACTED_DIR}"
log_info "=========================================="

# Check if video exists
if [[ ! -f "${VIDEO_PATH}" ]]; then
    log_error "Video not found at ${VIDEO_PATH}"
    exit 1
fi

# Create extraction directory
mkdir -p "${EXTRACTED_DIR}"

# Clean up any previous extractions
log_info "Cleaning up previous extractions..."
rm -f "${EXTRACTED_DIR}"/frame_*.png "${EXTRACTED_DIR}"/audio.wav

# Extract frames
log_info "Extracting ${NUM_FRAMES} frames from video..."
ffmpeg -i "${VIDEO_PATH}" -frames:v "${NUM_FRAMES}" "${EXTRACTED_DIR}/frame_%d.png" -y 2>/dev/null

if [[ $? -eq 0 ]]; then
    log_success "Extracted ${NUM_FRAMES} frames"
    ls -lh "${EXTRACTED_DIR}"/frame_*.png | tail -n +2
else
    log_error "Failed to extract frames"
    exit 1
fi

# Extract audio
log_info "Extracting audio from video (16kHz, mono)..."
ffmpeg -i "${VIDEO_PATH}" -vn -acodec pcm_s16le -ar 16000 -ac 1 "${EXTRACTED_DIR}/audio.wav" -y 2>/dev/null

if [[ $? -eq 0 ]]; then
    log_success "Extracted audio track"
    file "${EXTRACTED_DIR}/audio.wav"
    ls -lh "${EXTRACTED_DIR}/audio.wav"
else
    log_error "Failed to extract audio"
    exit 1
fi

# Verify extracted files
log_info "Verifying extracted files..."
FRAME_COUNT=$(ls -1 "${EXTRACTED_DIR}"/frame_*.png 2>/dev/null | wc -l)
if [[ ${FRAME_COUNT} -ne ${NUM_FRAMES} ]]; then
    log_error "Expected ${NUM_FRAMES} frames, found ${FRAME_COUNT}"
    exit 1
fi

if [[ ! -f "${EXTRACTED_DIR}/audio.wav" ]]; then
    log_error "Audio file not found"
    exit 1
fi

log_success "=========================================="
log_success "Data preparation completed successfully!"
log_success "Frames: ${FRAME_COUNT}"
log_success "Audio: $(basename "${EXTRACTED_DIR}/audio.wav")"
log_success "=========================================="
