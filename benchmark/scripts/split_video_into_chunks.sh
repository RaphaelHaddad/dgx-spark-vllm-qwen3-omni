#!/usr/bin/env bash
set -euo pipefail

# Load configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../benchmark_config.sh"

# Parameters
VIDEO_PATH="${1:-${VIDEO_PATH}}"
CLIP_DURATION="${2:-${CLIP_DURATION}}"  # Duration of each chunk in seconds
OUTPUT_DIR="${3:-${EXTRACTED_DIR}/chunks}"

log_info "=========================================="
log_info "Splitting Video into Chunks"
log_info "=========================================="
log_info "Video: ${VIDEO_PATH}"
log_info "Clip Duration: ${CLIP_DURATION}s"
log_info "Output: ${OUTPUT_DIR}"
log_info "=========================================="

# Check if video exists
if [[ ! -f "${VIDEO_PATH}" ]]; then
    log_error "Video not found at ${VIDEO_PATH}"
    exit 1
fi

# Get video duration
VIDEO_DURATION=$(ffprobe -v error -show_entries format=duration -of default=noprint_wrappers=1:nokey=1 "${VIDEO_PATH}")
VIDEO_DURATION_INT=${VIDEO_DURATION%.*}

log_info "Video duration: ${VIDEO_DURATION}s"

# Calculate number of chunks
if [[ "${CLIP_DURATION}" -gt 0 ]]; then
    NUM_CHUNKS=$(( (VIDEO_DURATION_INT + CLIP_DURATION - 1) / CLIP_DURATION ))
else
    NUM_CHUNKS=1
    CLIP_DURATION=${VIDEO_DURATION_INT}
fi

log_info "Number of chunks: ${NUM_CHUNKS}"

# Create output directory
mkdir -p "${OUTPUT_DIR}"

# Clean up previous chunks
rm -f "${OUTPUT_DIR}"/chunk_*.mp4 "${OUTPUT_DIR}"/chunk_*.wav "${OUTPUT_DIR}"/chunk_*/frame_*.png

# Split video into chunks
log_info "Splitting video into ${NUM_CHUNKS} chunks..."

for i in $(seq 0 $((NUM_CHUNKS - 1))); do
    START_TIME=$((i * CLIP_DURATION))
    CHUNK_NUM=$((i + 1))

    log_info "  - Creating chunk ${CHUNK_NUM} (start: ${START_TIME}s)..."

    # Extract video segment
    ffmpeg -ss "${START_TIME}" -i "${VIDEO_PATH}" \
        -t "${CLIP_DURATION}" \
        -c:v libx264 -c:a aac -y \
        "${OUTPUT_DIR}/chunk_${CHUNK_NUM}.mp4" \
        -loglevel error -stats

    # Extract frames from this chunk
    CHUNK_DIR="${OUTPUT_DIR}/chunk_${CHUNK_NUM}"
    mkdir -p "${CHUNK_DIR}"

    ffmpeg -i "${OUTPUT_DIR}/chunk_${CHUNK_NUM}.mp4" \
        -frames:v "${NUM_FRAMES}" \
        "${CHUNK_DIR}/frame_%d.png" \
        -y -loglevel error

    # Extract audio from this chunk
    ffmpeg -i "${OUTPUT_DIR}/chunk_${CHUNK_NUM}.mp4" \
        -vn -acodec pcm_s16le -ar 16000 -ac 1 \
        "${CHUNK_DIR}/audio.wav" \
        -y -loglevel error

    # Count extracted files
    FRAME_COUNT=$(ls -1 "${CHUNK_DIR}"/frame_*.png 2>/dev/null | wc -l)
    AUDIO_EXISTS=$([[ -f "${CHUNK_DIR}/audio.wav" ]] && echo "yes" || echo "no")

    log_info "    - Frames: ${FRAME_COUNT}, Audio: ${AUDIO_EXISTS}"
done

# Summary
log_success "=========================================="
log_success "Video split into ${NUM_CHUNKS} chunks!"
log_success "=========================================="
log_info "Chunk duration: ${CLIP_DURATION}s"
log_info "Output directory: ${OUTPUT_DIR}"
log_info ""
log_info "Chunks created:"
for i in $(seq 1 ${NUM_CHUNKS}); do
    CHUNK_DIR="${OUTPUT_DIR}/chunk_${i}"
    if [[ -d "${CHUNK_DIR}" ]]; then
        FRAME_COUNT=$(ls -1 "${CHUNK_DIR}"/frame_*.png 2>/dev/null | wc -l)
        AUDIO_SIZE=$([[ -f "${CHUNK_DIR}/audio.wav" ]] && du -h "${CHUNK_DIR}/audio.wav" | cut -f1 || echo "N/A")
        echo "  - Chunk ${i}: ${FRAME_COUNT} frames, audio: ${AUDIO_SIZE}"
    fi
done
log_success "=========================================="

# Export number of chunks for other scripts
export NUM_CHUNKS="${NUM_CHUNKS}"
