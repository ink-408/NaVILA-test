#!/bin/bash

if [ "$#" -lt 3 ]; then
    echo "Usage: $0 MODEL_PATH TOTAL_CHUNKS GPU_ID [CONFIG_KEY CONFIG_VALUE ...]"
    echo "Example: MAX_RETRIES=5 $0 ../checkpoints/navila-llama3-8b-8f 16 1 VIDEO_OPTION '[]'"
    exit 1
fi

MODEL_PATH=$1
TOTAL_CHUNKS=$2
GPU_ID=$3
shift 3
EXTRA_OPTS=("$@")

MAX_RETRIES=${MAX_RETRIES:-3}
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
LOG_DIR="$SCRIPT_DIR/../../logs"
mkdir -p "$LOG_DIR"

for CHUNK_IDX in $(seq 0 $((TOTAL_CHUNKS-1))); do
    ATTEMPT=1
    EXIT_CODE=1

    while [ "$ATTEMPT" -le "$MAX_RETRIES" ]; do
        TS=$(date +%Y%m%d_%H%M%S)
        LOG_FILE="$LOG_DIR/r2r_chunk-${TOTAL_CHUNKS}-${CHUNK_IDX}_attempt-${ATTEMPT}_${TS}.log"

        echo "Running chunk ${CHUNK_IDX}/${TOTAL_CHUNKS} on GPU ${GPU_ID}, attempt ${ATTEMPT}/${MAX_RETRIES}"
        bash "$SCRIPT_DIR/r2r.sh" "$MODEL_PATH" "$TOTAL_CHUNKS" "$CHUNK_IDX" "$GPU_ID" "${EXTRA_OPTS[@]}" \
            2>&1 | tee "$LOG_FILE"
        EXIT_CODE=${PIPESTATUS[0]}

        if [ "$EXIT_CODE" -eq 0 ]; then
            echo "Chunk ${CHUNK_IDX}/${TOTAL_CHUNKS} finished"
            break
        fi

        echo "Chunk ${CHUNK_IDX}/${TOTAL_CHUNKS} failed with exit code ${EXIT_CODE}; retrying after 5 seconds"
        ATTEMPT=$((ATTEMPT + 1))
        sleep 5
    done

    if [ "$EXIT_CODE" -ne 0 ]; then
        echo "Chunk ${CHUNK_IDX}/${TOTAL_CHUNKS} failed after ${MAX_RETRIES} attempts"
        exit "$EXIT_CODE"
    fi
done
