#!/bin/bash

if [ "$#" -lt 4 ]; then
    echo "Usage: $0 MODEL_PATH TOTAL_CHUNKS IDX_START GPU_LIST [CONFIG_KEY CONFIG_VALUE ...]"
    exit 1
fi

MODEL_PATH=$1
TOTAL_CHUNKS=$2
IDX_START=$3
GPU_LIST=$4  # GPU list as a string (e.g., "0,2,4,6")
shift 4
EXTRA_OPTS=("$@")

IFS=',' read -ra GPULIST <<< "$GPU_LIST"

CHUNKS=${#GPULIST[@]}

for IDX in $(seq 0 $((CHUNKS-1))); do
    CHUNK_IDX=$((IDX + IDX_START))
    if [ "$CHUNK_IDX" -ge "$TOTAL_CHUNKS" ]; then
        echo "Invalid chunk index: $CHUNK_IDX >= TOTAL_CHUNKS=$TOTAL_CHUNKS"
        exit 1
    fi
    echo "Total Chunks: $TOTAL_CHUNKS, Local Chunks: $CHUNKS, Chunk Index: $CHUNK_IDX, GPU: ${GPULIST[$IDX]}"

    PYTHONFAULTHANDLER=1 CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python run.py \
        --exp-config vlnce_baselines/config/r2r_baselines/navila.yaml \
        --run-type eval \
        --num-chunks $TOTAL_CHUNKS \
        --chunk-idx $CHUNK_IDX \
        EVAL_CKPT_PATH_DIR "$MODEL_PATH" \
        "${EXTRA_OPTS[@]}" &
done

wait
