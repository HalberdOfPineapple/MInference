#!/usr/bin/bash

# Run train_qwen2_3B_sparse_ratio.sh for TARGET_ITER_IDX from 0001 to 0039

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
TRAIN_SCRIPT="${SCRIPT_DIR}/train_qwen2_3B_sparse_ratio.sh"
NUM_NODES=4

for i in $(seq 0 39); do
    TARGET_ITER_IDX=$(printf "%04d" $i)
    echo "========================================"
    echo "Running with TARGET_ITER_IDX=${TARGET_ITER_IDX} (${i}/39)"
    echo "========================================"
    /blob/utils/dist_exec.sh $NUM_NODES "$TRAIN_SCRIPT" "$TARGET_ITER_IDX"
    echo "Finished TARGET_ITER_IDX=${TARGET_ITER_IDX}"
    echo ""
done

echo "All 39 runs completed."
