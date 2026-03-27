#!/usr/bin/bash

# Copyright (c) 2026 Microsoft
# Licensed under The MIT License [see LICENSE for details]

# ============================================================
# Parallel multi-GPU batch runner for sparse index collection.
#
# Distributes checkpoint iterations across available GPUs so that
# each GPU processes a disjoint subset concurrently.
#
# Usage:
#   bash run_all_infer_sparse_indices.sh                   # iters 1..39, auto-detect GPUs
#   bash run_all_infer_sparse_indices.sh 0 39              # iters 0..39
#   bash run_all_infer_sparse_indices.sh 0 39 1 4          # iters 0..39 step 1, 4 GPUs
#   NUM_GPUS=2 bash run_all_infer_sparse_indices.sh 0 19   # force 2 GPUs
#
# Each GPU runs its share of checkpoints sequentially; all GPUs run
# in parallel.  Pass NUM_GPUS to override auto-detection.
# ============================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
INFER_SCRIPT="${SCRIPT_DIR}/infer_sparse_indices.sh"

START_ITER="${1:-3}"
END_ITER="${2:-39}"
STEP="${3:-1}"
NUM_GPUS="${NUM_GPUS:-${4:-$(nvidia-smi -L 2>/dev/null | wc -l)}}"

# Fallback if nvidia-smi is unavailable
if [ "$NUM_GPUS" -le 0 ] 2>/dev/null; then
    NUM_GPUS=1
fi

# Build the full list of iteration indices
ITERS=()
for i in $(seq "$START_ITER" "$STEP" "$END_ITER"); do
    ITERS+=("$i")
done
TOTAL=${#ITERS[@]}

echo "============================================"
echo " Parallel sparse index collection"
echo "  Iterations : ${START_ITER}..${END_ITER} (step ${STEP}), total ${TOTAL}"
echo "  GPUs       : ${NUM_GPUS}"
echo "============================================"

# ---------------------------------------------------------------
# Worker function: runs a slice of iterations on a single GPU.
# ---------------------------------------------------------------
run_worker() {
    local gpu_id=$1
    shift
    local iters=("$@")

    echo "[GPU ${gpu_id}] Starting ${#iters[@]} iteration(s): ${iters[*]}"
    for i in "${iters[@]}"; do
        local tag
        tag=$(printf "%04d" "$i")
        echo "[GPU ${gpu_id}] Running iteration ${tag}"
        CUDA_VISIBLE_DEVICES=${gpu_id} bash "${INFER_SCRIPT}" "${tag}"
        echo "[GPU ${gpu_id}] Finished iteration ${tag}"
    done
    echo "[GPU ${gpu_id}] Worker done."
}

# ---------------------------------------------------------------
# Distribute iterations round-robin across GPUs and launch.
# ---------------------------------------------------------------
# Build per-GPU iteration lists
declare -A GPU_ITERS
for g in $(seq 0 $((NUM_GPUS - 1))); do
    GPU_ITERS[$g]=""
done

for idx in "${!ITERS[@]}"; do
    gpu=$((idx % NUM_GPUS))
    GPU_ITERS[$gpu]="${GPU_ITERS[$gpu]} ${ITERS[$idx]}"
done

# Launch one background worker per GPU
PIDS=()
for g in $(seq 0 $((NUM_GPUS - 1))); do
    iters_str="${GPU_ITERS[$g]}"
    if [ -z "${iters_str// /}" ]; then
        continue  # no iterations for this GPU
    fi
    # shellcheck disable=SC2086
    run_worker "$g" $iters_str &
    PIDS+=($!)
done

echo ""
echo "Launched ${#PIDS[@]} worker(s). Waiting for completion..."

# Wait for all workers; propagate any failure
FAILED=0
for pid in "${PIDS[@]}"; do
    if ! wait "$pid"; then
        echo "WARN: Worker PID ${pid} failed."
        FAILED=$((FAILED + 1))
    fi
done

echo ""
if [ "$FAILED" -eq 0 ]; then
    echo "All ${TOTAL} iterations completed successfully on ${NUM_GPUS} GPU(s)."
else
    echo "${FAILED} worker(s) failed out of ${#PIDS[@]}."
    exit 1
fi
echo "Analyse with:  python analyze_sparse_indices.py --base_dir <output_base> --mode frag_over_steps"
