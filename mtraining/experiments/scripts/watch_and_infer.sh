#!/usr/bin/bash

# Copyright (c) 2026 Microsoft
# Licensed under The MIT License [see LICENSE for details]

# ============================================================
# GPU-pool watcher for sparse index collection.
#
# Continuously monitors the merged checkpoint directory for newly
# merged models (produced by launch_auto_merge.sh / auto_merge_ckpt.py)
# and dispatches inference jobs to a pool of GPUs.
#
# Each GPU is treated as an independent worker.  When a worker is
# idle and a new merged checkpoint is discovered, the checkpoint is
# assigned to that worker.  This allows the inference pipeline to
# run alongside training + auto-merge without manual intervention.
#
# Usage:
#   bash watch_and_infer.sh                          # auto-detect GPUs
#   bash watch_and_infer.sh --num_gpus 4             # use 4 GPUs
#   bash watch_and_infer.sh --poll_interval 30       # poll every 30s
#   bash watch_and_infer.sh --start_iter 5           # skip iters < 5
#
# Environment variables (inherited by infer_sparse_indices.sh):
#   COLLECT_SPARSE_INDEX  — collect v_idx/s_idx (default: 1)
#   COLLECT_BLOCK_MASK    — collect block_mask/bar_cnt (default: 0)
#   NUM_SAMPLES           — samples per checkpoint (default: inherited)
#   SEED                  — random seed (default: inherited)
# ============================================================

set -euo pipefail

# -----------------------------------------------
# Parse arguments
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
INFER_SCRIPT="${SCRIPT_DIR}/infer_sparse_indices.sh"

NUM_GPUS=""
POLL_INTERVAL=15
START_EPOCH=0
START_ITER=0
MERGED_CKPT_DIR=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --num_gpus)      NUM_GPUS="$2";        shift 2 ;;
        --poll_interval) POLL_INTERVAL="$2";   shift 2 ;;
        --start_epoch)   START_EPOCH="$2";     shift 2 ;;
        --start_iter)    START_ITER="$2";      shift 2 ;;
        --merged_ckpt_dir) MERGED_CKPT_DIR="$2"; shift 2 ;;
        *)
            echo "Unknown argument: $1"
            echo "Usage: $0 [--num_gpus N] [--poll_interval S] [--start_epoch E] [--start_iter I] [--merged_ckpt_dir DIR]"
            exit 1 ;;
    esac
done

# Auto-detect GPU count if not specified
if [ -z "$NUM_GPUS" ]; then
    NUM_GPUS=$(nvidia-smi -L 2>/dev/null | wc -l)
    if [ "$NUM_GPUS" -le 0 ] 2>/dev/null; then
        NUM_GPUS=1
    fi
fi

# Default merged checkpoint directory (matches launch_auto_merge.sh layout)
if [ -z "$MERGED_CKPT_DIR" ]; then
    MERGED_CKPT_DIR="/blob/mtrain_expr_data_store/A100_32/mtrain_qwen/qwen_3B_best_pattern_512K/merged_ckpts"
fi

echo "============================================"
echo " GPU-pool watcher for sparse index collection"
echo "  Merged ckpt dir : ${MERGED_CKPT_DIR}"
echo "  GPUs            : ${NUM_GPUS}"
echo "  Poll interval   : ${POLL_INTERVAL}s"
echo "  Start filter    : epoch >= ${START_EPOCH}, iter >= ${START_ITER}"
echo "============================================"

# -----------------------------------------------
# State tracking
# GPU_PID[gpu_id] = PID of the running worker (0 = idle)
# PROCESSED tracks checkpoint tags we've already dispatched
declare -A GPU_PID
declare -A GPU_TAG
declare -A PROCESSED

for g in $(seq 0 $((NUM_GPUS - 1))); do
    GPU_PID[$g]=0
    GPU_TAG[$g]=""
done

# -----------------------------------------------
# Helper: check if a merged checkpoint is ready
# (pytorch_model.bin exists and is stable)
is_merged_ready() {
    local ckpt_dir="$1"
    local model_file="${ckpt_dir}/pytorch_model.bin"

    [ -f "$model_file" ] || return 1

    # Check file size stability over 3 seconds
    local size1
    size1=$(stat -c %s "$model_file" 2>/dev/null) || return 1
    sleep 3
    local size2
    size2=$(stat -c %s "$model_file" 2>/dev/null) || return 1

    [ "$size1" = "$size2" ]
}

# -----------------------------------------------
# Helper: find an idle GPU (returns gpu_id or -1)
find_idle_gpu() {
    for g in $(seq 0 $((NUM_GPUS - 1))); do
        local pid=${GPU_PID[$g]}
        if [ "$pid" -eq 0 ]; then
            echo "$g"
            return
        fi
        # Check if the process is still running
        if ! kill -0 "$pid" 2>/dev/null; then
            # Process finished — check exit status
            if wait "$pid" 2>/dev/null; then
                echo "[GPU $g] Finished ${GPU_TAG[$g]} successfully." >&2
            else
                echo "[GPU $g] FAILED ${GPU_TAG[$g]}!" >&2
            fi
            GPU_PID[$g]=0
            GPU_TAG[$g]=""
            echo "$g"
            return
        fi
    done
    echo "-1"
}

# -----------------------------------------------
# Helper: dispatch a checkpoint to a GPU
dispatch() {
    local gpu_id="$1"
    local tag="$2"     # full epoch-iter tag, e.g. "0000-0005"

    echo "[GPU $gpu_id] Dispatching iteration ${tag}"
    CUDA_VISIBLE_DEVICES=${gpu_id} bash "${INFER_SCRIPT}" "${tag}" &
    GPU_PID[$gpu_id]=$!
    GPU_TAG[$gpu_id]="$tag"
    PROCESSED[$tag]=1
}

# -----------------------------------------------
# Main loop: poll for new merged checkpoints
echo ""
echo "Watching ${MERGED_CKPT_DIR} for new merged checkpoints..."
echo ""

while true; do
    if [ -d "$MERGED_CKPT_DIR" ]; then
        # Collect candidate checkpoint tags, sorted
        CANDIDATES=()
        for entry in "$MERGED_CKPT_DIR"/*/; do
            [ -d "$entry" ] || continue
            tag=$(basename "$entry")

            # Must match epoch-iter format (e.g. 0000-0005)
            [[ "$tag" =~ ^([0-9]+)-([0-9]+)$ ]] || continue
            epoch_idx=$((10#${BASH_REMATCH[1]}))
            iter_idx=$((10#${BASH_REMATCH[2]}))

            # Filter by start epoch/iter
            if [ "$epoch_idx" -lt "$START_EPOCH" ]; then continue; fi
            if [ "$epoch_idx" -eq "$START_EPOCH" ] && [ "$iter_idx" -lt "$START_ITER" ]; then continue; fi

            # Skip already dispatched
            [ -z "${PROCESSED[$tag]+x}" ] || continue

            # Check if merge is complete
            if is_merged_ready "$entry"; then
                CANDIDATES+=("$tag")
            fi
        done

        # Sort candidates by epoch then iter
        if [ ${#CANDIDATES[@]} -gt 0 ]; then
            IFS=$'\n' SORTED=($(printf '%s\n' "${CANDIDATES[@]}" | sort)); unset IFS

            for tag in "${SORTED[@]}"; do
                gpu_id=$(find_idle_gpu)
                if [ "$gpu_id" = "-1" ]; then
                    # All GPUs busy — wait for next poll
                    break
                fi
                # Pass the full epoch-iter tag (infer_sparse_indices.sh parses it)
                dispatch "$gpu_id" "$tag"
            done
        fi
    fi

    sleep "$POLL_INTERVAL"
done
