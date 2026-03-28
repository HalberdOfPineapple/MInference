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
#   bash watch_and_infer.sh --merged_ckpt_dir /path/to/merged_ckpts
#
# All experiment settings (model, dataset, pattern config, etc.) are
# passed through to infer_sparse_indices.sh via environment variables.
# See --help or the argument list below for the full set.
#
# Environment variables also accepted (lower priority than CLI args):
#   COLLECT_SPARSE_INDEX  — collect v_idx/s_idx (default: 1)
#   COLLECT_BLOCK_MASK    — collect block_mask/bar_cnt (default: 0)
#   NUM_SAMPLES           — samples per checkpoint (default: 20)
#   SEED                  — random seed (default: 42)
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

# Experiment settings — passed through to infer_sparse_indices.sh as env vars.
# CLI args here override env vars; env vars override hardcoded defaults.
OPT_MERGED_CKPT_BASE=""
OPT_MODEL_ID=""
OPT_MODEL_CONFIG_PATH=""
OPT_PATTERN_CONFIG=""
OPT_DATASET_PATH=""
OPT_NUM_SAMPLES=""
OPT_SEED=""
OPT_GRANULARITY=""
OPT_SAVE_INTERVAL=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --num_gpus)          NUM_GPUS="$2";              shift 2 ;;
        --poll_interval)     POLL_INTERVAL="$2";         shift 2 ;;
        --start_epoch)       START_EPOCH="$2";           shift 2 ;;
        --start_iter)        START_ITER="$2";            shift 2 ;;
        --merged_ckpt_dir)   OPT_MERGED_CKPT_BASE="$2"; shift 2 ;;
        --model_id)          OPT_MODEL_ID="$2";          shift 2 ;;
        --model_config_path) OPT_MODEL_CONFIG_PATH="$2"; shift 2 ;;
        --pattern_config)    OPT_PATTERN_CONFIG="$2";    shift 2 ;;
        --dataset_path)      OPT_DATASET_PATH="$2";      shift 2 ;;
        --num_samples)       OPT_NUM_SAMPLES="$2";       shift 2 ;;
        --seed)              OPT_SEED="$2";              shift 2 ;;
        --granularity)       OPT_GRANULARITY="$2";       shift 2 ;;
        --save_interval)     OPT_SAVE_INTERVAL="$2";     shift 2 ;;
        *)
            echo "Unknown argument: $1"
            echo "Usage: $0 [--num_gpus N] [--poll_interval S] [--start_epoch E] [--start_iter I]"
            echo "          [--merged_ckpt_dir DIR] [--model_id ID] [--model_config_path PATH]"
            echo "          [--pattern_config NAME] [--dataset_path PATH] [--num_samples N]"
            echo "          [--seed N] [--granularity N] [--save_interval N]"
            exit 1 ;;
    esac
done

# -----------------------------------------------
# Export experiment settings as env vars for infer_sparse_indices.sh.
# CLI args take priority; otherwise fall through to the defaults in
# infer_sparse_indices.sh.
[ -n "$OPT_MERGED_CKPT_BASE" ]  && export MERGED_CKPT_BASE="$OPT_MERGED_CKPT_BASE"
[ -n "$OPT_MODEL_ID" ]          && export MODEL_ID="$OPT_MODEL_ID"
[ -n "$OPT_MODEL_CONFIG_PATH" ] && export MODEL_CONFIG_PATH="$OPT_MODEL_CONFIG_PATH"
[ -n "$OPT_PATTERN_CONFIG" ]    && export PATTERN_CONFIG="$OPT_PATTERN_CONFIG"
[ -n "$OPT_DATASET_PATH" ]      && export DATASET_PATH="$OPT_DATASET_PATH"
[ -n "$OPT_NUM_SAMPLES" ]       && export NUM_SAMPLES="$OPT_NUM_SAMPLES"
[ -n "$OPT_SEED" ]              && export SEED="$OPT_SEED"
[ -n "$OPT_GRANULARITY" ]       && export GRANULARITY="$OPT_GRANULARITY"
[ -n "$OPT_SAVE_INTERVAL" ]     && export SAVE_INTERVAL="$OPT_SAVE_INTERVAL"

# Resolve MERGED_CKPT_BASE for the watcher's own polling loop.
MERGED_CKPT_BASE="${MERGED_CKPT_BASE:-/blob/mtrain_expr_data_store/A100_32/mtrain_qwen/qwen_3B_best_pattern_512K/merged_ckpts}"

# Auto-detect GPU count if not specified
if [ -z "$NUM_GPUS" ]; then
    NUM_GPUS=$(nvidia-smi -L 2>/dev/null | wc -l)
    if [ "$NUM_GPUS" -le 0 ] 2>/dev/null; then
        NUM_GPUS=1
    fi
fi

echo "============================================"
echo " GPU-pool watcher for sparse index collection"
echo "  Merged ckpt dir : ${MERGED_CKPT_BASE}"
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
echo "Watching ${MERGED_CKPT_BASE} for new merged checkpoints..."
echo ""

while true; do
    if [ -d "$MERGED_CKPT_BASE" ]; then
        # Collect candidate checkpoint tags, sorted
        CANDIDATES=()
        for entry in "$MERGED_CKPT_BASE"/*/; do
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
