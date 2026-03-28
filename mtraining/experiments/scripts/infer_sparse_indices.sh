#!/usr/bin/bash

# Copyright (c) 2026 Microsoft
# Licensed under The MIT License [see LICENSE for details]

# Single-card inference script for collecting sparse attention data
# (indices, block masks, bar counts, sparse ratios).
#
# Unlike the training-based sparse ratio collection (train_qwen2_3B_sparse_ratio.sh),
# this runs on a single GPU without distributed training or nnscaler.
#
# All experiment settings can be overridden via environment variables,
# making this script composable with watch_and_infer.sh and
# run_all_infer_sparse_indices.sh.
#
# Usage:
#   bash infer_sparse_indices.sh [ITER_IDX | EPOCH-ITER_TAG]
#   bash infer_sparse_indices.sh 0005        # load checkpoint 0000-0005
#   bash infer_sparse_indices.sh 0000-0005   # same, explicit epoch-iter tag

set -euo pipefail

# -----------------------------------------------
# HuggingFace settings
export HF_HOME=/scratch/hf_cache/huggingface
mkdir -p $HF_HOME
export HF_TRUST_REMOTE_CODE=true
export HF_DATASETS_TRUST_REMOTE_CODE=true

# -----------------------------------------------
# Enable sparse data collection (each type is independently controllable).
# Set to 1 to collect; 0 (or unset) to skip.
#   COLLECT_SPARSE_INDEX — v_idx / s_idx index tensors (moderate size)
#   COLLECT_BLOCK_MASK   — block_mask / bar_cnt tensors (very large!)
# Sparse ratios are always recorded when any flag is active.
export COLLECT_SPARSE_INDEX=${COLLECT_SPARSE_INDEX:-0}
export COLLECT_BLOCK_MASK=${COLLECT_BLOCK_MASK:-0}

# -----------------------------------------------
# Paths
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
EXPR_HOME="$(cd "${SCRIPT_DIR}/../.." && pwd)"  # .../mtraining
REPO_ROOT="$(cd "${EXPR_HOME}/.." && pwd)"
cd "${EXPR_HOME}"

# Ensure both mtraining and minference are importable
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

# -----------------------------------------------
# Experiment settings (all overridable via env vars)
MODEL_ID="${MODEL_ID:-Qwen/Qwen2.5-3B}"
MODEL_CONFIG_PATH="${MODEL_CONFIG_PATH:-${EXPR_HOME}/model_configs/qwen2/lc_config_3B}"
PATTERN_CONFIG="${PATTERN_CONFIG:-Qwen2.5_3B_kv_out_v32_fit_o_best_pattern}"
DATASET_PATH="${DATASET_PATH:-/scratch/data_store/processed_datasets/long-context-524288}"
MERGED_CKPT_BASE="${MERGED_CKPT_BASE:-/blob/mtrain_expr_data_store/A100_32/mtrain_qwen/qwen_3B_best_pattern_512K/merged_ckpts}"
NUM_SAMPLES="${NUM_SAMPLES:-20}"
SEED="${SEED:-42}"
GRANULARITY="${GRANULARITY:-128}"
SAVE_INTERVAL="${SAVE_INTERVAL:-5}"

# -----------------------------------------------
# Checkpoint settings
TARGET_EPOCH_IDX="0000"
TARGET_ITER_IDX="${1:-0001}"

# Accept either a bare iter index (e.g. "0005") or a full epoch-iter
# tag (e.g. "0000-0005") so the script works with both
# run_all_infer_sparse_indices.sh and watch_and_infer.sh.
if [[ "$TARGET_ITER_IDX" == *-* ]]; then
    TARGET_EPOCH_IDX="${TARGET_ITER_IDX%%-*}"
    TARGET_ITER_IDX="${TARGET_ITER_IDX##*-}"
fi

TARGET_CKPT_TAG="${TARGET_EPOCH_IDX}-${TARGET_ITER_IDX}"
CKPT_PATH="${MERGED_CKPT_BASE}/${TARGET_CKPT_TAG}/pytorch_model.bin"

echo "Checkpoint tag: ${TARGET_CKPT_TAG}"
echo "Checkpoint path: ${CKPT_PATH}"

# -----------------------------------------------
# Output settings
OUTPUT_BASE="${MERGED_CKPT_BASE}/../sparse_indices"
OUTPUT_DIR="${OUTPUT_BASE}/${TARGET_CKPT_TAG}"
mkdir -p "${OUTPUT_DIR}"
echo "Output directory: ${OUTPUT_DIR}"

# -----------------------------------------------
# Logging
LOG_DIR="${OUTPUT_BASE}/${TARGET_CKPT_TAG}/logs"
mkdir -p "${LOG_DIR}"
LOG_FILE="${LOG_DIR}/infer.log"
echo "Log file: ${LOG_FILE}"

# Build collection flags for the Python script
COLLECT_FLAGS=""
if [ "${COLLECT_SPARSE_INDEX}" = "1" ]; then
    COLLECT_FLAGS="${COLLECT_FLAGS} --collect_indices"
fi
if [ "${COLLECT_BLOCK_MASK}" = "1" ]; then
    COLLECT_FLAGS="${COLLECT_FLAGS} --collect_block_mask"
fi

# -----------------------------------------------
# Run single-card inference
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0} \
python experiments/scripts/infer_sparse_indices.py \
    --model_id "${MODEL_ID}" \
    --model_config_path "${MODEL_CONFIG_PATH}" \
    --ckpt_path "${CKPT_PATH}" \
    --pattern_config "${PATTERN_CONFIG}" \
    --dataset_path "${DATASET_PATH}" \
    --output_dir "${OUTPUT_DIR}" \
    --num_samples ${NUM_SAMPLES} \
    --seed ${SEED} \
    --granularity ${GRANULARITY} \
    --save_interval ${SAVE_INTERVAL} \
    ${COLLECT_FLAGS} > ${LOG_FILE} 2>&1

echo "Done. Data saved to ${OUTPUT_DIR}"
echo "Log saved to ${LOG_FILE}"
