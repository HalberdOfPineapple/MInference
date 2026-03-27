#!/usr/bin/bash

# Copyright (c) 2026 Microsoft
# Licensed under The MIT License [see LICENSE for details]

# Single-card inference script for collecting sparse attention indices.
# Unlike the training-based sparse ratio collection (train_qwen2_3B_sparse_ratio.sh),
# this runs on a single GPU without distributed training or nnscaler.
#
# Usage:
#   bash infer_sparse_indices.sh [ITER_IDX]
#   bash infer_sparse_indices.sh 0005        # load checkpoint 0000-0005

set -euo pipefail

# -----------------------------------------------
# HuggingFace settings
export HF_HOME=/scratch/hf_cache/huggingface
mkdir -p $HF_HOME
export HF_TRUST_REMOTE_CODE=true
export HF_DATASETS_TRUST_REMOTE_CODE=true

# Enable sparse index collection
export COLLECT_SPARSE_INDEX=1

# -----------------------------------------------
# Paths
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
EXPR_HOME="$(cd "${SCRIPT_DIR}/../.." && pwd)"  # .../mtraining
cd "${EXPR_HOME}"

# -----------------------------------------------
# Model settings
MODEL_ID="Qwen/Qwen2.5-3B"
MODEL_CONFIG_PATH="${EXPR_HOME}/model_configs/qwen2/lc_config_3B"
PATTERN_CONFIG="Qwen2.5_3B_flex_0.90"

# -----------------------------------------------
# Checkpoint settings
MERGED_CKPT_BASE="/blob/mtrain_expr_data_store/A100_32/mtrain_qwen/qwen_3B_fp090_512K_tokenized_7B_4GPUS/merged_ckpts"
TARGET_EPOCH_IDX="0000"
TARGET_ITER_IDX="${1:-0000}"
TARGET_CKPT_TAG="${TARGET_EPOCH_IDX}-${TARGET_ITER_IDX}"
CKPT_PATH="${MERGED_CKPT_BASE}/${TARGET_CKPT_TAG}/pytorch_model.bin"

echo "Checkpoint tag: ${TARGET_CKPT_TAG}"
echo "Checkpoint path: ${CKPT_PATH}"

# -----------------------------------------------
# Dataset settings
DATASET_PATH="/scratch/data_store/processed_datasets/long-context-524288"
NUM_SAMPLES=20

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
    --granularity 128 \
    --save_interval 5 > ${LOG_FILE} 2>&1

echo "Done. Indices saved to ${OUTPUT_DIR}"
echo "Log saved to ${LOG_FILE}"
