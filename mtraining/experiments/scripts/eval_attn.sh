#!/usr/bin/bash

# Copyright (c) 2026 Microsoft
# Licensed under The MIT License [see LICENSE for details]

set -euo pipefail

# Example:
#   bash mtraining/experiments/scripts/eval_attn_qkv_dump_qwen2_3B.sh

i=$(hostname | awk -F'-' '{print $2}')
NODE_RANK=${i}

export GPU_NAME="A100"
export NUM_NODES=4
export GPU_PER_NODE=8
# world size = num_nodes * gpu_per_node
export WORLD_SIZE=$((NUM_NODES * GPU_PER_NODE))
export MASTER_ADDR="node-0"
export MASTER_PORT="12345"

export HF_HOME=/scratch/hf_cache/huggingface
mkdir -p "${HF_HOME}"
export HF_TRUST_REMOTE_CODE=true
export HF_DATASETS_TRUST_REMOTE_CODE=true

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
MTRAIN_HOME="$(cd "${SCRIPT_DIR}/../.." && pwd)" # .../mtraining
PROJECT_ROOT="$(cd "${MTRAIN_HOME}/.." && pwd)" # .../MInference
cd "${MTRAIN_HOME}"

GPU_SET="${GPU_NAME}_${WORLD_SIZE}"
EXPR_DATA_STORE="/blob/mtrain_expr_data_store/${GPU_SET}"
EXPR_DIR="dense_qwen"
EXPR_NAME="qwen_3B_dense_qkv"
QKV_DUMP_ROOT="${EXPR_DATA_STORE}/${EXPR_DIR}/${EXPR_NAME}/qkv_dump"

# -------------------------------------------------------------
NUM_Q_HEADS=16
NUM_KV_HEADS=2
GLOBAL_SEQ_LEN=524288

WARMUP_ITERS=20
BENCH_ITERS=50
DTYPE="bf16"
MEASURE_BACKWARD="true"
ENABLE_REGION_TIMER="true"
# Optional: set to 1 to emit torch profiler Chrome traces into RESULT_DIR.
export EVAL_ATTN_ENABLE_TORCH_PROFILER=0


# 0-34 for XAttn
# 1-32 for MTrain

# LAYER_IDX="34"          # e.g. "0,1,2"
# SAMPLE_IDX="0"         # e.g. "0,1,2,3"
LAYER_IDX="0"          # e.g. "0,1,2"
SAMPLE_IDX="0"         # e.g. "0,1,2,3"
MAX_PAIRS=0                  # 0 means no cap

# -------------------------------------------------------------
export EVAL_GPU_NAME="A100"
export EVAL_NUM_NODES=4
export EVAL_GPU_PER_NODE=8

export COLLECT_SPARSE_RATIO=1

# Supported choices: dense, zigzag_ring, stripe_ring, minfer, moba, xattn
ATTN_TYPE="minfer"
# TRAIN_ATTN_CONFIG_NAME="qwen_mf_zigzag"
# TRAIN_ATTN_CONFIG_NAME="qwen_mf_stripe"
TRAIN_ATTN_CONFIG_NAME="qwen_mf_dr_stripe"

# ATTN_TYPE="xattn"
# TRAIN_ATTN_CONFIG_NAME="xattn_dr_stripe_s16"


# -------------------------------------------------------------
# LOG_DIR="${EXPR_DATA_STORE}/${EXPR_DIR}/${EXPR_NAME}/attn_eval/${GPU_NAME}_${NUM_NODES}x${GPU_PER_NODE}/"
# mkdir -p "${LOG_DIR}"

RESULT_DIR="${EXPR_DATA_STORE}/${EXPR_DIR}/${EXPR_NAME}/attn_eval/${SAMPLE_IDX}_${LAYER_IDX}/${TRAIN_ATTN_CONFIG_NAME}/${EVAL_GPU_NAME}_${EVAL_NUM_NODES}x${EVAL_GPU_PER_NODE}"
mkdir -p "${RESULT_DIR}"

# -------------------------------------------------------------
declare -A CLI_ARGS=(
    ["qkv_dump_root"]="${QKV_DUMP_ROOT}"
    ["attn_type"]="${ATTN_TYPE}"
    ["train_attn_config_path"]="${MTRAIN_HOME}/train_attn_configs/${TRAIN_ATTN_CONFIG_NAME}.yaml"
    ["num_q_heads"]="${NUM_Q_HEADS}"
    ["num_kv_heads"]="${NUM_KV_HEADS}"
    ["global_seq_len"]="${GLOBAL_SEQ_LEN}"
    ["layer_indices"]="${LAYER_IDX}"
    ["sample_indices"]="${SAMPLE_IDX}"
    ["max_pairs"]="${MAX_PAIRS}"
    ["warmup_iters"]="${WARMUP_ITERS}"
    ["bench_iters"]="${BENCH_ITERS}"
    ["dtype"]="${DTYPE}"
    ["save_json"]="${RESULT_DIR}/rank_${NODE_RANK}_inner_128.json"
    ["save_csv"]="${RESULT_DIR}/rank_${NODE_RANK}_inner_128.csv"
)

# -------------------------------------------------------------
CMD=(
    torchrun
    --nproc_per_node="${EVAL_GPU_PER_NODE}"
    --nnodes="${EVAL_NUM_NODES}"
    --node_rank="${NODE_RANK}"
    --master_addr="${MASTER_ADDR}"
    --master_port="${MASTER_PORT}"
    experiments/scripts/eval_attn.py
)

for key in "${!CLI_ARGS[@]}"; do
    value="${CLI_ARGS[$key]}"
    if [ -n "${value}" ]; then
        CMD+=("--${key}" "${value}")
    fi
done

if [ "${MEASURE_BACKWARD}" = "true" ]; then
    CMD+=("--measure_backward")
fi
if [ "${ENABLE_REGION_TIMER}" = "true" ]; then
    CMD+=("--enable_region_timer")
fi

LOG_FILE="${RESULT_DIR}/rank_${NODE_RANK}.log"

# -------------------------------------------------------------
/blob/utils/kill_nv_local.sh true

# if GPU starts with 'A100', lock GPU frequency to 1410MHz for stable profiling
# else if GPU starts with 'H100', lock GPU frequency to 1980MHz
if [[ "$GPU_NAME" == A100* ]]; then
    /blob/utils/lock_freq.sh --sm 1410
elif [[ "$GPU_NAME" == H100* ]]; then
    /blob/utils/lock_freq.sh --sm 1980
else
    echo "GPU $GPU_NAME not supported for frequency lock. Skipping."
fi
 
echo "Logging directed to ${LOG_FILE}"

# printf 'Command:\n%s\n' "${CMD[*]}"
"${CMD[@]}" > "${LOG_FILE}" 2>&1

/blob/utils/kill_nv_local.sh


echo "Log saved to ${LOG_FILE}"
