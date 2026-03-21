#!/usr/bin/bash

# Copyright (c) 2026 Microsoft
# Licensed under The MIT License [see LICENSE for details]

set -euo pipefail

# Example:
#   bash mtraining/experiments/scripts/eval_attn_qkv_dump_qwen2_3B.sh

i=$(hostname | awk -F'-' '{print $2}')
NODE_RANK=${i}

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

export EFFI_EVAL_MODE=1

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
MTRAIN_HOME="$(cd "${SCRIPT_DIR}/../.." && pwd)" # .../mtraining
PROJECT_ROOT="$(cd "${MTRAIN_HOME}/.." && pwd)" # .../MInference
cd "${MTRAIN_HOME}"

GPU_SET="A100_${WORLD_SIZE}"
EXPR_DATA_STORE="/blob/mtrain_expr_data_store/${GPU_SET}"
EXPR_DIR="dense_qwen"
EXPR_NAME="qwen_3B_dense_qkv"

LOG_DIR="${EXPR_DATA_STORE}/${EXPR_DIR}/${EXPR_NAME}/rank_${NODE_RANK}"
mkdir -p "${LOG_DIR}"

RESULT_DIR="${EXPR_DATA_STORE}/${EXPR_DIR}/${EXPR_NAME}/attn_eval"
mkdir -p "${RESULT_DIR}"

QKV_DUMP_ROOT="${EXPR_DATA_STORE}/${EXPR_DIR}/${EXPR_NAME}/qkv_dump"


# Supported choices: dense, zigzag_ring, stripe_ring, minfer, moba, xattn
# ATTN_TYPE="minfer"
# TRAIN_ATTN_CONFIG_PATH="${MTRAIN_HOME}/train_attn_configs/qwen_mf_zigzag.yaml"
# TRAIN_ATTN_CONFIG_PATH="${MTRAIN_HOME}/train_attn_configs/qwen_mf_stripe.yaml"
# TRAIN_ATTN_CONFIG_PATH="${MTRAIN_HOME}/train_attn_configs/qwen_mf_dr_stripe.yaml"



# # Supported choices: dense, zigzag_ring, stripe_ring, minfer, moba, xattn
ATTN_TYPE="xattn"
TRAIN_ATTN_CONFIG_NAME="xattn_dr_stripe_s16"

NUM_Q_HEADS=16
NUM_KV_HEADS=2
GLOBAL_SEQ_LEN=524288

WARMUP_ITERS=20
BENCH_ITERS=50
DTYPE="bf16"
MEASURE_BACKWARD="true"
ENABLE_REGION_TIMER="true"

LAYER_INDICES="34"          # e.g. "0,1,2"
SAMPLE_INDICES="0"         # e.g. "0,1,2,3"
MAX_PAIRS=0                  # 0 means no cap

declare -A CLI_ARGS=(
    ["qkv_dump_root"]="${QKV_DUMP_ROOT}"
    ["attn_type"]="${ATTN_TYPE}"
    ["train_attn_config_path"]="${MTRAIN_HOME}/train_attn_configs/${TRAIN_ATTN_CONFIG_NAME}.yaml"
    ["num_q_heads"]="${NUM_Q_HEADS}"
    ["num_kv_heads"]="${NUM_KV_HEADS}"
    ["global_seq_len"]="${GLOBAL_SEQ_LEN}"
    ["layer_indices"]="${LAYER_INDICES}"
    ["sample_indices"]="${SAMPLE_INDICES}"
    ["max_pairs"]="${MAX_PAIRS}"
    ["warmup_iters"]="${WARMUP_ITERS}"
    ["bench_iters"]="${BENCH_ITERS}"
    ["dtype"]="${DTYPE}"
    ["save_json"]="${RESULT_DIR}/${ATTN_TYPE}_rank${NODE_RANK}.json"
    ["save_csv"]="${RESULT_DIR}/${ATTN_TYPE}_rank${NODE_RANK}.csv"
)

LOG_FILE="${LOG_DIR}/eval_attn_${TRAIN_ATTN_CONFIG_NAME}.log"
echo "Logging directed to ${LOG_FILE}"

CMD=(
    torchrun
    --nproc_per_node="${GPU_PER_NODE}"
    --nnodes="${NUM_NODES}"
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

# printf 'Command:\n%s\n' "${CMD[*]}"
"${CMD[@]}" > "${LOG_FILE}" 2>&1
echo "Log saved to ${LOG_FILE}"
