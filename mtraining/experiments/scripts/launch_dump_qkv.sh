#!/usr/bin/bash

# Copyright (c) 2026 Microsoft
# Licensed under The MIT License [see LICENSE for details]

set -euo pipefail

export HF_TRUST_REMOTE_CODE=1
export HF_HOME=/scratch/hf_cache/huggingface
export HF_TOKEN_PATH="/scratch/.hf_access_token"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
PYTHON_SCRIPT="${SCRIPT_DIR}/dump_qkv.py"

mkdir -p "${SCRIPT_DIR}/logs"
LOG_FILE="${SCRIPT_DIR}/logs/dump_qkv.log"

export HF_TRUST_REMOTE_CODE="${HF_TRUST_REMOTE_CODE:-1}"
export HF_HOME="${HF_HOME:-/scratch/hf_cache/huggingface}"
mkdir -p "${HF_HOME}"

declare -A CLI_ARGS=(
    [model_id]="${MODEL_ID:-Qwen/Qwen2.5-3B}"
    [dataset_path]="${DATASET_PATH:-/scratch/data_store/processed_datasets/long-context-524288/}"
    [output_root]="${OUTPUT_ROOT:-${SCRIPT_DIR}/qkv_outputs}"
    [num_samples]="${NUM_SAMPLES:-50}"
    [seed]="${SEED:-42}"
    [device]="${DEVICE:-cuda}"
    [dtype]="${DTYPE:-auto}"
    [attn_implementation]="${ATTN_IMPLEMENTATION:-flash_attention_2}"
)

if [[ -n "${DATASET_SPLIT:-}" ]]; then
    CLI_ARGS[dataset_split]="${DATASET_SPLIT}"
fi

CMD=(
    python -u "${PYTHON_SCRIPT}"
)

ARG_ORDER=(
    model_id
    dataset_path
    dataset_split
    output_root
    num_samples
    seed
    device
    dtype
    attn_implementation
)

for key in "${ARG_ORDER[@]}"; do
    if [[ -n "${CLI_ARGS[$key]:-}" ]]; then
        CMD+=("--${key}" "${CLI_ARGS[$key]}")
    fi
done

if [[ $# -gt 0 ]]; then
    CMD+=("$@")
fi

cd "${PROJECT_ROOT}"

{
    echo "Working directory: ${PROJECT_ROOT}"
    echo "Log file: ${LOG_FILE}"
    echo "Command: ${CMD[*]}"
    "${CMD[@]}"
} 2>&1 | tee "${LOG_FILE}"
