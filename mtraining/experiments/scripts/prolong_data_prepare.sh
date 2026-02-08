#!/usr/bin/bash

export HF_TRUST_REMOTE_CODE=1
export HF_HOME=/scratch/hf_cache/huggingface
export HF_TOKEN_PATH="/scratch/.hf_access_token"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
BASE_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
PROJECT_ROOT="$(cd "${BASE_DIR}/.." && pwd)"

# conda activate mtrain
source /opt/conda/etc/profile.d/conda.sh
conda activate mtrain

# # Ensure mtraining package is discoverable (parent of mtraining dir)
# export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH}"
# echo $PROJECT_ROOT

# ------------------------------------------
# Download data
# Prerequisite: sudo apt-get install git-lfs && git lfs install
# RAW_DATASET_DIR="/path/to/datasets"
RAW_DATASET_DIR="/scratch/datasets"
mkdir -p $RAW_DATASET_DIR

# git clone https://huggingface.co/datasets/princeton-nlp/prolong-data-512K $RAW_DATASET_DIR/long-context-524288
cd $RAW_DATASET_DIR/long-context-524288
git lfs fetch
git lfs checkout


# ------------------------------------------
# Data Processing
cd $BASE_DIR
MODEL_ID="Qwen/Qwen2.5-3B"
PROCESSED_DATA_DIR="${BASE_DIR}/experiments/processed_datasets"
mkdir -p $PROCESSED_DATA_DIR

torchrun --nproc_per_node=1 \
	utils/data_utils/prolong.py \
    --model_id $MODEL_ID \
    --dataset_mix fixed_524288 \
    --dataset_path $RAW_DATASET_DIR/long-context-524288 \
    --save_path $PROCESSED_DATA_DIR/long-context-524288  
