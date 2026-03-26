#!/usr/bin/bash

# Copyright (c) 2026 Microsoft
# Licensed under The MIT License [see LICENSE for details]

# ============================================================
# Shell wrapper for analyze_sparse_ratio.py
#
# Usage:
#   bash run_analysis.sh                  # default: summary
#   bash run_analysis.sh draw_all         # generate ALL figures
#   bash run_analysis.sh <mode>           # run a specific mode
#   bash run_analysis.sh heatmap 0-35:5   # mode + layer spec
#
# Positional shortcuts:
#   $1 = mode        (default: summary)
#   $2 = layers      (default: auto)
#   $3 = ckpts       (default: auto)
#   $4 = ckpt_tag    (for sample_dist mode)
#
# Or just edit the variables below for full control.
# ============================================================

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PYTHON_SCRIPT="${SCRIPT_DIR}/analyze_sparse_ratio.py"

# -----------------------------------------------------------
# Data settings
# -----------------------------------------------------------
BASE_DIR="/blob/mtrain_expr_data_store/A100_32/mtrain_qwen/qwen_3B_fp090_512K_sr"
ATTN_NAME="qwen_flex_090"
RANK=0

# -----------------------------------------------------------
# Analysis mode
# -----------------------------------------------------------
# Overridden by $1 if provided.
#   summary            – Print per-checkpoint and per-layer statistics
#   layer_over_steps   – Plot sparse ratio vs training step (per layer)
#   step_over_layers   – Plot sparse ratio vs layer (per checkpoint)
#   heatmap            – Heatmap: layers x checkpoints
#   delta_heatmap      – Heatmap of change from first checkpoint
#   sample_dist        – Per-sample distribution at one checkpoint
#   layer_trend        – Mean ± std bands for selected layers
#   global_trend       – Global mean sparse ratio over training
#   layer_histogram    – Histogram of sparse ratios at checkpoints
#   per_layer_grid     – Small-multiples: per-layer trend
#   compare_attn       – Compare multiple attn configs side-by-side
#   export_csv         – Export flat CSV
#   draw_all           – Generate ALL figure types at once
MODE="${1:-summary}"

# -----------------------------------------------------------
# Output settings
# -----------------------------------------------------------
OUTPUT_DIR="${SCRIPT_DIR}/outputs/${ATTN_NAME}"

# -----------------------------------------------------------
# Layer / checkpoint selection (optional)
# -----------------------------------------------------------
# Overridden by $2 / $3 if provided.
LAYERS="${2:-auto}"
CKPTS="${3:-auto}"
CKPT_TAG="${4:-}"

# -----------------------------------------------------------
# compare_attn mode: comma-separated attn config names
# -----------------------------------------------------------
ATTN_NAMES=""

# ============================================================
# Build and run the command
# ============================================================
mkdir -p "${OUTPUT_DIR}" 2>/dev/null

CMD=(python "${PYTHON_SCRIPT}"
    --base_dir "${BASE_DIR}"
    --attn_name "${ATTN_NAME}"
    --rank "${RANK}"
    --mode "${MODE}"
)

[[ -n "${OUTPUT_DIR}" ]] && CMD+=(--output "${OUTPUT_DIR}")
[[ -n "${LAYERS}" && "${LAYERS}" != "auto" ]] && CMD+=(--layers "${LAYERS}")
[[ -n "${CKPTS}" && "${CKPTS}" != "auto" ]] && CMD+=(--ckpts "${CKPTS}")
[[ -n "${CKPT_TAG}" ]] && CMD+=(--ckpt_tag "${CKPT_TAG}")
[[ -n "${ATTN_NAMES}" ]] && CMD+=(--attn_names "${ATTN_NAMES}")

echo "Running: ${CMD[*]}"
echo "---"
"${CMD[@]}"
