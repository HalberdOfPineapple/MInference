#!/usr/bin/bash

# Copyright (c) 2026 Microsoft
# Licensed under The MIT License [see LICENSE for details]

# ============================================================
# Shell wrapper for analyze_sparse_indices.py
#
# Usage:
#   bash run_index_analysis.sh                  # default: summary
#   bash run_index_analysis.sh draw_all         # generate ALL figures
#   bash run_index_analysis.sh <mode>           # run a specific mode
#   bash run_index_analysis.sh position_dist 0-35:5   # mode + layer spec
#
# Positional shortcuts:
#   $1 = mode        (default: summary)
#   $2 = layers      (default: auto)
#   $3 = samples     (default: all)
#
# Analysis modes:
#   --- Single-iteration (uses DATA_DIR) ---
#   summary          – Print per-layer valid-index statistics
#   count_heatmap    – Heatmap of index counts per (layer, head)
#   position_dist    – Histogram of where indices fall in the sequence
#   overlap          – Cross-sample Jaccard similarity of selected indices
#   sparsity         – Derive and plot sparsity ratio from index counts
#   head_variance    – Box plots of index counts across heads
#   per_head_grid    – Small-multiples: per-head count vs layer
#   discreteness     – Quantitative gap/run/fragmentation table per layer
#   gap_histogram    – Histogram of inter-index gaps (log-scale)
#   fragmentation    – Fragmentation index & mean run length across layers
#   raster           – Spike raster of selected positions per head
#   density_heatmap  – Binned spatial density heatmap (layers × regions)
#   export_csv       – Export flat CSV with indices per (sample, layer, head)
#   draw_all         – Generate ALL figure types at once
#   --- Cross-iteration dynamics (uses BASE_DIR) ---
#   frag_over_steps     – Fragmentation index vs training step, per layer
#   count_over_steps    – Valid index count vs training step, per layer
#   dynamics_heatmap    – Heatmap: layers × steps coloured by fragmentation
#   overlap_over_steps  – Jaccard similarity between consecutive checkpoints
# ============================================================

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PYTHON_SCRIPT="${SCRIPT_DIR}/analyze_sparse_indices.py"

# -----------------------------------------------------------
# Data settings — point to the output of infer_sparse_indices.sh
# -----------------------------------------------------------
CKPT_TAG="${CKPT_TAG:-0000-0000}"
BASE_DIR="/blob/mtrain_expr_data_store/A100_32/mtrain_qwen/qwen_3B_fp090_512K_tokenized_7B_4GPUS/sparse_indices"
DATA_DIR="${BASE_DIR}/${CKPT_TAG}"

# -----------------------------------------------------------
# Analysis mode
# -----------------------------------------------------------
MODE="${1:-discreteness}"

# -----------------------------------------------------------
# Output settings
# -----------------------------------------------------------
OUTPUT_DIR="${SCRIPT_DIR}/outputs/sparse_indices/${CKPT_TAG}"

# -----------------------------------------------------------
# Layer / sample selection (optional)
# -----------------------------------------------------------
LAYERS="${2:-auto}"
SAMPLES="${3:-all}"

# ============================================================
# Build and run the command
# ============================================================
mkdir -p "${OUTPUT_DIR}" 2>/dev/null

# Dynamics modes use --base_dir; single-iteration modes use --data_dir
DYNAMICS_MODES="frag_over_steps count_over_steps dynamics_heatmap overlap_over_steps"
if echo "${DYNAMICS_MODES}" | grep -qw "${MODE}"; then
    CMD=(python "${PYTHON_SCRIPT}"
        --base_dir "${BASE_DIR}"
        --mode "${MODE}"
        --output_dir "${OUTPUT_DIR}"
    )
else
    CMD=(python "${PYTHON_SCRIPT}"
        --data_dir "${DATA_DIR}"
        --mode "${MODE}"
        --output_dir "${OUTPUT_DIR}"
    )
fi

[[ "${LAYERS}" != "auto" ]] && CMD+=(--layers "${LAYERS}")
[[ "${SAMPLES}" != "all" ]] && CMD+=(--samples "${SAMPLES}")

echo "Running: ${CMD[*]}"
echo "---"
"${CMD[@]}"
