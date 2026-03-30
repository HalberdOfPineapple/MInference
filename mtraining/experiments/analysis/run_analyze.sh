#!/usr/bin/bash

# Copyright (c) 2026 Microsoft
# Licensed under The MIT License [see LICENSE for details]

# ============================================================
# Shell wrapper for the unified sparse attention analysis script.
#
# Usage:
#   bash run_analyze.sh ratio   --mode summary
#   bash run_analyze.sh ratio   --mode draw_all --output_dir ./plots
#   bash run_analyze.sh indices --mode fragmentation --ckpt_tag 0000-0005
#   bash run_analyze.sh indices --mode frag_over_steps --layers 0,17,35
#   bash run_analyze.sh masks   --mode block_pattern --ckpt_tag 0000-0005 --layer 10
#   bash run_analyze.sh --help
#
# Subcommands:
#   ratio    — Sparse ratio trends across checkpoints / layers
#              Modes: summary, layer_over_steps, step_over_layers, heatmap,
#                     delta_heatmap, global_trend, export_csv, draw_all, layer_avg
#
#   indices  — Vertical & slash index statistics
#              Modes: summary, count_heatmap, fragmentation, discreteness,
#                     position_dist, export_csv, draw_all,
#                     frag_over_steps, count_over_steps (cross-checkpoint)
#
#   masks    — Block mask density & bar count distributions
#              Modes: summary, density_heatmap, bar_distribution,
#                     block_pattern, sparsity_per_layer, export_csv, draw_all
# ============================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PYTHON_SCRIPT="${SCRIPT_DIR}/analyze.py"

# -----------------------------------------------
# Defaults
BASE_DIR="${BASE_DIR:-/blob/mtrain_expr_data_store/A100_32/mtrain_qwen/qwen_3B_best_pattern_512K/sparse_indices}"
OUTPUT_DIR=""

# -----------------------------------------------
# Extract subcommand
if [[ $# -lt 1 ]] || [[ "$1" == --help ]] || [[ "$1" == -h ]]; then
    echo "Usage: $0 <ratio|indices|masks> [OPTIONS]"
    echo ""
    echo "Options (passed through to analyze.py):"
    echo "  --base_dir DIR       Root directory with checkpoint-tag subdirs"
    echo "  --ckpt_tag TAG       Checkpoint tag (indices/masks, default: latest)"
    echo "  --mode MODE          Analysis mode (default: summary)"
    echo "  --layers SPEC        Layer selection: 'all', 'auto', '0,5,10', '0-35:5'"
    echo "  --ckpts SPEC         Checkpoint selection (ratio subcommand)"
    echo "  --output_dir DIR     Output directory for plots"
    echo "  --sample N           Sample index (masks block_pattern mode)"
    echo "  --layer N            Layer index (masks block_pattern mode)"
    echo "  --head N             Head index (masks block_pattern mode)"
    echo ""
    echo "Run '$0 <subcommand> --help' for subcommand-specific help."
    exit 0
fi

SUBCMD="$1"
shift

# -----------------------------------------------
# Parse remaining args — separate --base_dir and --output_dir
# for default handling, pass everything else through.
PASSTHROUGH=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --base_dir)    BASE_DIR="$2";    shift 2 ;;
        --output_dir)  OUTPUT_DIR="$2";  shift 2 ;;
        *)             PASSTHROUGH+=("$1"); shift ;;
    esac
done

# Auto-derive output_dir if not set
if [ -z "$OUTPUT_DIR" ]; then
    OUTPUT_DIR="${BASE_DIR}/outputs/${SUBCMD}"
fi
mkdir -p "${OUTPUT_DIR}" 2>/dev/null

# -----------------------------------------------
# Build and run
CMD=(python "${PYTHON_SCRIPT}" "${SUBCMD}"
    --base_dir "${BASE_DIR}"
    --output_dir "${OUTPUT_DIR}"
    "${PASSTHROUGH[@]}"
)

echo "Running: ${CMD[*]}"
echo "---"
"${CMD[@]}"
