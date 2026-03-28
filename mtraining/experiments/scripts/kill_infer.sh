#!/usr/bin/bash

# Copyright (c) 2026 Microsoft
# Licensed under The MIT License [see LICENSE for details]

# ============================================================
# Kill all running sparse index inference and watcher processes.
#
# Finds and terminates:
#   1. watch_and_infer.sh (the GPU-pool watcher)
#   2. infer_sparse_indices.sh (per-GPU workers)
#   3. infer_sparse_indices.py (the actual Python inference)
#   4. run_all_infer_sparse_indices.sh (batch runner, if any)
#
# Usage:
#   bash kill_infer.sh          # graceful SIGTERM
#   bash kill_infer.sh --force  # SIGKILL (immediate)
# ============================================================

set -euo pipefail

SIG="TERM"
if [[ "${1:-}" == "--force" || "${1:-}" == "-f" ]]; then
    SIG="KILL"
    echo "Using SIGKILL (force mode)"
else
    echo "Using SIGTERM (graceful mode, use --force for SIGKILL)"
fi

SELF_PID=$$
KILLED=0

# Patterns to match (order: parent → child so children don't get orphaned)
PATTERNS=(
    "watch_and_infer.sh"
    "run_all_infer_sparse_indices.sh"
    "infer_sparse_indices.sh"
    "infer_sparse_indices.py"
)

for pat in "${PATTERNS[@]}"; do
    # Find matching PIDs, excluding this script itself
    PIDS=$(ps aux | grep "[${pat:0:1}]${pat:1}" | awk '{print $2}' | grep -v "^${SELF_PID}$" || true)

    if [ -n "$PIDS" ]; then
        for pid in $PIDS; do
            CMD=$(ps -p "$pid" -o args= 2>/dev/null || echo "(already exited)")
            echo "  Killing PID ${pid} (${pat}): ${CMD}"
            kill -"${SIG}" "$pid" 2>/dev/null || true
            KILLED=$((KILLED + 1))
        done
    fi
done

echo ""
if [ "$KILLED" -eq 0 ]; then
    echo "No matching processes found."
else
    echo "Sent SIG${SIG} to ${KILLED} process(es)."
fi
