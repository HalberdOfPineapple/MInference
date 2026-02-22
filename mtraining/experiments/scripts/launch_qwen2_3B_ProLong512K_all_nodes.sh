#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# Usage:
#   bash launch_qwen2_3B_ProLong512K_all_nodes.sh [NUM_NODES] [script1.sh script2.sh ...]
# Examples:
#   bash launch_qwen2_3B_ProLong512K_all_nodes.sh
#   bash launch_qwen2_3B_ProLong512K_all_nodes.sh 4 train_qwen2_3B_ProLong512K.sh
#   NUM_NODES=4 bash launch_qwen2_3B_ProLong512K_all_nodes.sh train_qwen2_3B_ProLong512K.sh

NUM_NODES="${NUM_NODES:-4}"
if [[ $# -gt 0 && "$1" =~ ^[0-9]+$ ]]; then
    NUM_NODES="$1"
    shift
fi

if [[ $# -gt 0 ]]; then
    TARGET_SCRIPTS=("$@")
else
    TARGET_SCRIPTS=("train_qwen2_3B_ProLong512K.sh")
fi

NODES=()
for ((i=0; i<NUM_NODES; i++)); do
    NODES+=("node-${i}")
done

echo "Nodes: ${NODES[*]}"

for script in "${TARGET_SCRIPTS[@]}"; do
    if [[ "${script}" != /* ]]; then
        script="${SCRIPT_DIR}/${script}"
    fi

    script_name="$(basename "${script}")"
    echo "Launching ${script_name} on all nodes..."
    pids=()

    for node in "${NODES[@]}"; do
        if [[ "${node}" == "node-0" ]]; then
            bash "${script}" 2>&1 | sed -u "s/^/[${node}] /" &
            pids+=("$!")
            echo "[${node}] started"
        else
            ssh -n -o BatchMode=yes -o ConnectTimeout=10 "${node}" \
                "bash '${script}'" 2>&1 | sed -u "s/^/[${node}] /" &
            pids+=("$!")
            echo "[${node}] started"
        fi
    done

    echo "All nodes started for ${script_name}. Streaming logs below..."
    fail=0
    for pid in "${pids[@]}"; do
        if ! wait "${pid}"; then
            fail=1
        fi
    done

    if [[ "${fail}" -ne 0 ]]; then
        echo "${script_name} finished with at least one node failure."
    else
        echo "${script_name} finished successfully on all nodes."
    fi
done

echo "Done."
