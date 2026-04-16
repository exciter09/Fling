#!/bin/bash
#
# Launch 30 focused experiments (resnet8, 10 clients).
# Designed for a single 80GB NVIDIA GPU.
#
# resnet8 + 10 clients ≈ 2-3GB per run → 6 parallel jobs safely fit in 80GB.
#
# Usage:
#   bash flzoo/experimental/run_experiments.sh              # run all
#   bash flzoo/experimental/run_experiments.sh --dry-run    # preview commands only

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${PROJECT_DIR}"

MAX_PARALLEL=6
GPU_ID=0
DRY_RUN=false

for arg in "$@"; do
    case $arg in
        --dry-run)      DRY_RUN=true ;;
        --max-parallel=*) MAX_PARALLEL="${arg#*=}" ;;
        --gpu=*)        GPU_ID="${arg#*=}" ;;
    esac
done

PIDS=()
JOB_COUNT=0
LOG_DIR="${PROJECT_DIR}/logging/experimental"
mkdir -p "${LOG_DIR}"

wait_for_slot() {
    while [[ ${#PIDS[@]} -ge $MAX_PARALLEL ]]; do
        local new_pids=()
        for pid in "${PIDS[@]}"; do
            if kill -0 "$pid" 2>/dev/null; then
                new_pids+=("$pid")
            fi
        done
        PIDS=("${new_pids[@]}")
        if [[ ${#PIDS[@]} -ge $MAX_PARALLEL ]]; then
            sleep 5
        fi
    done
}

launch() {
    local config_name="$1"
    local config_path="flzoo/experimental/${config_name}_config"
    local run_log="${LOG_DIR}/${config_name}.log"

    local cmd="CUDA_VISIBLE_DEVICES=${GPU_ID} python -m ${config_path/\//.}"

    if [[ "$DRY_RUN" == "true" ]]; then
        echo "[DRY-RUN] ${config_name}"
        JOB_COUNT=$((JOB_COUNT + 1))
        return
    fi

    echo "[LAUNCH] ${config_name} -> ${run_log}"
    wait_for_slot
    CUDA_VISIBLE_DEVICES=${GPU_ID} python -m "${config_path//\//.}" > "${run_log}" 2>&1 &
    PIDS+=($!)
    JOB_COUNT=$((JOB_COUNT + 1))
}

echo "=============================================="
echo "  Generalized Aggregation Experiments (30)"
echo "  Model: resnet8 | Clients: 10"
echo "  GPU: ${GPU_ID} | Max parallel: ${MAX_PARALLEL}"
echo "=============================================="
echo ""

DATASETS=("cifar10" "cifar100")
SPLITS=("iid" "dir01" "dir05")
METHODS=("fedavg" "geomed" "gen15" "fedmedian" "fedprox")

for ds in "${DATASETS[@]}"; do
    for split in "${SPLITS[@]}"; do
        for method in "${METHODS[@]}"; do
            launch "${ds}_${method}_${split}"
        done
    done
done

if [[ "$DRY_RUN" == "false" ]]; then
    echo ""
    echo "[INFO] Waiting for all ${JOB_COUNT} jobs to finish..."
    for pid in "${PIDS[@]}"; do
        wait "$pid" 2>/dev/null || true
    done
    echo "[DONE] All ${JOB_COUNT} experiments completed. Logs in ${LOG_DIR}/"
else
    echo ""
    echo "[DRY-RUN] Would launch ${JOB_COUNT} experiments."
fi
