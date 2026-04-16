#!/bin/bash
#
# Launch script for generalized aggregation experiments.
# Designed for a single 80GB NVIDIA GPU (e.g., A100/H100).
#
# Strategy:
#   - ResNet-18 + 100 clients uses ~8-10GB per experiment
#   - CNN + 100 clients uses ~2-4GB per experiment
#   - We can run ~4 experiments in parallel on an 80GB GPU
#   - Experiments are grouped into batches; each batch runs up to MAX_PARALLEL jobs
#
# Usage:
#   bash flzoo/experimental/run_experiments.sh                # Run all experiments
#   bash flzoo/experimental/run_experiments.sh --generate     # Generate all config files first
#   bash flzoo/experimental/run_experiments.sh --dry-run      # Print commands without running

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${PROJECT_DIR}"

MAX_PARALLEL=4
GPU_ID=0
DRY_RUN=false

for arg in "$@"; do
    case $arg in
        --generate)
            echo "[INFO] Generating all config files..."
            python "${SCRIPT_DIR}/generate_configs.py"
            echo "[INFO] Config generation complete."
            exit 0
            ;;
        --dry-run)
            DRY_RUN=true
            ;;
        --max-parallel=*)
            MAX_PARALLEL="${arg#*=}"
            ;;
        --gpu=*)
            GPU_ID="${arg#*=}"
            ;;
    esac
done

DATASETS=("cifar10" "cifar100")
MODELS=("resnet18" "cnn")
CLIENT_NUMS=(10 50 100)
SAMPLE_KEYS=("iid" "dir_0.1" "dir_0.5" "dir_1.0" "dir_5.0")
AGGR_KEYS=("fedavg" "generalized_1.0" "generalized_1.2" "generalized_1.5" "generalized_1.8" "fedmedian" "fedprox")

AGGR_METHOD_MAP=(
    "fedavg:avg"
    "generalized_1.0:generalized"
    "generalized_1.2:generalized"
    "generalized_1.5:generalized"
    "generalized_1.8:generalized"
    "fedmedian:median"
    "fedprox:avg"
)

AGGR_ALPHA_MAP=(
    "generalized_1.0:1.0"
    "generalized_1.2:1.2"
    "generalized_1.5:1.5"
    "generalized_1.8:1.8"
)

AGGR_CLIENT_MAP=(
    "fedprox:fedprox_client"
)

DS_PATH_MAP=(
    "cifar10:./data/CIFAR10"
    "cifar100:./data/CIFAR100"
)

DS_CLASS_MAP=(
    "cifar10:10"
    "cifar100:100"
)

lookup() {
    local key="$1"
    shift
    for entry in "$@"; do
        local k="${entry%%:*}"
        local v="${entry#*:}"
        if [[ "$k" == "$key" ]]; then
            echo "$v"
            return
        fi
    done
    echo ""
}

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

run_experiment() {
    local ds="$1"
    local model="$2"
    local client_num="$3"
    local sample_key="$4"
    local aggr_key="$5"

    local aggr_method
    aggr_method=$(lookup "$aggr_key" "${AGGR_METHOD_MAP[@]}")
    local aggr_alpha
    aggr_alpha=$(lookup "$aggr_key" "${AGGR_ALPHA_MAP[@]}")
    local client_name
    client_name=$(lookup "$aggr_key" "${AGGR_CLIENT_MAP[@]}")
    [[ -z "$client_name" ]] && client_name="base_client"
    local data_path
    data_path=$(lookup "$ds" "${DS_PATH_MAP[@]}")
    local class_num
    class_num=$(lookup "$ds" "${DS_CLASS_MAP[@]}")

    local exp_name="${ds}_${aggr_key}_${model}_${sample_key}_c${client_num}"
    local log_path="./logging/experimental/${exp_name}"
    local run_log="${LOG_DIR}/${exp_name}.log"

    local sample_method_arg
    if [[ "$sample_key" == "iid" ]]; then
        sample_method_arg="data.sample_method.name:iid"
    else
        local beta="${sample_key#dir_}"
        sample_method_arg="data.sample_method.name:dirichlet -e data.sample_method.alpha:${beta}"
    fi

    local extra_args=""
    extra_args+=" -e data.dataset:${ds}"
    extra_args+=" -e data.data_path:${data_path}"
    extra_args+=" -e ${sample_method_arg}"
    extra_args+=" -e model.name:${model}"
    extra_args+=" -e model.input_channel:3"
    extra_args+=" -e model.class_number:${class_num}"
    extra_args+=" -e client.name:${client_name}"
    extra_args+=" -e client.client_num:${client_num}"
    extra_args+=" -e group.aggregation_method:${aggr_method}"
    extra_args+=" -e other.logging_path:${log_path}"

    if [[ -n "$aggr_alpha" ]]; then
        extra_args+=" -e group.aggregation_alpha:${aggr_alpha}"
    fi

    if [[ "$aggr_key" == "fedprox" ]]; then
        extra_args+=" -e learn.mu:0.01"
    fi

    local base_config="flzoo/experimental/cifar10_fedavg_resnet18_dir01_config"
    local cmd="CUDA_VISIBLE_DEVICES=${GPU_ID} fling run -c ${base_config} -p generic_model_pipeline ${extra_args}"

    if [[ "$DRY_RUN" == "true" ]]; then
        echo "[DRY-RUN] ${cmd}"
        return
    fi

    echo "[LAUNCH] ${exp_name} -> ${run_log}"
    wait_for_slot
    eval "${cmd}" > "${run_log}" 2>&1 &
    PIDS+=($!)
    JOB_COUNT=$((JOB_COUNT + 1))
}

echo "=============================================="
echo "  Generalized Aggregation Experiments"
echo "  GPU: ${GPU_ID} (80GB), Max parallel: ${MAX_PARALLEL}"
echo "=============================================="

echo ""
echo "[Phase 1] Alpha sensitivity analysis (Sec 5.1)"
echo "  Fixed: Dirichlet beta=0.1, varying alpha"
for ds in "${DATASETS[@]}"; do
    for model in "${MODELS[@]}"; do
        for aggr in "${AGGR_KEYS[@]}"; do
            run_experiment "$ds" "$model" 10 "dir_0.1" "$aggr"
        done
    done
done

echo ""
echo "[Phase 2] Heterogeneity-alpha interaction (Sec 5.2)"
echo "  Varying beta and alpha together"
for ds in "${DATASETS[@]}"; do
    for sample_key in "${SAMPLE_KEYS[@]}"; do
        for aggr in "fedavg" "generalized_1.0" "generalized_1.2" "generalized_1.5" "generalized_1.8" "fedmedian"; do
            run_experiment "$ds" "resnet18" 10 "$sample_key" "$aggr"
        done
    done
done

echo ""
echo "[Phase 3] Client number scaling"
echo "  Testing with different client numbers"
for ds in "${DATASETS[@]}"; do
    for client_num in "${CLIENT_NUMS[@]}"; do
        for aggr in "fedavg" "generalized_1.0" "generalized_1.5" "fedmedian"; do
            run_experiment "$ds" "resnet18" "$client_num" "dir_0.1" "$aggr"
        done
    done
done

echo ""
echo "[Phase 4] IID baseline verification"
for ds in "${DATASETS[@]}"; do
    for model in "${MODELS[@]}"; do
        for aggr in "fedavg" "generalized_1.0" "generalized_1.5" "fedmedian"; do
            run_experiment "$ds" "$model" 10 "iid" "$aggr"
        done
    done
done

if [[ "$DRY_RUN" == "false" ]]; then
    echo ""
    echo "[INFO] Waiting for all remaining jobs to complete..."
    for pid in "${PIDS[@]}"; do
        wait "$pid" 2>/dev/null || true
    done
    echo "[DONE] All ${JOB_COUNT} experiments completed."
else
    echo ""
    echo "[DRY-RUN] Would launch ${JOB_COUNT} experiments."
fi
