#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-python}"
DEVICE="${DEVICE:-cuda:0}"
NUM_WORKERS="${NUM_WORKERS:-0}"

MAIN_ROOT="${MAIN_ROOT:-./logging/fedmini_complete/main}"
ABLATION_ROOT="${ABLATION_ROOT:-./logging/fedmini_complete/ablation}"
THETA_ROOT="${THETA_ROOT:-./logging/fedmini_complete/theta}"

RUN_MAIN="${RUN_MAIN:-1}"
RUN_ABLATION="${RUN_ABLATION:-1}"
RUN_THETA="${RUN_THETA:-1}"
COLLECT_RESULTS="${COLLECT_RESULTS:-1}"

export PYTHONWARNINGS="${PYTHONWARNINGS:-ignore::FutureWarning}"
export PYTHONPATH="${ROOT_DIR}"

echo "Repository root: ${ROOT_DIR}"
echo "Python: ${PYTHON_BIN}"
echo "Device: ${DEVICE}"
echo "Workers: ${NUM_WORKERS}"

run_python() {
  echo
  echo ">>> $*"
  "$PYTHON_BIN" "$@"
}

if [[ "${RUN_MAIN}" == "1" ]]; then
  run_python scripts/run_fedmini_paper.py \
    --datasets all \
    --scenarios all \
    --seeds 0 1 2 \
    --dirichlet-alphas 0.1 0.3 0.5 \
    --device "${DEVICE}" \
    --logging-root "${MAIN_ROOT}" \
    --num-workers "${NUM_WORKERS}"
fi

if [[ "${RUN_ABLATION}" == "1" ]]; then
  # Figure 3-9: FedMini default.
  run_python scripts/run_fedmini_paper.py \
    --datasets cifar100 \
    --scenarios dirichlet \
    --dirichlet-alphas 0.1 \
    --seeds 0 \
    --device "${DEVICE}" \
    --logging-root "${ABLATION_ROOT}/fedmini" \
    --num-workers "${NUM_WORKERS}"

  # Figure 3-9: No Frozen = stay in warmup for all 300 rounds.
  run_python scripts/run_fedmini_paper.py \
    --datasets cifar100 \
    --scenarios dirichlet \
    --dirichlet-alphas 0.1 \
    --seeds 0 \
    --device "${DEVICE}" \
    --logging-root "${ABLATION_ROOT}/no_frozen" \
    --num-workers "${NUM_WORKERS}" \
    --warmup-rounds 300

  # Figure 3-9: Full Network Update = use full-model update every round.
  run_python scripts/run_fedmini_paper.py \
    --datasets cifar100 \
    --scenarios dirichlet \
    --dirichlet-alphas 0.1 \
    --seeds 0 \
    --device "${DEVICE}" \
    --logging-root "${ABLATION_ROOT}/full_network_update" \
    --num-workers "${NUM_WORKERS}" \
    --warmup-rounds 300 \
    --full-update-rounds 300 \
    --rounds-per-group 1
fi

if [[ "${RUN_THETA}" == "1" ]]; then
  for theta in 0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0; do
    theta_tag="${theta/./p}"
    run_python scripts/run_fedmini_paper.py \
      --datasets cifar100 \
      --scenarios dirichlet \
      --dirichlet-alphas 0.1 \
      --seeds 0 \
      --device "${DEVICE}" \
      --logging-root "${THETA_ROOT}/theta_${theta_tag}" \
      --num-workers "${NUM_WORKERS}" \
      --sensitivity-weight "${theta}"
  done
fi

if [[ "${COLLECT_RESULTS}" == "1" ]]; then
  run_python scripts/collect_fedmini_results.py \
    --logging-root "${MAIN_ROOT}" \
    --output "${MAIN_ROOT}/fedmini_runs.csv" \
    --aggregate-output "${MAIN_ROOT}/fedmini_aggregate.csv"
fi

cat <<'EOF'

Completed the FedMini suite currently supported by this repository:
- Main FedMini table runs: CIFAR-100 + Tiny-ImageNet, Dirichlet/Pathological, 3 seeds.
- Figure 3-9 ablation: FedMini / No Frozen / Full Network Update on CIFAR-100 alpha=0.1.
- Figure 3-10 theta sweep on CIFAR-100 alpha=0.1.

Not included by this script:
- FedPAC baseline (not implemented in this repo)
- pFedCE baseline (not implemented in this repo)
- CIFAR-10 13-client overlap experiment from Figure 3-11/3-12 (requires custom client split tooling)
EOF
