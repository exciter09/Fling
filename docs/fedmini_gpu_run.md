# FedMini GPU Run Guide

This document describes how to run the FedMini third-chapter experiments on a single CUDA machine.

## Environment

From the repository root:

```bash
pip install -e .
```

Recommended CUDA command pattern:

```bash
PYTHONPATH=. python scripts/run_fedmini_paper.py --device cuda:0
```

## Full Paper Sweep

Run the CIFAR-100 and Tiny-ImageNet experiments for both Dirichlet and Pathological settings, with seeds `0 1 2`:

```bash
PYTHONPATH=. python scripts/run_fedmini_paper.py \
  --datasets all \
  --scenarios all \
  --seeds 0 1 2 \
  --dirichlet-alphas 0.1 0.3 0.5 \
  --device cuda:0 \
  --logging-root ./logging/fedmini_paper \
  --num-workers 8
```

## Single Experiment

Run only the CIFAR-100 Dirichlet `alpha=0.1` experiment:

```bash
PYTHONPATH=. python scripts/run_fedmini_paper.py \
  --datasets cifar100 \
  --scenarios dirichlet \
  --dirichlet-alphas 0.1 \
  --seeds 0 \
  --device cuda:0
```

## Result Collection

After runs finish, collect per-run and aggregated CSV summaries:

```bash
PYTHONPATH=. python scripts/collect_fedmini_results.py \
  --logging-root ./logging/fedmini_paper
```

## Generated Logs

Each run directory contains:

- `fedmini_metadata.json`: config, git info, layer-group metadata.
- `round_metrics.jsonl`: per-round structured metrics.
- `round_metrics.csv`: flattened per-round metrics for analysis.
- `freeze_events.jsonl`: layer freeze events.
- `summary.json`: best metric, total communication cost, completion state.
- TensorBoard event files and `txt_logger_output.txt`.
