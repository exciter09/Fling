"""
Visualization script for generalized aggregation experiments.

Reads TensorBoard logs and plots after-aggregation test accuracy curves.
For each (dataset, split) setting, produces one figure with different
aggregation methods as separate curves.

Usage:
    python flzoo/experimental/plot_results.py
    python flzoo/experimental/plot_results.py --log-dir ./logging/experimental --output-dir ./figures
"""
import os
import argparse

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

DATASETS = ['cifar10', 'cifar100']
SPLITS = ['iid', 'dir01', 'dir05']
METHODS = ['fedavg', 'geomed', 'gen15', 'fedmedian', 'fedprox']

METHOD_DISPLAY = {
    'fedavg': 'FedAvg (α=2)',
    'geomed': 'GeoMed (α=1.0)',
    'gen15': 'Generalized (α=1.5)',
    'fedmedian': 'FedMedian',
    'fedprox': 'FedProx',
}

SPLIT_DISPLAY = {
    'iid': 'IID',
    'dir01': 'Dirichlet β=0.1',
    'dir05': 'Dirichlet β=0.5',
}

METHOD_COLORS = {
    'fedavg': '#1f77b4',
    'geomed': '#ff7f0e',
    'gen15': '#2ca02c',
    'fedmedian': '#d62728',
    'fedprox': '#9467bd',
}

METHOD_LINESTYLES = {
    'fedavg': '-',
    'geomed': '-',
    'gen15': '-',
    'fedmedian': '--',
    'fedprox': '--',
}


def load_scalar(log_dir, tag):
    ea = EventAccumulator(log_dir)
    ea.Reload()
    if tag not in ea.Tags().get('scalars', []):
        return [], []
    events = ea.Scalars(tag)
    steps = [e.step for e in events]
    values = [e.value for e in events]
    return steps, values


def plot_group(dataset, split, log_root, output_dir):
    tag = 'after_aggregation_test/test_acc'
    fig, ax = plt.subplots(figsize=(8, 5))

    has_data = False
    best_records = []
    for method in METHODS:
        exp_name = f'{dataset}_{method}_{split}'
        log_dir = os.path.join(log_root, exp_name)
        if not os.path.isdir(log_dir):
            continue
        steps, values = load_scalar(log_dir, tag)
        if not steps:
            continue
        has_data = True
        best_idx = max(range(len(values)), key=lambda i: values[i])
        best_records.append((METHOD_DISPLAY.get(method, method), steps[best_idx], values[best_idx]))
        ax.plot(
            steps, values,
            label=METHOD_DISPLAY.get(method, method),
            color=METHOD_COLORS.get(method, None),
            linestyle=METHOD_LINESTYLES.get(method, '-'),
            linewidth=2,
        )

    if not has_data:
        plt.close(fig)
        return

    ds_display = dataset.upper().replace('CIFAR', 'CIFAR-')
    title = f'{ds_display} / {SPLIT_DISPLAY[split]}'

    ax.set_xlabel('Communication Round', fontsize=13)
    ax.set_ylabel('Test Accuracy', fontsize=13)
    ax.set_title(title, fontsize=14)
    ax.legend(fontsize=11, loc='lower right')
    ax.grid(True, alpha=0.3)
    ax.tick_params(labelsize=11)

    fig.tight_layout()
    fname = f'{dataset}_{split}_test_acc.pdf'
    fig.savefig(os.path.join(output_dir, fname), dpi=150, bbox_inches='tight')
    print(f'Saved: {fname}')
    plt.close(fig)

    print(f'\n  [Best Acc] {title}')
    print(f'  {"Method":<25s} {"Round":>6s} {"Acc":>10s}')
    print(f'  {"-"*25} {"-"*6} {"-"*10}')
    for name, rnd, acc in best_records:
        print(f'  {name:<25s} {rnd:>6d} {acc:>10.4f}')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--log-dir', default='./logging/experimental')
    parser.add_argument('--output-dir', default='./figures')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    for dataset in DATASETS:
        for split in SPLITS:
            plot_group(dataset, split, args.log_dir, args.output_dir)

    print(f'\nAll figures saved to {args.output_dir}/')


if __name__ == '__main__':
    main()
