import argparse
import csv
import json
import os
from collections import defaultdict


def parse_args():
    parser = argparse.ArgumentParser(description='Collect FedMini structured results into CSV summaries.')
    parser.add_argument('--logging-root', type=str, default='./logging/fedmini_paper')
    parser.add_argument('--output', type=str, default='./logging/fedmini_paper/fedmini_runs.csv')
    parser.add_argument('--aggregate-output', type=str, default='./logging/fedmini_paper/fedmini_aggregate.csv')
    return parser.parse_args()


def load_json(path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def write_csv(path, rows):
    if len(rows) == 0:
        return
    fieldnames = sorted({key for row in rows for key in row.keys()})
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main():
    args = parse_args()
    run_rows = []
    grouped = defaultdict(list)

    for root, _, files in os.walk(args.logging_root):
        if 'summary.json' not in files or 'fedmini_metadata.json' not in files:
            continue
        summary = load_json(os.path.join(root, 'summary.json'))
        metadata = load_json(os.path.join(root, 'fedmini_metadata.json'))
        exp_args = metadata['args']
        sample_method = exp_args['data']['sample_method']
        row = dict(
            logging_path=root,
            dataset=exp_args['data']['dataset'],
            split_mode=sample_method['name'],
            split_value=sample_method['alpha'],
            seed=summary['seed'],
            best_after_aggregation_test_acc=summary['best_after_aggregation_test_acc'],
            best_after_aggregation_test_round=summary['best_after_aggregation_test_round'],
            total_trans_cost_mb=summary['total_trans_cost_mb'],
            executed_rounds=summary['executed_rounds'],
            finished_all_frozen=summary['finished_all_frozen'],
            git_commit=metadata.get('git_commit'),
            git_branch=metadata.get('git_branch'),
        )
        run_rows.append(row)
        grouped[(row['dataset'], row['split_mode'], row['split_value'])].append(row['best_after_aggregation_test_acc'])

    aggregate_rows = []
    for (dataset, split_mode, split_value), values in sorted(grouped.items()):
        mean = sum(values) / len(values)
        variance = sum((value - mean) ** 2 for value in values) / len(values)
        aggregate_rows.append(
            dict(
                dataset=dataset,
                split_mode=split_mode,
                split_value=split_value,
                run_count=len(values),
                best_after_aggregation_test_acc_mean=mean,
                best_after_aggregation_test_acc_std=variance ** 0.5,
            )
        )

    write_csv(args.output, run_rows)
    write_csv(args.aggregate_output, aggregate_rows)
    print(f'Collected {len(run_rows)} runs into {args.output}')
    print(f'Aggregated {len(aggregate_rows)} groups into {args.aggregate_output}')


if __name__ == '__main__':
    main()
