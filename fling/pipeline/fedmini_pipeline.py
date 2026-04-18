import csv
import json
import os
import subprocess
import time
import torch

from fling.component.client import get_client
from fling.component.group import get_group
from fling.component.server import get_server
from fling.dataset import get_dataset
from fling.utils import Logger, VariableMonitor, client_sampling, compile_config, get_launcher, LRScheduler
from fling.utils.data_utils import data_sampling


def _jsonify(value):
    if isinstance(value, dict):
        return {str(k): _jsonify(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonify(v) for v in value]
    if hasattr(value, 'items'):
        return {str(k): _jsonify(v) for k, v in value.items()}
    if isinstance(value, (int, float, str, bool)) or value is None:
        return value
    return str(value)


def _write_json(path: str, payload: dict) -> None:
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(_jsonify(payload), f, ensure_ascii=False, indent=2)


def _append_jsonl(path: str, payload: dict) -> None:
    with open(path, 'a', encoding='utf-8') as f:
        f.write(json.dumps(_jsonify(payload), ensure_ascii=False) + '\n')


def _write_csv(path: str, rows: list) -> None:
    if len(rows) == 0:
        return
    flattened_rows = [_flatten_record(row) for row in rows]
    fieldnames = sorted({key for row in flattened_rows for key in row.keys()})
    with open(path, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in flattened_rows:
            writer.writerow({key: _jsonify(row.get(key)) for key in fieldnames})


def _flatten_record(record: dict, prefix: str = '') -> dict:
    flattened = {}
    for key, value in record.items():
        flat_key = f'{prefix}.{key}' if prefix else str(key)
        if isinstance(value, dict):
            flattened.update(_flatten_record(value, flat_key))
        else:
            flattened[flat_key] = value
    return flattened


def _maybe_git_info() -> dict:
    try:
        commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()
        branch = subprocess.check_output(['git', 'branch', '--show-current'], text=True).strip()
        return dict(git_commit=commit, git_branch=branch)
    except Exception:
        return dict(git_commit=None, git_branch=None)


def fedmini_pipeline(args: dict, seed: int = 0) -> None:
    """
    Overview:
        Pipeline for the minimal FedMini reproduction used in this workspace.
    """
    args = compile_config(args, seed=seed)
    if str(args.learn.device).startswith('cuda'):
        torch.backends.cudnn.benchmark = bool(getattr(args.learn, 'cudnn_benchmark', True))
    logger = Logger(args.other.logging_path)
    round_metrics_path = os.path.join(args.other.logging_path, 'round_metrics.jsonl')
    freeze_events_path = os.path.join(args.other.logging_path, 'freeze_events.jsonl')
    round_csv_path = os.path.join(args.other.logging_path, 'round_metrics.csv')
    summary_path = os.path.join(args.other.logging_path, 'summary.json')
    metadata_path = os.path.join(args.other.logging_path, 'fedmini_metadata.json')
    for path in [round_metrics_path, freeze_events_path]:
        if os.path.exists(path):
            os.remove(path)

    train_set = get_dataset(args, train=True)
    test_set = get_dataset(args, train=False)
    train_sets = data_sampling(train_set, args, seed, train=True)
    test_sets = data_sampling(test_set, args, seed, train=False)

    group = get_group(args, logger)
    group.server = get_server(args, test_dataset=test_set)
    for i in range(args.client.client_num):
        group.append(get_client(args=args, client_id=i, train_dataset=train_sets[i], test_dataset=test_sets[i]))
    group.initialize()
    metadata = dict(
        seed=seed,
        pipeline='fedmini_pipeline',
        logging_path=args.other.logging_path,
        args=_jsonify(args),
        train_dataset_size=len(train_set),
        test_dataset_size=len(test_set),
        group_metadata=group.get_metadata(),
        start_time=time.strftime('%Y-%m-%d %H:%M:%S', time.localtime()),
    )
    metadata.update(_maybe_git_info())
    _write_json(metadata_path, metadata)

    lr_scheduler = LRScheduler(base_lr=args.learn.optimizer.lr, args=args.learn.scheduler)
    launcher = get_launcher(args)
    round_records = []
    freeze_event_records = []
    cumulative_trans_cost_mb = 0.0
    executed_rounds = 0

    for i in range(args.learn.global_eps):
        round_args = group.get_round_args(i)
        if round_args['stage'] == 'finished':
            logger.logging('All layer groups are frozen. Stop training early.')
            break

        logger.logging(f'Starting round: {i}')
        train_monitor = VariableMonitor()
        participated_clients = client_sampling(range(args.client.client_num), args.client.sample_rate)
        cur_lr = lr_scheduler.get_lr(train_round=i)

        train_results = launcher.launch(
            clients=[group.clients[j] for j in participated_clients],
            lr=cur_lr,
            task_name='train',
            train_args=round_args['train_args']
        )
        for item in train_results:
            train_monitor.append(item)
        participated_client_objs = [group.clients[j] for j in participated_clients]
        client_stats = [getattr(client, 'round_statistics', {}) for client in participated_client_objs]
        sensitive_ratio_values = [stat.get('sensitive_ratio', 0.0) for stat in client_stats]
        active_param_values = [stat.get('active_param_num', 0) for stat in client_stats]

        before_test_result = None
        if i % args.other.test_freq == 0 and "before_aggregation" in args.learn.test_place:
            test_monitor = VariableMonitor()
            test_results = launcher.launch(
                clients=[group.clients[j] for j in range(args.client.client_num)],
                task_name='test'
            )
            for item in test_results:
                test_monitor.append(item)
            before_test_result = test_monitor.variable_mean()
            logger.add_scalars_dict(prefix='before_aggregation_test', dic=before_test_result, rnd=i)

        trans_cost = group.aggregate(
            i,
            participate_clients_ids=participated_clients,
            aggr_parameter_args=round_args['aggr_args']
        )
        cumulative_trans_cost_mb += trans_cost / 1e6
        group_stats = group.get_last_round_stats()

        mean_train_variables = train_monitor.variable_mean()
        mean_train_variables.update(
            {
                'trans_cost(MB)': trans_cost / 1e6,
                'lr': cur_lr,
                'index': int(round_args['group_index']),
                'stage_id': 0 if round_args['stage'] == 'warmup' else 1,
                'frozen_groups': float(sum(group.frozen_groups)),
                'sensitive_ratio_mean': float(sum(sensitive_ratio_values) / len(sensitive_ratio_values)),
                'active_param_num_mean': float(sum(active_param_values) / len(active_param_values)),
            }
        )
        logger.add_scalars_dict(prefix='train', dic=mean_train_variables, rnd=i)
        fedmini_log_scalars = {
            key: value for key, value in group_stats.items()
            if isinstance(value, (int, float)) and value is not None
        }
        fedmini_log_scalars['cumulative_trans_cost_mb'] = cumulative_trans_cost_mb
        logger.add_scalars_dict(prefix='fedmini', dic=fedmini_log_scalars, rnd=i)

        after_test_result = None
        if i % args.other.test_freq == 0 and "after_aggregation" in args.learn.test_place:
            test_monitor = VariableMonitor()
            test_results = launcher.launch(
                clients=[group.clients[j] for j in range(args.client.client_num)],
                task_name='test'
            )
            for item in test_results:
                test_monitor.append(item)
            after_test_result = test_monitor.variable_mean()
            logger.add_scalars_dict(prefix='after_aggregation_test', dic=after_test_result, rnd=i)
            torch.save(group.server.glob_dict, os.path.join(args.other.logging_path, 'model.ckpt'))
        round_record = dict(
            round=i,
            lr=cur_lr,
            stage=round_args['stage'],
            group_index=round_args['group_index'],
            train=mean_train_variables,
            before_aggregation_test=before_test_result,
            after_aggregation_test=after_test_result,
            group_stats=group_stats,
            cumulative_trans_cost_mb=cumulative_trans_cost_mb,
            participated_client_num=len(participated_clients),
            participated_clients=list(participated_clients),
            sensitive_ratio_mean=float(sum(sensitive_ratio_values) / len(sensitive_ratio_values)),
            sensitive_ratio_std=float(torch.tensor(sensitive_ratio_values).std(unbiased=False).item()),
            active_param_num_mean=float(sum(active_param_values) / len(active_param_values)),
        )
        round_records.append(round_record)
        _append_jsonl(round_metrics_path, round_record)
        if group.last_freeze_event is not None:
            freeze_event_record = dict(round=i, **group.last_freeze_event)
            freeze_event_records.append(freeze_event_record)
            _append_jsonl(freeze_events_path, freeze_event_record)
        executed_rounds += 1

    _write_csv(round_csv_path, round_records)
    best_after_acc = None
    best_after_round = None
    for record in round_records:
        after_test = record.get('after_aggregation_test')
        if after_test is None:
            continue
        acc = after_test.get('test_acc')
        if acc is None:
            continue
        if best_after_acc is None or acc > best_after_acc:
            best_after_acc = acc
            best_after_round = record['round']
    summary = dict(
        seed=seed,
        executed_rounds=executed_rounds,
        configured_rounds=int(args.learn.global_eps),
        best_after_aggregation_test_acc=best_after_acc,
        best_after_aggregation_test_round=best_after_round,
        final_after_aggregation_test=round_records[-1].get('after_aggregation_test') if len(round_records) > 0 else None,
        total_trans_cost_mb=cumulative_trans_cost_mb,
        freeze_events=freeze_event_records,
        finished_all_frozen=all(group.frozen_groups),
        end_time=time.strftime('%Y-%m-%d %H:%M:%S', time.localtime()),
    )
    _write_json(summary_path, summary)
