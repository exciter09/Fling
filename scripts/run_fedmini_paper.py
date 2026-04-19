import argparse
import os
import sys
import warnings
from copy import deepcopy

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

warnings.filterwarnings(
    'ignore',
    message='Importing from timm.models.layers is deprecated, please import via timm.layers',
    category=FutureWarning
)

from tqdm import tqdm

from fling.pipeline import fedmini_pipeline
from flzoo.fedmini_utils import DATASET_META, build_fedmini_paper_exp_args


def build_run_plan(args):
    datasets = list(DATASET_META.keys()) if args.datasets == ['all'] else args.datasets
    scenarios = ['dirichlet', 'pathological'] if args.scenarios == ['all'] else args.scenarios
    run_plan = []
    for dataset in datasets:
        for scenario in scenarios:
            if scenario == 'dirichlet':
                for alpha in args.dirichlet_alphas:
                    for seed in args.seeds:
                        run_plan.append((dataset, scenario, alpha, seed))
            else:
                split_value = args.pathological_split[dataset]
                for seed in args.seeds:
                    run_plan.append((dataset, scenario, split_value, seed))
    return run_plan


def parse_args():
    parser = argparse.ArgumentParser(description='Run FedMini paper experiments on a single GPU machine.')
    parser.add_argument(
        '--datasets',
        nargs='+',
        default=['all'],
        choices=['all', 'cifar100', 'tiny_imagenet'],
        help='Datasets to run.'
    )
    parser.add_argument(
        '--scenarios',
        nargs='+',
        default=['all'],
        choices=['all', 'dirichlet', 'pathological'],
        help='Non-IID settings to run.'
    )
    parser.add_argument('--dirichlet-alphas', nargs='+', type=float, default=[0.1, 0.3, 0.5])
    parser.add_argument('--seeds', nargs='+', type=int, default=[0, 1, 2])
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--logging-root', type=str, default='./logging/fedmini_paper')
    parser.add_argument('--num-workers', type=int, default=8)
    parser.add_argument('--test-freq', type=int, default=1)
    parser.add_argument('--global-eps', type=int, default=300)
    parser.add_argument('--local-eps', type=int, default=5)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--amp-dtype', type=str, default='float16', choices=['float16', 'bfloat16'])
    parser.add_argument('--no-amp', action='store_true')
    parser.add_argument('--no-progress', action='store_true')
    parser.add_argument('--dry-run', action='store_true')
    parsed = parser.parse_args()
    parsed.pathological_split = {
        'cifar100': DATASET_META['cifar100']['pathological_alpha'],
        'tiny_imagenet': DATASET_META['tiny_imagenet']['pathological_alpha'],
    }
    return parsed


def main():
    args = parse_args()
    run_plan = build_run_plan(args)
    print(f'FedMini run plan size: {len(run_plan)}')
    run_progress = tqdm(total=len(run_plan), desc='FedMini runs', dynamic_ncols=True, disable=args.no_progress)
    try:
        for idx, (dataset, scenario, split_value, seed) in enumerate(run_plan, start=1):
            run_message = (
                f'[{idx}/{len(run_plan)}] dataset={dataset} scenario={scenario} split_value={split_value} '
                f'seed={seed} device={args.device}'
            )
            if args.no_progress:
                print(run_message)
            else:
                tqdm.write(run_message)
                run_progress.set_postfix(
                    dataset=dataset,
                    scenario=scenario,
                    split=split_value,
                    seed=seed
                )
                run_progress.refresh()
            if args.dry_run:
                run_progress.update(1)
                continue
            exp_args = build_fedmini_paper_exp_args(
                dataset=dataset,
                split_mode=scenario,
                split_value=split_value,
                device=args.device,
                logging_root=args.logging_root,
                seed_for_path=seed,
                num_workers=args.num_workers,
                test_freq=args.test_freq,
                global_eps=args.global_eps,
                local_eps=args.local_eps,
                batch_size=args.batch_size,
                lr=args.lr,
                use_amp=(not args.no_amp),
                amp_dtype=args.amp_dtype,
            )
            exp_args.other.progress_bar = not args.no_progress
            exp_args.other.experiment_name = f'{dataset}:{scenario}:{split_value}:seed{seed}'
            fedmini_pipeline(args=deepcopy(exp_args), seed=seed)
            run_progress.update(1)
    finally:
        run_progress.close()


if __name__ == '__main__':
    main()
