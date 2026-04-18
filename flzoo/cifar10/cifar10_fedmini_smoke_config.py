import os
import sys

from easydict import EasyDict

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

exp_args = dict(
    data=dict(
        dataset='cifar10',
        data_path='./data/CIFAR10',
        sample_method=dict(name='dirichlet', alpha=0.3, train_num=80, test_num=40)
    ),
    learn=dict(
        device='mps',
        local_eps=1,
        global_eps=8,
        batch_size=16,
        optimizer=dict(name='sgd', lr=0.01, momentum=0.9),
        scheduler=dict(name='fix'),
        test_place=['after_aggregation'],
        sensitivity_weight=0.7,
        sensitive_ratio_min=0.3,
        sensitive_ratio_max=0.5,
        sensitive_decay_rate=2.0,
        collaboration_decay_rate=2.0,
        full_update_rounds=1,
        rounds_per_group=1,
        warmup_rounds=4,
        freeze_threshold=0.2,
        freeze_ema=0.5,
        freeze_max_rounds=2,
        freeze_eps=1e-8,
    ),
    model=dict(
        name='resnet8',
        input_channel=3,
        class_number=10,
    ),
    client=dict(name='fedmini_client', client_num=4, sample_rate=1.0),
    server=dict(name='base_server'),
    group=dict(
        name='fedmini_group',
        aggregation_method='avg',
        aggregation_parameters=dict(name='all'),
        include_non_param=False,
    ),
    other=dict(test_freq=1, logging_path='./logging/cifar10_fedmini_smoke')
)

exp_args = EasyDict(exp_args)

if __name__ == '__main__':
    from fling.pipeline import fedmini_pipeline

    fedmini_pipeline(exp_args, seed=0)
