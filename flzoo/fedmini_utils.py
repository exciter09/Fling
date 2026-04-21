import os
from easydict import EasyDict


DATASET_META = {
    'cifar100': dict(data_path='./data/CIFAR100', class_number=100, pathological_alpha=10),
    'tiny_imagenet': dict(data_path='./data/tiny-imagenet-200', class_number=200, pathological_alpha=20),
}


def _format_split_value(split_value) -> str:
    if isinstance(split_value, float):
        if split_value.is_integer():
            split_value = int(split_value)
        else:
            return str(split_value).replace('.', 'p')
    return str(split_value)


def build_fedmini_paper_exp_args(
        dataset: str,
        split_mode: str = 'dirichlet',
        split_value=None,
        device: str = 'cuda:0',
        logging_root: str = './logging/fedmini_paper',
        client_num: int = 50,
        sample_rate: float = 1.0,
        global_eps: int = 300,
        local_eps: int = 5,
        batch_size: int = 32,
        lr: float = 0.01,
        test_freq: int = 1,
        seed_for_path: int = 0,
        num_workers: int = 8,
        use_amp: bool = True,
        amp_dtype: str = 'float16',
        sensitivity_weight: float = 0.7,
        full_update_rounds: int = 5,
        rounds_per_group: int = 2,
        warmup_rounds: int = 94,
        freeze_threshold: float = 0.2,
        freeze_ema: float = 0.5,
        freeze_max_rounds: int = 10,
) -> EasyDict:
    if dataset not in DATASET_META:
        raise ValueError(f'Unsupported FedMini paper dataset: {dataset}')
    if split_mode not in ['dirichlet', 'pathological']:
        raise ValueError(f'Unsupported FedMini split mode: {split_mode}')

    dataset_meta = DATASET_META[dataset]
    if split_value is None:
        split_value = 0.1 if split_mode == 'dirichlet' else dataset_meta['pathological_alpha']

    pin_memory = str(device).startswith('cuda')
    logging_path = os.path.join(
        logging_root,
        dataset,
        split_mode,
        f'value_{_format_split_value(split_value)}',
        f'seed_{seed_for_path}'
    )

    sample_method = dict(name=split_mode, train_num=500, test_num=100)
    sample_method['alpha'] = float(split_value) if split_mode == 'dirichlet' else int(split_value)

    exp_args = dict(
        data=dict(
            dataset=dataset,
            data_path=dataset_meta['data_path'],
            sample_method=sample_method,
        ),
        learn=dict(
            device=device,
            local_eps=local_eps,
            global_eps=global_eps,
            batch_size=batch_size,
            optimizer=dict(name='sgd', lr=lr, momentum=0.9),
            scheduler=dict(name='fix'),
            test_place=['after_aggregation', 'before_aggregation'],
            use_amp=use_amp,
            amp_dtype=amp_dtype,
            cudnn_benchmark=True,
            sensitivity_weight=sensitivity_weight,
            sensitive_ratio_min=0.3,
            sensitive_ratio_max=0.5,
            sensitive_decay_rate=2.0,
            collaboration_decay_rate=2.0,
            full_update_rounds=full_update_rounds,
            rounds_per_group=rounds_per_group,
            warmup_rounds=warmup_rounds,
            freeze_threshold=freeze_threshold,
            freeze_ema=freeze_ema,
            freeze_max_rounds=freeze_max_rounds,
            freeze_eps=1e-8,
        ),
        model=dict(
            name='resnet18',
            input_channel=3,
            class_number=dataset_meta['class_number'],
        ),
        client=dict(
            name='fedmini_client',
            client_num=client_num,
            sample_rate=sample_rate,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=(num_workers > 0),
            prefetch_factor=4,
        ),
        server=dict(name='base_server'),
        group=dict(
            name='fedmini_group',
            aggregation_method='avg',
            aggregation_parameters=dict(name='all'),
            include_non_param=False,
        ),
        launcher=dict(name='serial'),
        other=dict(
            test_freq=test_freq,
            logging_path=logging_path,
            progress_bar=True,
            experiment_name=f'{dataset}:{split_mode}:{_format_split_value(split_value)}:seed{seed_for_path}',
        )
    )
    return EasyDict(exp_args)
