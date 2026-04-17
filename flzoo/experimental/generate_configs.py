"""
Generate the 30 focused experiment configs.

Dimensions (fixed: resnet8, 10 clients):
    Dataset: cifar10, cifar100
    Data split: iid, dirichlet(0.1), dirichlet(0.5)
    Aggregation: fedavg, geomed(alpha=1.0), gen15(alpha=1.5), fedmedian, fedprox
"""
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

DATASETS = {
    'cifar10': dict(dataset='cifar10', data_path='./data/CIFAR10', class_number=10),
    'cifar100': dict(dataset='cifar100', data_path='./data/CIFAR100', class_number=100),
}

SPLITS = {
    'iid': "dict(name='iid', train_num=500, test_num=100)",
    'dir01': "dict(name='dirichlet', alpha=0.1, train_num=500, test_num=100)",
    'dir05': "dict(name='dirichlet', alpha=0.5, train_num=500, test_num=100)",
}

METHODS = {
    'fedavg': dict(
        client_name='base_client',
        group_line="aggregation_method='avg', aggregation_parameters=dict(name='except', keywords=['fc'])",
        learn_extra='\n        finetune_parameters=dict(name="contain", keywords=["fc"]),',
    ),
    'geomed': dict(
        client_name='base_client',
        group_line="aggregation_method='generalized', aggregation_alpha=1.0, aggregation_parameters=dict(name='except', keywords=['fc'])",
        learn_extra='\n        finetune_parameters=dict(name="contain", keywords=["fc"]),',
    ),
    'gen15': dict(
        client_name='base_client',
        group_line="aggregation_method='generalized', aggregation_alpha=1.5, aggregation_parameters=dict(name='except', keywords=['fc'])",
        learn_extra='\n        finetune_parameters=dict(name="contain", keywords=["fc"]),',
    ),
    'fedmedian': dict(
        client_name='base_client',
        group_line="aggregation_method='median', aggregation_parameters=dict(name='except', keywords=['fc'])",
        learn_extra='\n        finetune_parameters=dict(name="contain", keywords=["fc"]),',
    ),
    'fedprox': dict(
        client_name='fedprox_client',
        group_line="aggregation_method='avg', aggregation_parameters=dict(name='except', keywords=['fc'])",
        learn_extra='\n        mu=0.01,\n        finetune_parameters=dict(name="contain", keywords=["fc"]),',
    ),
}

TEMPLATE = """from easydict import EasyDict

exp_args = dict(
    data=dict(
        dataset='{dataset}',
        data_path='{data_path}',
        sample_method={sample_method},
    ),
    learn=dict(
        device='cuda:0',
        local_eps=5,
        global_eps=300,
        batch_size=100,
        optimizer=dict(name='sgd', lr=0.1, momentum=0.9),{learn_extra}
    ),
    model=dict(
        name='resnet8',
        input_channel=3,
        class_number={class_number},
    ),
    client=dict(name='{client_name}', client_num=30),
    server=dict(name='base_server'),
    group=dict(name='base_group', {group_line}),
    other=dict(test_freq=3, logging_path='./logging/experimental/{exp_name}'),
)

exp_args = EasyDict(exp_args)

if __name__ == '__main__':
    from fling.pipeline import personalized_model_pipeline
    personalized_model_pipeline(exp_args, seed=0)
"""

count = 0
for ds_key, ds in DATASETS.items():
    for split_key, sample_method in SPLITS.items():
        for method_key, method in METHODS.items():
            exp_name = f'{ds_key}_{method_key}_{split_key}'
            content = TEMPLATE.format(
                dataset=ds['dataset'],
                data_path=ds['data_path'],
                class_number=ds['class_number'],
                sample_method=sample_method,
                learn_extra=method['learn_extra'],
                client_name=method['client_name'],
                group_line=method['group_line'],
                exp_name=exp_name,
            )
            filepath = os.path.join(SCRIPT_DIR, f'{exp_name}_config.py')
            with open(filepath, 'w') as f:
                f.write(content)
            count += 1

print(f'Generated {count} config files in {SCRIPT_DIR}')
