"""
Config generator for the generalized aggregation experiments.
Generates all experiment configs described in idea_doc.md.

Dimensions:
    - Dataset: CIFAR-10, CIFAR-100
    - Model: ResNet-18, CNN
    - Client number: 10, 50, 100
    - Data split: IID, Dirichlet beta in {0.1, 0.5, 1.0, 5.0}
    - Aggregation: FedAvg, Generalized (alpha in {1.0, 1.2, 1.5, 1.8}), FedMedian, FedProx
"""
import os
import itertools

DATASETS = {
    'cifar10': dict(dataset='cifar10', data_path='./data/CIFAR10', class_number=10),
    'cifar100': dict(dataset='cifar100', data_path='./data/CIFAR100', class_number=100),
}

MODELS = {
    'resnet18': dict(name='resnet18', input_channel=3),
    'cnn': dict(name='cnn', input_channel=3),
}

CLIENT_NUMS = [10, 50, 100]

SAMPLE_METHODS = {
    'iid': dict(name='iid', train_num=500, test_num=100),
    'dir_0.1': dict(name='dirichlet', alpha=0.1, train_num=500, test_num=100),
    'dir_0.5': dict(name='dirichlet', alpha=0.5, train_num=500, test_num=100),
    'dir_1.0': dict(name='dirichlet', alpha=1.0, train_num=500, test_num=100),
    'dir_5.0': dict(name='dirichlet', alpha=5.0, train_num=500, test_num=100),
}

AGGREGATION_METHODS = {
    'fedavg': dict(
        aggregation_method='avg',
        client_name='base_client',
        pipeline='generic_model_pipeline',
    ),
    'generalized_1.0': dict(
        aggregation_method='generalized',
        aggregation_alpha=1.0,
        client_name='base_client',
        pipeline='generic_model_pipeline',
    ),
    'generalized_1.2': dict(
        aggregation_method='generalized',
        aggregation_alpha=1.2,
        client_name='base_client',
        pipeline='generic_model_pipeline',
    ),
    'generalized_1.5': dict(
        aggregation_method='generalized',
        aggregation_alpha=1.5,
        client_name='base_client',
        pipeline='generic_model_pipeline',
    ),
    'generalized_1.8': dict(
        aggregation_method='generalized',
        aggregation_alpha=1.8,
        client_name='base_client',
        pipeline='generic_model_pipeline',
    ),
    'fedmedian': dict(
        aggregation_method='median',
        client_name='base_client',
        pipeline='generic_model_pipeline',
    ),
    'fedprox': dict(
        aggregation_method='avg',
        client_name='fedprox_client',
        mu=0.01,
        pipeline='generic_model_pipeline',
    ),
}


def generate_config_string(dataset_key, model_key, client_num, sample_key, aggr_key):
    ds = DATASETS[dataset_key]
    model = MODELS[model_key]
    sample = SAMPLE_METHODS[sample_key]
    aggr = AGGREGATION_METHODS[aggr_key]

    exp_name = f'{dataset_key}_{aggr_key}_{model_key}_{sample_key}_c{client_num}'
    logging_path = f'./logging/experimental/{exp_name}'

    group_dict = f"aggregation_method='{aggr['aggregation_method']}'"
    if 'aggregation_alpha' in aggr:
        group_dict += f", aggregation_alpha={aggr['aggregation_alpha']}"

    learn_extra = ''
    if 'mu' in aggr:
        learn_extra = f'\n        mu={aggr["mu"]},'

    lines = f"""from easydict import EasyDict

exp_args = dict(
    data=dict(
        dataset='{ds["dataset"]}',
        data_path='{ds["data_path"]}',
        sample_method={repr(sample)},
    ),
    learn=dict(
        device='cuda:0',
        local_eps=8,
        global_eps=40,
        batch_size=32,
        optimizer=dict(name='sgd', lr=0.02, momentum=0.9),{learn_extra}
    ),
    model=dict(
        name='{model["name"]}',
        input_channel={model["input_channel"]},
        class_number={ds["class_number"]},
    ),
    client=dict(name='{aggr["client_name"]}', client_num={client_num}),
    server=dict(name='base_server'),
    group=dict(name='base_group', {group_dict}),
    other=dict(test_freq=3, logging_path='{logging_path}'),
)

exp_args = EasyDict(exp_args)

if __name__ == '__main__':
    from fling.pipeline import {aggr["pipeline"]}
    {aggr["pipeline"]}(exp_args, seed=0)
"""
    return exp_name, lines


def main():
    output_dir = os.path.dirname(os.path.abspath(__file__))
    count = 0
    for ds_key, model_key, client_num, sample_key, aggr_key in itertools.product(
        DATASETS.keys(), MODELS.keys(), CLIENT_NUMS, SAMPLE_METHODS.keys(), AGGREGATION_METHODS.keys()
    ):
        exp_name, config_str = generate_config_string(ds_key, model_key, client_num, sample_key, aggr_key)
        filepath = os.path.join(output_dir, f'{exp_name}_config.py')
        with open(filepath, 'w') as f:
            f.write(config_str)
        count += 1
    print(f'Generated {count} config files in {output_dir}')


if __name__ == '__main__':
    main()
