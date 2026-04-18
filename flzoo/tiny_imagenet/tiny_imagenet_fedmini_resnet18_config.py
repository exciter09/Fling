import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from flzoo.fedmini_utils import build_fedmini_paper_exp_args

exp_args = build_fedmini_paper_exp_args(
    dataset='tiny_imagenet',
    split_mode=os.getenv('FEDMINI_SPLIT_MODE', 'dirichlet'),
    split_value=float(os.getenv('FEDMINI_SPLIT_VALUE', '0.1'))
    if os.getenv('FEDMINI_SPLIT_MODE', 'dirichlet') == 'dirichlet'
    else int(os.getenv('FEDMINI_SPLIT_VALUE', '20')),
    device=os.getenv('FEDMINI_DEVICE', 'cuda:0'),
    logging_root=os.getenv('FEDMINI_LOGGING_ROOT', './logging/fedmini_paper'),
    seed_for_path=int(os.getenv('FEDMINI_SEED', '0')),
    num_workers=int(os.getenv('FEDMINI_NUM_WORKERS', '8')),
    use_amp=(os.getenv('FEDMINI_USE_AMP', '1') == '1'),
    amp_dtype=os.getenv('FEDMINI_AMP_DTYPE', 'float16'),
)

if __name__ == '__main__':
    from fling.pipeline import fedmini_pipeline

    fedmini_pipeline(exp_args, seed=int(os.getenv('FEDMINI_SEED', '0')))
