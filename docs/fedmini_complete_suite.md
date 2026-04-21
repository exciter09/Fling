# FedMini Complete Suite

This document describes the full FedMini experiment suite that is currently supported by this repository.

## Coverage

The shell script `scripts/run_fedmini_complete_paper_suite.sh` runs:

- Main FedMini experiments for `CIFAR-100` and `Tiny-ImageNet`
- `Pathological Non-IID` and `Dirichlet Non-IID`
- `Dirichlet alpha = 0.1, 0.3, 0.5`
- `3` random seeds for the main FedMini tables
- Figure 3-9 ablation on `CIFAR-100`, `alpha=0.1`
- Figure 3-10 sensitivity-weight (`theta`) sweep on `CIFAR-100`, `alpha=0.1`

## Not Covered

The repository still does not contain:

- `FedPAC`
- `pFedCE`
- the custom `CIFAR-10` 13-client overlap experiment from Figure 3-11 / Figure 3-12

Therefore, the script reproduces the full `FedMini` suite, but not every baseline or visualization in the thesis.

## Usage

Run everything:

```bash
bash scripts/run_fedmini_complete_paper_suite.sh
```

Run only the main table experiments:

```bash
RUN_MAIN=1 RUN_ABLATION=0 RUN_THETA=0 bash scripts/run_fedmini_complete_paper_suite.sh
```

Run only ablation:

```bash
RUN_MAIN=0 RUN_ABLATION=1 RUN_THETA=0 bash scripts/run_fedmini_complete_paper_suite.sh
```

Run only theta sweep:

```bash
RUN_MAIN=0 RUN_ABLATION=0 RUN_THETA=1 bash scripts/run_fedmini_complete_paper_suite.sh
```

Override device or worker count:

```bash
DEVICE=cuda:0 NUM_WORKERS=0 bash scripts/run_fedmini_complete_paper_suite.sh
```
