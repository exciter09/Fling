import copy
import math

import torch
import torch.nn as nn

from fling.utils import VariableMonitor, get_optimizer, get_weights
from fling.utils.registry_utils import CLIENT_REGISTRY
from .base_client import BaseClient


@CLIENT_REGISTRY.register('fedmini_client')
class FedMiniClient(BaseClient):
    """
    Overview:
        Minimal FedMini client implementation for parameter-wise sensitivity estimation.
    """

    def __init__(self, args, client_id, train_dataset, test_dataset=None):
        super(FedMiniClient, self).__init__(args, client_id, train_dataset, test_dataset)
        self.sensitive_mask = {}
        self.local_update = {}
        self.active_parameter_names = []
        self.round_statistics = {}

    def _get_amp_kwargs(self):
        use_amp = bool(getattr(self.args.learn, 'use_amp', False)) and str(self.device).startswith('cuda')
        amp_dtype_name = str(getattr(self.args.learn, 'amp_dtype', 'float16')).lower()
        amp_dtype = torch.bfloat16 if amp_dtype_name == 'bfloat16' else torch.float16
        return use_amp, amp_dtype

    def train(self, lr, device=None, train_args=None):
        if device is not None:
            device_bak = self.device
            self.device = device

        self.model.train()
        self.model.to(self.device)

        if train_args is None:
            active_parameter_dict = dict(self.model.named_parameters())
        else:
            active_parameter_dict = get_weights(self.model, parameter_args=train_args, return_dict=True)
        self.active_parameter_names = list(active_parameter_dict.keys())

        optimizer = get_optimizer(weights=active_parameter_dict.values(), **self.args.learn.optimizer)
        criterion = nn.CrossEntropyLoss()
        monitor = VariableMonitor()
        use_amp, amp_dtype = self._get_amp_kwargs()
        scaler = torch.amp.GradScaler('cuda', enabled=use_amp)

        prev_state = {
            name: param.detach().clone().to(self.device) for name, param in self.model.named_parameters()
            if name in self.active_parameter_names
        }
        fisher_diag = {
            name: torch.zeros_like(param, device=self.device) for name, param in active_parameter_dict.items()
        }
        batch_count = 0

        for _ in range(self.args.learn.local_eps):
            for _, data in enumerate(self.train_dataloader):
                preprocessed_data = self.preprocess_data(data)
                batch_x, batch_y = preprocessed_data['x'], preprocessed_data['y']
                with torch.amp.autocast(device_type='cuda', dtype=amp_dtype, enabled=use_amp):
                    o = self.model(batch_x)
                    loss = criterion(o, batch_y)
                y_pred = torch.argmax(o, dim=-1)

                monitor.append(
                    {
                        'train_acc': torch.mean((y_pred == batch_y).float()).item(),
                        'train_loss': loss.item()
                    },
                    weight=batch_y.shape[0]
                )

                optimizer.zero_grad(set_to_none=True)
                if use_amp:
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                else:
                    loss.backward()

                for name, param in active_parameter_dict.items():
                    if param.grad is not None:
                        fisher_diag[name] += param.grad.detach() ** 2

                if use_amp:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                batch_count += 1

        batch_count = max(batch_count, 1)
        sensitivity_weight = getattr(self.args.learn, 'sensitivity_weight', 0.7)
        beta_min = getattr(self.args.learn, 'sensitive_ratio_min', 0.3)
        beta_max = getattr(self.args.learn, 'sensitive_ratio_max', 0.5)
        decay_rate = getattr(self.args.learn, 'sensitive_decay_rate', 2.0)
        total_rounds = max(int(self.args.learn.global_eps), 1)
        cur_round = getattr(train_args, 'train_round', 0) if train_args is not None else 0
        beta_t = beta_max - (beta_max - beta_min) * math.exp(-decay_rate * float(cur_round + 1) / total_rounds)

        self.sensitive_mask = {}
        self.local_update = {}
        active_param_num = 0
        sensitive_param_num = 0
        for name, param in self.model.named_parameters():
            if name not in self.active_parameter_names:
                self.sensitive_mask[name] = torch.zeros_like(param, dtype=torch.int32, device='cpu')
                self.local_update[name] = torch.zeros_like(param, device='cpu')
                continue

            delta = param.detach() - prev_state[name]
            first_order = torch.abs(delta * param.detach())
            fisher = fisher_diag[name] / batch_count

            first_flat = first_order.view(-1)
            fisher_flat = fisher.view(-1)
            first_norm = (first_flat - first_flat.mean()) / (first_flat.std(unbiased=False) + 1e-12)
            fisher_norm = (fisher_flat - fisher_flat.mean()) / (fisher_flat.std(unbiased=False) + 1e-12)
            score = sensitivity_weight * first_norm + (1 - sensitivity_weight) * fisher_norm

            topk = min(score.numel(), max(1, int(beta_t * score.numel())))
            mask_flat = torch.zeros_like(score, dtype=torch.int32)
            if topk > 0:
                _, top_indices = torch.topk(score, k=topk)
                mask_flat[top_indices] = 1

            self.sensitive_mask[name] = mask_flat.view_as(param).detach().to('cpu')
            self.local_update[name] = delta.detach().to('cpu')
            active_param_num += mask_flat.numel()
            sensitive_param_num += int(mask_flat.sum().item())

        self.round_statistics = dict(
            active_param_num=active_param_num,
            sensitive_param_num=sensitive_param_num,
            sensitive_ratio=(sensitive_param_num / active_param_num) if active_param_num > 0 else 0.0,
            active_layer_count=len(self.active_parameter_names),
            use_amp=float(use_amp),
        )

        mean_monitor_variables = monitor.variable_mean()
        self.model.to('cpu')

        if device is not None:
            self.device = device_bak

        return mean_monitor_variables
