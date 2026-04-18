import copy
import math
from typing import Dict, List

import numpy as np
import torch
from easydict import EasyDict

from fling.component.group import ParameterServerGroup
from fling.utils.registry_utils import GROUP_REGISTRY


def _resnet18_layer_groups() -> List[List[str]]:
    return [
        ['pre_conv', 'pre_bn'],
        ['layers.0.0.conv1', 'layers.0.0.bn1'],
        ['layers.0.0.conv2', 'layers.0.0.bn2'],
        ['layers.0.1.conv1', 'layers.0.1.bn1'],
        ['layers.0.1.conv2', 'layers.0.1.bn2'],
        ['layers.1.0.conv1', 'layers.1.0.bn1'],
        ['layers.1.0.conv2', 'layers.1.0.bn2'],
        ['layers.1.0.downsample.0', 'layers.1.0.downsample.1'],
        ['layers.1.1.conv1', 'layers.1.1.bn1'],
        ['layers.1.1.conv2', 'layers.1.1.bn2'],
        ['layers.2.0.conv1', 'layers.2.0.bn1'],
        ['layers.2.0.conv2', 'layers.2.0.bn2'],
        ['layers.2.0.downsample.0', 'layers.2.0.downsample.1'],
        ['layers.2.1.conv1', 'layers.2.1.bn1'],
        ['layers.2.1.conv2', 'layers.2.1.bn2'],
        ['layers.3.0.conv1', 'layers.3.0.bn1'],
        ['layers.3.0.conv2', 'layers.3.0.bn2'],
        ['layers.3.0.downsample.0', 'layers.3.0.downsample.1'],
        ['layers.3.1.conv1', 'layers.3.1.bn1'],
        ['layers.3.1.conv2', 'layers.3.1.bn2'],
        ['fc'],
    ]


def _resnet8_layer_groups() -> List[List[str]]:
    return [
        ['pre_conv', 'pre_bn'],
        ['layers.0.0.conv1', 'layers.0.0.bn1'],
        ['layers.0.0.conv2', 'layers.0.0.bn2'],
        ['layers.1.0.conv1', 'layers.1.0.bn1'],
        ['layers.1.0.conv2', 'layers.1.0.bn2'],
        ['layers.1.0.downsample.0', 'layers.1.0.downsample.1'],
        ['layers.2.0.conv1', 'layers.2.0.bn1'],
        ['layers.2.0.conv2', 'layers.2.0.bn2'],
        ['layers.2.0.downsample.0', 'layers.2.0.downsample.1'],
        ['fc'],
    ]


def _build_train_args(keywords: List[str], train_round: int) -> EasyDict:
    return EasyDict(name='contain', keywords=keywords, train_round=train_round)


@GROUP_REGISTRY.register('fedmini_group')
class FedMiniServerGroup(ParameterServerGroup):
    """
    Overview:
        Minimal FedMini server implementation with warmup scheduling, parameter-wise collaboration,
        and stability-driven layer freezing.
    """

    def __init__(self, args: dict, logger):
        super(FedMiniServerGroup, self).__init__(args, logger)
        self.layer_groups = self._get_layer_groups(args.model.name)
        self.warmup_rounds = int(getattr(args.learn, 'warmup_rounds', 0))
        self.full_rounds = int(getattr(args.learn, 'full_update_rounds', 5))
        self.rounds_per_group = int(getattr(args.learn, 'rounds_per_group', 2))
        self.freeze_threshold = float(getattr(args.learn, 'freeze_threshold', 0.2))
        self.freeze_ema = float(getattr(args.learn, 'freeze_ema', 0.5))
        self.freeze_max_rounds = int(getattr(args.learn, 'freeze_max_rounds', 10))
        self.freeze_decay_eps = float(getattr(args.learn, 'freeze_eps', 1e-8))
        self.collaboration_decay_rate = float(getattr(args.learn, 'collaboration_decay_rate', 2.0))
        self.frozen_groups = [False] * len(self.layer_groups)
        self.freeze_round_counts = [0] * len(self.layer_groups)
        self.freeze_ema_update = {}
        self.freeze_ema_abs = {}
        self._current_stage = 'warmup'
        self._current_group_index = -1
        self._current_active_keys = list(self.server.glob_dict.keys()) if self.server is not None else []
        self.total_param_num = 0
        self.last_round_stats = {}
        self.last_freeze_event = None

    @staticmethod
    def _get_layer_groups(model_name: str) -> List[List[str]]:
        if model_name == 'resnet18':
            return _resnet18_layer_groups()
        if model_name == 'resnet8':
            return _resnet8_layer_groups()
        raise ValueError(f'FedMini layer scheduling is only implemented for resnet8/resnet18, got: {model_name}')

    def initialize(self) -> None:
        super(FedMiniServerGroup, self).initialize()
        self.total_param_num = sum(param.numel() for _, param in self.clients[0].model.named_parameters())

    def get_round_args(self, train_round: int) -> Dict:
        if all(self.frozen_groups):
            return dict(
                train_args=None,
                aggr_args=None,
                group_index=-2,
                stage='finished',
            )

        if train_round < self.warmup_rounds:
            cycle_length = self.full_rounds + len(self.layer_groups) * self.rounds_per_group
            position = train_round % cycle_length
            self._current_stage = 'warmup'
            if position < self.full_rounds:
                self._current_group_index = -1
                self._current_active_keys = [name for name, _ in self.clients[0].model.named_parameters()]
                return dict(
                    train_args=EasyDict(name='all', train_round=train_round),
                    aggr_args=EasyDict(name='all', train_round=train_round),
                    group_index=-1,
                    stage='warmup',
                )

            group_index = (position - self.full_rounds) // self.rounds_per_group
            group_index = min(group_index, len(self.layer_groups) - 1)
            self._current_group_index = group_index
            self._current_active_keys = self._keys_from_keywords(self.layer_groups[group_index])
            args = _build_train_args(self.layer_groups[group_index], train_round)
            return dict(train_args=args, aggr_args=args, group_index=group_index, stage='warmup')

        self._current_stage = 'freeze'
        group_index = self._first_unfrozen_group()
        self._current_group_index = group_index
        self._current_active_keys = self._keys_from_keywords(self.layer_groups[group_index])
        args = _build_train_args(self.layer_groups[group_index], train_round)
        return dict(train_args=args, aggr_args=args, group_index=group_index, stage='freeze')

    def get_last_round_stats(self) -> Dict:
        return copy.deepcopy(self.last_round_stats)

    def get_metadata(self) -> Dict:
        return dict(
            layer_groups=copy.deepcopy(self.layer_groups),
            warmup_rounds=self.warmup_rounds,
            full_update_rounds=self.full_rounds,
            rounds_per_group=self.rounds_per_group,
            freeze_threshold=self.freeze_threshold,
            freeze_max_rounds=self.freeze_max_rounds,
            total_param_num=self.total_param_num,
        )

    def _first_unfrozen_group(self) -> int:
        for idx, frozen in enumerate(self.frozen_groups):
            if not frozen:
                return idx
        return -1

    def _keys_from_keywords(self, keywords: List[str]) -> List[str]:
        keys = []
        for kw in keywords:
            for name, _ in self.clients[0].model.named_parameters():
                if kw in name:
                    keys.append(name)
        return list(dict.fromkeys(keys))

    def aggregate(self, train_round: int, participate_clients_ids: list = None, aggr_parameter_args: dict = None) -> int:
        if participate_clients_ids is None:
            participate_clients_ids = list(range(self.args.client.client_num))

        active_keys = list(self._current_active_keys)
        if len(active_keys) == 0:
            return 0

        participated_clients = [self.clients[i] for i in participate_clients_ids]
        collaboration_sets, collaboration_stats = self._build_collaboration_sets(
            participate_clients_ids, active_keys, train_round
        )
        global_average = {key: self._average_tensor(participate_clients_ids, key) for key in active_keys}
        freeze_stability = None
        freeze_event = None

        for client_id in participate_clients_ids:
            client = self.clients[client_id]
            state_dict = copy.deepcopy(client.model.state_dict())
            collaborator_ids = collaboration_sets[client_id]
            for key in active_keys:
                collab_average = self._average_tensor(collaborator_ids, key)
                mask = client.sensitive_mask[key].float()
                updated = mask * collab_average + (1.0 - mask) * global_average[key]
                state_dict[key] = updated.to(state_dict[key].dtype)
            client.model.load_state_dict(state_dict)

        if self._current_stage == 'freeze' and self._current_group_index >= 0:
            freeze_stability, freeze_event = self._update_freeze_state(participate_clients_ids, active_keys)

        for key in active_keys:
            self.server.glob_dict[key] = self._average_tensor(participate_clients_ids, key).to(
                self.server.glob_dict[key].dtype
            )

        trans_cost = 0
        for key in active_keys:
            trans_cost += len(participated_clients) * self.clients[0].model.state_dict()[key].numel()
        active_param_num = sum(self.clients[0].model.state_dict()[key].numel() for key in active_keys)
        self.last_round_stats = dict(
            round=train_round,
            stage=self._current_stage,
            group_index=self._current_group_index,
            active_key_count=len(active_keys),
            active_param_num=active_param_num,
            active_param_ratio=(active_param_num / self.total_param_num) if self.total_param_num > 0 else 0.0,
            active_layers='all' if self._current_group_index == -1 else '|'.join(self.layer_groups[self._current_group_index]),
            frozen_groups=float(sum(self.frozen_groups)),
            trans_cost_mb=(4 * trans_cost) / 1e6,
            freeze_stability=freeze_stability,
            freeze_event=0.0 if freeze_event is None else 1.0,
        )
        self.last_round_stats.update(collaboration_stats)
        if freeze_event is not None:
            self.last_round_stats['freeze_group_index'] = float(freeze_event['group_index'])
            self.last_round_stats['freeze_group_name'] = '|'.join(freeze_event['keywords'])
        self.last_freeze_event = freeze_event
        return 4 * trans_cost

    def _build_collaboration_sets(self, participate_clients_ids: List[int], active_keys: List[str], train_round: int):
        if len(participate_clients_ids) <= 1:
            return {cid: [cid] for cid in participate_clients_ids}, dict(
                overlap_avg=1.0,
                overlap_p90=1.0,
                collaboration_threshold=1.0,
                collaboration_size_mean=1.0,
                collaboration_size_min=1.0,
                collaboration_size_max=1.0,
            )

        overlap_scores = {cid: {} for cid in participate_clients_ids}
        pair_values = []
        vectors = {}
        for cid in participate_clients_ids:
            vectors[cid] = torch.cat([
                self.clients[cid].sensitive_mask[key].reshape(-1).float() for key in active_keys
            ])

        for cid in participate_clients_ids:
            for oid in participate_clients_ids:
                if cid == oid:
                    continue
                overlap = self._jaccard(vectors[cid], vectors[oid])
                overlap_scores[cid][oid] = overlap
                pair_values.append(overlap)

        overlap_avg = float(np.mean(pair_values)) if len(pair_values) > 0 else 1.0
        overlap_p90 = float(np.percentile(pair_values, 90)) if len(pair_values) > 0 else overlap_avg
        total_rounds = max(int(self.args.learn.global_eps), 1)
        progress = float(train_round + 1) / total_rounds
        threshold = overlap_p90 - (overlap_p90 - overlap_avg) * math.exp(-self.collaboration_decay_rate * progress)

        collaboration_sets = {}
        collaboration_sizes = []
        for cid in participate_clients_ids:
            collaborators = [cid]
            for oid, overlap in overlap_scores[cid].items():
                if overlap >= threshold:
                    collaborators.append(oid)
            collaboration_sets[cid] = sorted(set(collaborators))
            collaboration_sizes.append(len(collaboration_sets[cid]))
        stats = dict(
            overlap_avg=overlap_avg,
            overlap_p90=overlap_p90,
            collaboration_threshold=threshold,
            collaboration_size_mean=float(np.mean(collaboration_sizes)),
            collaboration_size_min=float(np.min(collaboration_sizes)),
            collaboration_size_max=float(np.max(collaboration_sizes)),
        )
        return collaboration_sets, stats

    @staticmethod
    def _jaccard(vec_a: torch.Tensor, vec_b: torch.Tensor) -> float:
        intersection = torch.logical_and(vec_a > 0, vec_b > 0).sum().item()
        union = torch.logical_or(vec_a > 0, vec_b > 0).sum().item()
        if union == 0:
            return 1.0
        return float(intersection) / float(union)

    def _average_tensor(self, client_ids: List[int], key: str) -> torch.Tensor:
        total_samples = sum(self.clients[cid].sample_num for cid in client_ids)
        avg = None
        for cid in client_ids:
            weight = self.clients[cid].sample_num / total_samples
            tensor = self.clients[cid].model.state_dict()[key].detach().float()
            avg = tensor * weight if avg is None else avg + tensor * weight
        return avg

    def _update_freeze_state(self, participate_clients_ids: List[int], active_keys: List[str]):
        group_index = self._current_group_index
        if group_index < 0:
            return None, None

        if group_index not in self.freeze_ema_update:
            self.freeze_ema_update[group_index] = {
                key: torch.zeros_like(self.clients[participate_clients_ids[0]].local_update[key].float())
                for key in active_keys
            }
            self.freeze_ema_abs[group_index] = {
                key: torch.zeros_like(self.clients[participate_clients_ids[0]].local_update[key].float())
                for key in active_keys
            }

        stability_values = []
        for key in active_keys:
            eff_update = None
            eff_abs = None
            total_samples = sum(self.clients[cid].sample_num for cid in participate_clients_ids)
            for cid in participate_clients_ids:
                client = self.clients[cid]
                weight = client.sample_num / total_samples
                non_sensitive = 1.0 - client.sensitive_mask[key].float()
                cur_eff = client.local_update[key].float() * non_sensitive
                cur_abs = cur_eff.abs()
                eff_update = cur_eff * weight if eff_update is None else eff_update + cur_eff * weight
                eff_abs = cur_abs * weight if eff_abs is None else eff_abs + cur_abs * weight

            self.freeze_ema_update[group_index][key] = (
                self.freeze_ema * self.freeze_ema_update[group_index][key] + (1 - self.freeze_ema) * eff_update
            )
            self.freeze_ema_abs[group_index][key] = (
                self.freeze_ema * self.freeze_ema_abs[group_index][key] + (1 - self.freeze_ema) * eff_abs
            )

            numerator = self.freeze_ema_update[group_index][key].abs()
            denominator = self.freeze_ema_abs[group_index][key] + self.freeze_decay_eps
            stability_values.append((numerator / denominator).mean().item())

        stability = float(np.mean(stability_values)) if len(stability_values) > 0 else 0.0
        self.freeze_round_counts[group_index] += 1
        freeze_event = None
        if stability < self.freeze_threshold or self.freeze_round_counts[group_index] >= self.freeze_max_rounds:
            self.frozen_groups[group_index] = True
            freeze_event = dict(
                group_index=group_index,
                keywords=copy.deepcopy(self.layer_groups[group_index]),
                stability=stability,
                freeze_round_count=self.freeze_round_counts[group_index],
            )
        return stability, freeze_event
