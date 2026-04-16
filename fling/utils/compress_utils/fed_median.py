import torch

from fling.component.server import ServerTemplate


def fed_median(clients: list, server: ServerTemplate) -> int:
    r"""
    Overview:
        FedMedian aggregation: compute the coordinate-wise median of client model parameters.
        For each parameter coordinate, the median across all clients is taken. This provides
        robustness against Byzantine or outlier clients.
    Arguments:
        clients: a list of clients that is needed to be aggregated in this round.
        server: The parameter server of these clients.
    Returns:
        trans_cost: the total uplink cost in this communication round.
    """
    fed_keys = clients[0].fed_keys
    device = clients[0].args.learn.device

    glob_dict = {}
    ref_sd = clients[0].model.state_dict()
    for k in fed_keys:
        stacked = torch.stack([client.model.state_dict()[k].float().to(device) for client in clients], dim=0)
        median_val = torch.median(stacked, dim=0).values
        glob_dict[k] = median_val.to(ref_sd[k].dtype)
    server.glob_dict = glob_dict

    trans_cost = 0
    for k in fed_keys:
        trans_cost += len(clients) * ref_sd[k].numel()
    return 4 * trans_cost
