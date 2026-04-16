import torch

from fling.component.server import ServerTemplate


def fed_generalized_aggregation(clients: list, server: ServerTemplate, alpha: float = 1.5, max_iter: int = 100, tol: float = 1e-6) -> int:
    r"""
    Overview:
        Generalized aggregation method based on the alpha-th power distance minimization.
        Solves: W* = argmin_W sum_i ||W - W_i||_2^alpha via fixed-point iteration:
            W_{t+1} = sum_i (||W_i - W_t||_2^{alpha-2} * W_i) / sum_i (||W_i - W_t||_2^{alpha-2})
        Special cases:
            - alpha=2: equivalent to FedAvg
            - alpha=1: geometric median (Weiszfeld iteration)
    Arguments:
        clients: a list of clients that is needed to be aggregated in this round.
        server: The parameter server of these clients.
        alpha: The exponent in the generalized distance. Default is 1.5.
        max_iter: Maximum number of fixed-point iterations. Default is 100.
        tol: Convergence tolerance. Default is 1e-6.
    Returns:
        trans_cost: the total uplink cost in this communication round.
    """
    eps = 1e-10
    fed_keys = clients[0].fed_keys
    device = clients[0].args.learn.device

    client_params = []
    for client in clients:
        sd = client.model.state_dict()
        flat = torch.cat([sd[k].float().reshape(-1).to(device) for k in fed_keys])
        client_params.append(flat)

    total_samples = sum([client.sample_num for client in clients])
    sample_weights = torch.tensor([client.sample_num / total_samples for client in clients], device=device)

    w_t = sum(sample_weights[i] * client_params[i] for i in range(len(clients)))

    for _ in range(max_iter):
        distances = torch.stack([torch.norm(client_params[i] - w_t, p=2) for i in range(len(clients))])
        weights = sample_weights * (distances + eps).pow(alpha - 2)
        weights = weights / weights.sum()
        w_new = sum(weights[i] * client_params[i] for i in range(len(clients)))

        if torch.norm(w_new - w_t) < tol:
            w_t = w_new
            break
        w_t = w_new

    offset = 0
    glob_dict = {}
    ref_sd = clients[0].model.state_dict()
    for k in fed_keys:
        numel = ref_sd[k].numel()
        glob_dict[k] = w_t[offset:offset + numel].reshape(ref_sd[k].shape).to(ref_sd[k].dtype)
        offset += numel
    server.glob_dict = glob_dict

    trans_cost = 0
    for k in fed_keys:
        trans_cost += len(clients) * ref_sd[k].numel()
    return 4 * trans_cost
