from __future__ import annotations

import torch


def load_balance_loss(router_probs: torch.Tensor) -> torch.Tensor:
    """Encourage uniform expert usage.

    router_probs: [B, N]
    Returns scalar loss.
    """
    if router_probs.numel() == 0:
        return torch.tensor(0.0, device=router_probs.device)
    # mean prob per expert
    mean_per_expert = router_probs.mean(dim=0)  # [N]
    n = mean_per_expert.shape[0]
    target = torch.full_like(mean_per_expert, 1.0 / n)
    return torch.mean((mean_per_expert - target) ** 2)


