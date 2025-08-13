from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class Router(nn.Module):
    def __init__(self, num_experts: int, gate_feature_dim: int = 2048, hidden_dim: int = 256, top_k: int = 1,
                 temperature: float = 1.0, gumbel_noise: bool = False):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = max(1, top_k)
        self.temperature = max(1e-6, float(temperature))
        self.gumbel_noise = gumbel_noise
        self.ff = nn.Sequential(
            nn.Linear(gate_feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_experts),
        )

    def forward(self, gate_features: torch.Tensor):
        # gate_features: [B, D]
        logits = self.ff(gate_features)
        if self.gumbel_noise and self.training:
            # sample Gumbel(0,1) noise
            eps = 1e-9
            u = torch.rand_like(logits).clamp_(eps, 1.0 - eps)
            g = -torch.log(-torch.log(u))
            logits = logits + g
        probs = F.softmax(logits / self.temperature, dim=-1)
        if self.top_k == 1:
            assignments = torch.argmax(probs, dim=-1)
            return assignments, probs, None
        else:
            topk_probs, topk_idx = torch.topk(probs, k=self.top_k, dim=-1)
            assignments = topk_idx[:, 0]
            return assignments, probs, (topk_idx, topk_probs)


