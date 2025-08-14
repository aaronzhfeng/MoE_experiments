from __future__ import annotations

from typing import Dict, List, Tuple

import torch
import torch.nn as nn

from hetero_moe.models.gating.router import Router


class MoEModel(nn.Module):
    def __init__(self, experts: Dict[str, nn.Module], gate_feature_dim: int, balance_lambda: float = 0.01, top_k: int = 1):
        super().__init__()
        self.expert_names: List[str] = list(experts.keys())
        self.experts = nn.ModuleDict(experts)
        self.router = Router(num_experts=len(self.expert_names), gate_feature_dim=gate_feature_dim, top_k=top_k)
        self.balance_lambda = balance_lambda

    def forward(self, batch: Dict) -> Tuple[torch.Tensor, Dict]:
        # Expect batch to contain 'gate_features' [B, D] and expert-specific inputs
        route = self.router(batch["gate_features"])  # top-1 or top-k
        if len(route) == 3 and route[2] is not None:
            assignments, probs, (topk_idx, topk_probs) = route
        else:
            assignments, probs = route[0], route[1]
            topk_idx = None
            topk_probs = None

        total_loss = torch.tensor(0.0, device=assignments.device)
        aux = {"assignments": assignments, "probs": probs, "topk": (topk_idx, topk_probs)}

        batch_size = assignments.shape[0]

        # Route indices per expert
        for idx, name in enumerate(self.expert_names):
            mask = (assignments == idx)
            if not torch.any(mask):
                continue
            # subset tensors (first-dimension equals batch) and also list/tuple fields of batch-size length
            selected_indices = torch.nonzero(mask, as_tuple=False).squeeze(-1).tolist()
            expert_batch: Dict = {}
            for k, v in batch.items():
                if torch.is_tensor(v) and v.shape[0] == batch_size:
                    expert_batch[k] = v[mask]
                elif isinstance(v, list) and len(v) == batch_size:
                    expert_batch[k] = [v[i] for i in selected_indices]
                elif isinstance(v, tuple) and len(v) == batch_size:
                    expert_batch[k] = tuple(v[i] for i in selected_indices)
                else:
                    expert_batch[k] = v
            outputs = self.experts[name](expert_batch)
            # Expect outputs to include 'loss'
            total_loss = total_loss + outputs["loss"]

        # Add load-balance loss if provided in batch or compute externally
        if "balance_loss" in batch:
            total_loss = total_loss + self.balance_lambda * batch["balance_loss"]

        return total_loss, aux

    @torch.no_grad()
    def predict_logits(self, batch: Dict) -> torch.Tensor:
        """Run gating and return per-sample logits [B, T, V] from assigned experts.

        Assumes each expert forward returns a dict with 'logits' shaped [b_i, T, V].
        """
        assignments, probs = self.router(batch["gate_features"])  # [B], [B, N]
        batch_size = assignments.shape[0]
        # infer time dimension and vocab from one expert by a dry forward on the whole batch
        # Here we do per-expert and assemble
        logits_out: torch.Tensor = None
        for idx, name in enumerate(self.expert_names):
            mask = (assignments == idx)
            if not torch.any(mask):
                continue
            selected_indices = torch.nonzero(mask, as_tuple=False).squeeze(-1).tolist()
            expert_batch: Dict = {}
            for k, v in batch.items():
                if torch.is_tensor(v) and v.shape[0] == batch_size:
                    expert_batch[k] = v[mask]
                elif isinstance(v, list) and len(v) == batch_size:
                    expert_batch[k] = [v[i] for i in selected_indices]
                elif isinstance(v, tuple) and len(v) == batch_size:
                    expert_batch[k] = tuple(v[i] for i in selected_indices)
                else:
                    expert_batch[k] = v
            out = self.experts[name](expert_batch)
            part = out.get("logits")
            if logits_out is None:
                # allocate
                device = part.device
                b, t, v = batch["target_ids"].shape[0], part.shape[1], part.shape[2]
                logits_out = torch.zeros((b, t, v), device=device, dtype=part.dtype)
            logits_out[mask] = part
        return logits_out


