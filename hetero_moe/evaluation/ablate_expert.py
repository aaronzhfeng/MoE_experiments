from __future__ import annotations

import argparse
import json
from typing import Dict

import torch
from torch.utils.data import DataLoader

from hetero_moe.data.dataset import USPTODataset
from hetero_moe.data.dataloader import collate_seq_batch
from hetero_moe.models.moe import MoEModel
from hetero_moe.models.experts.smiles_expert import SmilesExpert
from hetero_moe.models.experts.graph_expert import GraphExpert
from hetero_moe.models.experts.cond_expert import ConditionExpert
from hetero_moe.models.experts.gnn3d_expert import GNN3DExpert


def parse_args():
    p = argparse.ArgumentParser("ablate_expert")
    p.add_argument("--test_bin", required=True)
    p.add_argument("--disable", required=True, choices=["smiles", "graph", "cond", "gnn3d"]) 
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def main():
    args = parse_args()
    device = args.device

    ds = USPTODataset(args.test_bin)
    loader = DataLoader(ds, batch_size=8, shuffle=False, collate_fn=collate_seq_batch)

    experts = {
        "smiles": SmilesExpert(),
        "graph": GraphExpert(),
        "cond": ConditionExpert(),
        "gnn3d": GNN3DExpert(),
    }
    if args.disable in experts:
        del experts[args.disable]

    model = MoEModel(experts=experts, gate_feature_dim=2048).to(device)
    model.eval()

    total = 0.0
    iters = 0
    with torch.no_grad():
        for batch in loader:
            batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
            loss, _ = model(batch)
            total += float(loss.detach().cpu())
            iters += 1
    print(f"avg_loss_no_{args.disable}: {total / max(1, iters):.4f}")


if __name__ == "__main__":
    main()


