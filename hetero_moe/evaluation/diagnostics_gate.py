from __future__ import annotations

import argparse
from collections import Counter

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
    p = argparse.ArgumentParser("diagnostics_gate")
    p.add_argument("--bin", required=True)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def main():
    args = parse_args()
    device = args.device

    ds = USPTODataset(args.bin)
    loader = DataLoader(ds, batch_size=64, shuffle=False, collate_fn=collate_seq_batch)

    experts = {
        "smiles": SmilesExpert(),
        "graph": GraphExpert(),
        "cond": ConditionExpert(),
        "gnn3d": GNN3DExpert(),
    }
    model = MoEModel(experts=experts, gate_feature_dim=2048).to(device)
    model.eval()

    counts = Counter()
    with torch.no_grad():
        for batch in loader:
            batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
            _, aux = model(batch)
            assign = aux["assignments"].detach().cpu().tolist()
            counts.update(assign)

    total = sum(counts.values())
    print("Expert assignment frequencies:")
    for idx, name in enumerate(model.expert_names):
        c = counts.get(idx, 0)
        pct = 100.0 * c / max(1, total)
        print(f"  {idx} ({name}): {c} ({pct:.2f}%)")


if __name__ == "__main__":
    main()


