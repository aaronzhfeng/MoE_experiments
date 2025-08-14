from __future__ import annotations
import os
import sys

# Add the project root to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)


import argparse
from typing import Dict

import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from hetero_moe.data.dataset import USPTODataset
from hetero_moe.data.dataloader import collate_seq_batch, collate_graph_batch
from hetero_moe.models.moe import MoEModel
from hetero_moe.models.experts.smiles_expert import SmilesExpert
from hetero_moe.models.experts.graph_expert import GraphExpert
from hetero_moe.models.experts.cond_expert import ConditionExpert
from hetero_moe.models.experts.gnn3d_expert import GNN3DExpert
from hetero_moe.training.utils import load_balance_loss
from hetero_moe.utils.config import load_yaml, apply_overrides


def parse_args():
    p = argparse.ArgumentParser("train_moe")
    p.add_argument("--train_bin", required=True)
    p.add_argument("--valid_bin", required=True)
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-3)
    # model hyperparams
    p.add_argument("--hidden", type=int, default=256)
    p.add_argument("--layers", type=int, default=4)
    p.add_argument("--heads", type=int, default=8)
    p.add_argument("--ff", type=int, default=1024)
    p.add_argument("--balance_lambda", type=float, default=0.01)
    p.add_argument("--enabled_experts", type=str, default="smiles,graph,cond,gnn3d")
    p.add_argument("--freeze_experts", type=str, default="")
    p.add_argument("--router_warmup_epochs", type=int, default=0)
    p.add_argument("--top_k", type=int, default=1)
    p.add_argument("--router_temperature", type=float, default=1.0)
    p.add_argument("--router_gumbel_noise", action="store_true")
    p.add_argument("--balance_lambda_schedule", type=str, default="constant")  # constant|linear_warmup
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--save_path", type=str, default="")
    p.add_argument("--config", type=str, default="")
    p.add_argument("--pad_id", type=int, default=0)
    p.add_argument("--bos_id", type=int, default=2)
    p.add_argument("--eos_id", type=int, default=3)
    return p.parse_args()


def main():
    args = parse_args()
    if args.config:
        cfg = load_yaml(args.config)
        apply_overrides(args, cfg)
    device = args.device

    train_ds = USPTODataset(args.train_bin)
    valid_ds = USPTODataset(args.valid_bin)
    collate = collate_graph_batch if getattr(train_ds, "has_graph", False) else collate_seq_batch
    def _collate_with_smiles(batch):
        smiles = [s.get("smiles") for s in batch] if batch and isinstance(batch[0], dict) else None
        use_morgan = smiles is not None and all(isinstance(s, str) for s in smiles)
        return collate(batch, use_morgan=use_morgan, smiles=smiles)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=_collate_with_smiles)
    valid_loader = DataLoader(valid_ds, batch_size=args.batch_size, shuffle=False, collate_fn=_collate_with_smiles)

    # Parse expert lists
    def parse_list(val):
        if isinstance(val, list):
            return [str(x).strip() for x in val]
        return [x.strip() for x in str(val).split(",") if x.strip()]

    enabled = parse_list(args.enabled_experts)
    frozen = set(parse_list(args.freeze_experts))

    all_experts = {
        "smiles": SmilesExpert(hidden=args.hidden, layers=args.layers, heads=args.heads, ff=args.ff),
        "graph": GraphExpert(hidden=args.hidden, layers=args.layers, heads=args.heads, ff=args.ff),
        "cond": ConditionExpert(),
        "gnn3d": GNN3DExpert(),
    }
    experts = {name: all_experts[name] for name in enabled if name in all_experts}

    model = MoEModel(experts=experts, gate_feature_dim=2048, balance_lambda=args.balance_lambda, top_k=args.top_k).to(device)
    # Set router temperature/noise
    model.router.temperature = args.router_temperature
    model.router.gumbel_noise = args.router_gumbel_noise

    # Apply initial freezing based on config
    for name, module in model.experts.items():
        if name in frozen:
            for p in module.parameters():
                p.requires_grad = False

    def make_optimizer():
        params = [p for p in model.parameters() if p.requires_grad]
        return optim.AdamW(params, lr=args.lr)

    optimizer = make_optimizer()

    best_val = float("inf")
    best_payload = None
    for epoch in range(1, args.epochs + 1):
        # Router warmup: freeze all experts for initial epochs
        if epoch == 1 and args.router_warmup_epochs > 0:
            for module in model.experts.values():
                for p in module.parameters():
                    p.requires_grad = False
            optimizer = make_optimizer()

        # Train
        model.train(True)
        total = 0.0
        iters = 0
        for batch in train_loader:
            batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
            loss, aux = model(batch)
            balance = load_balance_loss(aux["probs"]) if isinstance(aux, dict) and "probs" in aux else torch.tensor(0.0, device=device)
            # schedule balance lambda
            if args.balance_lambda_schedule == "linear_warmup" and args.epochs > 0:
                lam = args.balance_lambda * (epoch / max(1, args.epochs))
            else:
                lam = args.balance_lambda
            total_loss = loss + lam * balance
            optimizer.zero_grad(set_to_none=True)
            total_loss.backward()
            optimizer.step()
            total += float(total_loss.detach().cpu())
            iters += 1
        print(f"epoch {epoch} | train_loss {total / max(1, iters):.4f}")

        # Unfreeze after warmup boundary
        if args.router_warmup_epochs > 0 and epoch == args.router_warmup_epochs:
            for name, module in model.experts.items():
                # Unfreeze unless explicitly frozen by config
                if name in frozen:
                    continue
                for p in module.parameters():
                    p.requires_grad = True
            optimizer = make_optimizer()

        # Valid
        model.train(False)
        total = 0.0
        iters = 0
        with torch.no_grad():
            for batch in valid_loader:
                batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
                loss, aux = model(batch)
                balance = load_balance_loss(aux["probs"]) if isinstance(aux, dict) and "probs" in aux else torch.tensor(0.0, device=device)
                total_loss = loss + model.balance_lambda * balance
                total += float(total_loss.detach().cpu())
                iters += 1
        valid_loss = total / max(1, iters)
        print(f"epoch {epoch} | valid_loss {valid_loss:.4f}")
        if args.save_path:
            payload = {"model": model.state_dict(), "epoch": epoch, "valid_loss": valid_loss}
            torch.save(payload, args.save_path)
            if valid_loss < best_val:
                best_val = valid_loss
                best_payload = payload
                torch.save(best_payload, args.save_path + ".best")


if __name__ == "__main__":
    main()


