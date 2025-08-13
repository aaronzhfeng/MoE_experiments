from __future__ import annotations

import argparse
from typing import Dict

import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from hetero_moe.data.dataset import USPTODataset
from hetero_moe.data.dataloader import collate_seq_batch, collate_graph_batch
from hetero_moe.models.experts.smiles_expert import SmilesExpert
from hetero_moe.models.experts.graph_expert import GraphExpert
from hetero_moe.models.experts.cond_expert import ConditionExpert
from hetero_moe.models.experts.gnn3d_expert import GNN3DExpert
from hetero_moe.utils.config import load_yaml, apply_overrides


def build_expert(name: str, vocab_size: int = 512, hidden: int = 256, layers: int = 4, heads: int = 8, ff: int = 1024):
    name = name.lower()
    if name == "smiles":
        return SmilesExpert(vocab_size=vocab_size, hidden=hidden, layers=layers, heads=heads, ff=ff)
    if name == "graph":
        return GraphExpert(vocab_size=vocab_size, hidden=hidden, layers=layers, heads=heads, ff=ff)
    if name == "cond":
        return ConditionExpert(vocab_size=vocab_size)
    if name == "gnn3d":
        return GNN3DExpert(vocab_size=vocab_size)
    raise ValueError(f"Unknown expert: {name}")


def parse_args():
    p = argparse.ArgumentParser("train_expert")
    p.add_argument("--expert", required=True, choices=["smiles", "graph", "cond", "gnn3d"])
    p.add_argument("--train_bin", required=True)
    p.add_argument("--valid_bin", required=True)
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--vocab_size", type=int, default=512)
    p.add_argument("--hidden", type=int, default=256)
    p.add_argument("--layers", type=int, default=4)
    p.add_argument("--heads", type=int, default=8)
    p.add_argument("--ff", type=int, default=1024)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--save_path", type=str, default="")
    p.add_argument("--config", type=str, default="")
    p.add_argument("--pad_id", type=int, default=0)
    p.add_argument("--bos_id", type=int, default=2)
    p.add_argument("--eos_id", type=int, default=3)
    return p.parse_args()


def run_epoch(model, loader, optimizer=None, device="cpu"):
    total = 0.0
    iters = 0
    train = optimizer is not None
    model.train(train)
    for batch in loader:
        batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
        out = model(batch)
        loss = out["loss"]
        if train:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        total += float(loss.detach().cpu())
        iters += 1
    return total / max(1, iters)


def main():
    args = parse_args()
    # Load YAML config if provided
    if args.config:
        cfg = load_yaml(args.config)
        apply_overrides(args, cfg)
    device = args.device

    train_ds = USPTODataset(args.train_bin)
    valid_ds = USPTODataset(args.valid_bin)
    # If graph features are present, use graph collate; else sequence collate
    collate = collate_graph_batch if getattr(train_ds, "has_graph", False) else collate_seq_batch
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate)
    valid_loader = DataLoader(valid_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate)

    model = build_expert(args.expert, vocab_size=args.vocab_size, hidden=args.hidden, layers=args.layers, heads=args.heads, ff=args.ff).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)

    best_val = float("inf")
    best_payload = None
    for epoch in range(1, args.epochs + 1):
        train_loss = run_epoch(model, train_loader, optimizer=optimizer, device=device)
        valid_loss = run_epoch(model, valid_loader, optimizer=None, device=device)
        print(f"epoch {epoch} | train_loss {train_loss:.4f} | valid_loss {valid_loss:.4f}")
        if args.save_path:
            payload = {"model": model.state_dict(), "epoch": epoch, "train_loss": train_loss, "valid_loss": valid_loss}
            # save last
            torch.save(payload, args.save_path)
            # save best
            if valid_loss < best_val:
                best_val = valid_loss
                best_payload = payload
                torch.save(best_payload, args.save_path + ".best")


if __name__ == "__main__":
    main()


