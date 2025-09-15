from __future__ import annotations
import os
import sys
import time
import math
import argparse
from typing import Dict, Any

import torch
import torch.optim as optim
from torch.utils.data import DataLoader

# Make project importable when run as a module
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

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
    # data
    p.add_argument("--train_bin", required=True)
    p.add_argument("--valid_bin", required=True)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--pin_memory", action="store_true")
    p.add_argument("--persistent_workers", action="store_true")
    # training
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=0.0)
    p.add_argument("--grad_clip", type=float, default=0.0, help="0 = disabled")
    p.add_argument("--log_every", type=int, default=100)
    p.add_argument("--max_train_steps", type=int, default=0, help="0 = all steps")
    p.add_argument("--max_valid_steps", type=int, default=0, help="0 = all steps")
    # model hyperparams
    p.add_argument("--hidden", type=int, default=256)
    p.add_argument("--layers", type=int, default=4)
    p.add_argument("--heads", type=int, default=8)
    p.add_argument("--ff", type=int, default=1024)
    p.add_argument("--enabled_experts", type=str, default="smiles,graph,cond,gnn3d")
    p.add_argument("--freeze_experts", type=str, default="")
    # routing / MoE
    p.add_argument("--balance_lambda", type=float, default=0.01)
    p.add_argument("--balance_lambda_schedule", type=str, default="constant", choices=["constant", "linear_warmup"])
    p.add_argument("--router_warmup_epochs", type=int, default=0, help="freeze non-router params for first N epochs")
    p.add_argument("--top_k", type=int, default=1)
    p.add_argument("--router_temperature", type=float, default=1.0)
    p.add_argument("--router_gumbel_noise", action="store_true")
    # misc
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--save_path", type=str, default="")
    p.add_argument("--config", type=str, default="")
    p.add_argument("--pad_id", type=int, default=0)
    p.add_argument("--bos_id", type=int, default=2)
    p.add_argument("--eos_id", type=int, default=3)
    # gpu debug
    p.add_argument("--gpu_debug", action="store_true")
    p.add_argument("--gpu_report_every", type=int, default=1000)
    return p.parse_args()


def parse_list(val) -> list[str]:
    if isinstance(val, list):
        return [str(x).strip() for x in val]
    return [x.strip() for x in str(val).split(",") if x.strip()]


def set_requires_grad(module: torch.nn.Module, flag: bool):
    for p in module.parameters():
        p.requires_grad = flag


def maybe_gpu_report(tag: str, step: int, args):
    if not args.gpu_debug or (step % max(1, args.gpu_report_every) != 0):
        return
    if torch.cuda.is_available() and args.device.startswith("cuda"):
        dev = torch.cuda.current_device()
        alloc = torch.cuda.memory_allocated(dev) / (1024 ** 2)
        reserved = torch.cuda.memory_reserved(dev) / (1024 ** 2)
        peak = torch.cuda.max_memory_allocated(dev) / (1024 ** 2)
        name = torch.cuda.get_device_name(dev)
        print(f"[gpu:{tag}@{step}] id={dev} {name} alloc={alloc:.1f}MB reserved={reserved:.1f}MB peak={peak:.1f}MB")


def build_loaders(args):
    train_ds = USPTODataset(args.train_bin)
    valid_ds = USPTODataset(args.valid_bin)

    # choose collate: if dataset carries graphs, we collate graphs (and optionally morgan for gate features)
    base_collate = collate_graph_batch if getattr(train_ds, "has_graph", False) else collate_seq_batch

    def _collate_with_smiles(batch):
        smiles = [s.get("smiles") for s in batch] if batch and isinstance(batch[0], dict) else None
        use_morgan = smiles is not None and all(isinstance(s, str) for s in smiles)
        return base_collate(batch, use_morgan=use_morgan, smiles=smiles)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        persistent_workers=args.persistent_workers if args.num_workers > 0 else False,
        collate_fn=_collate_with_smiles,
        drop_last=False,
    )
    valid_loader = DataLoader(
        valid_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        persistent_workers=args.persistent_workers if args.num_workers > 0 else False,
        collate_fn=_collate_with_smiles,
        drop_last=False,
    )
    return train_loader, valid_loader


def build_model(args) -> MoEModel:
    enabled = parse_list(args.enabled_experts)
    frozen = set(parse_list(args.freeze_experts))

    all_experts = {
        "smiles": SmilesExpert(hidden=args.hidden, layers=args.layers, heads=args.heads, ff=args.ff),
        "graph": GraphExpert(hidden=args.hidden, layers=args.layers, heads=args.heads, ff=args.ff),
        "cond": ConditionExpert(),
        "gnn3d": GNN3DExpert(),
    }
    experts = {name: all_experts[name] for name in enabled if name in all_experts}

    model = MoEModel(
        experts=experts,
        gate_feature_dim=2048,
        balance_lambda=args.balance_lambda,
        top_k=args.top_k,
    )

    # router settings
    model.router.temperature = args.router_temperature
    model.router.gumbel_noise = args.router_gumbel_noise

    # initial freezing (explicit user request)
    for name, module in model.experts.items():
        if name in frozen:
            set_requires_grad(module, False)

    return model


def make_optimizer(model: torch.nn.Module, args):
    params = [p for p in model.parameters() if p.requires_grad]
    return optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)


def to_device_batch(batch: Dict[str, Any], device: str) -> Dict[str, Any]:
    return {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}


def main():
    args = parse_args()
    if args.config:
        cfg = load_yaml(args.config)
        apply_overrides(args, cfg)

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.benchmark = True

    device = args.device
    print(f"[setup] device={device} | workers={args.num_workers} | pin_memory={args.pin_memory}")

    train_loader, valid_loader = build_loaders(args)

    model = build_model(args).to(device)
    frozen_names = set(parse_list(args.freeze_experts))
    opt = make_optimizer(model, args)

    best_val = float("inf")
    best_payload = None

    for epoch in range(1, args.epochs + 1):
        # ---- Warmup freezing schedule: freeze non-router params for first N epochs ----
        if args.router_warmup_epochs > 0:
            freeze_phase = epoch <= args.router_warmup_epochs
            for name, module in model.experts.items():
                # keep explicitly frozen ones frozen always
                if name in frozen_names:
                    set_requires_grad(module, False)
                else:
                    set_requires_grad(module, not freeze_phase)
            # router always trainable
            set_requires_grad(model.router, True)
            # refresh optimizer if any requires_grad flags changed
            opt = make_optimizer(model, args)

        # --------------------
        # Train
        # --------------------
        model.train(True)
        running = 0.0
        running_bal = 0.0
        start = time.perf_counter()

        for step, batch in enumerate(train_loader, start=1):
            batch = to_device_batch(batch, device)
            loss, aux = model(batch)

            # balance loss (safe)
            probs = aux.get("probs") if isinstance(aux, dict) else None
            bal = load_balance_loss(probs) if probs is not None else torch.tensor(0.0, device=device)

            # schedule lambda
            if args.balance_lambda_schedule == "linear_warmup":
                lam = args.balance_lambda * (epoch / max(1, args.epochs))
            else:
                lam = args.balance_lambda

            total_loss = loss + lam * bal

            # NaN/Inf guard
            if not torch.isfinite(total_loss):
                print(f"[warn] non-finite loss at step {step}: loss={float(loss):.4f}, bal={float(bal):.4f}. Skipping.")
                opt.zero_grad(set_to_none=True)
                continue

            opt.zero_grad(set_to_none=True)
            total_loss.backward()

            if args.grad_clip and args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

            opt.step()

            running += float(total_loss.detach().cpu())
            running_bal += float(bal.detach().cpu())

            if step % max(1, args.log_every) == 0:
                dt = time.perf_counter() - start
                it_s = step / max(dt, 1e-9)
                msg = (
                    f"[train] epoch {epoch} step {step} | "
                    f"loss {running/args.log_every:.4f} | "
                    f"balance {running_bal/args.log_every:.4f} | "
                    f"{it_s:.2f} it/s"
                )
                print(msg)
                maybe_gpu_report("train", step, args)
                running = 0.0
                running_bal = 0.0
                start = time.perf_counter()

            # optional cap for debugging
            if args.max_train_steps and step >= args.max_train_steps:
                break

        # --------------------
        # Validate
        # --------------------
        model.train(False)
        v_running = 0.0
        v_running_bal = 0.0
        v_steps = 0
        with torch.no_grad():
            for vstep, batch in enumerate(valid_loader, start=1):
                batch = to_device_batch(batch, device)
                v_loss, v_aux = model(batch)
                v_probs = v_aux.get("probs") if isinstance(v_aux, dict) else None
                v_bal = load_balance_loss(v_probs) if v_probs is not None else torch.tensor(0.0, device=device)
                v_total = v_loss + args.balance_lambda * v_bal
                v_running += float(v_total.detach().cpu())
                v_running_bal += float(v_bal.detach().cpu())
                v_steps += 1

                if args.max_valid_steps and vstep >= args.max_valid_steps:
                    break

                if vstep % max(1, args.log_every) == 0:
                    print(f"[valid] epoch {epoch} step {vstep} | loss {v_running/vstep:.4f} | balance {v_running_bal/vstep:.4f}")
                    maybe_gpu_report("valid", vstep, args)

        v_loss_mean = v_running / max(1, v_steps)
        print(f"epoch {epoch} | valid_loss {v_loss_mean:.4f}")

        # --------------------
        # Save
        # --------------------
        if args.save_path:
            payload = {"model": model.state_dict(), "epoch": epoch, "valid_loss": v_loss_mean}
            torch.save(payload, args.save_path)
            if v_loss_mean < best_val:
                best_val = v_loss_mean
                best_payload = payload
                torch.save(best_payload, args.save_path + ".best")


if __name__ == "__main__":
    main()

