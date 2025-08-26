from __future__ import annotations

import argparse, time, os, csv, math
from typing import Dict

import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader

try:
    from torch.utils.tensorboard import SummaryWriter
except Exception:
    SummaryWriter = None  # optional

from hetero_moe.data.dataset import USPTODataset
    # must expose USPTODataset with ._feat NPZ dict-like
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
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu", help="e.g., cuda, cuda:1, or cpu")
    p.add_argument("--save_path", type=str, default="")
    p.add_argument("--config", type=str, default="")
    p.add_argument("--pad_id", type=int, default=0)
    p.add_argument("--bos_id", type=int, default=2)
    p.add_argument("--eos_id", type=int, default=3)

    # dataloader + logging knobs
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--pin_memory", action="store_true")
    p.add_argument("--prefetch_factor", type=int, default=2)
    p.add_argument("--persistent_workers", action="store_true")
    p.add_argument("--log_every", type=int, default=100)
    p.add_argument("--warmup_check", action="store_true", help="Load a first batch and print its shapes/timing")

    # epoch-level metrics CSV
    p.add_argument("--metrics_csv", type=str, default="", help="CSV to append epoch metrics; default derives from save_path")
    # per-batch metrics CSV
    p.add_argument("--batch_metrics_csv", type=str, default="", help="Optional CSV for per-batch loss/EMA/skip flags")

    # stability knobs (FP32)
    p.add_argument("--grad_clip", type=float, default=1.0, help="Clip grad-norm to this value; 0 disables")

    # outlier loss guard
    p.add_argument("--loss_spike_factor", type=float, default=3.0,
                   help="Mark a batch as outlier if loss > factor * EMA(loss)")
    p.add_argument("--loss_spike_warmup", type=int, default=50,
                   help="Do not guard until this many steps are seen")
    p.add_argument("--outlier_policy", choices=["skip", "cap_ema", "cap_running"], default="skip",
                   help="skip: drop update; cap_ema: backprop EMA(loss); cap_running: backprop running mean loss")
    return p.parse_args()


def _eagerize_npz_in_dataset(ds):
    """Turn lazy NpzFile into in-memory arrays once, avoiding per-item decompression stalls."""
    feat = getattr(ds, "_feat", None)
    if feat is not None and type(feat).__name__ == "NpzFile":
        t0 = time.time()
        files = list(feat.files)
        eager = {k: feat[k] for k in files}  # one-time read/decompress
        ds._feat = eager
        try:
            getattr(feat, "close", lambda: None)()
        except Exception:
            pass
        print(f"[eagerize] materialized {len(files)} arrays in {time.time()-t0:.2f}s", flush=True)


def _find_target_tensor(batch: Dict):
    for k in ["tgt_token_ids", "tgt", "target", "y"]:
        if k in batch and torch.is_tensor(batch[k]):
            return batch[k]
    raise KeyError("Target tensor not found in batch (expected one of: tgt_token_ids/tgt/target/y).")


def _ensure_csv(csv_path: str, header: list[str]):
    if not csv_path:
        return
    d = os.path.dirname(csv_path)
    if d and not os.path.isdir(d):
        os.makedirs(d, exist_ok=True)
    if not os.path.exists(csv_path):
        with open(csv_path, "w", newline="") as f:
            csv.writer(f).writerow(header)


def _append_csv(csv_path: str, row):
    if not csv_path:
        return
    with open(csv_path, "a", newline="") as f:
        csv.writer(f).writerow(row)


def run_epoch(
    model,
    loader,
    optimizer=None,
    device="cpu",
    log_every=0,
    phase="train",
    pad_id=0,
    grad_clip=0.0,
    loss_spike_factor=3.0,
    loss_spike_warmup=50,
    outlier_policy="skip",
    batch_metrics_csv="",
):
    # EMA for loss (momentum form)
    ema = None
    ema_var = 0.0
    momentum = 0.98

    def ema_update(x, ema, ema_var):
        if ema is None:
            return x, 0.0
        delta = x - ema
        ema = ema + (1.0 - momentum) * delta
        ema_var = momentum * ema_var + (1.0 - momentum) * (delta * delta)
        return ema, ema_var

    if batch_metrics_csv:
        _ensure_csv(batch_metrics_csv, ["phase", "step", "raw_loss", "ema_loss", "ema_std", "running_mean", "skipped"])

    total_loss = 0.0
    total_tokens = 0
    total_correct_tokens = 0
    seq_matches = 0
    n_samples = 0
    total_batches = 0

    train = optimizer is not None
    model.train(train)
    t0 = time.time()

    for step, batch in enumerate(loader, start=1):
        batch = {k: (v.to(device, non_blocking=True) if torch.is_tensor(v) else v) for k, v in batch.items()}

        out = model(batch)                   # FP32 forward
        loss = out["loss"]                   # scalar loss (FP32)
        raw_loss = float(loss.detach().cpu())

        # stats before any update
        running_mean_before = (total_loss / total_batches) if total_batches > 0 else raw_loss

        # update EMA
        ema, ema_var = ema_update(raw_loss, ema, ema_var)
        ema_std = math.sqrt(max(ema_var, 1e-12))

        # decide outlier
        is_bad = (not math.isfinite(raw_loss)) or \
                 (step > loss_spike_warmup and raw_loss > max(1e-8, loss_spike_factor * max(ema, 1e-8)))

        if train:
            if is_bad and outlier_policy == "skip":
                _append_csv(batch_metrics_csv, [phase, step, raw_loss, ema, ema_std, running_mean_before, 1])
                if log_every and step % log_every == 0:
                    print(f"[{phase}] step {step} | SKIP outlier raw_loss={raw_loss:.4f} ema={ema:.4f}", flush=True)
                continue

            loss_to_backprop = loss
            if is_bad and outlier_policy in ("cap_ema", "cap_running"):
                cap_value = ema if outlier_policy == "cap_ema" else running_mean_before
                cap_tensor = torch.tensor(cap_value, dtype=loss.dtype, device=loss.device)
                loss_to_backprop = cap_tensor
                if log_every and step % log_every == 0:
                    print(f"[{phase}] step {step} | CAP({outlier_policy}) {raw_loss:.4f} -> {float(cap_value):.4f}", flush=True)

            optimizer.zero_grad(set_to_none=True)
            loss_to_backprop.backward()
            if grad_clip and grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)  # FP32 clipping keeps grads bounded :contentReference[oaicite:2]{index=2}
            optimizer.step()

        # aggregate raw loss
        total_loss += raw_loss
        total_batches += 1
        _append_csv(batch_metrics_csv, [phase, step, raw_loss, ema, ema_std, running_mean_before, 0])

        # token/seq accuracy if logits available
        logits = out.get("logits", None)
        if logits is not None:
            try:
                tgt = _find_target_tensor(batch)
                mask = tgt.ne(pad_id)
                preds = logits.argmax(dim=-1)
                total_tokens += int(mask.sum().item())
                total_correct_tokens += int((preds.eq(tgt) & mask).sum().item())
                seq_matches += int(((preds.eq(tgt) | (~mask)).all(dim=1)).sum().item())
                n_samples += tgt.size(0)
            except Exception:
                pass

        if log_every and (step % log_every == 0):
            it_s = total_batches / max(time.time() - t0, 1e-9)
            print(f"[{phase}] step {step} | loss {total_loss/total_batches:.4f} | {it_s:.2f} it/s", flush=True)

    avg_loss = total_loss / max(1, total_batches)
    token_acc = (total_correct_tokens / total_tokens) if total_tokens > 0 else float("nan")
    seq_acc = (seq_matches / n_samples) if n_samples > 0 else float("nan")
    elapsed = time.time() - t0
    return {
        "loss": avg_loss,
        "token_acc": token_acc,
        "seq_acc": seq_acc,
        "batches_per_sec": total_batches / max(elapsed, 1e-9),
        "tokens_per_sec": (total_tokens / max(elapsed, 1e-9)) if total_tokens > 0 else float("nan"),
    }


def main():
    args = parse_args()

    # YAML overrides
    if args.config:
        cfg = load_yaml(args.config)
        apply_overrides(args, cfg)

    # device
    device = args.device
    if device.startswith("cuda"):
        assert torch.cuda.is_available(), "CUDA requested but not available"
        if ":" in device:
            torch.cuda.set_device(int(device.split(":")[1]))

    torch.backends.cudnn.benchmark = True

    # epoch metrics CSV (default from save_path)
    metrics_csv = args.metrics_csv or (args.save_path + ".metrics.csv" if args.save_path else "runs/experts/metrics.csv")
    _ensure_csv(metrics_csv, ["epoch", "phase", "loss", "token_acc", "seq_acc", "tokens_per_sec", "batches_per_sec"])

    # TB writer
    writer = None
    logdir = getattr(args, "logdir", "")
    if logdir and SummaryWriter is not None:
        os.makedirs(logdir, exist_ok=True)
        writer = SummaryWriter(log_dir=logdir)
        print(f"[setup] TensorBoard -> {logdir}", flush=True)

    print(f"[setup] device={device} | workers={args.num_workers} | pin_memory={args.pin_memory}", flush=True)

    # datasets
    train_ds = USPTODataset(args.train_bin)
    valid_ds = USPTODataset(args.valid_bin)
    _eagerize_npz_in_dataset(train_ds)
    _eagerize_npz_in_dataset(valid_ds)

    collate = collate_graph_batch if getattr(train_ds, "has_graph", False) else collate_seq_batch

    dl_kwargs = dict(
        batch_size=args.batch_size,
        collate_fn=collate,
        pin_memory=args.pin_memory and device.startswith("cuda"),
        num_workers=args.num_workers,
        prefetch_factor=args.prefetch_factor if args.num_workers > 0 else None,  # standard PyTorch arg :contentReference[oaicite:3]{index=3}
        persistent_workers=args.persistent_workers if args.num_workers > 0 else False,
    )
    train_loader = DataLoader(train_ds, shuffle=True, **dl_kwargs)
    valid_loader = DataLoader(valid_ds, shuffle=False, **dl_kwargs)

    if args.warmup_check:
        print("[warmup] loading one training batch...", flush=True)
        t0 = time.time()
        b = next(iter(train_loader))
        shapes = {k: (tuple(v.shape) if torch.is_tensor(v) else type(v).__name__) for k, v in b.items()}
        print(f"[warmup] first batch in {time.time()-t0:.2f}s; keys={list(b.keys())}; shapes={shapes}", flush=True)

    # model & optimizer (FP32)
    model = build_expert(
        args.expert, vocab_size=args.vocab_size, hidden=args.hidden,
        layers=args.layers, heads=args.heads, ff=args.ff
    ).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)

    best_val = float("inf")
    best_payload = None
    for epoch in range(1, args.epochs + 1):
        tr = run_epoch(
            model, train_loader, optimizer=optimizer, device=device, log_every=args.log_every, phase="train",
            pad_id=args.pad_id, grad_clip=args.grad_clip,
            loss_spike_factor=args.loss_spike_factor, loss_spike_warmup=args.loss_spike_warmup,
            outlier_policy=args.outlier_policy, batch_metrics_csv=args.batch_metrics_csv
        )
        va = run_epoch(
            model, valid_loader, optimizer=None, device=device, log_every=args.log_every, phase="valid",
            pad_id=args.pad_id, grad_clip=0.0,
            loss_spike_factor=args.loss_spike_factor, loss_spike_warmup=args.loss_spike_warmup,
            outlier_policy="skip",  # never update on valid
            batch_metrics_csv=args.batch_metrics_csv.replace(".csv", ".valid.csv") if args.batch_metrics_csv else ""
        )

        print(
            f"epoch {epoch} | "
            f"train_loss {tr['loss']:.4f} | train_tok_acc {tr['token_acc']:.4f} | train_seq_acc {tr['seq_acc']:.4f} || "
            f"valid_loss {va['loss']:.4f} | valid_tok_acc {va['token_acc']:.4f} | valid_seq_acc {va['seq_acc']:.4f}",
            flush=True
        )

        _append_csv(metrics_csv, [epoch, "train", tr["loss"], tr["token_acc"], tr["seq_acc"], tr["tokens_per_sec"], tr["batches_per_sec"]])
        _append_csv(metrics_csv, [epoch, "valid", va["loss"], va["token_acc"], va["seq_acc"], va["tokens_per_sec"], va["batches_per_sec"]])

        if writer is not None:
            writer.add_scalar("Loss/train", tr["loss"], epoch)
            writer.add_scalar("Loss/valid", va["loss"], epoch)
            writer.add_scalar("AccToken/train", tr["token_acc"], epoch)
            writer.add_scalar("AccToken/valid", va["token_acc"], epoch)
            writer.add_scalar("AccSeq/train", tr["seq_acc"], epoch)
            writer.add_scalar("AccSeq/valid", va["seq_acc"], epoch)
            if not math.isnan(tr["tokens_per_sec"]):
                writer.add_scalar("Speed/tokens_per_sec_train", tr["tokens_per_sec"], epoch)
            if not math.isnan(va["tokens_per_sec"]):
                writer.add_scalar("Speed/tokens_per_sec_valid", va["tokens_per_sec"], epoch)

        if args.save_path:
            payload = {"model": model.state_dict(), "epoch": epoch, "train": tr, "valid": va}
            torch.save(payload, args.save_path)  # last
            if va["loss"] < best_val:
                best_val = va["loss"]
                best_payload = payload
                torch.save(best_payload, args.save_path + ".best")

    if writer is not None:
        writer.close()


if __name__ == "__main__":
    main()

