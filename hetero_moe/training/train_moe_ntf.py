# hetero_moe/training/train_moe_ntf.py
from __future__ import annotations

import os
import sys
import time
import json
import math
import argparse
import importlib
from typing import Any, Dict, List, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


# ---------------------------
# Utils
# ---------------------------

def set_seed(seed: int = 1337):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def human_mem(mb_float: float) -> str:
    return f"{mb_float:.1f}MB"


def make_valid_mask(gold: torch.Tensor, pad_id: int = 0, eos_id: Optional[int] = None) -> torch.Tensor:
    """
    Return (B, T) boolean mask that excludes PAD and stops counting at the first EOS (inclusive).
    """
    not_pad = (gold != pad_id)
    if eos_id is None:
        return not_pad
    is_eos = (gold == eos_id).to(torch.int32)         # (B,T)
    seen = torch.cumsum(is_eos, dim=1)                # (B,T)
    before = (seen == 0)
    at_first = (seen == 1) & (gold == eos_id)
    return (before | at_first) & not_pad


def first_eos_len(seq: torch.Tensor, eos_id: int, pad_id: int) -> int:
    s = seq.tolist()
    for i, t in enumerate(s):
        if t == eos_id:
            return i + 1
        if t == pad_id:
            return i
    return len(s)


def gpu_brief(prefix: str, device: torch.device):
    if device.type != "cuda" or not torch.cuda.is_available():
        return
    props = torch.cuda.get_device_properties(device)
    alloc = torch.cuda.memory_allocated(device) / (1024 ** 2)
    reserved = torch.cuda.memory_reserved(device) / (1024 ** 2)
    peak = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
    print(f"[gpu:{prefix}] id={device.index or 0} {props.name} "
          f"alloc={human_mem(alloc)} reserved={human_mem(reserved)} peak={human_mem(peak)}")


# ---------------------------
# Data
# ---------------------------

def _npz_to_arrays(path: str) -> Dict[str, Any]:
    t0 = time.time()
    with np.load(path, allow_pickle=True) as npz:
        data = {}
        for k in npz.keys():
            arr = npz[k]
            if arr.dtype == object:
                try:
                    data[k] = arr.tolist()
                except Exception:
                    data[k] = arr
            else:
                data[k] = arr
    print(f"[eagerize] materialized {len(data)} arrays in {time.time() - t0:.2f}s")
    return data


def infer_vocab_size_from_data(*dicts: Dict[str, Any], prefer: Optional[List[str]] = None, min_size: int = 8) -> int:
    """
    Look through dicts for token id arrays and infer vocab_size = max_token_id + 1.
    Preference order in `prefer`. Falls back to scanning all ndarrays.
    """
    keys = prefer or ["tgt_token_ids", "target_ids", "labels", "input_ids", "src_token_ids"]
    max_id = -1

    def scan_array(x: np.ndarray):
        nonlocal max_id
        try:
            if x.size == 0:
                return
            local_max = int(np.nanmax(x))
            if local_max > max_id:
                max_id = local_max
        except Exception:
            pass

    # try preferred keys first
    for d in dicts:
        for k in keys:
            if k in d and isinstance(d[k], np.ndarray):
                scan_array(d[k])

    # if still unknown, scan all numeric arrays
    if max_id < 0:
        for d in dicts:
            for v in d.values():
                if isinstance(v, np.ndarray) and np.issubdtype(v.dtype, np.integer):
                    scan_array(v)

    vocab_size = max(min_size, max_id + 1)
    return vocab_size


class NpzSeqDataset(Dataset):
    def __init__(self, data: Dict[str, Any]):
        super().__init__()
        self.data = data
        if "indices" in data:
            self.N = len(data["indices"])
        elif "input_ids" in data:
            self.N = len(data["input_ids"])
        elif "tgt_token_ids" in data:
            self.N = len(data["tgt_token_ids"])
        elif "target_ids" in data:
            self.N = len(data["target_ids"])
        else:
            # fallback: first 1D/2D array
            self.N = None
            for v in data.values():
                if isinstance(v, np.ndarray) and v.ndim >= 1:
                    self.N = len(v)
                    break
            if self.N is None:
                raise ValueError("Could not infer dataset length from NPZ.")

    def __len__(self) -> int:
        return int(self.N)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        ex = {}
        for k, v in self.data.items():
            if isinstance(v, list):
                ex[k] = v[idx]
            elif isinstance(v, np.ndarray):
                ex[k] = v[idx]
            else:
                ex[k] = v
        return ex


def pad_2d_long(batch_list: List[np.ndarray], pad: int = 0) -> torch.Tensor:
    if len(batch_list) == 0:
        return torch.empty(0, dtype=torch.long)
    rows = []
    maxT = 0
    for x in batch_list:
        x = np.asarray(x)
        if x.ndim == 1:
            x = x[None, :]
        assert x.ndim == 2 and x.shape[0] == 1, "Expect 1D sequences"
        row = x[0]
        rows.append(row)
        maxT = max(maxT, row.shape[0])
    B = len(rows)
    out = torch.full((B, maxT), pad, dtype=torch.long)
    for i, row in enumerate(rows):
        t = row.shape[0]
        out[i, :t] = torch.from_numpy(row.astype(np.int64))
    return out


def collate_batch(batch: List[Dict[str, Any]], pad_id: int) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    keys = batch[0].keys()
    seq_keys = {"input_ids", "target_ids", "tgt_token_ids", "labels", "src_token_ids"}
    for k in keys:
        vals = [b[k] for b in batch]
        if k in seq_keys:
            out[k] = pad_2d_long(vals, pad=pad_id)
        elif k == "gate_features":
            out[k] = torch.from_numpy(np.stack(vals, axis=0)).float()
        elif k == "indices":
            out[k] = torch.from_numpy(np.asarray(vals, dtype=np.int64))
        elif k == "graph_feature":
            out[k] = vals
        else:
            v0 = vals[0]
            if isinstance(v0, np.ndarray) and all(isinstance(x, np.ndarray) and x.shape == v0.shape for x in vals):
                out[k] = torch.from_numpy(np.stack(vals, axis=0))
            else:
                out[k] = vals
    return out


# ---------------------------
# Experts + Router
# ---------------------------

def dynamic_build_expert(name: str, cfg: Dict[str, Any]) -> nn.Module:
    """
    Import hetero_moe.models.experts.{name}_expert and build via common entrypoints.
    """
    module_name = f"hetero_moe.models.experts.{name}_expert"
    m = importlib.import_module(module_name)

    # Try factory functions first
    for fn_name in ("build_from_config", "build_model", "build", "get_model", "get_expert", "make", "create"):
        if hasattr(m, fn_name):
            try:
                return getattr(m, fn_name)(cfg)
            except TypeError:
                # allow zero-arg create()
                return getattr(m, fn_name)()

    # Try common class names
    for cls_name in ("Expert", "Model", "GraphExpert", "SmilesExpert"):
        if hasattr(m, cls_name):
            return getattr(m, cls_name)(cfg)

    raise RuntimeError(f"Could not build expert '{name}' via {module_name}.*")


class Router(nn.Module):
    def __init__(self, in_dim: int, num_experts: int):
        super().__init__()
        hid = max(64, in_dim // 2)
        self.in_dim = in_dim
        self.net = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, hid),
            nn.GELU(),
            nn.Linear(hid, num_experts),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def _last_step_logits(expert: nn.Module, batch_like: Dict[str, Any]) -> torch.Tensor:
    out = expert(batch_like)
    logits = out["logits"] if isinstance(out, dict) and "logits" in out else out
    if logits.dim() == 3:
        logits = logits[:, -1, :]
    return logits  # (B,V)


def _auto_target_key(batch: Dict[str, Any], prefer: Optional[str]) -> str:
    if prefer and prefer in batch:
        return prefer
    for k in ("tgt_token_ids", "target_ids", "labels", "tgt", "target", "y"):
        if k in batch:
            return k
    if "input_ids" in batch:
        print("[train] note: could not find target tensor (try keys like tgt_token_ids/labels).")
        print("[auto-detect] using 'input_ids' as target.")
        return "input_ids"
    raise KeyError("Target not found; expected keys like tgt_token_ids/labels/tgt/target/y.")


# ---------------------------
# NTF loop (with TF warmup)
# ---------------------------

def decode_and_loss_moe(
    experts: List[nn.Module],
    router: Router,
    batch: Dict[str, Any],
    *,
    device: torch.device,
    target_key: str,
    max_steps: int,
    stop_on_eos: bool,
    tf_warmup_steps: int,
    pad_id: int,
    eos_id: int,
    label_smoothing: float,
    router_temp: float,
    router_entropy_beta: float,
) -> Tuple[torch.Tensor, float, Dict[str, Any]]:
    gold: torch.Tensor = batch[target_key].to(device)  # (B,T)
    gate_feats: Optional[torch.Tensor] = batch.get("gate_features", None)
    if gate_feats is not None:
        gate_feats = gate_feats.to(device)

    B = gold.size(0)
    valid = make_valid_mask(gold, pad_id=pad_id, eos_id=eos_id)  # (B,T)

    loss_num = torch.tensor(0.0, device=device)
    loss_den = torch.tensor(0.0, device=device)
    tok_correct = torch.tensor(0.0, device=device)
    tok_total = torch.tensor(0.0, device=device)

    pred_prefix = torch.empty((B, 0), dtype=torch.long, device=device)
    em_correct = torch.tensor(0.0, device=device)
    em_total = torch.tensor(0.0, device=device)

    router_entropy_accum = torch.tensor(0.0, device=device)
    router_entropy_count = 0

    T_cap = min(max_steps, gold.shape[1])
    in_dim = router.in_dim

    for t in range(T_cap):
        use_tf = (t < tf_warmup_steps)
        gold_prefix = gold[:, :t]
        step_prefix = gold_prefix if use_tf else pred_prefix

        # build batch-like on device
        tmp: Dict[str, Any] = {}
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                tmp[k] = v.to(device, non_blocking=True)
            else:
                tmp[k] = v
        tmp["input_ids"] = step_prefix
        tmp["decoder_input_ids"] = step_prefix

        logits_list = []
        V = None
        for ex in experts:
            lg = _last_step_logits(ex, tmp)  # (B,V)
            if V is None:
                V = lg.shape[-1]
            logits_list.append(lg)

        # router input
        if gate_feats is not None and gate_feats.dim() == 2 and gate_feats.size(1) == in_dim:
            gate_in = gate_feats
        else:
            gate_in = torch.zeros(B, in_dim, device=device)

        gate_logits = router(gate_in)  # (B,E)
        p = torch.softmax(gate_logits / router_temp, dim=-1)
        router_entropy = -(p * (p.clamp_min(1e-8)).log()).sum(dim=-1).mean()
        router_entropy_accum += router_entropy
        router_entropy_count += 1

        lg = torch.stack(logits_list, dim=1)           # (B,E,V)
        lg = (p.unsqueeze(-1) * lg).sum(dim=1)         # (B,V)

        if t >= gold.shape[1]:
            break

        valid_t = valid[:, t]                          # (B,)
        ce_t = F.cross_entropy(
            lg, gold[:, t],
            reduction='none',
            ignore_index=pad_id,
            label_smoothing=label_smoothing
        )  # (B,)

        loss_num += (ce_t * valid_t.float()).sum()
        loss_den += valid_t.float().sum().clamp_min(1.0)

        pred_t = lg.argmax(dim=-1)
        tok_correct += ((pred_t == gold[:, t]) & valid_t).float().sum()
        tok_total   += valid_t.float().sum().clamp_min(1.0)

        # grow prefix
        next_token = pred_t
        pred_prefix = torch.cat([pred_prefix, next_token.unsqueeze(1)], dim=1) if t > 0 else next_token.unsqueeze(1)

        if stop_on_eos and (next_token == eos_id).all():
            break

    base_loss = loss_num / loss_den.clamp_min(1.0)
    if router_entropy_count > 0 and router_entropy_beta != 0.0:
        base_loss = base_loss + router_entropy_beta * (router_entropy_accum / router_entropy_count)

    # exact-match per sample up to first EOS (compare gold vs pred_prefix)
    for i in range(B):
        Lg = first_eos_len(gold[i], eos_id=eos_id, pad_id=pad_id)
        Lp = first_eos_len(pred_prefix[i], eos_id=eos_id, pad_id=pad_id)
        L = max(Lg, Lp)
        if torch.equal(gold[i][:L], pred_prefix[i][:L]):
            em_correct += 1.0
        em_total += 1.0

    tok_acc = (tok_correct / tok_total.clamp_min(1.0)).item()
    return base_loss, tok_acc, {
        "tok_correct": float(tok_correct.item()),
        "tok_total": float(tok_total.item()),
        "em_correct": float(em_correct.item()),
        "em_total": float(em_total.item()),
    }


@torch.no_grad()
def exact_match_eval(
    experts: List[nn.Module],
    router: Router,
    loader: DataLoader,
    *,
    device: torch.device,
    target_key: str,
    max_steps: int,
    stop_on_eos: bool,
    pad_id: int,
    eos_id: int,
    router_temp: float,
    max_batches: int = 100,
) -> float:
    em_correct = 0
    em_total = 0
    taken = 0
    in_dim = router.in_dim

    for batch in loader:
        if taken >= max_batches:
            break
        taken += 1

        gold = batch[target_key].to(device)
        gate_feats = batch.get("gate_features", None)
        if gate_feats is not None:
            gate_feats = gate_feats.to(device)

        B = gold.size(0)
        pred_prefix = torch.empty((B, 0), dtype=torch.long, device=device)
        T_cap = min(max_steps, gold.shape[1])

        for t in range(T_cap):
            tmp: Dict[str, Any] = {}
            for k, v in batch.items():
                if isinstance(v, torch.Tensor):
                    tmp[k] = v.to(device, non_blocking=True)
                else:
                    tmp[k] = v
            tmp["input_ids"] = pred_prefix
            tmp["decoder_input_ids"] = pred_prefix

            logits_list = []
            for ex in experts:
                lg = _last_step_logits(ex, tmp)
                logits_list.append(lg)

            if gate_feats is not None and gate_feats.dim() == 2 and gate_feats.size(1) == in_dim:
                gate_in = gate_feats
            else:
                gate_in = torch.zeros(B, in_dim, device=device)

            gate_logits = router(gate_in)
            p = torch.softmax(gate_logits / router_temp, dim=-1)
            lg = torch.stack(logits_list, dim=1)
            lg = (p.unsqueeze(-1) * lg).sum(dim=1)

            next_token = lg.argmax(dim=-1)
            pred_prefix = torch.cat([pred_prefix, next_token.unsqueeze(1)], dim=1) if t > 0 else next_token.unsqueeze(1)
            if stop_on_eos and (next_token == eos_id).all():
                break

        for i in range(B):
            gt_i = gold[i]
            pr_i = pred_prefix[i]
            Lg = first_eos_len(gt_i, eos_id=eos_id, pad_id=pad_id)
            Lp = first_eos_len(pr_i, eos_id=eos_id, pad_id=pad_id)
            L = max(Lg, Lp)
            if torch.equal(gt_i[:L], pr_i[:L]):
                em_correct += 1
            em_total += 1

    return em_correct / max(1, em_total)


# ---------------------------
# Main
# ---------------------------

def main():
    parser = argparse.ArgumentParser("Train N-expert MoE (non-teacher-forcing)")
    parser.add_argument("--experts", type=str, required=True, help="Comma-separated expert names (e.g. graph,smiles)")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--train_bin", type=str, required=True)
    parser.add_argument("--valid_bin", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=1337)

    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--grad_clip", type=float, default=1.0)

    parser.add_argument("--max_steps_per_seq", type=int, default=256)
    parser.add_argument("--stop_on_eos", action="store_true")

    parser.add_argument("--pad_id", type=int, default=0)
    parser.add_argument("--eos_id", type=int, default=2)
    parser.add_argument("--label_smoothing", type=float, default=0.0)
    parser.add_argument("--tf_warmup_steps", type=int, default=64)

    parser.add_argument("--router_use_gatefeats", action="store_true")
    parser.add_argument("--router_gate_dim", type=int, default=2048)
    parser.add_argument("--router_temp", type=float, default=1.5)
    parser.add_argument("--router_entropy_beta", type=float, default=1e-3)

    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--pin_memory", action="store_true")
    parser.add_argument("--persistent_workers", action="store_true")

    parser.add_argument("--log_every", type=int, default=100)
    parser.add_argument("--metrics_csv", type=str, default=None)
    parser.add_argument("--batch_metrics_csv", type=str, default=None)
    parser.add_argument("--save_path", type=str, default=None)

    parser.add_argument("--target_key", type=str, default=None)
    parser.add_argument("--inspect_batch", action="store_true")

    parser.add_argument("--gpu_debug", action="store_true")
    parser.add_argument("--gpu_report_every", type=int, default=1000)

    parser.add_argument("--valid_eval_em", action="store_true")
    parser.add_argument("--valid_em_batches", type=int, default=100)

    args = parser.parse_args()
    set_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"[setup] device={device.type} | workers={args.num_workers} | pin_memory={bool(args.pin_memory)}")
    if args.gpu_debug and device.type == "cuda":
        gpu_brief("setup", device)

    # Load data
    train_dict = _npz_to_arrays(args.train_bin)
    valid_dict = _npz_to_arrays(args.valid_bin)

    # Infer vocab size from the data (prevents None/str in cfg from breaking nn.Embedding)
    inferred_vocab = infer_vocab_size_from_data(train_dict, valid_dict,
                                                prefer=["tgt_token_ids", "target_ids", "labels", "input_ids"])
    # Ensure it's > special IDs
    inferred_vocab = max(inferred_vocab, args.eos_id + 1, args.pad_id + 1)

    # Datasets / loaders
    train_ds = NpzSeqDataset(train_dict)
    valid_ds = NpzSeqDataset(valid_dict)

    collate = lambda b: collate_batch(b, pad_id=args.pad_id)
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=args.pin_memory,
        persistent_workers=args.persistent_workers, collate_fn=collate, drop_last=False
    )
    valid_loader = DataLoader(
        valid_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=args.pin_memory,
        persistent_workers=args.persistent_workers, collate_fn=collate, drop_last=False
    )

    # Inspect one batch (for key names & shapes)
    if args.inspect_batch:
        try:
            b0 = next(iter(train_loader))
            pretty = {}
            for k, v in b0.items():
                if isinstance(v, torch.Tensor):
                    pretty[k] = (str(v.dtype).replace("torch.", ""), tuple(v.shape))
                elif isinstance(v, list):
                    pretty[k] = "list"
                else:
                    pretty[k] = type(v).__name__
            print(f"[inspect/train] keys: {pretty}")
        except Exception as e:
            print(f"[inspect/train] failed: {e}")

    # Load config
    def _load_cfg(path: str) -> Dict[str, Any]:
        if path.endswith(".json"):
            with open(path, "r") as f:
                return json.load(f)
        try:
            import yaml  # type: ignore
            with open(path, "r") as f:
                return yaml.safe_load(f)
        except Exception:
            return {}
    base_cfg = _load_cfg(args.config) or {}

    # Build experts with sanitized per-expert configs
    expert_names = [x.strip() for x in args.experts.split(",") if x.strip()]
    experts: List[nn.Module] = []

    def as_int(x, default: int) -> int:
        try:
            return int(x)
        except Exception:
            return int(default)

    for name in expert_names:
        cfg_ex = dict(base_cfg)  # shallow copy
        # Fill essentials robustly (avoid None/strings breaking torch layers)
        cfg_ex["vocab_size"] = as_int(cfg_ex.get("vocab_size", inferred_vocab), inferred_vocab)
        cfg_ex["hidden"] = as_int(cfg_ex.get("hidden", cfg_ex.get("hidden_size", 512)), 512)
        # Special IDs (keep consistent with flags)
        cfg_ex["pad_id"] = as_int(cfg_ex.get("pad_id", args.pad_id), args.pad_id)
        cfg_ex["eos_id"] = as_int(cfg_ex.get("eos_id", args.eos_id), args.eos_id)
        cfg_ex.setdefault("bos_id", 1)

        # Build and move
        ex = dynamic_build_expert(name, cfg_ex)
        ex.to(device)
        ex.train()
        experts.append(ex)

    # Build router
    if args.router_use_gatefeats:
        router_in_dim = int(args.router_gate_dim)
    else:
        # when no gate_features provided, route from a tiny learned input (zeros of right dim)
        router_in_dim = len(experts)
    router = Router(router_in_dim, num_experts=len(experts)).to(device)

    # Optimizer
    params = list(router.parameters())
    for ex in experts:
        params += list(ex.parameters())
    optim = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)

    # CSV headers
    if args.metrics_csv and not os.path.exists(args.metrics_csv):
        with open(args.metrics_csv, "w") as f:
            f.write("epoch,split,loss,tok_acc,em(optional)\n")
    if args.batch_metrics_csv and not os.path.exists(args.batch_metrics_csv):
        with open(args.batch_metrics_csv, "w") as f:
            f.write("step,split,loss,tok_acc,ips\n")

    global_step = 0
    for epoch in range(1, args.epochs + 1):
        # ---------- Train ----------
        for ex in experts: ex.train()
        router.train()

        loss_running = 0.0
        acc_running = 0.0
        run_count = 0
        last_log_t = time.time()

        for batch_idx, batch in enumerate(train_loader, start=1):
            global_step += 1
            target_key = _auto_target_key(batch, args.target_key)
            for k, v in batch.items():
                if isinstance(v, torch.Tensor):
                    batch[k] = v.to(device, non_blocking=True)

            loss, tok_acc, _ = decode_and_loss_moe(
                experts, router, batch,
                device=device,
                target_key=target_key,
                max_steps=args.max_steps_per_seq,
                stop_on_eos=args.stop_on_eos,
                tf_warmup_steps=args.tf_warmup_steps,
                pad_id=args.pad_id,
                eos_id=args.eos_id,
                label_smoothing=args.label_smoothing,
                router_temp=args.router_temp,
                router_entropy_beta=args.router_entropy_beta,
            )

            optim.zero_grad(set_to_none=True)
            loss.backward()
            if args.grad_clip and args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(params, args.grad_clip)
            optim.step()

            loss_running += float(loss.item())
            acc_running += float(tok_acc)
            run_count += 1

            if run_count % args.log_every == 0:
                dt = time.time() - last_log_t
                ips = args.log_every / max(dt, 1e-6)
                print(f"[train] step {global_step} | loss {loss_running / args.log_every:.4f} | "
                      f"tok_acc {acc_running / args.log_every:.4f} | {ips:.2f} it/s")
                if args.batch_metrics_csv:
                    with open(args.batch_metrics_csv, "a") as f:
                        f.write(f"{global_step},train,{loss_running / args.log_every:.6f},{acc_running / args.log_every:.6f},{ips:.2f}\n")
                loss_running = 0.0
                acc_running = 0.0
                last_log_t = time.time()

            if args.gpu_debug and (global_step % args.gpu_report_every == 0) and device.type == "cuda":
                gpu_brief(f"train@{global_step}", device)

        # ---------- Valid ----------
        for ex in experts: ex.eval()
        router.eval()

        with torch.no_grad():
            v_loss_sum, v_loss_cnt = 0.0, 0.0
            v_acc_sum, v_acc_cnt = 0.0, 0.0
            v_em_sum, v_em_cnt = 0.0, 0.0
            for batch in valid_loader:
                target_key = _auto_target_key(batch, args.target_key)
                for k, v in batch.items():
                    if isinstance(v, torch.Tensor):
                        batch[k] = v.to(device, non_blocking=True)
                loss, tok_acc, aux = decode_and_loss_moe(
                    experts, router, batch,
                    device=device,
                    target_key=target_key,
                    max_steps=args.max_steps_per_seq,
                    stop_on_eos=args.stop_on_eos,
                    tf_warmup_steps=0,                  # NO TF in eval
                    pad_id=args.pad_id,
                    eos_id=args.eos_id,
                    label_smoothing=0.0,                # no LS in eval
                    router_temp=args.router_temp,
                    router_entropy_beta=0.0,            # no router reg in eval
                )
                v_loss_sum += float(loss.item()); v_loss_cnt += 1.0
                v_acc_sum  += float(tok_acc);    v_acc_cnt  += 1.0
                if isinstance(aux, dict) and "em_correct" in aux and "em_total" in aux:
                    v_em_sum += float(aux["em_correct"]); v_em_cnt += float(aux["em_total"])

            valid_loss = v_loss_sum / max(1.0, v_loss_cnt)
            valid_tok_acc = v_acc_sum / max(1.0, v_acc_cnt)
            valid_em = (v_em_sum / max(1.0, v_em_cnt)) if v_em_cnt > 0 else None

        em_val = valid_em
        if args.valid_eval_em and em_val is None:
            # grab a target key from one fresh batch
            vb = next(iter(DataLoader(valid_ds, batch_size=args.batch_size, collate_fn=collate)))
            tkey = _auto_target_key(vb, args.target_key)
            em_val = exact_match_eval(
                experts, router, valid_loader, device=device,
                target_key=tkey,
                max_steps=args.max_steps_per_seq,
                stop_on_eos=args.stop_on_eos,
                pad_id=args.pad_id, eos_id=args.eos_id,
                router_temp=args.router_temp,
                max_batches=args.valid_em_batches,
            )

        if em_val is None:
            print(f"epoch {epoch} | valid_loss {valid_loss:.4f} | valid_tok_acc {valid_tok_acc:.4f}")
        else:
            print(f"epoch {epoch} | valid_loss {valid_loss:.4f} | valid_tok_acc {valid_tok_acc:.4f} | valid_em {em_val:.4f}")

        if args.metrics_csv:
            with open(args.metrics_csv, "a") as f:
                if em_val is None:
                    f.write(f"{epoch},valid,{valid_loss:.6f},{valid_tok_acc:.6f},\n")
                else:
                    f.write(f"{epoch},valid,{valid_loss:.6f},{valid_tok_acc:.6f},{em_val:.6f}\n")

        if args.save_path:
            os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
            torch.save({
                "router": router.state_dict(),
                "experts": [ex.state_dict() for ex in experts],
                "args": vars(args),
                "epoch": epoch,
                "inferred_vocab": inferred_vocab,
            }, args.save_path)


if __name__ == "__main__":
    main()

