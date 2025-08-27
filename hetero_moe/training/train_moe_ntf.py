from __future__ import annotations

import argparse, os, time, csv
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader

try:
    from torch.utils.tensorboard import SummaryWriter
except Exception:
    SummaryWriter = None

from hetero_moe.data.dataset import USPTODataset
from hetero_moe.data.dataloader import collate_seq_batch, collate_graph_batch
from hetero_moe.models.experts.smiles_expert import SmilesExpert
from hetero_moe.models.experts.graph_expert import GraphExpert
from hetero_moe.models.experts.cond_expert import ConditionExpert
from hetero_moe.models.experts.gnn3d_expert import GNN3DExpert
from hetero_moe.utils.config import load_yaml, apply_overrides


# -------------------- Expert factory --------------------
def build_expert(name: str, vocab_size: int, hidden: int, layers: int, heads: int, ff: int):
    n = name.lower()
    if n == "smiles":
        return SmilesExpert(vocab_size=vocab_size, hidden=hidden, layers=layers, heads=heads, ff=ff)
    if n == "graph":
        return GraphExpert(vocab_size=vocab_size, hidden=hidden, layers=layers, heads=heads, ff=ff)
    if n == "cond":
        return ConditionExpert(vocab_size=vocab_size)
    if n == "gnn3d":
        return GNN3DExpert(vocab_size=vocab_size)
    raise ValueError(f"Unknown expert: {name}")


# -------------------- Router --------------------
class RouterNet(nn.Module):
    """
    Fuses: (a) token-prefix embedding and (b) optional gate_features (B, gate_dim)
    """
    def __init__(self, vocab_size: int, num_experts: int, hidden: int = 128, take_tokens: int = 8,
                 use_gatefeats: bool = True, gate_dim: int = 2048):
        super().__init__()
        self.take_tokens = take_tokens
        self.use_gatefeats = use_gatefeats
        self.emb = nn.Embedding(vocab_size, hidden)
        self.proj_tok = nn.Linear(hidden, num_experts)
        if use_gatefeats:
            self.mlp_gate = nn.Sequential(
                nn.Linear(gate_dim, hidden),
                nn.GELU(),
                nn.Linear(hidden, num_experts),
            )

    def forward(self, prefix_ids: torch.Tensor, gate_feats: torch.Tensor | None = None) -> torch.Tensor:
        Tuse = min(prefix_ids.size(1), self.take_tokens)
        h = self.emb(prefix_ids[:, :Tuse]).mean(dim=1)      # (B,H)
        logits = self.proj_tok(h)                           # (B,K)
        if self.use_gatefeats and (gate_feats is not None):
            logits = logits + self.mlp_gate(gate_feats)     # fuse (sum)
        return logits


# -------------------- Argparse --------------------
def parse_args():
    p = argparse.ArgumentParser("train_moe_ntf (two experts, non-teacher-forcing)")
    # experts
    p.add_argument("--experts", type=str, required=True,
                   help="Comma-separated two experts, e.g. 'smiles,smiles' or 'graph,smiles'")
    p.add_argument("--vocab_size", type=int, default=512)
    p.add_argument("--hidden", type=int, default=256)
    p.add_argument("--layers", type=int, default=4)
    p.add_argument("--heads", type=int, default=8)
    p.add_argument("--ff", type=int, default=1024)

    # data
    p.add_argument("--train_bin", required=True)
    p.add_argument("--valid_bin", required=True)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--pin_memory", action="store_true")
    p.add_argument("--prefetch_factor", type=int, default=2)
    p.add_argument("--persistent_workers", action="store_true")

    # training
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--pad_id", type=int, default=0)
    p.add_argument("--bos_id", type=int, default=2)
    p.add_argument("--eos_id", type=int, default=3)
    p.add_argument("--grad_clip", type=float, default=1.0)

    # NTF controls
    p.add_argument("--max_steps_per_seq", type=int, default=256,
                   help="Max unrolled autoregressive steps per sequence in a batch (cap for speed).")
    p.add_argument("--stop_on_eos", action="store_true", help="Stop unrolling early when all in batch hit EOS.")
    p.add_argument("--tf_warmup_steps", type=int, default=32,
                   help="During NTF, for the first N steps per seq use gold tokens to build prefix.")
    p.add_argument("--aux_alpha", type=float, default=0.3,
                   help="Weight for per-expert auxiliary CE (encourages each expert to learn).")

    # logging / IO
    p.add_argument("--log_every", type=int, default=100)
    p.add_argument("--metrics_csv", type=str, default="")
    p.add_argument("--batch_metrics_csv", type=str, default="")
    p.add_argument("--save_path", type=str, default="runs/moe_ntf/ckpt.pt")
    p.add_argument("--logdir", type=str, default="")
    p.add_argument("--gpu_report_every", type=int, default=0)
    p.add_argument("--gpu_debug", action="store_true")

    # validation
    p.add_argument("--valid_eval_em", action="store_true",
                   help="During validation, compute free-running EM by generating full sequences.")
    p.add_argument("--valid_em_batches", type=int, default=100, help="Number of batches for EM (cap cost).")

    # config
    p.add_argument("--config", type=str, default="")

    # target resolution / inspection
    p.add_argument("--target_key", type=str, default="",
                   help="Explicit target tensor key in batch (e.g. target_ids, tgt_token_ids, labels).")
    p.add_argument("--inspect_batch", action="store_true",
                   help="Print first batch keys/dtypes/shapes (train & valid).")

    # router gating features
    p.add_argument("--router_use_gatefeats", action="store_true",
                   help="Let router use batch['gate_features'] if available.")
    p.add_argument("--router_gate_dim", type=int, default=2048)
    return p.parse_args()


# -------------------- Utils --------------------
def _ensure_csv(path: str, header: List[str]):
    if not path:
        return
    d = os.path.dirname(path)
    if d and not os.path.isdir(d):
        os.makedirs(d, exist_ok=True)
    if not os.path.exists(path):
        with open(path, "w", newline="") as f:
            csv.writer(f).writerow(header)


def _append_csv(path: str, row: List):
    if not path:
        return
    with open(path, "a", newline="") as f:
        csv.writer(f).writerow(row)


def _gpu_report(tag: str):
    if not torch.cuda.is_available():
        print(f"[gpu:{tag}] CUDA not available", flush=True); return
    idx = torch.cuda.current_device()
    prop = torch.cuda.get_device_properties(idx)
    alloc = torch.cuda.memory_allocated(idx) / (1024**2)
    reserved = torch.cuda.memory_reserved(idx) / (1024**2)
    peak = torch.cuda.max_memory_allocated(idx) / (1024**2)
    print(f"[gpu:{tag}] id={idx} {prop.name} alloc={alloc:.1f}MB reserved={reserved:.1f}MB peak={peak:.1f}MB", flush=True)
    torch.cuda.reset_peak_memory_stats()


def _find_target(batch: Dict, target_key: str = "") -> torch.Tensor:
    # user-specified
    if target_key and target_key in batch and torch.is_tensor(batch[target_key]):
        return batch[target_key].long()

    # Prefer your dataset's naming first
    priority = [
        "target_ids",               # <-- FIRST for graph2smiles_npz
        "tgt_token_ids", "labels", "tgt", "target", "y", "tgt_ids",
        "tgt_out_ids", "output_ids", "out_ids",
        "smiles_token_ids", "product_token_ids",
    ]
    for k in priority:
        t = batch.get(k, None)
        if torch.is_tensor(t) and t.dtype in (torch.int64, torch.int32) and t.ndim == 2:
            return t.long()

    # heuristic fallback: pick a 2D int tensor that's NOT obviously a source/graph tensor
    candidates: List[Tuple[str, torch.Tensor]] = []
    for k, v in batch.items():
        if torch.is_tensor(v) and v.dtype in (torch.int64, torch.int32) and v.ndim == 2:
            if all(s not in k.lower() for s in ["src", "input", "graph", "adj", "edge", "edge_index"]):
                candidates.append((k, v))
    if candidates:
        k, v = max(candidates, key=lambda kv: kv[1].shape[1])
        print(f"[auto-detect] using '{k}' as target.", flush=True)
        return v.long()

    raise KeyError(f"Target not found; pass --target_key. Batch keys: {list(batch.keys())}")


def _maybe_shift_targets(gold: torch.Tensor, bos_id: int, pad_id: int) -> torch.Tensor:
    """If most samples start with BOS, shift left by one so t=0 predicts the first real token."""
    if gold.ndim != 2 or gold.size(1) < 2:
        return gold
    frac_bos0 = (gold[:, 0] == bos_id).float().mean().item()
    if frac_bos0 > 0.5:
        gold = torch.cat([gold[:, 1:], torch.full((gold.size(0), 1), pad_id, device=gold.device, dtype=gold.dtype)], dim=1)
    return gold


def _apply_key_aliases(tmp: Dict[str, torch.Tensor], batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """
    Ensure experts see canonical names:
      - 'src_token_ids' for source (alias from 'input_ids' if present)
      - 'tgt_token_ids' is already the free-running prefix
      - 'labels' mirrors 'tgt_token_ids' for experts that read 'labels'
    """
    if "src_token_ids" not in tmp:
        if "input_ids" in batch and torch.is_tensor(batch["input_ids"]):
            tmp["src_token_ids"] = batch["input_ids"]
    if "labels" not in tmp and "tgt_token_ids" in tmp and torch.is_tensor(tmp["tgt_token_ids"]):
        tmp["labels"] = tmp["tgt_token_ids"]
    return tmp


def _last_step_logits(expert: nn.Module, batch_like: Dict) -> torch.Tensor:
    out = expert(batch_like)
    logits = out.get("logits", None)
    if logits is None:
        raise RuntimeError("Expert must return {'logits': ...}. Please modify experts to expose logits.")
    if logits.dim() == 2:      # (B,V)
        return logits
    if logits.dim() == 3:      # (B,t,V)
        return logits[:, -1, :]
    raise RuntimeError(f"Unsupported logits shape: {tuple(logits.shape)}")


@torch.no_grad()
def _greedy_generate_moe(
    experts: List[nn.Module], router: nn.Module, batch: Dict[str, torch.Tensor],
    bos_id: int, eos_id: int, pad_id: int, max_len: int, device: str,
    use_gatefeats: bool
) -> torch.Tensor:
    gold = _find_target(batch)  # just to get B
    B = gold.size(0)
    prefix = torch.full((B, 1), bos_id, dtype=torch.long, device=device)
    done = torch.zeros(B, dtype=torch.bool, device=device)
    for _ in range(max_len):
        tmp = {k: (v if not torch.is_tensor(v) else v) for k, v in batch.items()}
        tmp["tgt_token_ids"] = prefix
        tmp = _apply_key_aliases(tmp, batch)
        gate_feats = batch.get("gate_features", None)
        if not (isinstance(gate_feats, torch.Tensor) and use_gatefeats):
            gate_feats = None

        gates = router(prefix, gate_feats=gate_feats).softmax(dim=-1)         # (B,K)
        log_gates = torch.log(torch.clamp(gates, 1e-8))                       # (B,K)
        step_logits = []
        for ex in experts:
            lg = _last_step_logits(ex, tmp)                                   # (B,V)
            step_logits.append(lg)
        stacked_logits = torch.stack(step_logits, dim=-1)                     # (B,V,K)
        logits_mix = torch.logsumexp(stacked_logits + log_gates.unsqueeze(1), dim=-1)  # (B,V)
        nxt = logits_mix.argmax(dim=-1)                                       # (B,)
        nxt[done] = pad_id
        prefix = torch.cat([prefix, nxt.unsqueeze(1)], dim=1)
        done |= nxt.eq(eos_id)
        if torch.all(done):
            break
    return prefix


# ---------- Eagerize NPZ to avoid BadZipFile under workers ----------
def _eagerize_npz_in_dataset(ds):
    feat = getattr(ds, "_feat", None)
    if feat is not None and type(feat).__name__ == "NpzFile":
        t0 = time.time()
        files = list(feat.files)
        eager = {k: feat[k] for k in files}  # read/decompress once
        ds._feat = eager
        try:
            getattr(feat, "close", lambda: None)()
        except Exception:
            pass
        print(f"[eagerize] materialized {len(files)} arrays in {time.time()-t0:.2f}s", flush=True)


# -------------------- Training (NTF) --------------------
def train_epoch_ntf(
    experts: List[nn.Module], router: nn.Module, loader: DataLoader, optimizer: optim.Optimizer,
    device: str, pad_id: int, bos_id: int, eos_id: int, max_steps_per_seq: int, stop_on_eos: bool,
    grad_clip: float, log_every: int, gpu_report_every: int, batch_metrics_csv: str,
    target_key: str, inspect_batch: bool, router_use_gatefeats: bool,
    tf_warmup_steps: int, aux_alpha: float
) -> Dict[str, float]:
    for m in experts + [router]:
        m.train(True)

    _ensure_csv(batch_metrics_csv, ["phase", "global_step", "step_in_seq", "loss", "tok_acc", "active"])
    global_steps = 0
    total_loss, total_tok, total_correct = 0.0, 0, 0
    seq_em_hits, seq_em_total = 0, 0
    t0 = time.time()

    for batch_idx, batch in enumerate(loader, start=1):
        batch = {k: (v.to(device, non_blocking=True) if torch.is_tensor(v) else v) for k, v in batch.items()}

        if inspect_batch and batch_idx == 1:
            def _shape(x): return tuple(x.shape) if torch.is_tensor(x) else type(x).__name__
            print("[inspect/train] keys:", {k: (str(v.dtype), _shape(v)) if torch.is_tensor(v) else type(v).__name__
                  for k, v in batch.items()}, flush=True)

        gold = _find_target(batch, target_key)     # (B,T)
        gold = _maybe_shift_targets(gold, bos_id, pad_id)
        B, T = gold.shape
        prefix = torch.full((B, 1), bos_id, dtype=torch.long, device=device)
        alive = torch.ones(B, dtype=torch.bool, device=device)

        steps = min(T, max_steps_per_seq)
        gen_full = [prefix.clone()]

        for s in range(steps):
            tmp = {k: (v if not torch.is_tensor(v) else v) for k, v in batch.items()}
            tmp["tgt_token_ids"] = prefix
            tmp = _apply_key_aliases(tmp, batch)

            gate_feats = batch.get("gate_features", None)
            if not (isinstance(gate_feats, torch.Tensor) and router_use_gatefeats):
                gate_feats = None

            gates = router(prefix, gate_feats=gate_feats).softmax(dim=-1)      # (B,2)
            log_gates = torch.log(torch.clamp(gates, 1e-8))                    # (B,2)

            step_logits = []
            for ex in experts:
                lg = _last_step_logits(ex, tmp)                                # (B,V)
                step_logits.append(lg)

            stacked_logits = torch.stack(step_logits, dim=-1)                  # (B,V,2)
            logits_mix = torch.logsumexp(stacked_logits + log_gates.unsqueeze(1), dim=-1)  # (B,V)

            tgt_s = gold[:, s]
            active_mask = alive & tgt_s.ne(pad_id)
            if active_mask.any():
                # main CE on MoE mixed logits
                loss_main = F.cross_entropy(logits_mix[active_mask], tgt_s[active_mask], reduction="mean")

                # optional per-expert auxiliary CE (weighted by detached gate mass over active samples)
                loss = loss_main
                if aux_alpha and aux_alpha > 0:
                    with torch.no_grad():
                        w = gates[active_mask].detach().mean(dim=0)            # (2,)
                    aux = 0.0
                    for k, lg in enumerate(step_logits):
                        aux = aux + F.cross_entropy(lg[active_mask], tgt_s[active_mask], reduction="mean") * w[k]
                    loss = loss_main + aux_alpha * aux

                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                if grad_clip and grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_([p for m in experts+[router] for p in m.parameters()], grad_clip)
                optimizer.step()

                with torch.no_grad():
                    pred_s = logits_mix.argmax(dim=-1)
                    correct = (pred_s == tgt_s) & active_mask
                    total_correct += int(correct.sum().item())
                    total_tok += int(active_mask.sum().item())
                    total_loss += float(loss_main.detach().cpu())  # log primary CE
                    global_steps += 1

                _append_csv(batch_metrics_csv, ["train", global_steps, s+1,
                                                float(loss_main.detach().cpu()),
                                                float(correct.sum().item())/max(int(active_mask.sum().item()),1),
                                                int(active_mask.sum().item())])

            # greedy next token
            nxt = logits_mix.argmax(dim=-1)              # (B,)

            # tiny NTF warmup (build prefix with gold for early steps)
            if s < tf_warmup_steps:
                nxt_tf = gold[:, s]
                nxt = torch.where(alive, nxt_tf, nxt)

            nxt[~alive] = pad_id
            prefix = torch.cat([prefix, nxt.unsqueeze(1)], dim=1)
            gen_full.append(nxt.unsqueeze(1))

            if stop_on_eos:
                alive = alive & (~nxt.eq(eos_id))
                if not alive.any():
                    break

            if gpu_report_every and (global_steps % gpu_report_every == 0) and device.startswith("cuda"):
                _gpu_report(f"train@{global_steps}")
            if log_every and (global_steps % log_every == 0):
                it_s = global_steps / max(1e-9, (time.time() - t0))
                tok_acc = (total_correct / total_tok) if total_tok > 0 else float("nan")
                print(f"[train] step {global_steps} | loss {total_loss/max(1,global_steps):.4f} | tok_acc {tok_acc:.4f} | {it_s:.2f} it/s",
                      flush=True)

        with torch.no_grad():
            gen_seq = torch.cat(gen_full, dim=1)
            def strip(seq_2d: torch.Tensor) -> List[List[int]]:
                out = []
                for row in seq_2d.tolist():
                    cur = []
                    for t in row:
                        if t == eos_id:
                            break
                        cur.append(t)
                    out.append(cur)
                return out
            gold_list = strip(gold)
            gen_list = strip(gen_seq)
            for g, p in zip(gold_list, gen_list):
                seq_em_hits += int(g == p)
                seq_em_total += 1

    metrics = {
        "loss": total_loss / max(1, global_steps),
        "token_acc": (total_correct / total_tok) if total_tok > 0 else float("nan"),
        "seq_em": (seq_em_hits / max(1, seq_em_total)) if seq_em_total > 0 else float("nan"),
        "steps": global_steps,
    }
    return metrics


# -------------------- Validation (NTF loss + optional EM) --------------------
@torch.no_grad()
def valid_epoch_ntf(
    experts: List[nn.Module], router: nn.Module, loader: DataLoader, device: str,
    pad_id: int, bos_id: int, eos_id: int, max_steps_per_seq: int, stop_on_eos: bool,
    log_every: int, gpu_report_every: int, compute_em: bool, em_batches: int, batch_metrics_csv: str,
    target_key: str, inspect_batch: bool, router_use_gatefeats: bool,
    tf_warmup_steps: int
) -> Dict[str, float]:
    for m in experts + [router]:
        m.train(False)

    _ensure_csv(batch_metrics_csv, ["phase", "global_step", "step_in_seq", "loss", "tok_acc", "active"])
    global_steps = 0
    total_loss, total_tok, total_correct = 0.0, 0, 0
    seq_em_hits, seq_em_total = 0, 0
    t0 = time.time()

    for batch_idx, batch in enumerate(loader, start=1):
        batch = {k: (v.to(device, non_blocking=True) if torch.is_tensor(v) else v) for k, v in batch.items()}

        if inspect_batch and batch_idx == 1:
            def _shape(x): return tuple(x.shape) if torch.is_tensor(x) else type(x).__name__
            print("[inspect/valid] keys:", {k: (str(v.dtype), _shape(v)) if torch.is_tensor(v) else type(v).__name__
                  for k, v in batch.items()}, flush=True)

        gold = _find_target(batch, target_key)     # (B,T)
        gold = _maybe_shift_targets(gold, bos_id, pad_id)
        B, T = gold.shape
        prefix = torch.full((B, 1), bos_id, dtype=torch.long, device=device)
        alive = torch.ones(B, dtype=torch.bool, device=device)
        steps = min(T, max_steps_per_seq)
        gen_full = [prefix.clone()]

        for s in range(steps):
            tmp = {k: (v if not torch.is_tensor(v) else v) for k, v in batch.items()}
            tmp["tgt_token_ids"] = prefix
            tmp = _apply_key_aliases(tmp, batch)

            gate_feats = batch.get("gate_features", None)
            if not (isinstance(gate_feats, torch.Tensor) and router_use_gatefeats):
                gate_feats = None

            gates = router(prefix, gate_feats=gate_feats).softmax(dim=-1)
            log_gates = torch.log(torch.clamp(gates, 1e-8))
            step_logits = []
            for ex in experts:
                lg = _last_step_logits(ex, tmp)
                step_logits.append(lg)

            stacked_logits = torch.stack(step_logits, dim=-1)                  # (B,V,2)
            logits_mix = torch.logsumexp(stacked_logits + log_gates.unsqueeze(1), dim=-1)  # (B,V)

            tgt_s = gold[:, s]
            active_mask = alive & tgt_s.ne(pad_id)
            if active_mask.any():
                loss_main = F.cross_entropy(logits_mix[active_mask], tgt_s[active_mask], reduction="mean")
                total_loss += float(loss_main.detach().cpu())
                pred_s = logits_mix.argmax(dim=-1)
                correct = (pred_s == tgt_s) & active_mask
                total_correct += int(correct.sum().item())
                total_tok += int(active_mask.sum().item())
                global_steps += 1
                _append_csv(batch_metrics_csv, ["valid", global_steps, s+1,
                                                float(loss_main.detach().cpu()),
                                                float(correct.sum().item())/max(int(active_mask.sum().item()),1),
                                                int(active_mask.sum().item())])

            nxt = logits_mix.argmax(dim=-1)
            # match train: tiny TF warmup for prefix building (no effect on loss)
            if s < tf_warmup_steps:
                nxt_tf = gold[:, s]
                nxt = torch.where(alive, nxt_tf, nxt)

            nxt[~alive] = pad_id
            prefix = torch.cat([prefix, nxt.unsqueeze(1)], dim=1)
            gen_full.append(nxt.unsqueeze(1))
            if stop_on_eos:
                alive = alive & (~nxt.eq(eos_id))
                if not alive.any():
                    break

            if gpu_report_every and (global_steps % gpu_report_every == 0) and device.startswith("cuda"):
                _gpu_report(f"valid@{global_steps}")
            if log_every and (global_steps % log_every == 0):
                it_s = global_steps / max(1e-9, (time.time() - t0))
                tok_acc = (total_correct / total_tok) if total_tok > 0 else float("nan")
                print(f"[valid] step {global_steps} | loss {total_loss/max(1,global_steps):.4f} | tok_acc {tok_acc:.4f} | {it_s:.2f} it/s",
                      flush=True)

        if compute_em and batch_idx <= em_batches:
            full_gen = _greedy_generate_moe(
                experts, router, batch, bos_id, eos_id, pad_id, max_len=T, device=device,
                use_gatefeats=router_use_gatefeats
            )
            def strip(seq_2d: torch.Tensor) -> List[List[int]]:
                out = []
                for row in seq_2d.tolist():
                    cur = []
                    for t in row:
                        if t == eos_id:
                            break
                        cur.append(t)
                    out.append(cur)
                return out
            gold_list = strip(gold)
            gen_list = strip(full_gen)
            for g, p in zip(gold_list, gen_list):
                seq_em_hits += int(g == p)
                seq_em_total += 1

    return {
        "loss": total_loss / max(1, global_steps),
        "token_acc": (total_correct / total_tok) if total_tok > 0 else float("nan"),
        "seq_em": (seq_em_hits / max(1, seq_em_total)) if seq_em_total > 0 else float("nan"),
        "steps": global_steps,
    }


# -------------------- Main --------------------
def main():
    args = parse_args()
    if args.config:
        cfg = load_yaml(args.config)
        apply_overrides(args, cfg)

    device = args.device
    if device.startswith("cuda"):
        assert torch.cuda.is_available()
        if ":" in device:
            torch.cuda.set_device(int(device.split(":")[1]))
    torch.backends.cudnn.benchmark = True

    print(f"[setup] device={device} | workers={args.num_workers} | pin_memory={args.pin_memory}", flush=True)
    if args.gpu_debug and device.startswith("cuda"):
        _gpu_report("setup")

    # logging
    metrics_csv = args.metrics_csv or (args.save_path + ".metrics.csv")
    batch_csv = args.batch_metrics_csv or (args.save_path + ".batches.csv")
    _ensure_csv(metrics_csv, ["epoch", "phase", "loss", "token_acc", "seq_em"])

    writer = None
    if args.logdir and SummaryWriter is not None:
        os.makedirs(args.logdir, exist_ok=True)
        writer = SummaryWriter(log_dir=args.logdir)
        print(f"[setup] TensorBoard -> {args.logdir}", flush=True)

    # data
    train_ds = USPTODataset(args.train_bin)
    valid_ds = USPTODataset(args.valid_bin)
    _eagerize_npz_in_dataset(train_ds)
    _eagerize_npz_in_dataset(valid_ds)

    collate = collate_graph_batch if getattr(train_ds, "has_graph", False) else collate_seq_batch
    dl_kwargs = dict(
        batch_size=args.batch_size, collate_fn=collate,
        pin_memory=args.pin_memory and device.startswith("cuda"),
        num_workers=args.num_workers,
        prefetch_factor=args.prefetch_factor if args.num_workers > 0 else None,
        persistent_workers=args.persistent_workers if args.num_workers > 0 else False,
    )
    train_loader = DataLoader(train_ds, shuffle=True, **dl_kwargs)
    valid_loader = DataLoader(valid_ds, shuffle=False, **dl_kwargs)

    # experts + router
    names = [x.strip() for x in args.experts.split(",")]
    if len(names) != 2:
        raise ValueError("--experts must list exactly two experts, e.g. 'smiles,smiles' or 'graph,smiles'")
    experts = [
        build_expert(n, args.vocab_size, args.hidden, args.layers, args.heads, args.ff).to(device)
        for n in names
    ]
    router = RouterNet(args.vocab_size, num_experts=2,
                       hidden=args.hidden // 2, take_tokens=8,
                       use_gatefeats=args.router_use_gatefeats,
                       gate_dim=args.router_gate_dim).to(device)

    # optimizer (joint)
    params = list(router.parameters())
    for ex in experts:
        params += list(ex.parameters())
    optimizer = optim.AdamW(params, lr=args.lr)

    best_val = float("inf")
    for epoch in range(1, args.epochs + 1):
        tr = train_epoch_ntf(
            experts, router, train_loader, optimizer, device,
            pad_id=args.pad_id, bos_id=args.bos_id, eos_id=args.eos_id,
            max_steps_per_seq=args.max_steps_per_seq, stop_on_eos=args.stop_on_eos,
            grad_clip=args.grad_clip, log_every=args.log_every,
            gpu_report_every=args.gpu_report_every, batch_metrics_csv=batch_csv,
            target_key=args.target_key, inspect_batch=args.inspect_batch,
            router_use_gatefeats=args.router_use_gatefeats,
            tf_warmup_steps=args.tf_warmup_steps, aux_alpha=args.aux_alpha,
        )
        va = valid_epoch_ntf(
            experts, router, valid_loader, device,
            pad_id=args.pad_id, bos_id=args.bos_id, eos_id=args.eos_id,
            max_steps_per_seq=args.max_steps_per_seq, stop_on_eos=args.stop_on_eos,
            log_every=args.log_every, gpu_report_every=args.gpu_report_every,
            compute_em=args.valid_eval_em, em_batches=args.valid_em_batches,
            batch_metrics_csv=batch_csv.replace(".csv", ".valid.csv"),
            target_key=args.target_key, inspect_batch=args.inspect_batch,
            router_use_gatefeats=args.router_use_gatefeats,
            tf_warmup_steps=args.tf_warmup_steps,
        )

        print(f"epoch {epoch} | "
              f"train_loss {tr['loss']:.4f} | train_tok_acc {tr['token_acc']:.4f} | train_seq_em {tr['seq_em']:.4f} || "
              f"valid_loss {va['loss']:.4f} | valid_tok_acc {va['token_acc']:.4f} | valid_seq_em {va['seq_em']:.4f}",
              flush=True)

        _append_csv(metrics_csv, [epoch, "train", tr["loss"], tr["token_acc"], tr["seq_em"]])
        _append_csv(metrics_csv, [epoch, "valid", va["loss"], va["token_acc"], va["seq_em"]])

        if writer is not None:
            writer.add_scalar("moe_ntf/Loss/train", tr["loss"], epoch)
            writer.add_scalar("moe_ntf/Loss/valid", va["loss"], epoch)
            writer.add_scalar("moe_ntf/AccToken/train", tr["token_acc"], epoch)
            writer.add_scalar("moe_ntf/AccToken/valid", va["token_acc"], epoch)
            writer.add_scalar("moe_ntf/SeqEM/train", tr["seq_em"], epoch)
            writer.add_scalar("moe_ntf/SeqEM/valid", va["seq_em"], epoch)

        # checkpoint
        if args.save_path:
            ckpt = {
                "epoch": epoch,
                "experts_names": names,
                "router": router.state_dict(),
                "experts": [ex.state_dict() for ex in experts],
                "train": tr, "valid": va,
            }
            os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
            torch.save(ckpt, args.save_path)
            if va["loss"] < best_val:
                best_val = va["loss"]
                torch.save(ckpt, args.save_path + ".best")

    if writer is not None:
        writer.close()


if __name__ == "__main__":
    main()

