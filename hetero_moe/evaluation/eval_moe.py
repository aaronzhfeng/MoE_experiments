from __future__ import annotations

import argparse
import json
from typing import Dict, List

import torch
from torch.utils.data import DataLoader

from hetero_moe.data.dataset import USPTODataset
from hetero_moe.data.dataloader import collate_seq_batch
from hetero_moe.models.moe import MoEModel
from hetero_moe.models.experts.smiles_expert import SmilesExpert
from hetero_moe.models.experts.graph_expert import GraphExpert
from hetero_moe.models.experts.cond_expert import ConditionExpert
from hetero_moe.models.experts.gnn3d_expert import GNN3DExpert
from hetero_moe.evaluation.beam_search import greedy_decode_logits, topk_from_expert_beam
from hetero_moe.utils.smiles import ids_to_smiles, canonicalize_smiles, tanimoto_diversity


def parse_args():
    p = argparse.ArgumentParser("eval_moe")
    p.add_argument("--test_bin", required=True)
    p.add_argument("--beam_size", type=int, default=5)
    p.add_argument("--k", type=int, default=5)
    p.add_argument("--load_path", type=str, default="")
    p.add_argument("--vocab_file", type=str, default="")
    p.add_argument("--pad_id", type=int, default=0)
    p.add_argument("--bos_id", type=int, default=2)
    p.add_argument("--eos_id", type=int, default=3)
    p.add_argument("--out", type=str, default="evaluation_results.json")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def main():
    args = parse_args()
    device = args.device

    ds = USPTODataset(args.test_bin)
    # choose collate based on graph availability
    from hetero_moe.data.dataloader import collate_graph_batch
    collate = collate_graph_batch if getattr(ds, "has_graph", False) else collate_seq_batch
    loader = DataLoader(ds, batch_size=8, shuffle=False, collate_fn=collate)

    experts = {
        "smiles": SmilesExpert(),
        "graph": GraphExpert(),
        "cond": ConditionExpert(),
        "gnn3d": GNN3DExpert(),
    }
    model = MoEModel(experts=experts, gate_feature_dim=2048).to(device)
    if args.load_path:
        ckpt = torch.load(args.load_path, map_location=device)
        model.load_state_dict(ckpt.get("model", ckpt))
    model.eval()

    # Metrics: avg loss, token acc, exact match; optional validity if vocab provided
    total = 0.0
    iters = 0
    correct = 0
    total_tokens = 0
    exact_match_total = 0
    sample_total = 0

    itos: List[str] = []
    if args.vocab_file:
        with open(args.vocab_file, "r") as f:
            itos = [line.strip().split("\t")[0] for line in f]
    use_rdkit = False
    try:
        from rdkit import Chem
        use_rdkit = True
    except Exception:
        pass
    valid_smiles = 0
    decoded_total = 0
    with torch.no_grad():
        for batch in loader:
            batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
            loss, aux = model(batch)
            total += float(loss.detach().cpu())
            iters += 1
            logits = model.predict_logits(batch)
            pred = greedy_decode_logits(logits)
            tgt = batch["target_ids"]
            mask = tgt.ne(0)
            correct += (pred.eq(tgt) & mask).sum().item()
            total_tokens += mask.sum().item()
            # exact match (sequence-level top-1) up to EOS (3), ignoring PAD (0)
            eos_id = args.eos_id
            bsz, tlen = tgt.shape
            exact_flags = []
            for i in range(bsz):
                row_t = tgt[i]
                row_p = pred[i]
                eos_pos = (row_t == eos_id).nonzero(as_tuple=False)
                cutoff = int(eos_pos[0].item()) if eos_pos.numel() > 0 else tlen
                ok = torch.all(row_t[:cutoff].eq(row_p[:cutoff]))
                exact_flags.append(bool(ok.item()))
            exact_match = sum(1 for f in exact_flags if f)
            exact_match_total += exact_match
            sample_total += bsz

            # Validity and approximate top-k (requires vocab to detokenize)
            if itos:
                for i in range(bsz):
                    smi = ids_to_smiles(itos, pred[i].tolist())
                    smi = canonicalize_smiles(smi)
                    decoded_total += 1
                    if use_rdkit and smi:
                        mol = Chem.MolFromSmiles(smi)
                        if mol is not None:
                            valid_smiles += 1

                # Top-k union of experts with beam decoding (restricted by router top-k if available)
                if args.k > 1:
                    tgt_strings = [canonicalize_smiles(ids_to_smiles(itos, tgt[i].tolist())) for i in range(bsz)]
                    union_lists: List[List[str]] = [[] for _ in range(bsz)]
                    allowed_idx = None
                    if isinstance(aux, dict) and "topk" in aux and aux["topk"][0] is not None:
                        allowed_idx = aux["topk"][0].detach().cpu()  # [B, K]
                    for e_idx, name in enumerate(model.expert_names):
                        expert = experts[name]
                        cand_lists = topk_from_expert_beam(expert, batch, args.k, itos, token_ids=None, max_len=tgt.size(1))
                        for i in range(bsz):
                            if allowed_idx is not None and e_idx not in set(allowed_idx[i].tolist()):
                                continue
                            for smi in cand_lists[i]:
                                smi_c = canonicalize_smiles(smi)
                                if smi_c and smi_c not in union_lists[i]:
                                    union_lists[i].append(smi_c)
                    union_hit = 0
                    diversity_scores = []
                    for i in range(bsz):
                        topk = union_lists[i][: args.k]
                        if tgt_strings[i] in topk:
                            union_hit += 1
                        diversity_scores.append(tanimoto_diversity(topk))
                    # store rolling average in results object
                    if 'union_hits' not in locals():
                        union_hits = 0
                        union_seen = 0
                        union_div_sum = 0.0
                    union_hits += union_hit
                    union_seen += bsz
                    union_div_sum += sum(diversity_scores) / max(1, len(diversity_scores))
    results = {
        "avg_loss": total / max(1, iters),
        "token_accuracy": (correct / total_tokens) if total_tokens > 0 else 0.0,
        "exact_match": (exact_match_total / max(1, sample_total)) if sample_total > 0 else 0.0,
    }
    if itos:
        results["validity_rate"] = (valid_smiles / max(1, decoded_total)) if decoded_total > 0 else 0.0
        if 'union_hits' in locals() and union_seen > 0:
            results["topk_union_exact"] = union_hits / union_seen
            results["topk_union_diversity"] = union_div_sum / (union_seen / bsz) if bsz > 0 else 0.0
    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()


