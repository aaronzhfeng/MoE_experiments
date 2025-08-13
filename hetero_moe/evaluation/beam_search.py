from __future__ import annotations

from typing import Dict, List, Tuple

import torch


def greedy_decode_logits(logits: torch.Tensor, eos_id: int = 3) -> torch.Tensor:
    """Greedy decode from per-step logits [B, T, V] -> ids [B, T].
    Stops at EOS; keeps padding after EOS as EOS.
    """
    ids = logits.argmax(-1)
    if ids.ndim != 2:
        return ids
    # enforce EOS tail
    b, t = ids.shape
    for i in range(b):
        row = ids[i]
        done = False
        for j in range(t):
            if row[j].item() == eos_id:
                done = True
            elif done:
                row[j] = eos_id
    return ids


@torch.no_grad()
def topk_from_expert_beam(expert, batch: Dict, k: int, itos: List[str], token_ids=None, max_len: int = 128) -> List[List[str]]:
    """Beam search decoding (simple log-prob beam) returning top-k SMILES per sample.

    Assumes expert.forward can provide logits given partial targets via teacher forcing.
    """
    from hetero_moe.utils.smiles import ids_to_smiles
    if token_ids is None:
        from hetero_moe.utils.tokens import TokenIds
        token_ids = TokenIds()
    bsz = batch["input_ids"].size(0)
    device = batch["input_ids"].device
    results: List[List[str]] = [[] for _ in range(bsz)]

    # initialize beams per sample
    beams = [([(token_ids.bos,)], 0.0)] * bsz  # not used directly; we keep per-sample lists below
    per_sample_beams: List[List[Tuple[List[int], float]]] = [[([token_ids.bos], 0.0)] for _ in range(bsz)]

    for _step in range(max_len - 1):
        new_beams: List[List[Tuple[List[int], float]]] = [[] for _ in range(bsz)]
        for i in range(bsz):
            # expand each partial hypothesis by one token
            candidates: List[Tuple[List[int], float]] = []
            for seq, score in per_sample_beams[i]:
                if seq and seq[-1] == token_ids.eos:
                    candidates.append((seq, score))
                    continue
                # form a temporary batch with this single sample and seq as target
                tgt_ids = torch.tensor([seq], dtype=torch.long, device=device)
                out = expert({"input_ids": batch["input_ids"][i:i+1], "target_ids": tgt_ids})
                logits = out["logits"]  # [1, t, V]
                next_logit = logits[:, -1, :]  # [1, V]
                logp = torch.log_softmax(next_logit, dim=-1)
                topk_logp, topk_ids = torch.topk(logp, k, dim=-1)
                for j in range(k):
                    nid = int(topk_ids[0, j].item())
                    nsc = float(topk_logp[0, j].item())
                    candidates.append((seq + [nid], score + nsc))
            # prune to top-k
            candidates.sort(key=lambda x: x[1], reverse=True)
            new_beams[i] = candidates[:k]
        per_sample_beams = new_beams

    # finalize: convert top-k ids to SMILES
    for i in range(bsz):
        seqs = per_sample_beams[i]
        smi_list = []
        for seq, score in seqs:
            smi = ids_to_smiles(itos, seq)
            smi_list.append(smi)
        results[i] = smi_list
    return results


