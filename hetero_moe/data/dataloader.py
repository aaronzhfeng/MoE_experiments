from __future__ import annotations

from typing import Any, Dict, List, Tuple

import numpy as np
import torch
from hetero_moe.utils.gating_features import (
    compute_gate_features_from_tokens,
    try_compute_morgan_from_smiles,
)


def collate_seq_batch(samples: List[Dict[str, Any]], use_morgan: bool = False, smiles: List[str] = None) -> Dict[str, torch.Tensor]:
    # samples: list of dicts {src_ids, tgt_ids, idx}
    src = [torch.as_tensor(s["src_ids"], dtype=torch.long) for s in samples]
    tgt = [torch.as_tensor(s["tgt_ids"], dtype=torch.long) for s in samples]
    # right trim padding length to max in batch
    max_src = max(s.ne(0).sum().item() for s in src)
    max_tgt = max(t.ne(0).sum().item() for t in tgt)
    src = torch.stack([s[:max_src] for s in src])
    tgt = torch.stack([t[:max_tgt] for t in tgt])
    # Gate features: prefer Morgan from SMILES if present in a side-channel (not available here),
    # otherwise hash token ids to a bag-of-ids feature as a fallback.
    if use_morgan and smiles is not None:
        mf = try_compute_morgan_from_smiles(smiles, n_bits=2048)
        gate_feats = mf if mf is not None else compute_gate_features_from_tokens(src, num_bits=2048)
    else:
        gate_feats = compute_gate_features_from_tokens(src, num_bits=2048)
    batch = {
        "input_ids": src,
        "target_ids": tgt,
        "gate_features": gate_feats,
        "indices": torch.tensor([s["idx"] for s in samples], dtype=torch.long),
    }
    return batch


def collate_graph_batch(samples: List[Dict[str, Any]], use_morgan: bool = False, smiles: List[str] = None) -> Dict[str, torch.Tensor]:
    # Collate graph features from per-sample tuples to dense tensors compatible with G2S stubs.
    # Fallback to seq collation for tokens.
    out = collate_seq_batch(samples, use_morgan=use_morgan, smiles=smiles)
    graph_features = [s.get("graph_feature") for s in samples]
    if any(g is None for g in graph_features):
        return out
    # Concatenate per-batch with offsets like references/Graph2SMILES collate
    a_scopes_list, b_scopes_list, a_feat_list, b_feat_list, a_graph_list, b_graph_list = [], [], [], [], [], []
    atom_offset = 1  # reserve 0 as padding per Graph2SMILES convention
    bond_offset = 1
    for (a_scope, b_scope, a_feat, b_feat, a_graph, b_graph) in graph_features:
        a_scope = a_scope.copy()
        b_scope = b_scope.copy()
        a_feat = a_feat.copy()
        b_feat = b_feat.copy()
        a_graph = a_graph.copy()
        b_graph = b_graph.copy()
        a_scope[:, 0] += atom_offset
        b_scope[:, 0] += bond_offset
        a_graph[a_graph >= 999999999] = 0
        b_graph[b_graph >= 999999999] = 0
        a_scopes_list.append(a_scope)
        b_scopes_list.append(b_scope)
        a_feat_list.append(a_feat)
        b_feat_list.append(b_feat)
        a_graph_list.append(a_graph)
        b_graph_list.append(b_graph)
        atom_offset += a_feat.shape[0]
        bond_offset += b_feat.shape[0]

    a_scopes = torch.as_tensor(np.concatenate(a_scopes_list, axis=0), dtype=torch.int32)
    b_scopes = torch.as_tensor(np.concatenate(b_scopes_list, axis=0), dtype=torch.int32)
    a_features = torch.as_tensor(np.concatenate(a_feat_list, axis=0), dtype=torch.int32)
    b_features = torch.as_tensor(np.concatenate(b_feat_list, axis=0), dtype=torch.int32)
    a_graphs = torch.as_tensor(np.concatenate(a_graph_list, axis=0), dtype=torch.int32)
    b_graphs = torch.as_tensor(np.concatenate(b_graph_list, axis=0), dtype=torch.int32)
    out.update({
        "a_scopes": a_scopes,
        "b_scopes": b_scopes,
        "a_features": a_features,
        "b_features": b_features,
        "a_graphs": a_graphs,
        "b_graphs": b_graphs,
        # for G2S collate parity, also pass per-sample tuples when available
        "graph_feature": [s["graph_feature"] for s in samples],
    })
    return out


