from __future__ import annotations

import os
import sys
from typing import List, Tuple

import torch


def _import_g2s_collate():
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    g2s_dir = os.path.join(repo_root, "references", "Graph2SMILES")
    if g2s_dir not in sys.path:
        sys.path.insert(0, g2s_dir)
    from utils.data_utils import collate_graph_features  # type: ignore
    return collate_graph_features


def collate_g2s_compat(graph_features: List[Tuple]):
    """Use Graph2SMILES' collate to produce (fnode, fmess, agraph, bgraph, atom_scope, bond_scope)."""
    collate_graph_features = _import_g2s_collate()
    fnode, fmess, agraph, bgraph, atom_scope, bond_scope = collate_graph_features(
        graph_features, directed=True, use_rxn_class=False
    )
    # ensure tensors
    if not isinstance(fnode, torch.Tensor):
        fnode = torch.as_tensor(fnode)
    if not isinstance(fmess, torch.Tensor):
        fmess = torch.as_tensor(fmess)
    if not isinstance(agraph, torch.Tensor):
        agraph = torch.as_tensor(agraph)
    if not isinstance(bgraph, torch.Tensor):
        bgraph = torch.as_tensor(bgraph)
    return fnode, fmess, agraph, bgraph, atom_scope, bond_scope


