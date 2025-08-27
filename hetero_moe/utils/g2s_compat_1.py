from __future__ import annotations

import os
import sys
import importlib.util
from typing import List, Tuple

import torch


def _import_g2s_collate():
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    g2s_dir = os.path.join(repo_root, "references", "Graph2SMILES")
    utils_dir = os.path.join(g2s_dir, "utils")
    data_utils_path = os.path.join(utils_dir, "data_utils.py")

    if not os.path.exists(data_utils_path):
        raise FileNotFoundError(f"Graph2SMILES data_utils.py not found at {data_utils_path}")

    # Temporarily ensure Graph2SMILES/utils is importable for absolute imports 'chem_utils', 'rxn_graphs'
    prev_sys_path = list(sys.path)
    sys.path.insert(0, utils_dir)
    try:
        spec = importlib.util.spec_from_file_location("graph2smiles_data_utils", data_utils_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Failed to load spec for {data_utils_path}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module.collate_graph_features
    finally:
        # Restore sys.path to its previous state
        sys.path = prev_sys_path


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


