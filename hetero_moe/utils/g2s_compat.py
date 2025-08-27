from __future__ import annotations

import os
import sys
import importlib.util
from typing import List, Tuple

import torch


def _import_g2s_collate():
    """
    Dynamically import Graph2SMILES' collate from
    references/Graph2SMILES/utils/data_utils.py

    We temporarily extend sys.path with BOTH:
      - .../references/Graph2SMILES           (so 'utils.*' works as a package)
      - .../references/Graph2SMILES/utils     (so absolute 'chem_utils' also works)

    Then we load data_utils.py and return its `collate_graph_features` function.
    """
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    g2s_dir = os.path.join(repo_root, "references", "Graph2SMILES")
    utils_dir = os.path.join(g2s_dir, "utils")
    data_utils_path = os.path.join(utils_dir, "data_utils.py")

    if not os.path.exists(data_utils_path):
        raise FileNotFoundError(f"Graph2SMILES data_utils.py not found at {data_utils_path}")

    prev_sys_path = list(sys.path)
    try:
        # Ensure both package and module-level imports work
        if g2s_dir not in sys.path:
            sys.path.insert(0, g2s_dir)
        if utils_dir not in sys.path:
            sys.path.insert(0, utils_dir)

        spec = importlib.util.spec_from_file_location("graph2smiles_data_utils", data_utils_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Failed to load spec for {data_utils_path}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        if not hasattr(module, "collate_graph_features"):
            raise AttributeError("data_utils.py does not define 'collate_graph_features'")

        return module.collate_graph_features
    finally:
        # Restore sys.path to its previous state to avoid polluting global import state
        sys.path = prev_sys_path


def collate_g2s_compat(graph_features: List[Tuple]):
    """
    Use Graph2SMILES' collate to produce:
      (fnode, fmess, agraph, bgraph, atom_scope, bond_scope)
    """
    collate_graph_features = _import_g2s_collate()
    fnode, fmess, agraph, bgraph, atom_scope, bond_scope = collate_graph_features(
        graph_features, directed=True, use_rxn_class=False
    )

    # Ensure tensors (keep original dtypes; experts can cast as needed)
    if not isinstance(fnode, torch.Tensor):
        fnode = torch.as_tensor(fnode)
    if not isinstance(fmess, torch.Tensor):
        fmess = torch.as_tensor(fmess)
    if not isinstance(agraph, torch.Tensor):
        agraph = torch.as_tensor(agraph)
    if not isinstance(bgraph, torch.Tensor):
        bgraph = torch.as_tensor(bgraph)

    return fnode, fmess, agraph, bgraph, atom_scope, bond_scope

