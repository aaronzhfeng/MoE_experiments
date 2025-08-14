from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List, Optional, Tuple


@dataclass
class ReactionSample:
    reactants_smiles: List[str]
    product_smiles: str
    reactant_tokens: Optional[List[List[str]]] = None
    product_tokens: Optional[List[str]] = None
    graph_data: Optional[Any] = None
    condition_tokens: Optional[List[str]] = None
    precomputed_3d: Optional[Any] = None


class USPTODataset:
    """Placeholder dataset that wraps NPZ files produced by Graph2SMILES bridge.

    Parameters
    ----------
    npz_path:
        Path to the `.npz` archive containing token and optional graph features.
    smiles_file:
        Optional path to a text file with aligned SMILES strings. Used for gate
        feature computation but can be omitted.
    load_graph:
        When ``True`` (default) graph arrays are loaded if present.  Setting
        this to ``False`` avoids materializing large graph tensors when they are
        not needed (e.g. training the SMILES expert), speeding up startup and
        reducing memory usage.
    """

    def __init__(self, npz_path: str, smiles_file: Optional[str] = None, load_graph: bool = True):
        import numpy as np  # local import to avoid import cost when unused

        self._np = np
        self._file = npz_path
        self._feat = np.load(npz_path)
        self.size = len(self._feat["tgt_lens"]) if "tgt_lens" in self._feat else 0
        # Optional aligned SMILES (reactants) for gate features
        self._smiles = None
        if smiles_file is not None:
            try:
                with open(smiles_file, 'r') as f:
                    lines = [line.strip() for line in f]
                if len(lines) >= self.size:
                    self._smiles = lines
            except Exception:
                self._smiles = None
        # Optional graph arrays from Graph2SMILES preprocessing. These can be
        # extremely large, so allow callers to opt out of loading them.
        graph_keys = [
            "a_scopes",
            "b_scopes",
            "a_features",
            "b_features",
            "a_graphs",
            "b_graphs",
            "a_scopes_lens",
            "b_scopes_lens",
            "a_features_lens",
            "b_features_lens",
        ]
        self.has_graph = load_graph and all(k in self._feat for k in graph_keys)
        if self.has_graph:
            self.a_scopes = self._feat["a_scopes"]
            self.b_scopes = self._feat["b_scopes"]
            self.a_features = self._feat["a_features"]
            self.b_features = self._feat["b_features"]
            self.a_graphs = self._feat["a_graphs"]
            self.b_graphs = self._feat["b_graphs"]
            self.a_scopes_indices = self._len2idx(self._feat["a_scopes_lens"])
            self.b_scopes_indices = self._len2idx(self._feat["b_scopes_lens"])
            self.a_features_indices = self._len2idx(self._feat["a_features_lens"])
            self.b_features_indices = self._len2idx(self._feat["b_features_lens"])

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, idx: int):
        # Returns a dict with ids and index for advanced collate functions
        src = self._feat["src_token_ids"][idx]
        tgt = self._feat["tgt_token_ids"][idx]
        sample = {"src_ids": src, "tgt_ids": tgt, "idx": idx}
        if self._smiles is not None:
            sample["smiles"] = self._smiles[idx]
        if self.has_graph:
            a_st, a_en = self.a_scopes_indices[idx]
            b_st, b_en = self.b_scopes_indices[idx]
            af_st, af_en = self.a_features_indices[idx]
            bf_st, bf_en = self.b_features_indices[idx]
            a_scope = self.a_scopes[a_st:a_en]
            b_scope = self.b_scopes[b_st:b_en]
            a_feat = self.a_features[af_st:af_en]
            b_feat = self.b_features[bf_st:bf_en]
            a_graph = self.a_graphs[af_st:af_en]
            b_graph = self.b_graphs[bf_st:bf_en]
            sample["graph_feature"] = (a_scope, b_scope, a_feat, b_feat, a_graph, b_graph)
        return sample

    def _len2idx(self, lens) -> Any:
        end_indices = self._np.cumsum(lens)
        start_indices = self._np.concatenate([[0], end_indices[:-1]], axis=0)
        indices = self._np.stack([start_indices, end_indices], axis=1)
        return indices


