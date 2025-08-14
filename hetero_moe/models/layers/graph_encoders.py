from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class GraphNodeFeatureEncoder(nn.Module):
    """Embeds Graph2SMILES sparse atom feature columns and pools to a graph embedding.

    Expects batch dict to contain:
      - a_features: [N_atoms_total, F_cols] int32, where 999999999 denotes padding sentinel
      - a_scopes: [N_mols, 2] int32 with (start_idx, length) per molecule

    This module is a lightweight placeholder for a full D-MPNN.
    """

    def __init__(self, hidden_size: int = 256, per_col_dim: int = 32):
        super().__init__()
        self.hidden_size = hidden_size
        self.per_col_dim = per_col_dim
        # We create embeddings lazily per column based on observed maxima
        self._embeddings = nn.ModuleList()
        self.proj = nn.Linear(per_col_dim, hidden_size)

    def _ensure_embeddings(self, a_features: torch.Tensor):
        num_cols = a_features.shape[1]
        if len(self._embeddings) < num_cols:
            # initialize missing columns as simple hash-embedding tables
            for _ in range(len(self._embeddings), num_cols):
                # Use a moderate vocab size with modulo hashing in forward
                emb = nn.Embedding(4096, self.per_col_dim)
                nn.init.normal_(emb.weight, mean=0.0, std=0.02)
                self._embeddings.append(emb)

    def forward(self, batch: Dict) -> torch.Tensor:
        a_features: Optional[torch.Tensor] = batch.get("a_features")
        a_scopes: Optional[torch.Tensor] = batch.get("a_scopes")
        if a_features is None or a_scopes is None:
            # Fallback: zeros graph embedding
            bsz = batch["input_ids"].shape[0]
            return torch.zeros(bsz, self.hidden_size, device=batch["input_ids"].device)

        a_features = a_features.to(torch.long)
        self._ensure_embeddings(a_features)

        sentinel = 999999999
        # Hash feature values per column into embedding indices
        col_embeds = []
        for col, emb in enumerate(self._embeddings):
            vals = a_features[:, col]
            idx = torch.remainder(vals, emb.num_embeddings)
            # mask sentinel to zero index safely
            idx = torch.where(vals >= sentinel, torch.zeros_like(idx), idx)
            col_vec = emb(idx)  # [N_atoms, per_col_dim]
            col_embeds.append(col_vec)
        atom_repr = sum(col_embeds)  # simple additive combination per atom
        atom_repr = self.proj(atom_repr)  # [N_atoms, hidden]

        # Pool per-molecule using a_scopes (start, length)
        bsz = a_scopes.shape[0]
        graph_embeds = []
        for i in range(bsz):
            start, length = a_scopes[i].tolist()
            h = atom_repr[start:start + length]
            g = h.mean(dim=0) if length > 0 else torch.zeros_like(atom_repr[0])
            graph_embeds.append(g)
        graph_embeds = torch.stack(graph_embeds, dim=0)  # [B, hidden]
        return graph_embeds


class GraphMPNEncoder(nn.Module):
    """Lightweight message passing over atom features using bond connectivity.

    Uses a simple neighborhood aggregation for a few steps.
    Inputs expected in batch:
      - a_features: [N_atoms, F_cols] int32 (categorical columns with sentinel)
      - b_features: [N_bonds, 2 + F_bond] int32 where first two columns are (u, v)
      - a_scopes: [B, 2] int32 start,length per molecule
    """

    def __init__(self, hidden_size: int = 256, per_col_dim: int = 32, steps: int = 3):
        super().__init__()
        self.hidden_size = hidden_size
        self.per_col_dim = per_col_dim
        self.steps = steps
        self._embeddings = nn.ModuleList()
        self.proj_in = nn.Linear(per_col_dim, hidden_size)
        self.w_self = nn.Linear(hidden_size, hidden_size)
        self.w_neigh = nn.Linear(hidden_size, hidden_size)
        self.proj_out = nn.Linear(hidden_size, hidden_size)

    def _ensure_embeddings(self, a_features: torch.Tensor):
        num_cols = a_features.shape[1]
        if len(self._embeddings) < num_cols:
            for _ in range(len(self._embeddings), num_cols):
                emb = nn.Embedding(4096, self.per_col_dim)
                nn.init.normal_(emb.weight, mean=0.0, std=0.02)
                self._embeddings.append(emb)

    def forward(self, batch: Dict) -> torch.Tensor:
        a_features: Optional[torch.Tensor] = batch.get("a_features")
        b_features: Optional[torch.Tensor] = batch.get("b_features")
        a_scopes: Optional[torch.Tensor] = batch.get("a_scopes")
        if a_features is None or a_scopes is None or b_features is None:
            # fallback to zeros if connectivity not available
            bsz = batch["input_ids"].shape[0]
            return torch.zeros(bsz, self.hidden_size, device=batch["input_ids"].device)

        a_features = a_features.to(torch.long)
        b_features = b_features.to(torch.long)
        self._ensure_embeddings(a_features)

        sentinel = 999999999
        # atom initial embeddings via categorical columns
        col_embeds = []
        for col, emb in enumerate(self._embeddings):
            vals = a_features[:, col]
            idx = torch.remainder(vals, emb.num_embeddings)
            idx = torch.where(vals >= sentinel, torch.zeros_like(idx), idx)
            col_vec = emb(idx)
            col_embeds.append(col_vec)
        h = self.proj_in(sum(col_embeds))  # [N_atoms, H]

        # build directed edge index from bond endpoints
        u = b_features[:, 0].to(torch.long)
        v = b_features[:, 1].to(torch.long)
        src = torch.cat([u, v], dim=0)
        dst = torch.cat([v, u], dim=0)

        for _ in range(self.steps):
            agg = torch.zeros_like(h)
            # sum h[src] into dst positions
            agg.index_add_(0, dst, h[src])
            h = torch.relu(self.w_self(h) + self.w_neigh(agg))

        h = self.proj_out(h)

        # pool per molecule
        bsz = a_scopes.shape[0]
        graph_embeds = []
        for i in range(bsz):
            start, length = a_scopes[i].tolist()
            hi = h[start:start + length]
            g = hi.mean(dim=0) if length > 0 else torch.zeros_like(h[0])
            graph_embeds.append(g)
        return torch.stack(graph_embeds, dim=0)


class G2SMPNEncoder(nn.Module):
    """Encoder that expects pre-collated Graph2SMILES tensors (fnode, fmess, agraph, bgraph).

    This aligns with the D-GCN message passing in spirit but remains lightweight.
    """

    def __init__(self, hidden_size: int = 256):
        super().__init__()
        self.hidden_size = hidden_size
        # project one-hot fnode to hidden
        self.proj_in = nn.Linear( sum_dim_placeholder :=  sum([] or [1]) , hidden_size)  # placeholder overwritten at first forward
        self.w_msg = nn.Linear(hidden_size, hidden_size)
        self.w_update = nn.GRUCell(hidden_size, hidden_size)
        self.proj_out = nn.Linear(hidden_size, hidden_size)
        self._init_done = False

    def _maybe_init(self, fnode: torch.Tensor):
        if not self._init_done:
            in_dim = fnode.shape[1]
            self.proj_in = nn.Linear(in_dim, self.hidden_size).to(fnode.device)
            self._init_done = True

    def forward(self, fnode: torch.Tensor, fmess: torch.Tensor, agraph: torch.Tensor, bgraph: torch.Tensor,
                atom_scope: list, bond_scope: list) -> torch.Tensor:
        # fnode: [N_atoms, F], fmess: [N_bonds*2, 2 + BOND_FDIM] (per G2S collate), agraph: [N_atoms, max_deg]
        device = fnode.device
        self._maybe_init(fnode)
        h = torch.relu(self.proj_in(fnode))  # [N_atoms, H]

        # Build simple bond messages by gathering source atom states via agraph
        # agraph gives incoming edge ids; we derive adjacency by fmess first two columns hold (u, v)
        # In G2S, fmess columns are [u, v] + bond features. We'll create per-edge messages from u->v.
        u = fmess[:, 0].long()
        v = fmess[:, 1].long()
        # build messages m_e = W(h_u)
        m = torch.relu(self.w_msg(h[u]))
        # aggregate messages into atoms via destination indices v
        agg = torch.zeros_like(h)
        agg.index_add_(0, v, m)
        # update atom states using GRUCell per atom
        h = self.w_update(agg, h)
        h = torch.relu(h)
        g_embeds = []
        # atom_scope is a list; each element is an array of [start, length] rows for that sample's segments
        for scope in atom_scope:
            if isinstance(scope, torch.Tensor):
                s = scope.to(device)
            else:
                s = torch.as_tensor(scope, device=device)
            # Derive overall [start, length] covering this sample from first and last rows
            st = int(s[0, 0].item())
            last_start = int(s[-1, 0].item())
            last_len = int(s[-1, 1].item())
            ln = last_start + last_len - st
            hi = h[st:st + ln]
            g = hi.mean(dim=0) if ln > 0 else torch.zeros(self.hidden_size, device=device)
            g_embeds.append(g)
        g_embeds = torch.stack(g_embeds, dim=0)
        return self.proj_out(g_embeds)


