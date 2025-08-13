from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from hetero_moe.models.layers.graph_encoders import GraphNodeFeatureEncoder, GraphMPNEncoder, G2SMPNEncoder
from hetero_moe.utils.g2s_compat import collate_g2s_compat
from hetero_moe.models.layers.transformer_blocks import SimpleTransformerDecoder


class GraphExpert(nn.Module):
    """Graph2SMILES-style expert (simplified): graph encoder + Transformer decoder.

    Expects graph features in batch (fnode/fmess/agraph/bgraph) but this stub
    only validates shapes by decoding from token embeddings.
    """

    def __init__(self, vocab_size: int = 512, hidden: int = 256, layers: int = 4, heads: int = 8, ff: int = 1024):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, hidden)
        self.decoder = nn.Linear(hidden, vocab_size)
        # Use G2S-compatible encoder when possible; fallback to simple MPN otherwise
        self.graph_encoder = G2SMPNEncoder(hidden_size=hidden)
        self.graph_encoder_fallback = GraphMPNEncoder(hidden_size=hidden)
        self.decoder = SimpleTransformerDecoder(vocab_size=vocab_size, dim=hidden, layers=layers, heads=heads, ff=ff)

    def forward(self, batch: Dict):
        # Expect 'input_ids' [B, T] and 'target_ids' [B, T]
        input_ids = batch.get("input_ids")
        target_ids = batch.get("target_ids")
        if input_ids is None or target_ids is None:
            raise ValueError("GraphExpert expects 'input_ids' and 'target_ids' in batch")

        # Prefer full G2S tensors if present to build memory; otherwise fallback
        if all(k in batch for k in ["a_scopes", "b_scopes", "a_features", "b_features", "a_graphs", "b_graphs"]):
            # Reconstruct per-sample graph_features for G2S collate
            # Note: our collate_graph_batch already concatenates per-batch; create list of tuples per sample
            # For parity, we reuse the pre-collated per-batch features by slicing per sample via scopes
            # Here, we simply re-run G2S collate on per-sample tuples prepared in dataset; if absent, fallback.
            if "graph_feature" in batch:
                graph_features = batch["graph_feature"]
            else:
                # fallback to simple MPN
                g = self.graph_encoder_fallback(batch).unsqueeze(1)
                logits = self.decoder(x + g)
                loss = F.cross_entropy(logits.view(-1, logits.size(-1)), target_ids.view(-1), ignore_index=0)
                return {"logits": logits, "loss": loss}
            fnode, fmess, agraph, bgraph, atom_scope, bond_scope = collate_g2s_compat(graph_features)
            memory = self.graph_encoder(fnode.to(input_ids.device), fmess.to(input_ids.device), agraph.to(input_ids.device), bgraph.to(input_ids.device), atom_scope, bond_scope)
            memory = memory.unsqueeze(1)  # [B, 1, H]
        else:
            memory = self.graph_encoder_fallback(batch).unsqueeze(1)  # [B, 1, H]
        # Decode conditioned on graph memory using teacher forcing with target_ids
        src_pad_mask = None
        logits = self.decoder(target_ids, memory, memory_key_padding_mask=src_pad_mask)
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), target_ids.view(-1), ignore_index=0)
        return {"logits": logits, "loss": loss}


