from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConditionExpert(nn.Module):
    """Stubbed condition-aware expert with segment positional idea left for later.
    """

    def __init__(self, vocab_size: int = 512, hidden: int = 256):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, hidden)
        self.decoder = nn.Linear(hidden, vocab_size)

    def forward(self, batch: Dict):
        input_ids = batch.get("input_ids")
        target_ids = batch.get("target_ids")
        if input_ids is None or target_ids is None:
            raise ValueError("ConditionExpert expects 'input_ids' and 'target_ids' in batch")

        # Produce logits aligned with target length to avoid shape mismatches
        x_tgt = self.embed(target_ids)  # [B, T, H]
        logits = self.decoder(x_tgt)    # [B, T, V]
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), target_ids.view(-1), ignore_index=0)
        return {"logits": logits, "loss": loss}


