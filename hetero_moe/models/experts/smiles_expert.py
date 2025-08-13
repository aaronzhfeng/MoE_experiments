from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from hetero_moe.models.layers.transformer_blocks import (
    SimpleTransformerEncoder,
    SimpleTransformerDecoder,
)


class SmilesExpert(nn.Module):
    def __init__(self, vocab_size: int = 512, hidden: int = 256, layers: int = 4, heads: int = 8, ff: int = 1024):
        super().__init__()
        self.encoder = SimpleTransformerEncoder(vocab_size=vocab_size, dim=hidden, layers=layers, heads=heads, ff=ff)
        self.decoder = SimpleTransformerDecoder(vocab_size=vocab_size, dim=hidden, layers=layers, heads=heads, ff=ff)

    def forward(self, batch: Dict):
        # Expect 'input_ids' [B, S] and 'target_ids' [B, T]
        input_ids = batch.get("input_ids")
        target_ids = batch.get("target_ids")
        if input_ids is None or target_ids is None:
            raise ValueError("SmilesExpert expects 'input_ids' and 'target_ids' in batch")

        src_pad_mask = input_ids.eq(0)
        memory = self.encoder(input_ids, src_key_padding_mask=src_pad_mask)
        logits = self.decoder(target_ids, memory, memory_key_padding_mask=src_pad_mask)
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), target_ids.view(-1), ignore_index=0)
        return {"logits": logits, "loss": loss}


