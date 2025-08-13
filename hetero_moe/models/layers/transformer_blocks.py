from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class LearnedPositionalEmbedding(nn.Module):
    def __init__(self, max_len: int, dim: int, padding_idx: int = 0):
        super().__init__()
        self.pe = nn.Embedding(max_len, dim, padding_idx=padding_idx)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, D]
        b, t, d = x.shape
        pos = torch.arange(t, device=x.device, dtype=torch.long).unsqueeze(0).expand(b, t)
        return x + self.pe(pos)


class SimpleTransformerEncoder(nn.Module):
    def __init__(self, vocab_size: int, dim: int = 256, layers: int = 4, heads: int = 8, ff: int = 1024,
                 dropout: float = 0.1, max_len: int = 512, pad_id: int = 0):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, dim, padding_idx=pad_id)
        self.pos = LearnedPositionalEmbedding(max_len=max_len, dim=dim, padding_idx=pad_id)
        enc_layer = nn.TransformerEncoderLayer(d_model=dim, nhead=heads, dim_feedforward=ff, dropout=dropout,
                                               batch_first=True)
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=layers)

    def forward(self, input_ids: torch.Tensor, src_key_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = self.embed(input_ids)
        x = self.pos(x)
        mem = self.encoder(x, src_key_padding_mask=src_key_padding_mask)
        return mem  # [B, S, D]


class SimpleTransformerDecoder(nn.Module):
    def __init__(self, vocab_size: int, dim: int = 256, layers: int = 4, heads: int = 8, ff: int = 1024,
                 dropout: float = 0.1, max_len: int = 512, pad_id: int = 0):
        super().__init__()
        self.pad_id = pad_id
        self.embed = nn.Embedding(vocab_size, dim, padding_idx=pad_id)
        self.pos = LearnedPositionalEmbedding(max_len=max_len, dim=dim, padding_idx=pad_id)
        dec_layer = nn.TransformerDecoderLayer(d_model=dim, nhead=heads, dim_feedforward=ff, dropout=dropout,
                                               batch_first=True)
        self.decoder = nn.TransformerDecoder(dec_layer, num_layers=layers)
        self.out = nn.Linear(dim, vocab_size)

    def _generate_square_subsequent_mask(self, sz: int, device) -> torch.Tensor:
        return torch.triu(torch.ones(sz, sz, device=device) * float('-inf'), diagonal=1)

    def forward(self, target_ids: torch.Tensor, memory: torch.Tensor,
                memory_key_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # target_ids: [B, T], memory: [B, S, D]
        B, T = target_ids.shape
        tgt = self.embed(target_ids)
        tgt = self.pos(tgt)
        attn_mask = self._generate_square_subsequent_mask(T, device=target_ids.device)
        logits = self.out(self.decoder(tgt=tgt, memory=memory, tgt_mask=attn_mask,
                                       memory_key_padding_mask=memory_key_padding_mask))
        return logits  # [B, T, V]

    @torch.no_grad()
    def generate_greedy(self, bos_id: int, eos_id: int, max_len: int, memory: torch.Tensor,
                         memory_key_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        B = memory.size(0)
        out_ids = torch.full((B, 1), bos_id, dtype=torch.long, device=memory.device)
        for _ in range(max_len - 1):
            logits = self.forward(out_ids, memory, memory_key_padding_mask)
            next_id = logits[:, -1, :].argmax(-1, keepdim=True)
            out_ids = torch.cat([out_ids, next_id], dim=1)
            if torch.all(next_id.squeeze(1) == eos_id):
                break
        return out_ids


