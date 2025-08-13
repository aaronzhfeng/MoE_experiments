from __future__ import annotations

from dataclasses import dataclass


@dataclass
class TokenIds:
    pad: int = 0
    unk: int = 1
    bos: int = 2
    eos: int = 3


