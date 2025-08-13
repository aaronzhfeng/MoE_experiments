from __future__ import annotations

import re
from typing import List


_SMILES_REGEX = re.compile(r"(\[[^\]]+]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\|\/|:|~|@|\?|>|\*|\$|\%[0-9]{2}|[0-9])")


def tokenize_smiles(smi: str) -> List[str]:
    tokens = [t for t in _SMILES_REGEX.findall(smi)]
    return tokens


