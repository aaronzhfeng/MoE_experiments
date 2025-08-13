from __future__ import annotations

from typing import List


def ids_to_tokens(itos: List[str], ids: List[int], eos_id: int = 3, pad_id: int = 0) -> List[str]:
    toks: List[str] = []
    for tid in ids:
        if tid == pad_id:
            continue
        if tid == eos_id:
            break
        if 0 <= tid < len(itos):
            toks.append(itos[tid])
    return toks


def tokens_to_smiles(tokens: List[str]) -> str:
    return "".join(tokens)


def ids_to_smiles(itos: List[str], ids: List[int], eos_id: int = 3, pad_id: int = 0) -> str:
    return tokens_to_smiles(ids_to_tokens(itos, ids, eos_id=eos_id, pad_id=pad_id))


def canonicalize_smiles(smi: str) -> str:
    try:
        from rdkit import Chem
    except Exception:
        return smi
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return ""
    return Chem.MolToSmiles(mol, isomericSmiles=True)


def tanimoto_diversity(smiles_list: List[str]) -> float:
    try:
        from rdkit import Chem, DataStructs
        from rdkit.Chem import AllChem
    except Exception:
        return 0.0
    fps = []
    for s in smiles_list:
        m = Chem.MolFromSmiles(s)
        if m is None:
            continue
        fps.append(AllChem.GetMorganFingerprintAsBitVect(m, 2, nBits=2048))
    if len(fps) < 2:
        return 0.0
    sims = []
    for i in range(len(fps)):
        for j in range(i + 1, len(fps)):
            sims.append(DataStructs.TanimotoSimilarity(fps[i], fps[j]))
    if not sims:
        return 0.0
    # diversity = 1 - similarity
    diversities = [1.0 - s for s in sims]
    return sum(diversities) / len(diversities)


