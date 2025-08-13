from __future__ import annotations

from typing import Optional

import torch


def compute_gate_features_from_tokens(input_ids: torch.Tensor, num_bits: int = 2048) -> torch.Tensor:
    """Hash token ids into a fixed-length bag-of-ids vector per sample.

    input_ids: [B, T] tensor of token ids
    returns: [B, num_bits] float tensor
    """
    bsz, _ = input_ids.shape
    feats = torch.zeros(bsz, num_bits, dtype=torch.float, device=input_ids.device)
    # simple modulo hashing and count accumulation
    idx = (input_ids % num_bits).long()
    for i in range(bsz):
        vals, counts = torch.unique(idx[i], return_counts=True)
        feats[i, vals] = counts.float()
    # normalize
    feats = feats / (feats.norm(p=2, dim=1, keepdim=True) + 1e-6)
    return feats


def try_compute_morgan_from_smiles(smiles_batch: Optional[list], n_bits: int = 2048) -> Optional[torch.Tensor]:
    """Attempt RDKit Morgan fingerprint computation when SMILES are provided.
    Returns a float tensor [B, n_bits] or None on failure.
    """
    if smiles_batch is None:
        return None
    try:
        import numpy as np
        from rdkit import Chem
        from rdkit.Chem import AllChem
    except Exception:
        return None
    arrs = []
    for smi in smiles_batch:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            arrs.append(np.zeros(n_bits, dtype=float))
            continue
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=n_bits)
        arr = np.zeros((n_bits,), dtype=float)
        # On old RDKit versions, ConvertToNumpyArray might be under DataStructs
        try:
            from rdkit import DataStructs
            DataStructs.ConvertToNumpyArray(fp, arr)
        except Exception:
            # fallback: set bits by enumerating
            onbits = list(fp.GetOnBits())
            arr[onbits] = 1.0
        arrs.append(arr)
    feats = torch.as_tensor(np.stack(arrs, axis=0), dtype=torch.float)
    feats = feats / (feats.norm(p=2, dim=1, keepdim=True) + 1e-6)
    return feats


