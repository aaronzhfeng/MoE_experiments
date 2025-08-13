# Data layout for USPTO

Upload raw USPTO splits here:

```
hetero_moe/data/raw/uspto/
  train.src    # reactants/reagents SMILES (one per line)
  train.tgt    # product SMILES (one per line)
  val.src
  val.tgt
  test.src
  test.tgt
```

To preprocess using the Graph2SMILES pipeline into NPZ files:

```
python -m hetero_moe.preprocess.graph2smiles_bridge \
  --raw_dir hetero_moe/data/raw/uspto \
  --out_dir hetero_moe/data/processed/uspto/graph2smiles_npz \
  --model g2s --repr smiles --max_src_len 512 --max_tgt_len 512 --workers 4
```

Outputs (in `hetero_moe/data/processed/uspto/graph2smiles_npz`):
- `vocab_smiles.txt`
- `{train,val,test}_0.npz`

Notes:
- The bridge reuses `references/Graph2SMILES` preprocessing (tokenization, vocab, graph features).
- If you prefer sequence-only preprocessing, set `--model s2s`.
- The bridge supports either `train.src/train.tgt` style or `src-train.txt/tgt-train.txt` filenames.


