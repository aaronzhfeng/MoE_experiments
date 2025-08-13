## Heterogeneous MoE for Reaction Prediction

End-to-end guide to set up data, preprocess, train experts, train the MoE, and evaluate.

### 1) Environment

Use a fresh environment with RDKit compatible NumPy.

PowerShell (CPU example):

```powershell
python -m venv .venv
.\.venv\Scripts\pip install --upgrade pip wheel setuptools
.\.venv\Scripts\pip install numpy==1.26.4 rdkit-pypi==2022.9.5 selfies==2.2.0 networkx==3.3
.\.venv\Scripts\pip install torch --index-url https://download.pytorch.org/whl/cpu
```

Conda (alternative):

```powershell
conda create -n moe python=3.10 -y
conda activate moe
conda install -c rdkit rdkit=2022.09.5 -y
pip install numpy==1.26.4 selfies==2.2.0 networkx==3.3
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

### 2) Data layout

Place USPTO splits under:

```
hetero_moe/data/raw/uspto/
  src-train.txt  tgt-train.txt
  src-val.txt    tgt-val.txt
  src-test.txt   tgt-test.txt
```

The bridge also supports `train.src/train.tgt` naming.

### 3) Preprocess (Graph2SMILES bridge)

Generates token IDs and graph features (NPZ) using `references/Graph2SMILES`:

```powershell
.\.venv\Scripts\python -m hetero_moe.preprocess.graph2smiles_bridge \
  --raw_dir hetero_moe/data/raw/uspto \
  --out_dir hetero_moe/data/processed/uspto/graph2smiles_npz \
  --model g2s --repr smiles --max_src_len 512 --max_tgt_len 512 --workers 4
```

Outputs: `hetero_moe/data/processed/uspto/graph2smiles_npz/{train,val,test}_0.npz`, `vocab_smiles.txt`.

Details also in `hetero_moe/data/README.md`.

### 4) Configs

Config files live under `hetero_moe/configs/`. They specify expert hyperparameters, MoE gating options, and training options. You can also override most values via CLI flags as shown below.

### 5) Train single experts (Phase A)

Pretrain each expert individually. Commands will look like:

```powershell
.\.venv\Scripts\python -m hetero_moe.training.train_expert \
  --expert smiles \
  --config hetero_moe/configs/smiles_expert.yaml \
  --train_bin hetero_moe/data/processed/uspto/graph2smiles_npz/train_0.npz \
  --valid_bin hetero_moe/data/processed/uspto/graph2smiles_npz/val_0.npz \
  --save_path runs/experts/smiles.pt \
  --hidden 256 --layers 4 --heads 8 --ff 1024
```

Graph expert (optional):

```powershell
.\.venv\Scripts\python -m hetero_moe.training.train_expert \
  --expert graph \
  --config hetero_moe/configs/graph_expert.yaml \
  --train_bin hetero_moe/data/processed/uspto/graph2smiles_npz/train_0.npz \
  --valid_bin hetero_moe/data/processed/uspto/graph2smiles_npz/val_0.npz \
  --save_path runs/experts/graph.pt \
  --hidden 256 --layers 4 --heads 8 --ff 1024
```

Supported experts (planned): `smiles`, `graph`, `cond`, `gnn3d`.

### 6) Train MoE (Phase B)

Jointly train router + experts (optionally load/freeze experts first):

```powershell
.\.venv\Scripts\python -m hetero_moe.training.train_moe \
  --config hetero_moe/configs/moe.yaml \
  --train_bin hetero_moe/data/processed/uspto/graph2smiles_npz/train_0.npz \
  --valid_bin hetero_moe/data/processed/uspto/graph2smiles_npz/val_0.npz \
  --hidden 256 --layers 4 --heads 8 --ff 1024 \
  --top_k 1
```

The router uses Morgan fingerprints (or provided features) and a load-balancing loss.

Notes:
- Use `enabled_experts` and `freeze_experts` in `configs/moe.yaml` to pick and/or freeze experts.
- `router_warmup_epochs > 0` freezes experts initially to train the router.
- `router_temperature`, `router_gumbel_noise`, and `balance_lambda_schedule` can be tuned in the config.

### 7) Evaluation and diagnostics

Compute top-1/top-k, validity, diversity, and gate utilization:

```powershell
.\.venv\Scripts\python -m hetero_moe.evaluation.eval_moe \
  --test_bin hetero_moe/data/processed/uspto/graph2smiles_npz/test_0.npz \
  --beam_size 5 --k 5 --load_path runs/moe/best.pt \
  --vocab_file hetero_moe/data/processed/uspto/graph2smiles_npz/vocab_smiles.txt \
  --pad_id 0 --bos_id 2 --eos_id 3

.\.venv\Scripts\python -m hetero_moe.evaluation.diagnostics_gate \
  --bin hetero_moe/data/processed/uspto/graph2smiles_npz/val_0.npz
```

### 8) Ablations

Disable experts or compare to no-gating ensemble:

```powershell
.\.venv\Scripts\python -m hetero_moe.evaluation.ablate_expert \
  --checkpoint runs/moe/best.pt \
  --disable graph
```

### Current implementation status

- Preprocess bridge: `hetero_moe/preprocess/graph2smiles_bridge.py`
- Data stubs: `hetero_moe/data/dataset.py`, `hetero_moe/utils/tokenizer.py`
- Gating: `hetero_moe/models/gating/router.py`, balance loss in `hetero_moe/training/utils.py`
- MoE wrapper: `hetero_moe/models/moe.py`
- Expert stub: `hetero_moe/models/experts/smiles_expert.py` (others coming)

As training and evaluation scripts land, this README will be updated with exact commands and config examples.


