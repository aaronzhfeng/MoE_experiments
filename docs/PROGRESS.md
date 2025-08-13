### docs/PROGRESS.md

Heterogeneous MoE (hetero_moe) – Project Progress

This file tracks what is implemented, what remains, and polishing tasks.

## Implemented

### Data and preprocessing
- Layout under `hetero_moe/data/` (`raw/`, `processed/`).
- Graph2SMILES bridge: `hetero_moe/preprocess/graph2smiles_bridge.py` (supports `src-*.txt`/`tgt-*.txt` and `train.src/tgt`).
- `USPTODataset` with optional graph arrays and SMILES sidecar.
- Collate functions: `collate_seq_batch`, `collate_graph_batch` (auto-selected in training/eval).
- Gate features: token-hash fallback; optional Morgan FP from SMILES.

### Models and layers
- `SmilesExpert`: Transformer encoder–decoder (teacher forcing).
- `GraphExpert`: G2S-compatible MPN encoder + Transformer decoder.
  - `G2SMPNEncoder` (consumes Graph2SMILES tensors via `utils/g2s_compat.py`).
  - Fallback `GraphMPNEncoder` (lightweight message passing).
- Stubs: `ConditionExpert`, `GNN3DExpert`.
- Layers/utilities: `models/layers/transformer_blocks.py`, `models/layers/graph_encoders.py`, `utils/g2s_compat.py`, `utils/gating_features.py`, `utils/tokens.py`, `utils/smiles.py`.

### MoE and routing
- `models/moe.py`: wraps experts + router; routes, aggregates losses; `predict_logits`.
- Router: top-k routing, temperature scaling, optional Gumbel noise.
- Training scripts:
  - `training/train_expert.py` (last and `.best` checkpoints).
  - `training/train_moe.py` (enabled/frozen experts, router warmup, balance-loss schedule, top-k routing, temperature/noise).

### Evaluation and diagnostics
- `evaluation/eval_moe.py`: avg loss, token accuracy, exact match; validity (with vocab); top-k union via per-expert beam; diversity; respects router top-k.
- `evaluation/diagnostics_gate.py`: assignment frequencies.
- `evaluation/ablate_expert.py`: disable expert and evaluate.
- `evaluation/beam_search.py`: greedy + per-expert beam utilities.

### Configs and docs
- Annotated YAMLs: `configs/{smiles_expert,graph_expert,moe}.yaml`.
- `hetero_moe/README.md`: end-to-end commands (env, preprocess, train experts, train MoE, eval, diagnostics).

## Remaining (core)
- ConditionExpert
  - Preprocessing: role/condition tokens; segment positional encodings.
  - Condition-aware Transformer; `configs/cond_expert.yaml`.
- 3D Expert
  - Load 3DInfomax (or similar) pretrained GNN; freeze/fine-tune options.
  - Integrate as expert; `configs/gnn3d_expert.yaml`.
- GraphExpert parity
  - Multi-node graph memory with cross-attention (attend over node/fragment embeddings).
  - Richer D-MPNN semantics (edge messages, depth, residuals, norms).

## Remaining (polish)
- Gate features
  - Persist SMILES with NPZ or sidecar to reliably compute Morgan FPs; cache FPs.
  - Config switch for token-hash vs Morgan.
- Router training
  - Separate LR for router vs experts (param groups).
  - Capacity factor and overflow (Switch-style); entropy regularization.
- Decoding/metrics
  - Length penalty/normalization in beam; unified scoring merging per-expert beams; per-class metrics.
- Tests/CI
  - Unit tests (dataset/graph collation, router determinism/top-k, beam); smoke tests; optional CI.
- Ops
  - Checkpoint load/resume UX (save args/config snapshot); multi-GPU notes.
- Docs
  - Architecture diagram; gate usage plots; experiments table in `docs/`.

## Quick commands
- Preprocess: see `hetero_moe/data/README.md`.
- Train expert:
```powershell
python -m hetero_moe.training.train_expert ^
  --expert smiles ^
  --config hetero_moe/configs/smiles_expert.yaml ^
  --train_bin .../train_0.npz ^
  --valid_bin .../val_0.npz ^
  --save_path runs/experts/smiles.pt ^
  --hidden 256 --layers 4 --heads 8 --ff 1024
```
- Train MoE:
```powershell
python -m hetero_moe.training.train_moe ^
  --config hetero_moe/configs/moe.yaml ^
  --train_bin .../train_0.npz ^
  --valid_bin .../val_0.npz ^
  --hidden 256 --layers 4 --heads 8 --ff 1024 ^
  --top_k 1
```
- Evaluate:
```powershell
python -m hetero_moe.evaluation.eval_moe ^
  --test_bin .../test_0.npz ^
  --beam_size 5 --k 5 ^
  --load_path runs/moe/best.pt ^
  --vocab_file .../vocab_smiles.txt ^
  --pad_id 0 --bos_id 2 --eos_id 3
```
- Diagnostics:
```powershell
python -m hetero_moe.evaluation.diagnostics_gate --bin .../val_0.npz
```