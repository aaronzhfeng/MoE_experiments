# 6. MoE Wrapper & Training Loop

## 6.1 MoEModel Wrapper
- File: models/moe.py
- Holds ModuleDict of experts + Router
- Forward: compute gate features (e.g., batch[ fingerprint]) route samples, run per-expert forwards, combine losses; add load-balance loss
- Acceptance: dummy two-expert routing test passes; correct outputs per assignment

## 6.2 Training Phase A  Expert Pretraining
- File: training/train_expert.py (args: --expert, --config, --output_dir, --pretrained)
- Steps: load dataset, init expert (optionally load weights), train/val, save best ckpt
- Configs: configs/graph_expert.yaml, smiles_expert.yaml, cond_expert.yaml, gnn3d_expert.yaml
- Acceptance: script runs and saves checkpoints (smoke test)

## 6.3 Training Phase B  MoE Joint Training
- File: training/train_moe.py (config: experts list, checkpoint paths, lrs, λ)
- Stage 1: optionally freeze experts; train router (aux loss, or soft routing trick)
- Stage 2: unfreeze/all-train; total loss = CE (per-sample expert) + λbalance
- Optimizer param groups (router higher LR)
- Acceptance: training runs, non-degenerate expert usage, checkpoints saved

## 6.4 CLI and Config Examples
- README commands for pretraining and MoE training; resuming instructions
- Acceptance: docs list exact command lines
