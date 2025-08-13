# 2. Repository Structure Changes

## 2.1 Restructure Directories

MoE_experiments/hetero_moe/
- data/  Raw and processed datasets (USPTO, etc.)
- preprocess/  Data preprocessing scripts (e.g., convert CSV to tokens)
- configs/  YAML config files for training (experts, MoE, evaluation)
- models/
  - experts/  Expert model definitions
    - graph_expert.py  Graph-based expert (Graph2SMILES encoderdecoder)
    - smiles_expert.py  SMILES-based expert (MolTransformer/Chemformer)
    - cond_expert.py  Condition-aware expert (role tokens)
    - gnn3d_expert.py  3D-infused expert (GNN + Transformer)
  - layers/  Shared network layers or components
    - graph_encoders.py  Graph encoders (MPNNs, GAT layers, positional encodings)
    - transformer_blocks.py  Transformer encoder/decoder blocks for seq2seq
    - gating_layers.py  Router feed-forward layers, gating-specific layers
  - gating/
    - router.py  Router module (expert selection logic)
- training/
  - train_expert.py  Train a single expert model
  - train_moe.py  Train the MoE (router + experts)
  - utils.py  Training utilities (loss functions, schedules, etc.)
- evaluation/
  - eval_moe.py  Evaluate top-n accuracy, beam search, validity
  - ablate_expert.py  Run ablation tests by disabling experts
  - diagnostics_gate.py  Analyze gate utilization and expert stats
- utils/
  - tokenizer.py  Tokenizer and vocabulary handling (SMILES, SELFIES)
  - chem_utils.py  RDKit helpers, fingerprints, graph features
- tests/  Unit and integration tests for components
- docs/
  - ARCHITECTURE.md  Architecture description (with diagrams)
  - EXPERIMENTS.md  Planned experiments and ablation details
  - README.md  High-level README with installation and quickstart

Acceptance Criteria:
- Repository contains the above directories and placeholder files as needed
- Old reference code remains in 
eferences/ but is not used at runtime

## 2.2 Directory Purpose Descriptions

Update README.md to include a Project Structure section describing each top-level directorys purpose per the comments above.

Acceptance Criteria:
- README.md lists each directory with a one-line description matching the structure

