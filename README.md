# Heterogeneous Mixture-of-Experts for Chemical Reaction Prediction

<p align="center">
  <img src="https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?logo=pytorch&logoColor=white" alt="PyTorch">
  <img src="https://img.shields.io/badge/RDKit-2022.9+-3776AB?logo=python&logoColor=white" alt="RDKit">
  <img src="https://img.shields.io/badge/License-MIT-green.svg" alt="License">
</p>

A research framework implementing **heterogeneous Mixture-of-Experts (MoE)** architectures for chemical reaction outcome prediction. This system combines multiple specialized expert models—each leveraging different molecular representations (SMILES sequences, molecular graphs, 3D geometry, reaction conditions)—with a learned gating network to achieve state-of-the-art reaction prediction.

## 🎯 Key Features

- **Multi-Modal Experts**: Specialized models for different chemical representations
  - **SMILES Expert**: Transformer-based sequence model for tokenized SMILES
  - **Graph Expert**: D-MPNN encoder with Graph2SMILES architecture
  - **3D Expert**: 3DInfomax-pretrained GNN for geometry-aware prediction
  - **Condition Expert**: Handles catalysts, solvents, temperatures via role tokens

- **Learned Gating**: Neural router using Morgan fingerprints to dynamically select experts
- **Load Balancing**: Auxiliary losses to prevent expert collapse and ensure specialization
- **End-to-End Training**: Joint optimization of experts and gating network

## 📁 Repository Structure

```
MoE_experiments/
├── hetero_moe/                    # Main heterogeneous MoE implementation
│   ├── configs/                   # YAML configuration files
│   │   ├── moe.yaml              # MoE training configuration
│   │   ├── smiles_expert.yaml    # SMILES expert config
│   │   └── graph_expert.yaml     # Graph expert config
│   ├── data/                      # Data handling
│   │   ├── dataset.py            # PyTorch dataset classes
│   │   ├── dataloader.py         # Data loading utilities
│   │   └── raw/uspto/            # Place USPTO data here
│   ├── models/                    # Model implementations
│   │   ├── moe.py                # MoE wrapper model
│   │   ├── experts/              # Expert architectures
│   │   │   ├── smiles_expert.py
│   │   │   ├── graph_expert.py
│   │   │   ├── cond_expert.py
│   │   │   └── gnn3d_expert.py
│   │   ├── gating/
│   │   │   └── router.py         # Gating network implementation
│   │   └── layers/               # Shared layers (attention, etc.)
│   ├── preprocess/                # Data preprocessing
│   │   └── graph2smiles_bridge.py
│   ├── training/                  # Training scripts
│   │   ├── train_expert.py       # Single expert training
│   │   ├── train_moe.py          # Full MoE training
│   │   └── utils.py              # Training utilities
│   ├── evaluation/                # Evaluation & diagnostics
│   │   ├── eval_moe.py           # Top-k accuracy evaluation
│   │   ├── diagnostics_gate.py   # Gating analysis
│   │   ├── ablate_expert.py      # Expert ablation studies
│   │   └── beam_search.py        # Beam search decoding
│   └── utils/                     # Utility functions
│       ├── tokenizer.py          # SMILES tokenization
│       ├── smiles.py             # SMILES utilities
│       └── gating_features.py    # Feature extraction for gating
├── references/                    # Reference implementations
│   ├── Graph2SMILES/             # Original Graph2SMILES code
│   ├── 3Dinfomax/                # 3D molecular pretraining
│   ├── Chemformer/               # BART-based chemistry model
│   └── MolecularTransformer/     # Seq2seq baseline
├── docs/                          # Documentation
│   └── heterogeneous_moe_reaction_prediction_plan.md
├── runs/                          # Training outputs & checkpoints
└── requirements.txt               # Python dependencies
```

## 🚀 Quick Start

### 1. Environment Setup

```bash
# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# or: .\.venv\Scripts\activate  # Windows

# Install dependencies
pip install --upgrade pip wheel setuptools
pip install numpy==1.26.4 rdkit-pypi==2022.9.5 selfies==2.2.0 networkx==3.3
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118  # CUDA 11.8
pip install -r requirements.txt
```

### 2. Data Preparation

Place USPTO reaction data in `hetero_moe/data/raw/uspto/`:

```
hetero_moe/data/raw/uspto/
├── src-train.txt    # Reactant SMILES (train)
├── tgt-train.txt    # Product SMILES (train)
├── src-val.txt      # Reactant SMILES (validation)
├── tgt-val.txt      # Product SMILES (validation)
├── src-test.txt     # Reactant SMILES (test)
└── tgt-test.txt     # Product SMILES (test)
```

### 3. Preprocessing

Generate tokenized data and graph features:

```bash
python -m hetero_moe.preprocess.graph2smiles_bridge \
  --raw_dir hetero_moe/data/raw/uspto \
  --out_dir hetero_moe/data/processed/uspto/graph2smiles_npz \
  --model g2s --repr smiles --max_src_len 512 --max_tgt_len 512 --workers 4
```

### 4. Training

#### Phase A: Pre-train Individual Experts

```bash
# Train SMILES expert
python -m hetero_moe.training.train_expert \
  --expert smiles \
  --config hetero_moe/configs/smiles_expert.yaml \
  --train_bin hetero_moe/data/processed/uspto/graph2smiles_npz/train_0.npz \
  --valid_bin hetero_moe/data/processed/uspto/graph2smiles_npz/val_0.npz \
  --save_path runs/experts/smiles.pt \
  --hidden 256 --layers 4 --heads 8 --ff 1024

# Train Graph expert
python -m hetero_moe.training.train_expert \
  --expert graph \
  --config hetero_moe/configs/graph_expert.yaml \
  --train_bin hetero_moe/data/processed/uspto/graph2smiles_npz/train_0.npz \
  --valid_bin hetero_moe/data/processed/uspto/graph2smiles_npz/val_0.npz \
  --save_path runs/experts/graph.pt
```

#### Phase B: Train Full MoE

```bash
python -m hetero_moe.training.train_moe \
  --config hetero_moe/configs/moe.yaml \
  --train_bin hetero_moe/data/processed/uspto/graph2smiles_npz/train_0.npz \
  --valid_bin hetero_moe/data/processed/uspto/graph2smiles_npz/val_0.npz \
  --hidden 256 --layers 4 --heads 8 --ff 1024 \
  --top_k 1  # Hard gating (select 1 expert per sample)
```

### 5. Evaluation

```bash
# Evaluate MoE
python -m hetero_moe.evaluation.eval_moe \
  --test_bin hetero_moe/data/processed/uspto/graph2smiles_npz/test_0.npz \
  --beam_size 5 --k 5 --load_path runs/moe/best.pt \
  --vocab_file hetero_moe/data/processed/uspto/graph2smiles_npz/vocab_smiles.txt

# Analyze gating behavior
python -m hetero_moe.evaluation.diagnostics_gate \
  --bin hetero_moe/data/processed/uspto/graph2smiles_npz/val_0.npz

# Ablation: disable specific expert
python -m hetero_moe.evaluation.ablate_expert \
  --checkpoint runs/moe/best.pt --disable graph
```

## 🏗️ Architecture Overview

```
                    ┌─────────────────────────────┐
                    │       Input Reaction        │
                    │  Reactants → ? → Product    │
                    └─────────────┬───────────────┘
                                  │
                    ┌─────────────▼───────────────┐
                    │      Gating Network         │
                    │  (Morgan FP → Expert Prob)  │
                    └─────────────┬───────────────┘
                                  │
         ┌────────────┬───────────┼───────────┬────────────┐
         ▼            ▼           ▼           ▼            ▼
    ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐
    │ SMILES  │ │  Graph  │ │   3D    │ │  Cond   │ │  More   │
    │ Expert  │ │ Expert  │ │ Expert  │ │ Expert  │ │ Experts │
    └────┬────┘ └────┬────┘ └────┬────┘ └────┬────┘ └────┬────┘
         │            │           │           │            │
         └────────────┴───────────┼───────────┴────────────┘
                                  │
                    ┌─────────────▼───────────────┐
                    │     Weighted/Selected       │
                    │       Product Output        │
                    └─────────────────────────────┘
```

## ⚙️ Configuration

Key configuration options in `hetero_moe/configs/moe.yaml`:

```yaml
# Expert selection
enabled_experts: [smiles, graph]  # Which experts to use
freeze_experts: false             # Freeze expert weights

# Gating configuration
router_warmup_epochs: 5           # Train router before experts
router_temperature: 1.0           # Softmax temperature
router_gumbel_noise: 0.0          # Gumbel noise for exploration
top_k: 1                          # Number of experts per sample

# Load balancing
balance_lambda: 0.01              # Load balance loss weight
balance_lambda_schedule: linear   # Loss weight schedule
```

## 📊 Expected Results

| Model | USPTO-480k Top-1 | USPTO-480k Top-5 |
|-------|-----------------|-----------------|
| Molecular Transformer | 88.6% | 94.2% |
| Graph2SMILES | 90.3% | 95.1% |
| Chemformer (base) | 90.9% | 95.5% |
| **Hetero-MoE (ours)** | **~91-92%** | **~96%** |

## 📚 References

- [Graph2SMILES](https://arxiv.org/abs/2110.09681) - Tu & Coley, 2021
- [Molecular Transformer](https://pubs.acs.org/doi/10.1021/acscentsci.9b00576) - Schwaller et al., 2019
- [Chemformer](https://pubs.rsc.org/en/content/articlehtml/2022/sc/d2sc01118b) - Irwin et al., 2022
- [3D Infomax](https://arxiv.org/abs/2110.04126) - Stärk et al., 2022
- [Switch Transformer](https://arxiv.org/abs/2101.03961) - Fedus et al., 2021

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit issues and pull requests.

---

*This research is part of ongoing work on multi-modal chemical reaction prediction.*
