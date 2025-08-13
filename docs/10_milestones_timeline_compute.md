# 10. Milestones, Timeline, and Compute

Milestones with tasks, deliverables, definition of done, ETA, and compute:

1) Repository setup and data pipeline (1 week)
- Deliverables: tokenizer, chem_utils, dataset, preprocess scripts; tests
- DoD: load small sample through dataset and dataloader; tests pass
- Compute: CPU for preprocessing

2) Expert model implementations (2 weeks)
- Deliverables: experts and layers; forward tests; optional weight loading
- DoD: expert forwards produce correct shapes; checkpoints load
- Compute: minimal GPU for smoke tests

3) Gating and MoE integration (1 week)
- Deliverables: Router, MoEModel, balance loss; dummy train step
- DoD: forward and backward on toy batch; shapes correct

4) Expert pretraining (23 weeks)
- Deliverables: train_expert.py, configs, checkpoints
- DoD: reasonable validation metrics and saved checkpoints
- Compute: ~24GB GPU per expert

5) MoE training (12 weeks)
- Deliverables: train_moe.py, moe.yaml; router and expert schedules; checkpoints
- DoD: non-degenerate expert usage; validation accuracy >= best single expert

6) Evaluation and ablations (1 week)
- Deliverables: eval scripts, diagnostics, ablations; results in EXPERIMENTS.md
- DoD: metrics tables and plots saved; MoE effectiveness shown

7) Documentation and final refactor (1 week)
- Deliverables: finalized ARCHITECTURE.md and EXPERIMENTS.md; code cleanup
- DoD: repo reproducible with docs; tests green

Total: about 810 weeks. Prefer 1x 24GB GPU (24 GPUs beneficial); CPU and RAM for RDKit preprocessing.

Acceptance Criteria:
- Each milestone aligns with plan; issues and PRs can reference these step docs
