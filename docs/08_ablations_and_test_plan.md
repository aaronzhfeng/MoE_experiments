# 8. Ablations and Test Plan

## 8.1 Ablation Experiments
- No-gating ensemble vs MoE routed model
- Single large model vs MoE (parameter count comparable)
- Expert removal (disable one expert)
- Ablate pretraining (train all jointly from scratch)
- Top-k routing variant (optional)

## 8.2 Unit Tests
- Data pipeline tests for tokenization, graphs, dataset schema
- Expert forward shape tests
- Router determinism tests
- Integration smoke test for MoE forward and loss
- Optional device placement checks

## 8.3 Continuous Integration
- Optional GitHub Actions to run pytest and linting

Acceptance Criteria:
- Tests pass locally; ablation scripts report metrics
