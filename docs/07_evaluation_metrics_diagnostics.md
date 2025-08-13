# 7. Evaluation, Metrics, and Diagnostics

## 7.1 Top-k Accuracy and Beam Search
- File: evaluation/eval_moe.py with --beam_size and --k
- Metrics: top-1 and top-k accuracy using canonical SMILES matching
- Validity rate: fraction of valid predicted SMILES
- Per-class accuracy if labels available
- Output: print summary and save evaluation_results.json

## 7.2 Diversity Metrics
- Pairwise Tanimoto similarity of fingerprints across top-k outputs per input
- Report average diversity score across dataset

## 7.3 Gate Utilization Diagnostics
- File: evaluation/diagnostics_gate.py
- Collect expert assignment frequencies and per-expert accuracy
- Save histogram to docs/gate_usage.png

## 7.4 Ablation Hooks
- evaluation/ablate_expert.py or flag in eval_moe.py to disable an expert
- No-gating baseline by union of experts predictions for comparison

## 7.5 Error Analysis (optional)
- evaluation/error_analysis.py to dump failures CSV for manual review

Acceptance Criteria:
- Scripts run and produce metrics files and optional plots
