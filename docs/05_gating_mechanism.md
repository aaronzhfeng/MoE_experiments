# 5. Gating Mechanism

## 5.1 Router (Hard Top-1)
- File: models/gating/router.py
- Router(num_experts, gate_feature_dim) -> (assignments [B], probabilities [B,N])
- Gate features: Morgan fingerprint (2048-bit) + 2-layer MLP (2048->256->N)
- Acceptance: forward yields assignments == argmax(probs)

## 5.2 Load-Balancing Auxiliary Loss
- File: training/utils.py::load_balance_loss(router_probs)
- Encourage uniform expert usage (e.g., squared deviation from 1/N per expert)
- Weighted in total loss (e.g., λ=0.01)
- Acceptance: loss high when collapsed, low when uniform; gradients flow

## 5.3 Alternate Gating (Future)
- Document top-k (k=2) extension for later use; adjust shapes accordingly
- Acceptance: commented documentation in router.py
