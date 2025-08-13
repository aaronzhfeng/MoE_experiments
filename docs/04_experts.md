# 4. Experts (Modular Submodels)

Files under models/experts/:

## 4.1 GraphExpert (Graph2SMILES-style)
- File: models/experts/graph_expert.py
- Encoder: D-MPNN/GAT with positional encodings; Decoder: TransformerDecoder (6L, 8H, d256)
- Forward: forward(graph_batch, target_seq=None) -> {logits}
- Acceptance: dummy forward produces [B, T, V] logits

## 4.2 SmilesExpert (Text Transformer)
- File: models/experts/smiles_expert.py
- Option A: Load Chemformer (MolBART) checkpoint; Option B: smaller Transformer (Molecular Transformer style)
- Shared embeddings; tie decoder weights if training from scratch
- Forward: forward(input_ids, target_ids=None) -> {logits}
- Acceptance: loads checkpoint if provided; forward shape ok

## 4.3 ConditionExpert (Condition-aware Seq2Seq)
- File: models/experts/cond_expert.py
- Like SmilesExpert but input includes role tokens and uses SegmentPositionalEncoding (reset positions per segment)
- Forward: forward(input_ids_with_roles, target_ids=None)
- Acceptance: position indices reset after role tokens (unit-testable)

## 4.4 GNN3DExpert (3D-infused Graph)
- File: models/experts/gnn3d_expert.py
- Encoder: 2D GNN pretrained via 3DInfomax (PNA/GIN), optional weight freeze; Decoder: TransformerDecoder
- Forward: forward(graph_batch, target_seq=None)
- Acceptance: loads pretrained GNN state; forward works

## 4.5 Vocabulary & Output Heads
- Common target vocabulary across experts; each expert has its own output projection to same vocab size
- Alignment with Chemformer vocab if loading its weights
- Acceptance: vocab size consistent; token ids align across pipeline
