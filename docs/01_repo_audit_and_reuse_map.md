# 1. Repo Audit & Reuse Map (references/)

| Reference Component | Import into | Refactor & License Notes |
| --- | --- | --- |
| Graph2SMILES  Graph encoder & Transformer decoder (eferences/Graph2SMILES/) | models/experts/graph_expert.py, models/layers/graph_encoders.py | Reuse D-MPNN graph layers and shortest-path pos encoding. Drop OpenNMT dependency; integrate with our training loop and vocabulary. License: MIT. |
| MolecularTransformer  Seq2Seq baseline (eferences/MolecularTransformer/) | models/experts/smiles_expert.py | Reimplement a standard Transformer enc-dec (4 layer pairs) with original tokenization scheme. License: open-source; reimplement tokenizer. |
| Chemformer (MolBART)  Pre-trained SMILES Transformer (eferences/Chemformer/molbart/) | models/experts/smiles_expert.py, models/utils/tokenizer.py | Port MolBART model and tokenizer utils. Simplify config; integrate with argparse. License: Apache 2.0 (with attribution). |
| 3D Infomax  3D-GNN pretraining (eferences/3Dinfomax/) | models/experts/gnn3d_expert.py, models/layers/gnn_layers.py | Reuse GNN architecture to load provided pretrained weights. Focus on 2D GNN encoder pretrained with 3DInfomax. License: MIT. |
| Internal ChemTransformer  roles & conditions (eferences/ChemTransformer/) | data/reaction_roles.py, models/utils/position.py | Adapt reaction2events preprocessing for role tokens and temperature tokens. Implement ReactionRolesTokenizer and segment positional embeddings. License: internal reuse. |

Acceptance Criteria:
- Stubbed classes/functions with citations to sources
- License headers retained where required
