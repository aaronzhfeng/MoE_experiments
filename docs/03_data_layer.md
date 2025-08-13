# 3. Data Layer (Unified Views)

## 3.1 Define Sample Schema
- Create ReactionSample dataclass with fields: reactants_smiles, product_smiles, reactant_tokens, product_tokens, graph_data, condition_tokens?, precomputed_3d?
- File: data/dataset.py
- Test: tests/test_data_schema.py

## 3.2 Dataset Classes
- USPTODataset (and optional ORDDataset) yielding ReactionSample
- Supports __len__/__getitem__, on-the-fly tokenization and graph construction
- Test on small sample file

## 3.3 Preprocessing & Tokenization
- utils/chem_utils.py::canonicalize_smiles(smiles, remove_mapping)
- utils/tokenizer.py:
  - SmilesTokenizer (RDKit/regex-based), optional SELFIES
  - Special tokens: <PAD>, <BOS>, <EOS>, <UNK>, role tokens, <SEP>
  - preprocess/build_vocab.py builds vocab file
- Tests for tokenizing example SMILES

## 3.4 Graph Construction
- utils/chem_utils.py::mol_to_graph(mol) -> Data (PyG) with node/edge features, optional shortest-path distances
- Test node/edge counts on simple molecules; ensure atom mapping removal

## 3.5 Role & Condition Augmentation
- preprocess/reaction_roles.py::augment_reaction_inputs(reactants, roles, temp) inserts role/temperature tokens (ChemTransformer style)
- Test role token insertion order and content

## 3.6 Dataloaders
- data/dataloader.py with collate functions for sequences (padding) and graphs (PyG Batch)
- Batch dict contains: graph_data, input_ids, target_ids, etc.

**Acceptance Criteria:**
- Dataset returns consistent ReactionSample
- Vocab built successfully; tokenization tests pass
- Graph conversion verified; dataloader batches shape-check
