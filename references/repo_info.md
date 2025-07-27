# Literature review of selected chemical reaction prediction models

This review summarises several deep‑learning methods for predicting chemical reaction outcomes and includes an architectural analysis of an internal model (ChemTransformer).  For each method we describe the datasets used, the data representation, the model architecture, the results reported in the literature and the key novelty.  When possible, statements are supported by citations to the original papers.

## Graph2SMILES (Coley group, 2021)

### Dataset

Graph2SMILES was evaluated on multiple variants of the USPTO reaction datasets.  The authors trained and tested the model on **USPTO‐480k_mixed** and **USPTO‐STEREO_mixed**, which contain 409,035 and 479,035 atom‑mapped reactions respectively:contentReference[oaicite:0]{index=0}.  These datasets were created by mixing reactants and reagents (no separation token); **USPTO‐STEREO_mixed** retains stereochemical information.  For retrosynthesis experiments they used the **USPTO_full** dataset (full set of 479,035 reactions) and the smaller **USPTO‐50k** dataset, which is a 50,016‑reaction subset with annotated reaction classes.  In all cases the data were filtered to remove duplicates and invalid reactions.

### Data representation

Instead of tokenising SMILES strings, Graph2SMILES treats each reaction as a graph.  A **directed message passing neural network (D‑MPNN)** encodes the molecular graph, producing atom‑level embeddings.  A graph‑aware positional encoding is added by computing the shortest‑path distance between each pair of nodes; this distance matrix is then transformed into positional embeddings and added to the node features:contentReference[oaicite:1]{index=1}.  The target (product) is represented as a SMILES sequence and decoded using a Transformer decoder.

### Architecture

Graph2SMILES has an encoder–decoder architecture.  The **encoder** consists of a D‑MPNN with attention‑based updates similar to a graph attention network.  After several graph message‑passing layers, a **global attention encoder** (Transformer) with graph‑aware positional embeddings generates a fixed‑size context vector:contentReference[oaicite:2]{index=2}.  The **decoder** is a standard Transformer decoder operating on SMILES tokens; it attends over the graph‑encoded context to generate the product sequence.  Hyper‑parameters include an embedding size of 256, 8 attention heads and six layers in both encoder and decoder; the feed‑forward dimension is 2048:contentReference[oaicite:3]{index=3}.

### Results

On forward reaction prediction, Graph2SMILES achieved **90.3 % top‑1 accuracy** on the USPTO‑480k_mixed dataset and **78.1 %** on USPTO‑STEREO_mixed:contentReference[oaicite:4]{index=4}.  These accuracies were 1.7 and 1.9 percentage points higher than those of the Molecular Transformer baseline.  In retrosynthesis experiments, Graph2SMILES attained **45.7 % top‑1 accuracy** on the full USPTO dataset and **52.9 %** on USPTO‑50k:contentReference[oaicite:5]{index=5}.  In addition to improved accuracy, the model exhibited better beam diversity and robustness to SMILES enumeration.

### Novelty

Graph2SMILES introduces a **graph‑based encoder** for reaction prediction, enabling permutation‑invariant encoding and capturing local and global structure.  The **graph‑aware positional encoding** uses shortest‑path distances to embed atom positions, removing the need for SMILES augmentation and making the model invariant to the order of atoms or molecules:contentReference[oaicite:6]{index=6}.  Compared with token‑based models, it improves top‑1 accuracy and stereochemical prediction while maintaining computational efficiency.

## Molecular Transformer (Schwaller et al., 2019)

### Dataset

The Molecular Transformer was trained on several subsets of the USPTO patents.  The main set, **USPTO_MIT** (also known as USPTO‑480k), contains 479,035 atom‑mapped single‑product reactions extracted from Lowe’s patent corpus:contentReference[oaicite:7]{index=7}.  The authors created two preprocessing versions: **separated**, where reactants and reagents are separated by a `>` token, and **mixed**, where all inputs are concatenated.  Additional datasets include **USPTO_STEREO** (902 k reactions with stereochemical information) and a time‑split **Pistachio 2017** test set:contentReference[oaicite:8]{index=8}.

### Data representation

Reactions are expressed as **SMILES strings**.  The authors introduced a tokenisation scheme that separates atoms, rings and special symbols according to a regular expression.  Reactants and reagents are optionally separated by a `>` token.  The inputs are canonicalised and atom mapping is removed; reagents outside a set of 76 common molecules are discarded.  The sequences are augmented through SMILES enumeration during training.

### Architecture

The model uses the **Transformer** encoder–decoder architecture originally developed for natural‑language translation.  Four layers of multi‑head self‑attention are employed in both the encoder and decoder to reduce parameter count (around 12 M parameters):contentReference[oaicite:9]{index=9}.  Positional encodings are added to represent token positions, and beam search is used during decoding.  Training uses a combination of teacher forcing and label smoothing; inference uses greedy or beam search.

### Results

On the **USPTO_MIT** dataset, the Molecular Transformer achieved **90.4 % top‑1 accuracy** with separated preprocessing and **88.6 %** with mixed preprocessing:contentReference[oaicite:10]{index=10}.  On **USPTO_STEREO**, the model obtained 78.1 % (separated) and 76.2 % (mixed).  The model outperformed template‑based methods such as Schwaller’s seq‑to‑seq baseline by more than 10 percentage points.  It also demonstrated competitive performance on the Pistachio 2017 test set.

### Novelty

Molecular Transformer was the **first application of the Transformer** architecture to chemical reaction prediction.  It removed the recurrence of previous seq‑to‑seq models and introduced a tokenisation scheme tailored to SMILES.  The model showed that purely data‑driven neural sequence translation can outperform template‑based approaches.  Its success inspired many subsequent works and provided a strong baseline for reaction prediction tasks.

## Chemformer (Irwin et al., 2022)

### Dataset

Chemformer uses large‑scale pre‑training followed by fine‑tuning on downstream tasks.  For **pre‑training**, the authors selected **100 M SMILES strings** from the ZINC‑15 database of purchasable molecules (molecular weight ≤500 and log P ≤5):contentReference[oaicite:11]{index=11}.  Fine‑tuning tasks include: 

- **Forward reaction prediction** on USPTO‑MIT (≈470 k reactions) in mixed and separated settings:contentReference[oaicite:12]{index=12}.  
- **Retrosynthesis** on USPTO‑50k (≈50 k reactions).  
- **Molecular optimisation** tasks (e.g., logD, solubility, clearance) with ~160 k matched molecular pairs from ChEMBL:contentReference[oaicite:13]{index=13}.  
- Discriminative property prediction on MoleculeNet datasets (ESOL, FreeSolvation, Lipophilicity etc.).

### Data representation

Chemformer uses SMILES strings as both pre‑training and fine‑tuning inputs.  **Self‑supervised pre‑training tasks** are applied to the SMILES: 
- **Masking**, where random tokens are masked and the model learns to reconstruct them.  
- **Augmentation**, where random SMILES enumerations are generated and the model reconstructs the canonical form.  
- **Combined** masking and augmentation to capture both local and global context:contentReference[oaicite:14]{index=14}.  

During fine‑tuning the model acts as a sequence‑to‑sequence translator for reactions or as a sequence classifier for property prediction.  On‑the‑fly augmentation is used to avoid storing all augmented sequences.

### Architecture

Chemformer is based on **BART**, an encoder–decoder Transformer with bidirectional encoders and autoregressive decoders.  Two model sizes are provided: **base** (~45 M parameters) and **large** (~230 M).  Both use 12 attention heads and gelu‑activated feed‑forward networks.  Pre‑training uses dynamic masking and augmentation; fine‑tuning uses beam search with width 10 and top‑k sampling.  Dropout is applied to mitigate overfitting.

### Results

On USPTO‑MIT, **Chemformer** achieved **90.9 % top‑1 accuracy (mixed)** and **92.5 % (separated)**:contentReference[oaicite:15]{index=15}.  The **large** model attained slightly higher top‑1 accuracies of **91.3 % (mixed)** and **92.8 % (separated):contentReference[oaicite:16]{index=16}.  In retrosynthesis on USPTO‑50k, Chemformer reached **53.6 % top‑1** and **61.1 % top‑5 accuracy**; Chemformer‑large improved to **54.3 % top‑1** and **62.3 % top‑5**:contentReference[oaicite:17]{index=17}.  On discriminative tasks the model matched or surpassed state‑of‑the‑art results.  Pre‑training improved sample efficiency, reducing the number of reactions required for fine‑tuning.

### Novelty

Chemformer demonstrates that **large‑scale self‑supervised pre‑training on SMILES** yields transferable representations.  By combining masking and SMILES augmentation tasks, the model learns robust sequence embeddings that generalise across diverse tasks.  The inclusion of both generative (reaction prediction) and discriminative (property prediction) downstream tasks shows the versatility of the pre‑trained chemical language model.

## 3DInfomax (Stark et al., 2022)

### Dataset

3DInfomax aims to infuse 3D structural knowledge into 2D graph neural networks.  **Pre‑training** uses three 3D datasets: **QM9** (≈134 k molecules), **GEOM‑Drugs** (≈304 k drug‑like molecules with multiple conformers) and **QMugs** (≈665 k molecules):contentReference[oaicite:18]{index=18}.  For fine‑tuning the authors used multiple tasks: quantum property prediction on subsets of QM9 and GEOM‑Drugs; and classification/regression tasks from MoleculeNet (ESOL, FreeSolv, Lipophilicity, HIV, BACE, BBBP, Tox21, ToxCast, SIDER, ClinTox):contentReference[oaicite:19]{index=19}.  Each dataset is split into train/validation/test sets using random or scaffold split.

### Data representation

Two representations are used simultaneously: 
- A **2D molecular graph** processed by a graph neural network (GNN) to produce a representation \(z_a\).  
- A **3D conformer** representation processed by another network \(z_b\).  The 3D network takes pairwise Euclidean distances between atoms, encodes them with sinusoidal positional encodings and processes them with a GNN:contentReference[oaicite:20]{index=20}.  Positive pairs consist of the same molecule’s 2D and 3D representations; negative pairs are different molecules.

### Architecture

The core idea is to **maximise the mutual information** between the 2D and 3D representations.  Two neural networks \(f_a\) and \(f_b\) generate representations \(z_a\) and \(z_b\) for a molecule’s 2D and 3D inputs.  A **contrastive InfoMax objective** (NT‑Xent loss) encourages \(z_a\) to be predictive of \(z_b\) and vice‑versa:contentReference[oaicite:21]{index=21}.  A **multi‑conformer extension** aggregates positive pairs from all conformers of a molecule:contentReference[oaicite:22]{index=22}.  After pre‑training, \(f_a\) is used as a 2D GNN for downstream tasks; \(f_b\) is discarded.

### Results

Pre‑training yields large improvements on downstream tasks.  On quantum property prediction tasks (e.g., dipole moment, HOMO/LUMO energies) the MAE is reduced by ~22 % relative to training from scratch:contentReference[oaicite:23]{index=23}.  On MoleculeNet tasks the pre‑trained model outperforms or matches other self‑supervised methods.  The 3DInfomax approach is particularly beneficial when only 2D structures are available at fine‑tuning time.

### Novelty

3DInfomax is novel in that it leverages **mutual information maximisation** between 2D and 3D representations to endow 2D GNNs with implicit 3D knowledge.  Unlike approaches that require 3D coordinates for each molecule during inference, 3DInfomax trains a 2D encoder that benefits from 3D information but operates solely on 2D graphs at downstream time.  The method also introduces a **multi‑conformer contrastive objective**, enabling the model to learn from multiple conformers per molecule:contentReference[oaicite:24]{index=24}.

## ChemTransformer (internal model) – Architecture analysis

### Dataset

ChemTransformer is an internal project designed to process chemical reaction data and train a Transformer‑based model.  The preprocessing scripts expect a CSV file of **USPTO‑480k** reactions (named `reactions_unique.csv`) containing dictionaries of reactants, catalysts, solvents, reagents, conditions and products.  The training script splits the data into train/validation sets (default 85 %/15 %) and stores the processed data in `train.npz` and `val.npz`.  The raw dataset is the same **USPTO‑480k** set used by Jin et al., containing 479,035 atom‑mapped reactions:contentReference[oaicite:25]{index=25}.  Preprocessing filters out reactions with very short sequences and optionally limits the sequence length to 300 tokens for the input and 150 tokens for the target:contentReference[oaicite:26]{index=26}.

### Data representation

The preprocessing pipeline converts each reaction into an **event‑based sequence**:

- **Compound‑type tokens** are inserted to indicate the role of each molecule (e.g., `[REACTANT]`, `[CATALYST]`, `[SOLVENT]`, `[REAGENT]`):contentReference[oaicite:27]{index=27}.
- Each molecule’s SMILES string is tokenised using a simple SMILES tokenizer or SELFIES; optional Selfies encoding can be selected via an argument:contentReference[oaicite:28]{index=28}.  
- The temperature and amount of each reagent can be added as numeric tokens with appropriate units (e.g., `[CELSIUS]`, `[GRAM]`):contentReference[oaicite:29]{index=29}.  
- A **position index** is assigned to each token to mark the start of a new compound or section; the `get_position_indices` function increments the position index whenever a sequence‑start token (e.g., `[REACTANT]`, `<EOS>`) is encountered:contentReference[oaicite:30]{index=30}.  This allows the model to distinguish different reactants and reagents.

Token sequences are padded to fixed lengths (300 for inputs and 150 for targets) and converted to integer indices using a dictionary built from the corpus:contentReference[oaicite:31]{index=31}.  The final dataset comprises arrays of token indices (`X`, `Y`) and corresponding position indices (`x_pos`, `y_pos`) stored in compressed `.npz` files:contentReference[oaicite:32]{index=32}.

### Architecture

ChemTransformer implements a **Transformer encoder–decoder** using PyTorch.  Key architectural details include:

- **Embeddings**: separate embedding layers for source and target tokens (`src_embedding` and `tgt_embedding`) with dimension \(d_model=512\).  Each is multiplied by \(\sqrt{d_model}\) during forward propagation.
- **Learned positional encoding**: a `CompoundPositionalEncoding` module uses an embedding layer to generate position embeddings.  Unlike the sinusoidal positional encoding used in the original Transformer, this module learns embeddings for up to `max_len` positions and adds them to the token embeddings:contentReference[oaicite:33]{index=33}.
- **Transformer module**: a standard PyTorch `nn.Transformer` is configured with **6 encoder and 6 decoder layers**, **8 attention heads** and a feed‑forward dimension of **2048**:contentReference[oaicite:34]{index=34}.  Dropout (0.1 by default) is applied in the Transformer.
- **Output layer**: a linear layer projects the decoder output to vocabulary size.  Weights are initialised uniformly to avoid overfitting:contentReference[oaicite:35]{index=35}.
- **Forward pass**: during training the model receives token indices and position indices for both source and target sequences.  Embeddings are looked up, positional encodings are added and the token embeddings are passed to the Transformer.  A causal mask ensures that the decoder only attends to previous tokens, and padding masks mask out padded positions:contentReference[oaicite:36]{index=36}.

### Results and novelty

As this project is a template for reaction modelling and does not include a published manuscript, we could not locate official metrics.  The architecture largely follows the **original Transformer** but introduces **learned compound‑level positional encodings**.  By embedding positions based on the start of each reactant/reagent, the model can better capture the segmentation of the reaction equation.  The pipeline also emphasises data preprocessing: converting reactions to event‑based token sequences with explicit role tokens and optional numerical conditions.  This design makes it straightforward to incorporate additional information (e.g., amounts, temperature) and to experiment with SMILES versus SELFIES tokenisation.  The model therefore serves as a flexible framework for exploring Transformer‑based reaction prediction and can be extended with pre‑training or graph encoders.

## USPTO‑480k dataset (also known as USPTO‑MIT)

The **USPTO‑480k** dataset is a widely used benchmark for forward reaction prediction.  It was extracted from Lowe’s US patent database and contains **479,035 pairs of reactants and the major product**:contentReference[oaicite:37]{index=37}.  Each reaction is atom‑mapped and single‑step; the dataset excludes stereochemistry but includes the major product of each reaction.  Most studies split the data into training, validation and test sets using a **scaffold split** (often 8:1:1 or 80 / 10 / 10).  The dataset is used for both forward prediction (mapping reactants to product) and retrosynthesis (mapping product to reactants).  Variants include **USPTO‑480k_mixed** (reactants and reagents mixed), **USPTO‑480k_separated** (reactants/reagents separated by `>`), **USPTO‑STEREO** (retaining stereochemistry) and **USPTO‑50k** (subset of 50 k reactions with reaction class annotations).  The dataset is publicly available from Jin et al.’s repository:contentReference[oaicite:38]{index=38}.

### Novelty and relevance

USPTO‑480k is one of the largest open collections of atom‑mapped reaction data.  Its size allows training deep learning models without pre‑training.  The dataset’s quality and the availability of variants (mixed vs separated, stereo vs non‑stereo, full vs subsets) have made it the de facto benchmark for forward reaction prediction.  Many models in this review—including Molecular Transformer, Graph2SMILES and Chemformer—were initially evaluated on this dataset.  Moreover, the event‑based processing used in our **ChemTransformer** shows how USPTO‑480k can be enriched with additional reaction condition information (e.g., solvents, catalysts and temperatures) for more realistic modelling.

## Summary

Modern approaches to chemical reaction prediction leverage different data representations and learning objectives.  The **Molecular Transformer** treats reactions as tokenised SMILES sequences and uses a lightweight Transformer encoder–decoder to translate reactants to products; it established a strong baseline for forward prediction.  **Chemformer** extends this idea by pre‑training on millions of SMILES, demonstrating that general chemical language modelling improves performance and sample efficiency.  **Graph2SMILES** breaks away from purely text‑based methods by encoding the reaction as a graph, achieving higher top‑1 accuracy and better stereochemical predictions through graph‑aware positional embeddings.  **3DInfomax** addresses the limitation of 2D representations by maximising mutual information between 2D and 3D molecular views, pre‑training a GNN that can incorporate 3D structural information without requiring 3D coordinates at inference time.  Finally, the internal **ChemTransformer** project illustrates how to build a flexible Transformer model with learned positional encodings and event‑level tokenisation, providing a foundation for further experimentation on reaction modelling.
