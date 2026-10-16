# AyurKOSH-GNN: Evaluation Audit Report

**Audit Date**: October 2026  
**Repository**: `Abhishekvjadav/GNN_Ayurveda_project`  
**Target Task**: Compound/Formulation $\to$ `TREATS` $\to$ Disease / Symptom link prediction  

---

## 1. Executive Summary

This audit inspects the existing implementation of data loading, graph construction, train/validation/test splitting, negative sampling, model architecture, decoders, training loop, checkpoint selection, baselines, and evaluation scripts in the AyurKOSH-GNN codebase.

The repository implementation follows a sound, leakage-controlled protocol for compound-held-out cross-validation. However, the audit identified important architectural nuances, naming discrepancies, and a major bug in the existing `baseline_guna_group.py` script that rendered its previous benchmark outputs meaningless (scores tied at 0.0).

---

## 2. Graph & Data Audit

### 2.1 Graph Statistics (Frozen Phase 1 Graph)
- **Source files**: `results/phase1/nodes.csv` and `results/phase1/edges.csv`
- **Total Nodes**: 13,584
- **Total Edges**: 22,329
- **Isolated Nodes**: 0
- **Node Types (11)**:
  - `Herb`: 9,351
  - `Symptom`: 1,924
  - `Sub_Formulation`: 1,502
  - `Other_Unclassified`: 195
  - `Ayurvedic_Concept`: 129
  - `Group`: 203
  - `Mineral`: 93
  - `Compound`: 83 (82 with TREATS labels, 1 unlabeled)
  - `Disease`: 59
  - `Quality`: 35
  - `Vehicle`: 10
- **Relation Types (8)**:
  - `TREATED_BY`: 14,502 edges (Disease $\to$ Herb/Mineral/Sub_Formulation/Vehicle/Other)
  - `HAS_LAKSHAN`: 3,853 edges (Disease $\to$ Symptom, Ayurvedic_Concept $\to$ Ayurvedic_Concept)
  - `ALTERNATE_OF`: 1,328 edges
  - `HAS_QUALITY`: 836 edges
  - `CONTAINS`: 788 edges (Compound $\to$ Herb/Mineral/Sub_Formulation/etc.)
  - `BELONGS_TO`: 452 edges
  - `HAS_KALP`: 338 edges
  - `TREATS`: 232 edges (Compound $\to$ Disease: 39; Compound $\to$ Symptom: 193)

---

## 3. Model Architecture & Decoder Audit

### 3.1 Actual Model Architecture (`train_rgcn.py`)
- **Class Name**: `HeteroRGCNModel`
- **Actual Layer Implementation**: `torch_geometric.nn.HeteroConv` containing relation-specific `SAGEConv((-1, -1), hidden_dim)` operators with `aggr="mean"`.
  - **Discrepancy Note**: Despite the class and script names referencing "RGCN", the actual implementation utilizes **relation-specific GraphSAGE convolutions (`SAGEConv`)** over typed relation bipartite subgraphs, not `RGCNConv`. A separate script `train_true_rgcn.py` tested homogeneous `RGCNConv`, but the primary model in the repository and saved checkpoints is the `HeteroConv(SAGEConv)` model.
- **Layers**: 2 message-passing layers with `F.relu` non-linearity between layer 1 and layer 2.
- **Embedding Dimensions**:
  - `embed_dim`: 64
  - `hidden_dim`: 64
- **Compound Representation**:
  - `shared_compound_embedding`: Trainable parameter tensor of shape `(1, 64)`.
  - **Crucial Design Choice**: Because test compounds are completely held out, individual compound IDs cannot have memorized embedding rows. Every compound begins with the exact same type-level vector `[1, 64]`. Their final representations are derived entirely via message passing from their graph neighbors (chiefly `CONTAINS` edges pointing to ingredients).
  - Other 10 node types have individual trainable embedding tables `(num_nodes, 64)`.
- **Decoder**:
  - `DistMultDecoder`: Parameter vector $r \in \mathbb{R}^{64}$.
  - Scoring: $\text{score}(h, t) = \sum_{i=1}^{64} h_i \cdot r_i \cdot t_i = h^\top \text{diag}(r) t$.
  - This is a diagonal bilinear link-prediction decoder.
- **Loss Function**:
  - Weighted Binary Cross Entropy with logits:
    $$\mathcal{L} = \frac{w_{\text{pos}} \cdot \mathcal{L}_{\text{pos}} + \mathcal{L}_{\text{neg}}}{w_{\text{pos}} + 1}$$
    where $w_{\text{pos}} = 2.0$.
- **Optimizer & Training Schedule**:
  - Adam optimizer, learning rate $= 0.005$, weight decay $= 10^{-4}$.
  - Maximum epochs $= 150$, early stopping patience $= 20$, seed $= 42$.

---

## 4. Split Strategy & Leakage Controls

### 4.1 Outer Split (`generate_compound_split.py`)
- **Method**: Stratified 5-Fold Cross Validation.
- **Unit of Partitioning**: Entire `Compound` nodes (82 labeled compounds).
- **Stratification**: Degree-bucketed based on number of TREATS edges (bucket 1: 1 edge, bucket 2: 2-3 edges, bucket 3: 4+ edges). Round-robin allocation within buckets using `np.random.RandomState(42)`.
- **Fold sizes**:
  - Fold 0: 16 compounds, 43 test TREATS edges (11 disease, 32 symptom)
  - Fold 1: 17 compounds, 60 test TREATS edges (5 disease, 55 symptom)
  - Fold 2: 16 compounds, 38 test TREATS edges (5 disease, 33 symptom)
  - Fold 3: 17 compounds, 50 test TREATS edges (11 disease, 39 symptom)
  - Fold 4: 16 compounds, 41 test TREATS edges (7 disease, 34 symptom)
  - Total: 82 compounds, 232 test TREATS edges (39 disease, 193 symptom)

### 4.2 Leakage Controls
1. **Target Leakage**:
   - `TREATS` is explicitly in `NO_REVERSE`. No reverse `rev_TREATS` edge exists in the HeteroData graph.
   - For fold $k$, only the training TREATS edges (`fold != k`) are included in the graph. The test TREATS edges are completely excluded.
   - Structural non-TREATS edges (such as `CONTAINS`) remain in the training graph for test compounds, which is the intended scientific design.
2. **Inner Validation Protocol**:
   - For fold $k$, 15% of the outer training TREATS edges are deterministically sampled (`seed=42`) as an inner validation set.
   - In `mask_validation_treats_edges()`, these validation TREATS edges are removed from the training graph during inner validation training and early stopping.
   - The best epoch is chosen based on inner validation loss.
   - A fresh model is then retrained from scratch on 100% of outer training TREATS edges for `best_epoch + 1` epochs before evaluating on the untouched outer test set.
3. **Popularity Baseline Leakage Control**:
   - The popularity baseline counts target occurrences strictly from `train_treats_df` (`fold != k`). It never uses test or full-graph TREATS edges.

---

## 5. Negative Sampling & Evaluation Protocols

### 5.1 Training Negative Sampling
- 5 Guna/Group-informed negatives per positive edge generated in `build_eval_and_negatives.py`.
- Candidate targets with lowest Jaccard overlap of Guna/Group profiles with the compound are chosen as hard negatives. Known positive targets for the compound are excluded from being chosen as negatives.

### 5.2 Evaluation Protocol (Filtered Ranking)
- Evaluation does NOT sample negatives; it evaluates over the complete universe of same-type candidate entities.
- For each test association $(c, t^*)$ where $t^*$ is Disease:
  - Candidate universe: all 59 Disease nodes.
- For each test association $(c, t^*)$ where $t^*$ is Symptom:
  - Candidate universe: all 1,924 Symptom nodes.
- **Candidate Filtering**: All known positive targets of compound $c$ (other than the true target $t^*$ being evaluated) are filtered out from the candidate pool.
- **Ranking**: Tie-aware average rank:
  $$\text{Rank} = 1 + \sum_{c \in \mathcal{C}} \mathbb{I}(s_c > s_{t^*}) + 0.5 \cdot \left(\sum_{c \in \mathcal{C}} \mathbb{I}(s_c = s_{t^*}) - 1\right)$$
- Metrics:
  - $\text{MRR} = \frac{1}{|\mathcal{D}_{\text{test}}|} \sum \frac{1}{\text{Rank}}$
  - $\text{Hits@K} = \frac{1}{|\mathcal{D}_{\text{test}}|} \sum \mathbb{I}(\text{Rank} \le K)$ for $K \in \{1, 5, 10\}$.

---

## 6. Bugs and Inconsistencies Discovered

### Bug 1: Complete Inversion of Guna/Group Baseline (`baseline_guna_group.py`)
- **Location**: `src/phase2/baseline_guna_group.py`, lines 68–71.
- **Code in repo**:
  ```python
  target_herbs = defaultdict(set)
  for _, r in treated_by.iterrows():
      target_herbs[r["target"]].add(r["source"])
  ```
- **The Bug**:
  In `results/phase1/edges.csv`, the relation `TREATED_BY` has `source` as Disease and `target` as Herb (11,881 edges).
  By storing `target_herbs[r["target"]].add(r["source"])`, the dictionary used **Herb IDs as keys** and **Disease IDs as values**.
  Subsequently, line 84 attempted:
  ```python
  disease_profile = {d: profile(target_herbs.get(d, set())) for d in all_disease_ids}
  ```
  Because `d` is a Disease ID and `target_herbs` had only Herb IDs as keys, `target_herbs.get(d, set())` returned an **empty set for every disease**.
  Furthermore, the script completely omitted the `HAS_LAKSHAN` relation for symptoms, so `symptom_profile` was also an empty set for every symptom.
- **Consequence**:
  Every candidate target in the baseline received a Jaccard score of `0.0`.
  Because all candidate scores were tied at `0.0`, tie-aware ranking placed every disease at rank $30.0$ (out of 59) and every symptom at rank $962.5$ (out of 1,924).
  This produced artificial baseline metrics: $\text{Hits@1} = 0$, $\text{Hits@5} = 0$, $\text{Hits@10} = 0$, and $\text{MRR} \approx 0.0067$.
- **Correct Formulation**:
  Aligning `baseline_guna_group.py` with the methodology in `build_eval_and_negatives.py` (which correctly maps `Disease -> TREATED_BY -> Herb` and `Symptom <- HAS_LAKSHAN - Disease - TREATED_BY -> Herb`) gives real non-zero Jaccard profiles for 49/59 diseases and 1,847/1,924 symptoms.

### Bug 2: Hardcoded Paths
- Scripts such as `src/phase2/predict.py`, `src/phase2/train_rgcn.py`, and `src/phase2/train_true_rgcn.py` have hardcoded default paths pointing to `/content/GNN_Ayurveda_project/...` (Google Colab paths), requiring command-line argument overrides when run locally on Windows.

---

## 7. Audit Conclusion & Recommendations

1. **GNN Pipeline Integrity**: The GNN model training, inner validation early stopping, retraining, and outer test evaluation are mathematically sound, leakage-free, and 100% reproducible. The saved checkpoints in `models/gnn_rgcn_fold0.pt` through `models/gnn_rgcn_fold4.pt` perfectly regenerate the exact numbers reported in the project README.
2. **Baseline Correction**: The Guna/Group Jaccard baseline must be reported in both its existing uncorrected form (as saved in the repo) and its corrected form to provide a scientifically truthful benchmark.
3. **Model Naming Clarity**: The documentation should explicitly state that the heterogeneous model utilizes `HeteroConv(SAGEConv)` rather than standard `RGCNConv`.
