# AyurKOSH GNN — Compound-to-Disease/Symptom Link Prediction

A heterogeneous graph-neural-network pipeline over the AyurKOSH knowledge graph for predicting which Ayurvedic compound formulations are associated with treated diseases and symptoms.

The project focuses on **compound-to-disease/symptom link prediction** using graph structure, formulation composition, Ayurvedic knowledge relations, and leakage-controlled evaluation.

---

## Result Summary

The final evaluation uses **compound-held-out 5-fold cross-validation** with leakage-controlled inner validation for selecting the popularity/GNN blend weight.

The supervised dataset contains only **232 labeled `Compound --TREATS--> Disease/Symptom` edges across 82 compounds**, making this a relatively small graph-learning problem.

| Fold | Selected Alpha | Popularity MRR | GNN MRR | Combined MRR |
|---:|---:|---:|---:|---:|
| 0 | 0.75 | 0.1159 | 0.1242 | 0.1254 |
| 1 | 0.00 | 0.1027 | 0.1410 | 0.1027 |
| 2 | 0.75 | 0.1268 | 0.1120 | 0.1166 |
| 3 | 0.00 | 0.1480 | 0.1190 | 0.1480 |
| 4 | 0.25 | 0.1502 | 0.1130 | 0.1412 |
| **Mean** | | **0.1287 ± 0.0183** | **0.1218 ± 0.0105** | **0.1268 ± 0.0164** |

The results indicate that the GNN does **not show a statistically robust improvement over the leakage-safe popularity baseline** at this dataset scale.

This result is reported directly rather than hiding the baseline comparison.

The project also documents the discovery and correction of a popularity-baseline leakage issue before finalizing the evaluation. This correction is an important part of the experimental methodology.

---

# 1. Project Objective

The objective is to investigate whether heterogeneous graph representation learning can improve retrieval of diseases and symptoms associated with Ayurvedic compound formulations.

The main supervised task is:

```text
Compound ── TREATS ──> Disease
Compound ── TREATS ──> Symptom

The model uses information from the broader AyurKOSH knowledge graph rather than relying only on direct compound-treatment labels.

The system is designed as a research prototype for graph-based knowledge discovery and formulation retrieval.

It is not a clinical diagnostic system and the predictions should not be interpreted as clinical evidence or treatment recommendations.

2. Knowledge Graph

The graph is constructed from the AyurKOSH dataset and integrates multiple entity and relationship types.

Node Types

The graph contains 11 node types:

Disease
Symptom
Herb
Mineral
Vehicle
Sub_Formulation
Compound
Group
Quality
Ayurvedic_Concept
Other_Unclassified
Edge Types

The graph contains 8 relation types:

Relation	Meaning
TREATED_BY	Disease associated with an herb
HAS_LAKSHAN	Disease associated with a symptom
ALTERNATE_OF	Herb alternate relationship
HAS_QUALITY	Herb associated with a quality
CONTAINS	Compound contains an ingredient
BELONGS_TO	Herb belongs to a group
HAS_KALP	Herb associated with a formulation/preparation
TREATS	Compound treats a disease/symptom
Frozen Phase 1 Graph

The finalized Phase 1 graph contains:

Property	Value
Nodes	13,584
Edges	22,329
Node types	11
Edge types	8
TREATS edges	232
Labeled compounds	82
Isolated nodes	0

The graph is frozen after the Phase 1 integrity audit. Subsequent experiments operate on this fixed graph.

3. Architecture

The overall experimental pipeline is:

                    AyurKOSH Dataset
                           │
                           ▼
              Phase 1 Knowledge Graph
                           │
                           ▼
              ┌──────────────────────┐
              │  13,584 Nodes        │
              │  22,329 Edges        │
              │  11 Node Types       │
              │  8 Relation Types    │
              └──────────────────────┘
                           │
                           ▼
              Compound-Held-Out 5-Fold CV
                           │
          ┌────────────────┼────────────────┐
          ▼                ▼                ▼
   Guna/Group       Popularity         Heterogeneous
    Baseline         Baseline             GNN
          │                │                │
          └────────────────┼────────────────┘
                           ▼
                Filtered Ranking
                    Evaluation
                           │
                           ▼
                 MRR / Hits@K
                           │
                           ▼
              Inner Validation Alpha
                    Selection
                           │
                           ▼
                 Outer Test Fold
4. GNN Architecture

The graph-learning model uses a heterogeneous message-passing encoder with a DistMult decoder.

Encoder

The implementation uses:

Heterogeneous graph convolution
Relation-specific GraphSAGE-style message passing
2 message-passing layers
Hidden dimension: 64
Embedding dimension: 64
Decoder

The link-prediction decoder is:

DistMult

For a compound c, relation r, and target t:

score(c, r, t) = Σ h_c × r × h_t

The decoder produces a ranking score rather than a calibrated clinical probability.

Training

The model uses:

Weighted Binary Cross Entropy
Positive class weight: 2.0
Learning rate: 0.005
Weight decay: 1e-4
Maximum epochs: 150
Early stopping
Deterministic validation split
CPU-compatible PyTorch Geometric implementation
5. Evaluation Protocol

The primary evaluation uses compound-held-out 5-fold cross-validation.

This is important because randomly splitting individual TREATS edges could allow the same compound to appear in both training and testing.

Instead, compounds are divided into five folds.

For each outer fold:

Test compounds are completely held out from the supervised TREATS labels.
Only the corresponding TREATS edges are removed from message passing.
Other graph relations remain available to the encoder.
Candidate targets are restricted to the same target type:
Disease
Symptom
Known alternative positives for the test compound are filtered from ranking.
Popularity statistics are calculated using training-fold data only.
GNN hyperparameters/early stopping are selected using inner validation.
The popularity/GNN blend weight is selected using inner validation.
The untouched outer test fold is evaluated only after model and alpha selection.
6. Ranking Metrics

The primary metrics are:

Mean Reciprocal Rank — MRR

Measures how high the correct target appears in the ranked candidate list.

MRR = mean(1 / rank)
Hits@1

Fraction of cases where the correct target is ranked first.

Hits@5

Fraction of cases where the correct target appears within the top five.

Hits@10

Fraction of cases where the correct target appears within the top ten.

The evaluation uses filtered ranking and tie-aware ranking where required.

7. Baselines

The GNN is compared against simpler approaches using the same evaluation folds and candidate pools.

7.1 Guna/Group Structural Baseline

This baseline computes Jaccard similarity between:

Compound ingredient profile
          ↓
Quality + Group information
          ↓
Candidate Disease/Symptom profile

The baseline does not learn parameters.

Its low performance is partly explained by limited Quality/Group coverage in the source graph.

This is treated as a data-coverage limitation rather than evidence that Guna/Group information itself is meaningless.

7.2 Popularity Baseline

The popularity baseline ranks candidate diseases/symptoms according to their frequency in the training-fold TREATS edges.

Importantly:

Held-out test labels are never used when calculating popularity.

A previous implementation accidentally calculated popularity using the full TREATS dataset. That implementation was identified and corrected before the final results were produced.

The final reported popularity results use training-fold-only information.

8. GNN + Popularity Combination

A combined ranking score was investigated:

combined =
    (1 - alpha) × popularity_norm
    + alpha × gnn_norm

where:

alpha = 0

means popularity only, and:

alpha = 1

means GNN only.

The value of alpha is not selected using the outer test set.

Instead, it is selected using an inner validation split.

Final selected values:

Fold	Selected Alpha
0	0.75
1	0.00
2	0.75
3	0.00
4	0.25

The outer test fold remains untouched during alpha selection.

9. Explainability

The project includes structural explanation utilities.

For a predicted:

Compound → Disease/Symptom

relationship, the explanation module can inspect:

Direct TREATS relationships
Shared neighboring entities
Short graph paths
Structural context around the compound and candidate target

The explanation should be interpreted as graph-structural evidence, not as:

causal evidence,
clinical reasoning,
feature-level attribution,
or proof that the predicted treatment relationship is medically valid.

The explanation tools include:

src/phase2/explain_prediction_v3.py
src/phase2/diagnose_prediction.py
src/phase2/predict.py
10. Repository Structure
GNN_Ayurveda_project/
│
├── data/
│   ├── raw/
│   │   └── ayurkosh.xlsx
│   │
│   └── audit/
│       └── phase0_multisource_classification_review_v4.xlsx
│
├── results/
│   ├── phase1/
│   │   ├── nodes.csv
│   │   └── edges.csv
│   │
│   └── phase2/
│       ├── treats_edges_with_fold.csv
│       ├── eval_candidates_fold*.csv
│       ├── train_hard_negatives_fold*.csv
│       └── evaluation outputs
│
├── models/
│   └── gnn_rgcn_fold*.pt
│
├── src/
│   ├── data/
│   │   └── phase1_cleaning.py
│   │
│   └── phase2/
│       ├── graph_integrity_audit.py
│       ├── generate_compound_split.py
│       ├── build_eval_and_negatives.py
│       ├── baseline_guna_group.py
│       ├── popularity_baseline.py
│       ├── build_hetero_data.py
│       ├── train_rgcn.py
│       ├── alpha_validation_fold0.py
│       ├── evaluate_outer_fold0_alpha.py
│       ├── predict.py
│       ├── predict_all.py
│       ├── predict_explain.py
│       ├── explain_prediction_v3.py
│       └── diagnose_prediction.py
│
├── docs/
│   ├── METHODOLOGY.md
│   ├── DATA_DICTIONARY.md
│   └── EXPERIMENTS.md
│
├── requirements.txt
├── README.md
└── .gitignore
11. Important Script Notes
Alpha validation

The file:

src/phase2/alpha_validation_fold0.py

contains fold0 in its filename because it originated from early single-fold development.

It accepts the fold number as a command-line argument:

python src/phase2/alpha_validation_fold0.py <fold_num>

For example:

python src/phase2/alpha_validation_fold0.py 0
python src/phase2/alpha_validation_fold0.py 1
python src/phase2/alpha_validation_fold0.py 2
python src/phase2/alpha_validation_fold0.py 3
python src/phase2/alpha_validation_fold0.py 4

The same applies to:

src/phase2/evaluate_outer_fold0_alpha.py

The filenames are legacy names and do not restrict the scripts to Fold 0.

12. Important Prediction Warning

predict.py is intended for demonstration/deployment-style inference.

It averages predictions from the available fold models.

For a real compound, most of those models may have seen that compound's treatment labels during training.

Therefore:

predict.py is not an evaluation tool and its predictions must not be reported as held-out generalization performance.

The research evaluation numbers come only from the compound-held-out outer-fold evaluation.

13. Setup
Requirements

Recommended environment:

Python 3.10+

Install dependencies:

pip install -r requirements.txt

For PyTorch Geometric, installation may depend on the local PyTorch/CUDA environment.

14. Dataset Setup

The original AyurKOSH workbook is intentionally not committed to GitHub.

Place it at:

data/raw/ayurkosh.xlsx

The Phase 0 classification review should be available at:

data/audit/phase0_multisource_classification_review_v4.xlsx

The source workbook should be obtained from the appropriate AyurKOSH dataset release.

15. Running the Pipeline
Step 1 — Phase 1 Knowledge Graph Construction
python src/data/phase1_cleaning.py

This generates:

results/phase1/nodes.csv
results/phase1/edges.csv
Step 2 — Graph Integrity Audit
python src/phase2/graph_integrity_audit.py \
    results/phase1/nodes.csv \
    results/phase1/edges.csv

The audit checks for:

Duplicate node IDs
Dangling edges
Self-loops
Duplicate edges
Invalid relation directions
Graph connectivity
Step 3 — Generate Compound-Held-Out Split
python src/phase2/generate_compound_split.py \
    results/phase1/edges.csv

This generates the five compound-held-out folds.

Step 4 — Build Evaluation Candidates and Negatives

Run for each fold:

python src/phase2/build_eval_and_negatives.py \
    results/phase1/nodes.csv \
    results/phase1/edges.csv \
    treats_edges_with_fold.csv \
    <fold_num>

Example:

python src/phase2/build_eval_and_negatives.py \
    results/phase1/nodes.csv \
    results/phase1/edges.csv \
    treats_edges_with_fold.csv \
    0

Repeat for:

0
1
2
3
4
16. Run Baselines
Guna/Group Baseline

For each fold:

python src/phase2/baseline_guna_group.py \
    results/phase1/nodes.csv \
    results/phase1/edges.csv \
    <fold_num> \
    eval_candidates_fold<fold_num>.csv
Popularity Baseline

For each fold:

python src/phase2/popularity_baseline.py \
    results/phase1/nodes.csv \
    results/phase1/edges.csv \
    treats_edges_with_fold.csv \
    <fold_num> \
    eval_candidates_fold<fold_num>.csv
17. Train the GNN

For each fold:

python src/phase2/train_rgcn.py \
    results/phase1/nodes.csv \
    results/phase1/edges.csv \
    treats_edges_with_fold.csv \
    <fold_num> \
    eval_candidates_fold<fold_num>.csv \
    train_hard_negatives_fold<fold_num>.csv

Example:

python src/phase2/train_rgcn.py \
    results/phase1/nodes.csv \
    results/phase1/edges.csv \
    treats_edges_with_fold.csv \
    0 \
    eval_candidates_fold0.csv \
    train_hard_negatives_fold0.csv

Repeat for folds 0–4.

18. Inner Alpha Selection

After training, select the popularity/GNN blend weight using inner validation.

python src/phase2/alpha_validation_fold0.py <fold_num>

Example:

python src/phase2/alpha_validation_fold0.py 0

Repeat for all five folds.

The outer test set is not used for alpha selection.

19. Outer Test Evaluation

After alpha selection:

python src/phase2/evaluate_outer_fold0_alpha.py <fold_num>

Example:

python src/phase2/evaluate_outer_fold0_alpha.py 0

Repeat for:

0
1
2
3
4

The resulting outer-fold files contain the final evaluation results.

20. Reproducibility

The main experiments use deterministic settings where practical.

Important settings include:

Seed: 42
Embedding dimension: 64
Hidden dimension: 64
Learning rate: 0.005
Weight decay: 1e-4
Maximum epochs: 150
Validation fraction: 0.15
Positive class weight: 2.0

The exact software environment may still affect numerical results because of differences between PyTorch, PyTorch Geometric, CPU/GPU implementations, and dependency versions.

21. Data-Quality Limitations

The source AyurKOSH data contains several challenges that affect graph construction and evaluation.

Ingredient classification

A large portion of entities originally appearing under the Herbs category required best-effort classification.

Approximately 75% of the relevant Herb-category nodes in the frozen graph fall into the Other_Unclassified bucket rather than being fully manually verified.

This is treated as a source-data/classification limitation.

Source contradictions

Some entities appear with contradictory semantic categories in the raw source.

For example, approximately 95 entities are tagged as both Herbs and Lakshan in different records.

These contradictions originate from the source data rather than being introduced by the GNN model.

Small supervised dataset

Only approximately 82 compounds have both ingredient and treatment information.

The supervised task therefore contains only:

232 Compound → TREATS → Disease/Symptom edges

This is the primary limitation on the statistical strength of the evaluation.

22. Interpretation of Results

The current results should be interpreted carefully.

The GNN obtains:

MRR = 0.1218 ± 0.0105

while the leakage-safe popularity baseline obtains:

MRR = 0.1287 ± 0.0183

The combined model obtains:

MRR = 0.1268 ± 0.0164

Therefore, the current experiment does not establish a robust advantage of the GNN over the popularity baseline.

Possible reasons include:

Small number of labeled compounds
Sparse supervised TREATS relationships
Uneven target frequency
Limited Guna/Group coverage
Noisy or contradictory source classifications
Strong popularity structure in the treatment labels
Limited amount of supervised training data

These are experimental observations and limitations rather than claims about the clinical validity of Ayurvedic treatments.

23. Research Integrity

This repository intentionally preserves negative and corrective findings.

In particular:

A popularity-baseline leakage issue was discovered.
The popularity calculation was corrected to use training-fold-only labels.
Alpha selection was moved to inner validation.
Outer test folds were kept untouched during model/alpha selection.
The final comparison includes the corrected popularity baseline.
The GNN's lack of clear improvement is reported rather than hidden.

This makes the final evaluation more reproducible and easier to audit.

24. Scope and Safety

This project is intended for:

Ayurvedic knowledge graph research
Graph representation learning
Formulation-to-disease/symptom retrieval research
Knowledge discovery
Machine learning experimentation
Explainable graph-based retrieval

It is not intended for:

Medical diagnosis
Clinical decision-making
Patient-specific treatment recommendation
Automated prescription
Determining treatment effectiveness
Replacing qualified medical or Ayurvedic practitioners

A predicted graph relationship represents a model-generated ranking based on the available dataset and should not be interpreted as clinical evidence.

25. Future Research Directions

Potential future work includes:

Expanding the number of labeled formulations
Improving manual entity normalization
Increasing coverage of Quality and Group relations
Adding additional validated Ayurvedic knowledge sources
Evaluating stronger heterogeneous GNN architectures
Testing relation-aware attention mechanisms
Calibrating prediction scores
Conducting larger-scale external validation
Evaluating temporal or source-specific generalization
Improving explanation quality with formal attribution methods

These are future research directions and are not part of the current reported evaluation.

26. Citation

If this repository or its methodology is used in academic work, please cite the corresponding project/paper once the final publication information is available.

27. Project Status

Current status: Research prototype / experimental evaluation complete

The main pipeline includes:

 AyurKOSH graph construction
 Phase 1 graph integrity audit
 Compound-held-out 5-fold evaluation
 Guna/Group baseline
 Leakage-safe popularity baseline
 Heterogeneous GNN
 DistMult link prediction
 Inner validation
 Leakage-controlled alpha selection
 Outer-fold evaluation
 Structural prediction explanations
 Reproducible experiment scripts
 Documentation of data limitations
 Documentation of evaluation corrections
Final Takeaway

This project investigates whether a heterogeneous graph neural network can learn useful compound-to-disease/symptom relationships from the AyurKOSH knowledge graph.

The current evaluation shows that, at the available dataset scale, the GNN does not clearly outperform a simple popularity-based ranking baseline.

Rather than treating this as a failure, the repository preserves the complete evaluation process—including the discovery and correction of a baseline leakage issue, inner validation for alpha selection, and the final outer-fold comparison.

The resulting system therefore serves as a reproducible experimental framework for studying graph-based Ayurvedic knowledge discovery and formulation retrieval.