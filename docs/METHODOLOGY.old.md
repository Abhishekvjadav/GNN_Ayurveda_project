\# Methodology



\## 1. Overview



This project investigates heterogeneous graph neural network methods for compound-to-disease/symptom link prediction using the AyurKOSH knowledge graph.



The primary supervised task is:



```text

Compound ── TREATS ──> Disease

Compound ── TREATS ──> Symptom



The objective is to determine whether graph-based representation learning can retrieve disease/symptom targets associated with Ayurvedic compound formulations using the surrounding knowledge graph structure.



The evaluation is designed around compound-held-out cross-validation to prevent the same compound from appearing in both the supervised training and outer test sets.



2\. Source Dataset



The knowledge graph is constructed from the AyurKOSH Excel workbook.



The source contains multiple relational tables describing:



Diseases

Symptoms

Herbs

Compound formulations

Herb groups

Herb qualities

Alternate herbs

Formulation relationships

Compound ingredients

Compound treatment relationships



The source workbook is not committed to the repository.



Expected location:



data/raw/ayurkosh.xlsx

3\. Phase 1 — Knowledge Graph Construction



Phase 1 converts the source AyurKOSH tables into a unified heterogeneous knowledge graph.



The graph contains 11 node types:



Disease

Symptom

Herb

Mineral

Vehicle

Sub\_Formulation

Compound

Group

Quality

Ayurvedic\_Concept

Other\_Unclassified



The graph contains 8 relation types:



TREATED\_BY

HAS\_LAKSHAN

ALTERNATE\_OF

HAS\_QUALITY

CONTAINS

BELONGS\_TO

HAS\_KALP

TREATS



The frozen Phase 1 graph contains:



Property	Value

Nodes	13,584

Edges	22,329

Node types	11

Edge types	8

TREATS edges	232

Labeled compounds	82

Isolated nodes	0



The graph is frozen after construction and integrity validation so that subsequent experiments operate on the same graph.



4\. Entity Normalization and Classification



The source contains inconsistent entity representations and category conflicts.



Phase 1 therefore performs normalization and classification before graph construction.



The processing includes:



Unicode normalization

Whitespace normalization

Removal of zero-width characters

Controlled normalization of selected spelling variants

Ingredient classification

Detection of source conflicts

Removal/prevention of self-loop relationships where appropriate

Deduplication of graph entities and relations



The classification process uses controlled categories rather than assuming that every entity appearing under a source Herb label is necessarily a botanical herb.



Unresolved or ambiguous entities are retained using the appropriate fallback category rather than silently discarding source information.



5\. Graph Integrity Audit



The finalized graph is checked before supervised learning.



The integrity audit verifies:



Duplicate node IDs

Dangling source node IDs

Dangling target node IDs

Self-loops

Exact duplicate edges

Relation directions

Graph connectivity

Node-type consistency

Edge-type consistency



The finalized graph contains no isolated nodes and no remaining self-loops.



6\. Supervised Task Definition



The primary supervised task is compound-to-disease/symptom link prediction.



Each labeled relationship has the form:



Compound → TREATS → Disease



or:



Compound → TREATS → Symptom



The final supervised dataset contains:



82 labeled compounds

232 TREATS edges



The task is treated as a ranking problem.



For each compound, candidate diseases or symptoms are ranked according to their predicted relevance.



7\. Compound-Held-Out Cross-Validation



A five-fold compound-level split is used.



The split is performed over compounds rather than individual TREATS edges.



This prevents treatment labels belonging to the same compound from being distributed between training and outer testing.



The five outer folds contain:



Fold 0

Fold 1

Fold 2

Fold 3

Fold 4



For each outer fold:



Test compounds are identified.

Their TREATS labels are held out.

Training compounds provide the supervised training labels.

The graph encoder retains non-TREATS structural relationships.

The held-out TREATS relationships are excluded from message passing.

The trained model ranks candidate Disease/Symptom targets.

The final outer-fold metrics are calculated.

8\. Structural Information Available to the GNN



The graph encoder can use structural relationships such as:



Disease → TREATED\_BY → Herb

Disease → HAS\_LAKSHAN → Symptom

Compound → CONTAINS → Herb

Herb → HAS\_QUALITY → Quality

Herb → BELONGS\_TO → Group

Herb → ALTERNATE\_OF → Herb

Herb → HAS\_KALP → Sub\_Formulation



The held-out TREATS labels are masked for the outer test compounds.



Other structural relationships remain available because they represent graph context rather than the target label itself.



9\. Candidate Ranking



Evaluation uses filtered ranking.



For each test case:



The target type is identified.

Candidate targets are restricted to the same type.

Known alternative positive targets for the compound are excluded from the ranking.

The true held-out target remains in the candidate set.

The model scores all remaining candidates.

Candidates are sorted according to their score.

The rank of the true target is recorded.



Disease and Symptom candidates are therefore evaluated separately according to their target type.



10\. Evaluation Metrics



The primary metrics are:



Mean Reciprocal Rank



For a true target with rank r:



Reciprocal Rank = 1 / r



The mean reciprocal rank is:



MRR = mean(1 / rank)



Higher values indicate that the correct target tends to appear higher in the ranking.



Hits@1



The percentage of cases where the correct target is ranked first.



Hits@5



The percentage of cases where the correct target appears within the top five.



Hits@10



The percentage of cases where the correct target appears within the top ten.



11\. Guna/Group Structural Baseline



A non-learning structural baseline is used to evaluate whether simple Ayurvedic graph attributes can retrieve treatment targets.



The baseline constructs profiles using:



Quality

Group



information associated with formulation ingredients and candidate targets.



The similarity between compound and candidate profiles is calculated using Jaccard similarity.



The baseline uses the same compound-held-out folds and filtered candidate pools as the GNN.



This ensures that the comparison is performed under the same ranking conditions.



The low performance of this baseline is interpreted partly in the context of limited Quality/Group coverage in the source graph.



12\. Popularity Baseline



A popularity-frequency baseline is used as a stronger non-GNN comparison.



For each outer fold, target popularity is calculated using only the training-fold TREATS relationships.



The popularity baseline therefore does not use the held-out outer test labels when calculating target frequency.



This is essential for preventing label leakage.



The popularity baseline uses the same filtered candidate pools as the GNN.



13\. GNN Architecture



The graph model uses a heterogeneous message-passing encoder with a DistMult link-prediction decoder.



The encoder is implemented using PyTorch Geometric.



The architecture uses relation-specific GraphSAGE-style message passing through heterogeneous convolution layers.



The model contains two message-passing layers.



Main dimensions:



Embedding dimension: 64

Hidden dimension: 64

14\. Compound Representation



The implementation uses a shared type-level compound embedding parameter for the Compound node type.



This provides a common initial representation for compound nodes before graph message passing.



Other node types use trainable node representations.



The representations are subsequently updated using heterogeneous graph message passing.



15\. DistMult Decoder



The final link score is produced using a DistMult decoder.



For source representation h, relation representation r, and target representation t:



score(h, r, t) = Σ(h × r × t)



For the supervised task, the relevant relation is:



TREATS



The decoder produces a ranking score.



The score is not interpreted as a calibrated probability.



16\. Training Objective



The model is trained using weighted Binary Cross Entropy.



The training configuration includes:



Learning rate: 0.005

Weight decay: 1e-4

Maximum epochs: 150

Positive class weight: 2.0

Random seed: 42



Early stopping is used based on validation performance.



17\. Negative Samples



Training requires negative compound-target examples.



The negative-generation procedure avoids known positive TREATS relationships.



The project also uses structural information from the graph when constructing training negatives.



The negative-sampling procedure is separate from the final filtered-ranking evaluation candidates.



The evaluation candidate pool is reconstructed independently for the outer test cases.



18\. Inner Validation



Model selection is separated from outer test evaluation.



For each outer fold, the outer training compounds are divided into an inner training and validation portion.



The inner validation split is deterministic.



The validation TREATS relationships are masked during validation.



The validation data is used for:



Early stopping

Selecting the popularity/GNN blend weight



The outer test set is not used for these decisions.



19\. Popularity/GNN Combination



A combined ranking score is defined as:



combined =

&#x20;   (1 - alpha) × popularity\_norm

&#x20;   + alpha × gnn\_norm



where:



alpha = 0



corresponds to popularity-only ranking, and:



alpha = 1



corresponds to GNN-only ranking.



Intermediate values combine the two normalized scores.



The value of alpha is selected using inner validation.



It is not selected using outer test performance.



20\. Final Selected Alpha Values



The inner-validation procedure selected:



Outer Fold	Selected Alpha

0	0.75

1	0.00

2	0.75

3	0.00

4	0.25



These values are then fixed before evaluating the corresponding outer test fold.



21\. Outer Evaluation



After model training and inner validation:



The model is retrained using the outer training data.

The selected alpha is retained.

Popularity is calculated using outer-training labels only.

The held-out outer test cases are scored.

Popularity, GNN, and combined rankings are evaluated.

MRR and Hits@K metrics are calculated.



The outer test fold is not used during model or alpha selection.



22\. Final Outer-Fold Results



The saved outer-fold evaluation produces the following results:



Fold	Alpha	Popularity MRR	GNN MRR	Combined MRR

0	0.75	0.1159	0.1242	0.1254

1	0.00	0.1027	0.1410	0.1027

2	0.75	0.1268	0.1120	0.1166

3	0.00	0.1480	0.1190	0.1480

4	0.25	0.1502	0.1130	0.1412



Aggregate MRR:



Popularity:

0.1287 ± 0.0205



GNN:

0.1218 ± 0.0118



Popularity + GNN:

0.1268 ± 0.0183



The reported uncertainty is the sample standard deviation across the five outer folds.



23\. Interpretation



The final experiment does not establish a robust improvement of the GNN over the popularity baseline.



The results indicate that simple target-frequency information is already a strong signal in this small supervised dataset.



The GNN still provides a graph-based modeling framework and obtains non-trivial ranking performance, but the current data does not provide sufficient evidence to claim that the graph model consistently outperforms the popularity shortcut.



The combined model also does not establish a consistent improvement over popularity alone.



These conclusions are limited to the available AyurKOSH data and the evaluation protocol used in this project.



24\. Data Limitations



The dataset has several limitations that affect interpretation.



Small number of supervised compounds



Only 82 compounds have treatment labels that can be used for the supervised task.



The complete supervised relation set contains 232 TREATS edges.



This limits the amount of training data available for graph link prediction.



Entity ambiguity



The source contains inconsistent entity labels and terminology.



Some entities appear under different categories in different records.



Herb classification



Some source entities originally labeled as herbs could not be confidently assigned to the Herb category using the available evidence.



These are retained in fallback categories where appropriate.



Quality/Group sparsity



Quality and Group information does not cover all ingredients and compounds uniformly.



This limits the effectiveness of the Guna/Group structural baseline.



25\. Leakage Controls



Leakage prevention is a central part of the evaluation design.



The following controls are applied:



Compound-level holdout



Compounds are split rather than individual treatment edges.



Held-out TREATS masking



The test compound's held-out TREATS edges are removed from message passing.



Training-only popularity



Popularity is calculated using training labels only.



Inner validation for alpha



The popularity/GNN blend weight is selected using inner validation rather than outer test performance.



Filtered ranking



Known alternative positives are excluded from candidate ranking.



Outer test isolation



The outer test fold is evaluated only after model and alpha selection.



26\. Previous Evaluation Issue



An earlier popularity-baseline implementation calculated target popularity using the full TREATS dataset.



That could allow held-out test labels to influence popularity scores.



The issue was identified and corrected.



The final reported popularity results use training-fold-only TREATS labels.



The corrected evaluation is the one used for the final comparison in this repository.



27\. Explainability



The project includes structural explanation tools.



For a compound-to-target prediction, the system can inspect:



Direct TREATS relationships

Shared graph neighbors

Short structural paths

Local graph context



These explanations describe graph structure surrounding a prediction.



They should not be interpreted as:



causal explanations,

clinical reasoning,

proof of therapeutic effectiveness,

calibrated probabilities,

or feature-level attribution.

28\. Prediction vs Evaluation



The repository contains prediction utilities for demonstrating model inference on compounds.



These tools should not be confused with the research evaluation.



A multi-fold prediction utility may combine information from models trained on different folds.



For a compound that was present in the training set of some folds, those models have already observed its labels.



Therefore:



Prediction/demo output ≠ held-out evaluation



Only the compound-held-out outer-fold evaluation should be used when reporting generalization performance.



29\. Reproducibility



The main experiments use:



Random seed: 42

Embedding dimension: 64

Hidden dimension: 64

Learning rate: 0.005

Weight decay: 1e-4

Maximum epochs: 150

Validation fraction: 0.15

Positive class weight: 2.0



The repository contains scripts for:



Phase 1 graph construction

Phase 2 graph auditing

Compound splitting

Evaluation candidate generation

Negative generation

Baseline evaluation

GNN training

Inner validation

Alpha selection

Outer evaluation

Prediction

Structural explanation

30\. Research Scope



This methodology is intended for research into:



Ayurvedic knowledge graphs

Heterogeneous graph learning

Compound-to-disease/symptom retrieval

Link prediction

Graph representation learning

Knowledge discovery

Explainable graph-based retrieval



The system is not designed or validated as a clinical diagnostic or treatment recommendation system.



31\. Summary



The methodology follows a controlled experimental sequence:



AyurKOSH

&#x20;  ↓

Entity normalization

&#x20;  ↓

Heterogeneous knowledge graph

&#x20;  ↓

Graph integrity validation

&#x20;  ↓

Compound-held-out 5-fold split

&#x20;  ↓

Training / validation / outer testing

&#x20;  ↓

Popularity baseline

&#x20;  ↓

Guna/Group structural baseline

&#x20;  ↓

Heterogeneous GNN

&#x20;  ↓

Inner alpha selection

&#x20;  ↓

Filtered outer-fold ranking

&#x20;  ↓

MRR / Hits@1 / Hits@5 / Hits@10



The final experiment shows that the current GNN does not demonstrate a robust improvement over the leakage-safe popularity baseline.



The result is therefore reported as an empirical finding of the current dataset and methodology rather than as evidence that graph neural networks are ineffective for Ayurvedic knowledge discovery in general.

