\# Experiments



\## 1. Overview



This document records the main experiments performed for the AyurKOSH GNN project.



The purpose is to preserve the experimental history, evaluation corrections, baseline comparisons, and final results.



The primary research question is:



> Does heterogeneous graph representation learning improve compound-to-disease/symptom retrieval compared with simpler ranking approaches?



The experiments use the same frozen Phase 1 graph and compound-held-out five-fold evaluation framework unless explicitly stated otherwise.



\---



\# 2. Experimental Setup



The finalized graph contains:



| Property | Value |

|---|---:|

| Nodes | 13,584 |

| Edges | 22,329 |

| Node types | 11 |

| Edge types | 8 |

| TREATS edges | 232 |

| Labeled compounds | 82 |



The supervised task is:



```text

Compound → TREATS → Disease/Symptom



The primary metrics are:



MRR

Hits@1

Hits@5

Hits@10

3\. Experiment 1 — Graph Construction

Objective



Construct a heterogeneous AyurKOSH knowledge graph from the source workbook.



Procedure



The source workbook was processed through:



Entity normalization

&#x20;       ↓

Entity classification

&#x20;       ↓

Relationship construction

&#x20;       ↓

Deduplication

&#x20;       ↓

Graph validation



The resulting frozen graph contains:



13,584 nodes

22,329 edges

11 node types

8 edge types



The graph contains no isolated nodes and no remaining self-loops after integrity correction.



Outcome



The Phase 1 graph was frozen and used for all subsequent experiments.



4\. Experiment 2 — Compound-Held-Out Cross-Validation

Objective



Prevent the same compound from appearing in both supervised training and outer testing.



Procedure



The 82 labeled compounds were divided into five folds.



Each fold is evaluated independently.



For each outer fold:



Training compounds

&#x20;       ↓

GNN training

&#x20;       ↓

Inner validation

&#x20;       ↓

Alpha selection

&#x20;       ↓

Outer held-out compound evaluation



The held-out compound's TREATS labels are excluded from message passing during evaluation.



Other graph relations remain available as structural context.



Outcome



Five valid outer folds were produced.



The number of outer test cases differs between folds because compounds have different numbers of TREATS edges.



5\. Experiment 3 — Guna/Group Structural Baseline

Objective



Evaluate whether simple Ayurvedic structural information can retrieve the correct disease/symptom target without a learned GNN.



Method



The baseline constructs profiles from:



Quality

Group



information associated with formulation ingredients and candidate targets.



Jaccard similarity is then used to rank candidates.



The baseline uses the same filtered-ranking candidate pools as the GNN.



Observation



The baseline performs poorly overall.



This is interpreted partly as a consequence of incomplete Quality/Group coverage in the source graph.



The result is therefore treated as a structural baseline rather than evidence that Guna/Group information is intrinsically uninformative.



6\. Experiment 4 — Popularity Baseline

Objective



Measure how well a simple target-frequency ranking performs without graph learning.



Method



For each outer fold:



Training-fold TREATS labels

&#x20;       ↓

Target frequency

&#x20;       ↓

Popularity score

&#x20;       ↓

Candidate ranking



Only training-fold labels are used.



Held-out test labels are excluded from popularity calculation.



Final Result



The leakage-safe popularity baseline achieves:



Mean MRR = 0.1287 ± 0.0205



This establishes an important reference point for the GNN.



7\. Experiment 5 — Initial GNN

Objective



Train a heterogeneous graph neural network for compound-to-disease/symptom link prediction.



Model



The model uses:



Heterogeneous GraphSAGE-style message passing

Two graph convolution layers

64-dimensional embeddings

64-dimensional hidden representation

DistMult decoder

Weighted Binary Cross Entropy

Early stopping



The supervised relation is:



TREATS

Final GNN Result



The five-fold evaluation produces:



Mean MRR = 0.1218 ± 0.0118



The GNN therefore does not consistently exceed the popularity baseline.



8\. Experiment 6 — Popularity-Aware Negative Sampling

Objective



Investigate whether the GNN could be improved by explicitly including popular-but-wrong targets as training negatives.



Method



An alternative negative-generation strategy was tested.



For each training positive, negatives included:



Popular but incorrect targets

Low-overlap structural negatives



The popularity information was calculated using training-fold data.



Result



The alternative training configuration produced a lower mean MRR than the original GNN configuration.



The experiment was therefore not adopted as the final training configuration.



Interpretation



This experiment demonstrates that making negative samples more popularity-aware did not automatically improve ranking performance under the current dataset and model configuration.



No claim is made that popularity-aware negative sampling is generally ineffective.



9\. Experiment 7 — Initial Combined Ranking Sweep

Objective



Investigate whether combining popularity and GNN scores could improve ranking.



The combined score was defined as:



combined =

&#x20;   (1 - alpha) × popularity\_norm

&#x20;   + alpha × gnn\_norm



The following alpha values were investigated:



0

0.25

0.50

0.75

1.00

Important Evaluation Rule



An initial sensitivity sweep on outer-test data was useful for understanding score behavior, but those outer-test results were not used to select the final alpha.



Using the outer test set to choose alpha would constitute test-set tuning.



Therefore, the initial sweep was treated as exploratory analysis only.



10\. Experiment 8 — Leakage-Controlled Alpha Selection

Objective



Select alpha without using the outer test set.



Method



For each outer fold:



Outer training data

&#x20;       ↓

Inner train / validation split

&#x20;       ↓

Train / validate

&#x20;       ↓

Evaluate alpha candidates

&#x20;       ↓

Select alpha

&#x20;       ↓

Retrain on outer training data

&#x20;       ↓

Evaluate untouched outer test



The inner validation procedure evaluates candidate alpha values using:



MRR

Hits@10

Hits@5

Hits@1



with deterministic tie-breaking.



Selected Alpha

Outer Fold	Alpha

0	0.75

1	0.00

2	0.75

3	0.00

4	0.25



These values are selected before the corresponding outer test fold is evaluated.



11\. Experiment 9 — Final Outer Evaluation



The final outer-fold results are:



Fold	Test Cases	Alpha	Popularity MRR	GNN MRR	Combined MRR

0	43	0.75	0.1159	0.1242	0.1254

1	60	0.00	0.1027	0.1410	0.1027

2	38	0.75	0.1268	0.1120	0.1166

3	50	0.00	0.1480	0.1190	0.1480

4	41	0.25	0.1502	0.1130	0.1412

12\. Final Aggregate Results



The final five-fold aggregate results are:



Method	MRR	Hits@1	Hits@5	Hits@10

Popularity	0.1287 ± 0.0205	0.0146 ± 0.0327	0.2492 ± 0.0561	0.3190 ± 0.0464

GNN	0.1218 ± 0.0118	0.0556 ± 0.0125	0.1547 ± 0.0559	0.2742 ± 0.0493

Popularity + GNN	0.1268 ± 0.0183	0.0391 ± 0.0365	0.1986 ± 0.0720	0.3030 ± 0.0453



The uncertainty values are sample standard deviations across the five outer folds.



13\. Experiment 10 — Popularity Baseline Leakage Correction

Issue



An earlier popularity-baseline implementation calculated target frequency using the complete TREATS dataset.



This included labels from the outer test fold.



Therefore, the earlier result was not suitable for final reporting.



Correction



The popularity calculation was changed to:



Training-fold TREATS edges only



The held-out outer test labels are no longer used when calculating popularity.



Consequence



The corrected popularity baseline is the one used in the final comparison.



Earlier leaky popularity results are not reported as final performance.



14\. Why the Leakage Correction Matters



The popularity baseline is intentionally simple.



If the baseline is allowed to observe test labels, it can gain information about which targets are common in the test set.



That violates the intended train/test separation.



The corrected pipeline therefore follows:



Outer training labels

&#x20;       ↓

Popularity statistics

&#x20;       ↓

Test candidate ranking



rather than:



All labels

&#x20;       ↓

Popularity statistics

&#x20;       ↓

Test candidate ranking

15\. Experiment 11 — Structural Explanation

Objective



Provide interpretable graph context for predictions.



Method



The explanation utilities inspect:



Direct TREATS relationships

Shared graph neighbors

Short structural paths

Local graph context



The explanations are generated after prediction.



Interpretation



The explanations identify graph structure surrounding a prediction.



They do not provide:



Causal explanations

Clinical evidence

Feature-level attribution

Calibrated probabilities



Therefore, they are described as structural explanations.



16\. Example Structural Explanation



For a predicted compound-target pair, the explanation system may find shared neighboring entities such as herbs, vehicles, or other graph entities.



This provides context such as:



Compound

&#x20;  │

&#x20;  ├── CONTAINS ──> Herb

&#x20;  │

&#x20;  └── structural path ──> Disease/Symptom



or:



Compound

&#x20;  │

&#x20;  └── shared neighbor

&#x20;            │

&#x20;            ▼

&#x20;      Candidate target



The presence of a structural path does not prove that the compound clinically treats the target.



17\. Final Findings



The final experiments support the following observations.



Finding 1



The popularity baseline is a strong reference on this dataset.



Finding 2



The GNN achieves non-trivial ranking performance but does not consistently outperform popularity.



Finding 3



Combining popularity and GNN scores does not produce a consistent improvement over popularity alone.



Finding 4



The selected alpha varies across outer folds:



0.75, 0.00, 0.75, 0.00, 0.25



This indicates that the relative usefulness of the two ranking signals varies across folds.



Finding 5



The small number of supervised compounds limits the strength of conclusions that can be drawn from the current evaluation.



18\. Limitations



The main experimental limitations are:



Small supervised dataset

Sparse TREATS relationships

Uneven target popularity

Incomplete Quality/Group coverage

Entity ambiguity in the source data

Source-category contradictions

Limited external validation

No clinical outcome validation



The results should therefore be treated as a graph-learning research evaluation rather than clinical validation.



19\. Reproducibility Rules



The following principles should be maintained when extending the experiments:



Do not modify the frozen Phase 1 graph without documenting a new graph version.

Keep compound-level outer folds fixed when comparing models.

Do not use outer test labels for model selection.

Calculate popularity from training data only.

Select hyperparameters using inner validation.

Use the same candidate-pool definition when comparing models.

Report fold-level sample sizes.

Preserve negative experimental results.

Do not report exploratory outer-test tuning as final evaluation.

Distinguish prediction/demo outputs from held-out evaluation.

20\. Recommended Future Experiments



Future work can investigate:



Larger validated formulation datasets

Improved entity normalization

Manual verification of ambiguous ingredients

Additional validated Ayurvedic knowledge sources

Stronger heterogeneous GNN architectures

Relation-aware attention mechanisms

Alternative link-prediction decoders

Calibrated ranking probabilities

External validation

Temporal validation

Source-specific generalization

Formal attribution methods



Any future experiment should preserve the same leakage-control principles.



21\. Final Experimental Statement



The current experiments show that a heterogeneous GNN can be trained for compound-to-disease/symptom ranking on the AyurKOSH knowledge graph.



However, under the current compound-held-out evaluation, the GNN does not demonstrate a robust improvement over a simple leakage-safe popularity baseline.



The combined popularity/GNN model also does not establish a consistent advantage.



The main contribution of the current experimental setup is therefore the construction of a reproducible, leakage-controlled evaluation framework for studying graph-based Ayurvedic knowledge discovery on a small heterogeneous knowledge graph.



The findings should be interpreted within the limitations of the available dataset and should not be generalized to clinical effectiveness or to graph neural networks in Ayurvedic knowledge discovery as a whole.

