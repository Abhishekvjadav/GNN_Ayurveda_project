\# Data Dictionary



\## 1. Overview



This document describes the main data files, node types, edge types, and supervised relationships used in the AyurKOSH GNN project.



The project converts the AyurKOSH source workbook into a heterogeneous knowledge graph and then evaluates compound-to-disease/symptom link prediction.



\---



\# 2. Source Workbook



The original source file is:



```text

data/raw/ayurkosh.xlsx



The workbook contains multiple relational sheets.



The main source tables used by the graph construction pipeline include:



Disease-Herb-Symptom

Herb-Group relationship

Herb-AlternateHerb relationship

Herb-Quality relationship

Herb-Kalp relationship

Compound-Herb relationship

Compound-Treats relationship

Lakshan-Prakruti-Ras-Dhatu rela



The source workbook is not committed to the Git repository.



3\. Phase 1 Output Files



The finalized graph is stored as:



results/phase1/nodes.csv

results/phase1/edges.csv

nodes.csv



Each row represents one graph node.



Important fields include:



Field	Description

node\_id	Unique identifier assigned to the graph node

node\_type	Semantic type of the node

name	Normalized entity name

canonical\_name	Canonicalized representation where available

source	Source information where retained



The exact columns may depend on the Phase 1 graph-building implementation.



4\. Node Types



The frozen graph contains 11 node types.



4.1 Disease



Represents a disease or disorder entity extracted from the AyurKOSH data.



Examples may include Ayurvedic disease terminology appearing in the source dataset.



4.2 Symptom



Represents a symptom or Lakshan entity associated with a disease.



Symptoms can participate in:



Disease → HAS\_LAKSHAN → Symptom



relationships.



4.3 Herb



Represents an ingredient classified as a botanical/herbal entity.



Herbs can participate in:



Disease → TREATED\_BY → Herb

Compound → CONTAINS → Herb

Herb → HAS\_QUALITY → Quality

Herb → BELONGS\_TO → Group

Herb → ALTERNATE\_OF → Herb



relationships.



4.4 Mineral



Represents a mineral or mineral-derived ingredient identified during ingredient classification.



This category prevents non-botanical ingredients from automatically being treated as herbs.



4.5 Vehicle



Represents a vehicle, carrier, or administration-related ingredient/category identified during ingredient classification.



4.6 Sub\_Formulation



Represents a formulation or preparation entity that functions as a component/sub-formulation rather than a primary botanical herb.



4.7 Compound



Represents an Ayurvedic compound formulation.



Compounds are the source entities for the primary supervised prediction task.



A compound can have:



Compound → CONTAINS → Ingredient



and:



Compound → TREATS → Disease/Symptom



relationships.



4.8 Group



Represents a group/category associated with herbs.



Relationship:



Herb → BELONGS\_TO → Group

4.9 Quality



Represents a quality/property associated with an herb.



Relationship:



Herb → HAS\_QUALITY → Quality

4.10 Ayurvedic\_Concept



Represents a normalized Ayurvedic concept that does not fit the more specific graph entity categories.



4.11 Other\_Unclassified



Fallback category for entities that cannot be assigned confidently to a more specific node type using the available source information.



These nodes are retained rather than silently discarded.



5\. Edge Types



The frozen graph contains 8 relation types.



Edge Type	Source	Target	Meaning

TREATED\_BY	Disease	Herb	Disease associated with an herb

HAS\_LAKSHAN	Disease	Symptom	Disease associated with a symptom

ALTERNATE\_OF	Herb	Herb	Alternate herb relationship

HAS\_QUALITY	Herb	Quality	Herb associated with a quality

CONTAINS	Compound	Ingredient	Compound contains an ingredient

BELONGS\_TO	Herb	Group	Herb belongs to a group

HAS\_KALP	Herb	Sub\_Formulation/Compound	Herb associated with a preparation/formulation

TREATS	Compound	Disease/Symptom	Compound associated with treatment of a disease/symptom

6\. Graph Statistics



The frozen Phase 1 graph contains:



Property	Value

Total nodes	13,584

Total edges	22,329

Node types	11

Edge types	8

Isolated nodes	0

TREATS edges	232

Labeled compounds	82

7\. Node Counts



The finalized node-type distribution is:



Node Type	Count

Herb	9,351

Symptom	1,924

Sub\_Formulation	1,502

Group	203

Other\_Unclassified	195

Ayurvedic\_Concept	129

Mineral	93

Compound	83

Disease	59

Quality	35

Vehicle	10



Total:



13,584 nodes

8\. Edge Counts



The finalized edge-type distribution is:



Edge Type	Count

TREATED\_BY	14,502

HAS\_LAKSHAN	3,196

ALTERNATE\_OF	1,927

HAS\_QUALITY	860

CONTAINS	680

BELONGS\_TO	512

HAS\_KALP	420

TREATS	232



Total:



22,329 edges

9\. Supervised Dataset



The primary supervised relationship is:



Compound → TREATS → Disease/Symptom



The final dataset contains:



82 labeled compounds

232 TREATS edges



The task is therefore substantially smaller than the complete graph.



The complete graph provides structural context, while the TREATS relationships provide the supervised labels.



10\. Target Types



The TREATS target can be one of two types:



Disease

Symptom



The evaluation maintains target-type consistency.



A Disease target is ranked against Disease candidates.



A Symptom target is ranked against Symptom candidates.



This avoids mixing semantically different target types in the same ranking pool.



11\. Compound Ingredients



The CONTAINS relationship represents ingredient composition.



Example:



Compound → CONTAINS → Herb



Ingredient information is used by the graph model as structural context.



The Phase 1 classification process also allows ingredient entities to be categorized as:



Herb

Mineral

Vehicle

Sub\_Formulation

Other\_Unclassified



rather than forcing every ingredient into the Herb category.



12\. Disease-Herb Information



The TREATED\_BY relationship connects diseases to herbs.



Example:



Disease → TREATED\_BY → Herb



This provides graph structure connecting diseases to their associated herbal entities.



This structure can indirectly connect compound ingredients to disease nodes through the graph.



13\. Disease-Symptom Information



The HAS\_LAKSHAN relationship connects diseases and symptoms:



Disease → HAS\_LAKSHAN → Symptom



This allows the graph to represent disease-symptom context.



For example, a compound may be connected to a disease through the supervised TREATS relation while the disease itself has multiple symptom relationships.



14\. Quality and Group Information



Two structural attributes used by the Guna/Group baseline are:



Herb → HAS\_QUALITY → Quality

Herb → BELONGS\_TO → Group



These relationships are used to construct herb-associated structural profiles.



The Guna/Group baseline compares the profile of formulation ingredients with the profile associated with candidate Disease/Symptom targets.



Coverage is incomplete, which limits the baseline.



15\. Phase 2 Fold Assignment



The compound-level fold assignment is stored in:



results/phase2/compound\_fold\_assignment.csv



Each labeled compound is assigned to one of five outer folds.



The fold assignment is generated using a fixed random seed.



The project uses:



Seed = 42

Number of folds = 5

16\. TREATS Fold File



The file:



results/phase2/treats\_edges\_with\_fold.csv



contains the supervised TREATS edges together with their compound fold assignment.



This file is used to determine which TREATS relationships belong to:



Outer training

Outer testing

Inner validation



depending on the current experiment.



17\. Evaluation Candidate Files



The files:



results/phase2/eval\_candidates\_fold0.csv

results/phase2/eval\_candidates\_fold1.csv

results/phase2/eval\_candidates\_fold2.csv

results/phase2/eval\_candidates\_fold3.csv

results/phase2/eval\_candidates\_fold4.csv



represent the evaluation cases for the five outer folds.



Each case records information such as:



Compound

True target

Target type

Candidate count



The complete candidate ranking is reconstructed during evaluation.



18\. Training Negative Files



The training negative files are:



results/phase2/train\_hard\_negatives\_fold0.csv

results/phase2/train\_hard\_negatives\_fold1.csv

results/phase2/train\_hard\_negatives\_fold2.csv

results/phase2/train\_hard\_negatives\_fold3.csv

results/phase2/train\_hard\_negatives\_fold4.csv



These contain negative compound-target examples used during GNN training.



Known positive TREATS relationships are excluded from negative sampling.



19\. Outer Test Result Files



The final evaluation outputs are stored as:



results/phase2/outer\_test\_results\_fold0\_alpha075.csv

results/phase2/outer\_test\_results\_fold1\_alpha075.csv

results/phase2/outer\_test\_results\_fold2\_alpha075.csv

results/phase2/outer\_test\_results\_fold3\_alpha000.csv

results/phase2/outer\_test\_results\_fold4\_alpha025.csv



These files contain:



Fold number

Method

Alpha

Number of evaluation cases

MRR

Hits@1

Hits@5

Hits@10

20\. Outer Test Case Files



The corresponding case-level files are:



results/phase2/outer\_test\_cases\_fold0\_alpha075.csv

results/phase2/outer\_test\_cases\_fold1\_alpha075.csv

results/phase2/outer\_test\_cases\_fold2\_alpha075.csv

results/phase2/outer\_test\_cases\_fold3\_alpha000.csv

results/phase2/outer\_test\_cases\_fold4\_alpha025.csv



These provide case-level ranking information used to calculate the aggregate evaluation metrics.



21\. Alpha Validation Files



The inner validation procedure produces:



results/phase2/alpha\_selection\_fold\*.csv

results/phase2/alpha\_validation\_cases\_fold\*.csv

results/phase2/alpha\_validation\_metrics\_fold\*.csv



These files record the validation performance for different alpha values and the selected blend weight.



The outer test set is not used to select alpha.



22\. Final Evaluation Metrics



The final saved fold results produce the following aggregate MRR values:



Method	Mean MRR	Sample SD

Popularity	0.1287	0.0205

GNN	0.1218	0.0118

Popularity + GNN	0.1268	0.0183



The complete evaluation also reports:



Method	Hits@1	Hits@5	Hits@10

Popularity	0.0146	0.2492	0.3190

GNN	0.0556	0.1547	0.2742

Popularity + GNN	0.0391	0.1986	0.3030



These are means across the five outer folds.



23\. Important Interpretation Note



The graph contains substantially more structural information than the number of supervised TREATS labels.



Therefore:



Graph size ≠ supervised dataset size



The graph contains:



13,584 nodes

22,329 edges



while the primary supervised relationship contains:



82 labeled compounds

232 TREATS edges



This distinction is important when interpreting model performance.



The GNN is trained in the context of a large heterogeneous graph, but the direct supervised signal is relatively small.



24\. Data Quality Notes



The source dataset contains ambiguous and contradictory entity information.



Important limitations include:



Inconsistent source terminology

Conflicting source categories

Incomplete herb classification

Sparse Quality/Group coverage

Small number of labeled compounds

Uneven target frequencies

Unresolved ingredient entities



These limitations are retained and documented rather than hidden.



25\. Intended Use of the Data



The graph and derived datasets are intended for:



Graph learning research

Knowledge graph construction

Link prediction experiments

Ayurvedic knowledge discovery

Formulation retrieval research

Explainability experiments



The data and model outputs should not be interpreted as clinical validation or evidence of treatment effectiveness.

