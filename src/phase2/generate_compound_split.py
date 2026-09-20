"""
Phase 2 -- Compound-held-out split generator

Design (agreed in project discussion, correcting the earlier
"random edge split" default):

  - The unit being held out is the COMPOUND, not the edge. This tests
    whether the model generalizes to formulations it has never seen
    any TREATS label for -- a much harder and more honest test than
    holding out random edges from compounds the model has partially
    seen, given only 82-83 labeled compounds total.

  - Only TREATS edges are masked per fold. A held-out compound's
    CONTAINS edges (its herb composition) remain visible during
    training -- that's input structure describing what the compound
    IS, not the label being predicted. Only "what does this compound
    treat" is hidden for held-out compounds.

  - 5-fold CV, not a single train/test split, because 82 compounds
    is too small for a single held-out set to give a stable estimate.

  - Folds are stratified by TREATS-degree bucket (how many
    disease/symptom targets each compound has: 1, 2-3, 4+) so no
    fold is accidentally all "easy" (many treats-targets) or all
    "hard" (single treats-target) compounds -- degree ranges from
    1 to 19 in this dataset, heavily skewed toward 1-3.

Output: compound_fold_assignment.csv (compound_node_id -> fold 0-4),
and treats_edges_with_fold.csv (every TREATS edge tagged with which
fold its source compound belongs to, so train/test edges for any
given fold are a one-line filter).
"""

import sys
import numpy as np
import pandas as pd

EDGES_PATH = sys.argv[1] if len(sys.argv) > 1 else "edges.csv"
N_FOLDS = 5
SEED = 42

np.random.seed(SEED)


def degree_bucket(n):
    if n == 1:
        return "1"
    elif n <= 3:
        return "2-3"
    else:
        return "4+"


def main():
    edges = pd.read_csv(EDGES_PATH)
    treats = edges[edges["relation"] == "TREATS"].copy()

    compounds = treats["source"].unique()
    print(f"Compounds with TREATS edges: {len(compounds)}")
    print(f"Total TREATS edges: {len(treats)}")

    degree = treats.groupby("source").size()
    bucket = degree.map(degree_bucket)

    print("\nDegree bucket distribution:")
    print(bucket.value_counts())

    # Stratified fold assignment: shuffle within each bucket, then
    # round-robin assign fold numbers so each fold gets a proportional
    # share of easy/medium/hard compounds.
    fold_assignment = {}
    for b, group in bucket.groupby(bucket):
        members = list(group.index)
        rng = np.random.RandomState(SEED)
        rng.shuffle(members)
        for i, compound in enumerate(members):
            fold_assignment[compound] = i % N_FOLDS

    fold_df = pd.DataFrame([
        {"compound_node_id": c, "treats_degree": int(degree[c]),
         "degree_bucket": bucket[c], "fold": fold_assignment[c]}
        for c in compounds
    ]).sort_values(["fold", "degree_bucket"]).reset_index(drop=True)

    print("\nFold sizes (compounds):")
    print(fold_df["fold"].value_counts().sort_index())

    print("\nFold x degree-bucket cross-tab (check for balance):")
    print(pd.crosstab(fold_df["fold"], fold_df["degree_bucket"]))

    treats["fold"] = treats["source"].map(fold_assignment)

    print("\nFold sizes (TREATS edges -- test-set size per fold):")
    print(treats["fold"].value_counts().sort_index())

    fold_df.to_csv("compound_fold_assignment.csv", index=False)
    treats.to_csv("treats_edges_with_fold.csv", index=False)

    print("\nWrote compound_fold_assignment.csv")
    print("Wrote treats_edges_with_fold.csv")

    print("\n" + "=" * 60)
    print("HOW TO USE THIS FOR FOLD i:")
    print("=" * 60)
    print("""
  test_treats  = treats_edges_with_fold[fold == i]
  train_treats = treats_edges_with_fold[fold != i]

  # Training graph for this fold:
  #   ALL edges of every OTHER relation type (TREATED_BY, HAS_LAKSHAN,
  #   ALTERNATE_OF, HAS_QUALITY, BELONGS_TO, HAS_KALP, CONTAINS)
  #   PLUS train_treats
  #   MINUS test_treats (these must not appear anywhere in training)

  # Held-out compounds' CONTAINS edges stay in the training graph --
  # only their TREATS edges are removed.
""")


if __name__ == "__main__":
    main()
