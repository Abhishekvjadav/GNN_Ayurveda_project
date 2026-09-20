"""
Phase 2 -- Evaluation candidates (filtered ranking) + training negatives

This script builds two separate things:

1. EVALUATION CANDIDATES
   ---------------------
   Standard filtered link-prediction ranking protocol.

   For every held-out Compound -> TREATS -> Disease/Symptom edge,
   the true target is ranked against ALL nodes of the same type.

   Other known TREATS targets of the same compound are filtered out.

   Disease target:
       candidate pool = all Disease nodes

   Symptom target:
       candidate pool = all Symptom nodes

   These candidates are later used identically by the rule-based
   baseline and the GNN.

2. TRAINING GUNA/GROUP-INFORMED NEGATIVES
   --------------------------------------
   Training negatives are generated separately.

   Compound profile:
       Compound
           |
           | CONTAINS
           v
         Herb
           |
           +-- HAS_QUALITY --> Quality / Guna
           |
           +-- BELONGS_TO --> Group

   Disease target profile:
       Disease
           |
           | TREATED_BY
           v
          Herb

   Symptom target profile:
       Symptom
           ^
           | HAS_LAKSHAN
           |
        Disease
           |
           | TREATED_BY
           v
          Herb

   Therefore:

       Symptom -> Disease -> Herb

   is used to construct the herb profile for symptom targets.

   Candidates with LOW Guna/Group Jaccard overlap are selected
   as training negatives.

IMPORTANT:
    Evaluation candidates are NOT sampled hard negatives.
    Evaluation uses the full filtered candidate pool.

Run for each fold:

    python src/phase2/build_eval_and_negatives.py
        <nodes.csv>
        <edges.csv>
        <treats_edges_with_fold.csv>
        <fold_number>

Example:

    python src/phase2/build_eval_and_negatives.py ^
        results/phase1/nodes.csv ^
        results/phase1/edges.csv ^
        results/phase2/treats_edges_with_fold.csv ^
        0
"""

import sys
from collections import defaultdict

import pandas as pd


# ================================================================
# PATHS
# ================================================================

NODES_PATH = (
    sys.argv[1]
    if len(sys.argv) > 1
    else "results/phase1/nodes.csv"
)

EDGES_PATH = (
    sys.argv[2]
    if len(sys.argv) > 2
    else "results/phase1/edges.csv"
)

FOLD_TREATS_PATH = (
    sys.argv[3]
    if len(sys.argv) > 3
    else "results/phase2/treats_edges_with_fold.csv"
)

FOLD_NUM = (
    int(sys.argv[4])
    if len(sys.argv) > 4
    else 0
)

OUTPUT_DIR = "results/phase2"


# ================================================================
# HELPER FUNCTIONS
# ================================================================

def profile(herb_ids, herb_quality, herb_group):
    """
    Build a Guna/Group profile for a set of herbs.

    Profile = union of:
        HAS_QUALITY targets
        BELONGS_TO targets
    """

    prof = set()

    for herb_id in herb_ids:
        prof.update(herb_quality.get(herb_id, set()))
        prof.update(herb_group.get(herb_id, set()))

    return prof


def jaccard(set_a, set_b):
    """
    Jaccard similarity between two sets.
    """

    if not set_a and not set_b:
        return 0.0

    union = set_a | set_b

    if not union:
        return 0.0

    return len(set_a & set_b) / len(union)


# ================================================================
# MAIN
# ================================================================

def main():

    print("=" * 70)
    print("PHASE 2 -- EVALUATION + TRAINING NEGATIVES")
    print("=" * 70)

    print("\nLoading data...")

    nodes = pd.read_csv(NODES_PATH)
    edges = pd.read_csv(EDGES_PATH)
    treats_folded = pd.read_csv(FOLD_TREATS_PATH)

    print(f"Nodes loaded : {len(nodes):,}")
    print(f"Edges loaded : {len(edges):,}")
    print(f"Folded TREATS: {len(treats_folded):,}")

    # ------------------------------------------------------------
    # Node type lookup
    # ------------------------------------------------------------

    node_type = dict(
        zip(
            nodes["node_id"],
            nodes["node_type"]
        )
    )

    # ------------------------------------------------------------
    # Candidate pools
    # ------------------------------------------------------------

    all_disease_ids = (
        nodes.loc[
            nodes["node_type"] == "Disease",
            "node_id"
        ]
        .tolist()
    )

    all_symptom_ids = (
        nodes.loc[
            nodes["node_type"] == "Symptom",
            "node_id"
        ]
        .tolist()
    )

    print("\nCandidate pools:")
    print(f"Disease : {len(all_disease_ids):,}")
    print(f"Symptom : {len(all_symptom_ids):,}")

    # ============================================================
    # PART 1
    # FILTERED EVALUATION CANDIDATES
    # ============================================================

    print("\n" + "=" * 70)
    print(
        f"PART 1 -- FILTERED EVALUATION CANDIDATES "
        f"FOR FOLD {FOLD_NUM}"
    )
    print("=" * 70)

    # ------------------------------------------------------------
    # ALL known TREATS positives
    #
    # Used only for filtering.
    #
    # compound_id -> set(target_ids)
    # ------------------------------------------------------------

    known_positive = defaultdict(set)

    all_treats = edges[
        edges["relation"] == "TREATS"
    ]

    for _, row in all_treats.iterrows():

        compound_id = row["source"]
        target_id = row["target"]

        known_positive[compound_id].add(target_id)

    print(
        f"Total TREATS edges: {len(all_treats):,}"
    )

    # ------------------------------------------------------------
    # Held-out TREATS edges
    # ------------------------------------------------------------

    test_rows = treats_folded[
        treats_folded["fold"] == FOLD_NUM
    ]

    print(
        f"Held-out TREATS edges in fold {FOLD_NUM}: "
        f"{len(test_rows)}"
    )

    eval_rows = []

    # ------------------------------------------------------------
    # Build one evaluation record per held-out positive edge
    # ------------------------------------------------------------

    for _, row in test_rows.iterrows():

        compound_id = row["source"]
        true_target = row["target"]

        target_type = node_type.get(true_target)

        # --------------------------------------------------------
        # Select same-type candidate universe
        # --------------------------------------------------------

        if target_type == "Disease":

            pool = all_disease_ids

        elif target_type == "Symptom":

            pool = all_symptom_ids

        else:

            # Current supervised evaluation is Disease/Symptom.
            # Ignore unexpected target types safely.
            continue

        # --------------------------------------------------------
        # Filter other known positives for this compound.
        #
        # Keep the true target currently being evaluated.
        # --------------------------------------------------------

        candidates = [
            candidate
            for candidate in pool
            if (
                candidate == true_target
                or candidate not in known_positive[compound_id]
            )
        ]

        eval_rows.append(
            {
                "compound_id": compound_id,
                "true_target": true_target,
                "target_type": target_type,
                "n_candidates": len(candidates),
            }
        )

    eval_df = pd.DataFrame(eval_rows)

    eval_output = (
        f"{OUTPUT_DIR}/eval_candidates_fold{FOLD_NUM}.csv"
    )

    eval_df.to_csv(
        eval_output,
        index=False
    )

    print(
        f"\nWrote: {eval_output}"
    )

    print(
        f"Evaluation test cases: {len(eval_df)}"
    )

    if len(eval_df) > 0:

        print(
            "Mean candidate pool size: "
            f"{eval_df['n_candidates'].mean():.1f}"
        )

        print("\nCandidate pool by target type:")

        print(
            eval_df.groupby("target_type")[
                "n_candidates"
            ]
            .agg(
                [
                    "count",
                    "mean",
                    "min",
                    "max"
                ]
            )
            .to_string()
        )

    print(
        "\nEvaluation protocol:"
    )

    print(
        "  Rank true target against all filtered "
        "same-type candidates."
    )

    print(
        "  Disease targets -> Disease candidate pool."
    )

    print(
        "  Symptom targets -> Symptom candidate pool."
    )

    print(
        "  Other known positives for the compound "
        "are filtered."
    )

    # ============================================================
    # PART 2
    # BUILD GUNA/GROUP PROFILES
    # ============================================================

    print("\n" + "=" * 70)
    print("PART 2 -- BUILD TARGET GUNA/GROUP PROFILES")
    print("=" * 70)

    # ------------------------------------------------------------
    # Relevant relations
    # ------------------------------------------------------------

    contains = edges[
        edges["relation"] == "CONTAINS"
    ]

    treated_by = edges[
        edges["relation"] == "TREATED_BY"
    ]

    has_quality = edges[
        edges["relation"] == "HAS_QUALITY"
    ]

    belongs_to = edges[
        edges["relation"] == "BELONGS_TO"
    ]

    has_lakshan = edges[
        edges["relation"] == "HAS_LAKSHAN"
    ]

    # ------------------------------------------------------------
    # Herb -> Quality/Guna
    # ------------------------------------------------------------

    herb_quality = defaultdict(set)

    for _, row in has_quality.iterrows():

        source = row["source"]
        target = row["target"]

        herb_quality[source].add(target)

    # ------------------------------------------------------------
    # Herb -> Group
    # ------------------------------------------------------------

    herb_group = defaultdict(set)

    for _, row in belongs_to.iterrows():

        source = row["source"]
        target = row["target"]

        herb_group[source].add(target)

    # ------------------------------------------------------------
    # Compound -> Herbs
    #
    # Compound --CONTAINS--> Herb
    # ------------------------------------------------------------

    compound_herbs = defaultdict(set)

    for _, row in contains.iterrows():

        compound_id = row["source"]
        ingredient_id = row["target"]

        # Only herbs are used for this Guna/Group profile.
        if node_type.get(ingredient_id) == "Herb":

            compound_herbs[compound_id].add(
                ingredient_id
            )

    # ============================================================
    # TARGET -> HERB PROFILES
    # ============================================================

    target_herbs = defaultdict(set)

    # ------------------------------------------------------------
    # 1. DISEASE -> HERB
    #
    # Disease --TREATED_BY--> Herb
    #
    # IMPORTANT:
    # TREATED_BY direction is:
    #
    # source = Disease
    # target = Herb
    # ------------------------------------------------------------

    for _, row in treated_by.iterrows():

        source_id = row["source"]
        target_id = row["target"]

        source_type = node_type.get(source_id)
        target_type = node_type.get(target_id)

        if (
            source_type == "Disease"
            and target_type == "Herb"
        ):

            target_herbs[source_id].add(
                target_id
            )

    # ------------------------------------------------------------
    # 2. DISEASE -> SYMPTOM
    #
    # Disease --HAS_LAKSHAN--> Symptom
    #
    # disease_symptoms:
    #
    # disease_id -> set(symptom_ids)
    # ------------------------------------------------------------

    disease_symptoms = defaultdict(set)

    for _, row in has_lakshan.iterrows():

        source_id = row["source"]
        target_id = row["target"]

        source_type = node_type.get(source_id)
        target_type = node_type.get(target_id)

        if (
            source_type == "Disease"
            and target_type == "Symptom"
        ):

            disease_symptoms[source_id].add(
                target_id
            )

    # ------------------------------------------------------------
    # 3. SYMPTOM -> HERB
    #
    # Disease -> Symptom
    # Disease -> Herb
    #
    # Therefore:
    #
    # Symptom <- Disease -> Herb
    #
    # For every symptom, take the union of herbs belonging
    # to all diseases associated with that symptom.
    # ------------------------------------------------------------

    for disease_id, symptom_ids in disease_symptoms.items():

        disease_herbs = target_herbs.get(
            disease_id,
            set()
        )

        if not disease_herbs:
            continue

        for symptom_id in symptom_ids:

            target_herbs[symptom_id].update(
                disease_herbs
            )

    # ------------------------------------------------------------
    # Profile statistics
    # ------------------------------------------------------------

    disease_profile_count = 0
    symptom_profile_count = 0

    for disease_id in all_disease_ids:

        if target_herbs.get(disease_id):

            disease_profile_count += 1

    for symptom_id in all_symptom_ids:

        if target_herbs.get(symptom_id):

            symptom_profile_count += 1

    print(
        "\nTarget herb profiles:"
    )

    print(
        f"Diseases with herb profiles : "
        f"{disease_profile_count}/{len(all_disease_ids)}"
    )

    print(
        f"Symptoms with herb profiles : "
        f"{symptom_profile_count}/{len(all_symptom_ids)}"
    )

    # ------------------------------------------------------------
    # Sanity check
    # ------------------------------------------------------------

    if symptom_profile_count == 0:

        print(
            "\nWARNING:"
        )

        print(
            "No symptom herb profiles were constructed."
        )

        print(
            "Check Disease -> HAS_LAKSHAN -> Symptom "
            "and Disease -> TREATED_BY -> Herb relations."
        )

    # ============================================================
    # PART 3
    # TRAINING GUNA/GROUP-INFORMED NEGATIVES
    # ============================================================

    print("\n" + "=" * 70)
    print(
        "PART 3 -- GUNA/GROUP-INFORMED "
        "TRAINING NEGATIVES"
    )
    print("=" * 70)

    # ------------------------------------------------------------
    # Only training TREATS edges for this fold.
    #
    # Held-out compounds are excluded.
    # ------------------------------------------------------------

    train_rows = treats_folded[
        treats_folded["fold"] != FOLD_NUM
    ]

    print(
        f"Training TREATS edges: {len(train_rows)}"
    )

    hard_neg_rows = []

    # ------------------------------------------------------------
    # Process each training positive
    # ------------------------------------------------------------

    for _, row in train_rows.iterrows():

        compound_id = row["source"]
        true_target = row["target"]

        target_type = node_type.get(
            true_target
        )

        # --------------------------------------------------------
        # Compound Guna/Group profile
        # --------------------------------------------------------

        comp_herbs = compound_herbs.get(
            compound_id,
            set()
        )

        comp_profile = profile(
            comp_herbs,
            herb_quality,
            herb_group
        )

        # --------------------------------------------------------
        # Same-type candidate pool
        # --------------------------------------------------------

        if target_type == "Disease":

            pool = all_disease_ids

        elif target_type == "Symptom":

            pool = all_symptom_ids

        else:

            continue

        scored = []

        # --------------------------------------------------------
        # Score possible negatives
        # --------------------------------------------------------

        for candidate in pool:

            # Never use a known positive as a negative.
            if candidate in known_positive[compound_id]:

                continue

            candidate_herbs = target_herbs.get(
                candidate,
                set()
            )

            candidate_profile = profile(
                candidate_herbs,
                herb_quality,
                herb_group
            )

            overlap = jaccard(
                comp_profile,
                candidate_profile
            )

            scored.append(
                (
                    candidate,
                    overlap
                )
            )

        # --------------------------------------------------------
        # Select 5 lowest-overlap candidates.
        #
        # NOTE:
        # These are Guna/Group-informed low-overlap negatives.
        # --------------------------------------------------------

        scored.sort(
            key=lambda x: (
                x[1],
                x[0]
            )
        )

        top_negatives = [
            candidate
            for candidate, _ in scored[:5]
        ]

        # --------------------------------------------------------
        # Save negative pairs
        # --------------------------------------------------------

        for negative in top_negatives:

            hard_neg_rows.append(
                {
                    "compound_id": compound_id,
                    "true_target": true_target,
                    "hard_negative": negative,
                    "target_type": target_type,
                }
            )

    # ------------------------------------------------------------
    # Save training negatives
    # ------------------------------------------------------------

    hard_neg_df = pd.DataFrame(
        hard_neg_rows
    )

    hard_neg_output = (
        f"{OUTPUT_DIR}/"
        f"train_hard_negatives_fold{FOLD_NUM}.csv"
    )

    hard_neg_df.to_csv(
        hard_neg_output,
        index=False
    )

    print(
        f"\nWrote: {hard_neg_output}"
    )

    print(
        f"Training negative rows: "
        f"{len(hard_neg_df)}"
    )

    if len(train_rows) > 0:

        expected = len(train_rows) * 5

        print(
            f"Expected maximum: {expected}"
        )

    # ============================================================
    # FINAL SANITY CHECKS
    # ============================================================

    print("\n" + "=" * 70)
    print("FINAL SANITY CHECKS")
    print("=" * 70)

    # ------------------------------------------------------------
    # Check negative pairs against known positives
    # ------------------------------------------------------------

    accidental_positive_count = 0

    for _, row in hard_neg_df.iterrows():

        compound_id = row["compound_id"]
        negative = row["hard_negative"]

        if negative in known_positive[compound_id]:

            accidental_positive_count += 1

    print(
        "Training negatives that are actually known positives: "
        f"{accidental_positive_count}"
    )

    # ------------------------------------------------------------
    # Evaluation rows
    # ------------------------------------------------------------

    print(
        "Evaluation cases generated: "
        f"{len(eval_df)}"
    )

    # ------------------------------------------------------------
    # Expected test edges
    # ------------------------------------------------------------

    print(
        "Expected held-out TREATS edges: "
        f"{len(test_rows)}"
    )

    if len(eval_df) == len(test_rows):

        print(
            "Evaluation edge count check: PASS"
        )

    else:

        print(
            "Evaluation edge count check: WARNING"
        )

    # ------------------------------------------------------------
    # Negative count
    # ------------------------------------------------------------

    if len(hard_neg_df) == len(train_rows) * 5:

        print(
            "Training negative count check: PASS"
        )

    else:

        print(
            "Training negative count check: WARNING"
        )

    print("\n" + "=" * 70)
    print(
        f"FOLD {FOLD_NUM} COMPLETE"
    )
    print("=" * 70)


# ================================================================
# ENTRY POINT
# ================================================================

if __name__ == "__main__":
    main()