
import sys
import os
import pandas as pd

# Reuse the validated inference implementation
from predict import (
    NODES_PATH,
    EDGES_PATH,
    FOLD_TREATS_PATH,
    CHECKPOINT_DIR,
    build_node_index,
    build_fold_graph,
    load_fold_model,
    score_candidates,
)

from collections import defaultdict
import numpy as np


def predict_all(compound_id, top_k=10):

    print("\n" + "=" * 70)
    print("AYURKOSH-GNN — FULL INFERENCE")
    print("=" * 70)
    print(f"Compound: {compound_id}")

    nodes = pd.read_csv(NODES_PATH)
    edges = pd.read_csv(EDGES_PATH)
    treats_folded = pd.read_csv(FOLD_TREATS_PATH)

    (
        node_id_to_local,
        node_id_to_type,
        _
    ) = build_node_index(nodes)

    if compound_id not in node_id_to_type:
        raise ValueError(
            f"Unknown node ID: {compound_id}"
        )

    if node_id_to_type[compound_id] != "Compound":
        raise ValueError(
            f"{compound_id} is not a Compound node."
        )

    # Human-readable names
    name_map = (
        nodes
        .set_index("node_id")["canonical_name"]
        .to_dict()
    )

    raw_name_map = (
        nodes
        .set_index("node_id")["raw_name"]
        .to_dict()
    )

    # Candidate pools
    disease_ids = [
        node_id
        for node_id, node_type
        in node_id_to_type.items()
        if node_type == "Disease"
    ]

    symptom_ids = [
        node_id
        for node_id, node_type
        in node_id_to_type.items()
        if node_type == "Symptom"
    ]

    # Known TREATS relationships
    known_positive = defaultdict(set)

    treats = edges[
        edges["relation"] == "TREATS"
    ]

    for _, row in treats.iterrows():

        known_positive[
            row["source"]
        ].add(
            row["target"]
        )

    known_targets = known_positive[
        compound_id
    ]

    print(
        f"Known TREATS targets: {len(known_targets)}"
    )

    fold_results = []

    # ========================================================
    # FIVE-FOLD ENSEMBLE
    # ========================================================

    for fold_num in range(5):

        print(
            f"\nLoading fold {fold_num}..."
        )

        (
            fold_node_id_to_local,
            fold_node_id_to_type,
            fold_node_counts,
            edge_index_dict
        ) = build_fold_graph(
            nodes,
            edges,
            treats_folded,
            fold_num
        )

        node_types = list(
            fold_node_counts.keys()
        )

        edge_types = list(
            edge_index_dict.keys()
        )

        model, checkpoint = load_fold_model(
            fold_num,
            node_types,
            fold_node_counts,
            edge_types
        )

        # ----------------------------------------------------
        # Score ALL candidates
        # ----------------------------------------------------

        for target_type, pool in [
            ("Disease", disease_ids),
            ("Symptom", symptom_ids)
        ]:

            candidates = list(pool)

            scores = score_candidates(
                model,
                fold_node_counts,
                edge_index_dict,
                compound_id,
                candidates,
                target_type,
                fold_node_id_to_local
            )

            fold_results.append(
                pd.DataFrame(
                    {
                        "candidate_id": candidates,
                        "target_type": target_type,
                        "score": scores,
                        "fold": fold_num
                    }
                )
            )

    # ========================================================
    # ENSEMBLE
    # ========================================================

    all_scores = pd.concat(
        fold_results,
        ignore_index=True
    )

    ensemble = (
        all_scores
        .groupby(
            [
                "candidate_id",
                "target_type"
            ],
            as_index=False
        )["score"]
        .mean()
        .rename(
            columns={
                "score": "mean_score"
            }
        )
    )

    score_std = (
        all_scores
        .groupby(
            [
                "candidate_id",
                "target_type"
            ]
        )["score"]
        .std()
        .reset_index()
        .rename(
            columns={
                "score": "score_std"
            }
        )
    )

    ensemble = ensemble.merge(
        score_std,
        on=[
            "candidate_id",
            "target_type"
        ],
        how="left"
    )

    ensemble["score_std"] = (
        ensemble["score_std"].fillna(0.0)
    )

    # ========================================================
    # NAMES + STATUS
    # ========================================================

    ensemble["canonical_name"] = (
        ensemble["candidate_id"]
        .map(name_map)
    )

    ensemble["raw_name"] = (
        ensemble["candidate_id"]
        .map(raw_name_map)
    )

    ensemble["status"] = np.where(
        ensemble["candidate_id"].isin(
            known_targets
        ),
        "KNOWN_TREATS",
        "NOVEL_PREDICTION"
    )

    # ========================================================
    # RANK
    # ========================================================

    disease_ranked = (
        ensemble[
            ensemble["target_type"] == "Disease"
        ]
        .sort_values(
            "mean_score",
            ascending=False
        )
        .head(top_k)
        .reset_index(drop=True)
    )

    symptom_ranked = (
        ensemble[
            ensemble["target_type"] == "Symptom"
        ]
        .sort_values(
            "mean_score",
            ascending=False
        )
        .head(top_k)
        .reset_index(drop=True)
    )

    disease_ranked["rank"] = (
        np.arange(len(disease_ranked)) + 1
    )

    symptom_ranked["rank"] = (
        np.arange(len(symptom_ranked)) + 1
    )

    columns = [
        "rank",
        "candidate_id",
        "canonical_name",
        "raw_name",
        "target_type",
        "mean_score",
        "score_std",
        "status"
    ]

    disease_ranked = disease_ranked[
        columns
    ]

    symptom_ranked = symptom_ranked[
        columns
    ]

    # ========================================================
    # DISPLAY
    # ========================================================

    print(
        "\n" + "=" * 70
    )

    print(
        "TOP DISEASE TARGETS"
    )

    print(
        "=" * 70
    )

    print(
        disease_ranked.to_string(
            index=False
        )
    )

    print(
        "\n" + "=" * 70
    )

    print(
        "TOP SYMPTOM TARGETS"
    )

    print(
        "=" * 70
    )

    print(
        symptom_ranked.to_string(
            index=False
        )
    )

    # ========================================================
    # SAVE
    # ========================================================

    output_dir = os.path.join(
        os.path.dirname(
            CHECKPOINT_DIR
        ),
        "predictions"
    )

    os.makedirs(
        output_dir,
        exist_ok=True
    )

    safe_name = (
        str(compound_id)
        .replace("/", "_")
        .replace("\\", "_")
        .replace(" ", "_")
    )

    disease_path = os.path.join(
        output_dir,
        f"{safe_name}_all_disease_predictions.csv"
    )

    symptom_path = os.path.join(
        output_dir,
        f"{safe_name}_all_symptom_predictions.csv"
    )

    full_path = os.path.join(
        output_dir,
        f"{safe_name}_all_targets.csv"
    )

    disease_ranked.to_csv(
        disease_path,
        index=False
    )

    symptom_ranked.to_csv(
        symptom_path,
        index=False
    )

    ensemble.to_csv(
        full_path,
        index=False
    )

    print(
        f"\nSaved:"
        f"\n{disease_path}"
        f"\n{symptom_path}"
        f"\n{full_path}"
    )

    return (
        disease_ranked,
        symptom_ranked,
        ensemble
    )


if __name__ == "__main__":

    if len(sys.argv) < 2:

        print(
            "Usage:"
            " python predict_all.py <compound_id> [top_k]"
        )

        sys.exit(1)

    compound_id = sys.argv[1]

    top_k = (
        int(sys.argv[2])
        if len(sys.argv) > 2
        else 10
    )

    predict_all(
        compound_id,
        top_k
    )
