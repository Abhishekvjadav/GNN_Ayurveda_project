import os
import sys
import copy
import numpy as np
import pandas as pd
import torch

ROOT = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        "..",
        ".."
    )
)

PHASE2_DIR = os.path.join(
    ROOT,
    "src",
    "phase2"
)

if PHASE2_DIR not in sys.path:
    sys.path.insert(0, PHASE2_DIR)

import train_rgcn

from build_hetero_data import build_fold_hetero_data


# ============================================================
# CONFIG
# ============================================================

FOLD_NUM = 0

ALPHAS = [
    0.00,
    0.25,
    0.50,
    0.75,
    1.00
]

NODES_PATH = os.path.join(
    ROOT,
    "results",
    "phase1",
    "nodes.csv"
)

EDGES_PATH = os.path.join(
    ROOT,
    "results",
    "phase1",
    "edges.csv"
)

FOLD_TREATS_PATH = os.path.join(
    ROOT,
    "results",
    "phase2",
    "treats_edges_with_fold.csv"
)

TRAIN_NEGATIVES_PATH = os.path.join(
    ROOT,
    "results",
    "phase2",
    f"train_hard_negatives_fold{FOLD_NUM}.csv"
)

RESULTS_DIR = os.path.join(
    ROOT,
    "results",
    "phase2"
)

os.makedirs(
    RESULTS_DIR,
    exist_ok=True
)


# ============================================================
# RANK NORMALIZATION
# ============================================================

def rank_normalize(scores):

    scores = np.asarray(
        scores,
        dtype=float
    )

    order = np.argsort(
        -scores,
        kind="mergesort"
    )

    ranks = np.zeros(
        len(scores),
        dtype=float
    )

    i = 0

    while i < len(scores):

        j = i + 1

        while (
            j < len(scores)
            and scores[order[j]]
            == scores[order[i]]
        ):
            j += 1

        avg_rank = (
            (i + 1) + j
        ) / 2.0

        ranks[
            order[i:j]
        ] = avg_rank

        i = j

    if len(scores) <= 1:
        return np.ones(
            len(scores),
            dtype=float
        )

    return (
        1.0
        -
        (
            (ranks - 1.0)
            /
            (len(scores) - 1.0)
        )
    )


# ============================================================
# TIE-AWARE RANK
# ============================================================

def average_rank(
    true_score,
    scores
):

    greater = sum(
        score > true_score
        for score in scores
    )

    equal = (
        sum(
            score == true_score
            for score in scores
        )
        - 1
    )

    equal = max(
        equal,
        0
    )

    return (
        1
        + greater
        + 0.5 * equal
    )


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 70)
print("PROPER ALPHA VALIDATION -- FOLD 0")
print("=" * 70)

print("\nLoading data...")

nodes = pd.read_csv(
    NODES_PATH
)

edges = pd.read_csv(
    EDGES_PATH
)

treats = pd.read_csv(
    FOLD_TREATS_PATH
)

train_neg_full = pd.read_csv(
    TRAIN_NEGATIVES_PATH
)

print(
    "Nodes loaded        :",
    len(nodes)
)

print(
    "Edges loaded        :",
    len(edges)
)

print(
    "Folded TREATS       :",
    len(treats)
)

print(
    "Training negatives  :",
    len(train_neg_full)
)


# ============================================================
# BUILD OFFICIAL OUTER-FOLD GRAPH
# ============================================================

print(
    "\nBuilding official Fold 0 graph..."
)

(
    data,
    node_id_to_local,
    node_id_to_type,
    train_treats,
    test_treats
) = build_fold_hetero_data(
    NODES_PATH,
    EDGES_PATH,
    FOLD_TREATS_PATH,
    FOLD_NUM,
    embed_dim=train_rgcn.EMBED_DIM
)

print(
    f"\nTraining TREATS edges : "
    f"{len(train_treats)}"
)

print(
    f"Held-out TREATS edges : "
    f"{len(test_treats)}"
)


# ============================================================
# OUTER EDGE INDEX
# ============================================================

outer_edge_index_dict = {
    edge_type:
        data[edge_type].edge_index.clone()
    for edge_type in data.edge_types
}


# ============================================================
# INNER VALIDATION
#
# IMPORTANT:
# This calls the EXISTING official function.
# Therefore the following are exactly the same as the
# official training procedure:
#
# - RandomState(42)
# - 15% inner validation
# - validation TREATS masking
# - weighted BCE
# - POS_WEIGHT = 2.0
# - early stopping
# - best validation epoch
# ============================================================

print(
    "\nRunning official inner validation..."
)

(
    best_epoch,
    best_state,
    best_val_loss
) = train_rgcn.select_best_epoch(
    data,
    {
        node_type:
            data[node_type].num_nodes
        for node_type in data.node_types
    },
    outer_edge_index_dict,
    train_neg_full,
    node_id_to_local
)

print(
    "\nBest inner epoch:",
    best_epoch
)

print(
    "Best inner loss :",
    f"{best_val_loss:.6f}"
)


# ============================================================
# RECREATE THE SAME INNER VALIDATION SPLIT
#
# We reproduce the exact split used by select_best_epoch()
# so that alpha is evaluated on those same validation cases.
# ============================================================

unique_positives = (
    train_neg_full[
        [
            "compound_id",
            "true_target",
            "target_type"
        ]
    ]
    .drop_duplicates()
    .reset_index(drop=True)
)

rng = np.random.RandomState(
    train_rgcn.SEED
)

n_val = max(
    1,
    int(
        train_rgcn.VAL_FRACTION
        *
        len(unique_positives)
    )
)

val_indices = rng.choice(
    len(unique_positives),
    size=n_val,
    replace=False
)

validation = (
    unique_positives
    .iloc[val_indices]
    .reset_index(drop=True)
)

print(
    "\nInner validation cases:",
    len(validation)
)


# ============================================================
# RESTORE BEST INNER MODEL
# ============================================================

model, node_counts = train_rgcn.create_model(
    data
)

model.load_state_dict(
    best_state
)

model.eval()


# ============================================================
# GET INNER-VALIDATION EMBEDDINGS
# ============================================================

with torch.no_grad():

    out = model(
        node_counts,
        outer_edge_index_dict
    )


# ============================================================
# TRAINING-ONLY POPULARITY
# ============================================================

train_treats_df = pd.DataFrame(
    train_treats,
    columns=[
        "source",
        "target"
    ]
) if not isinstance(
    train_treats,
    pd.DataFrame
) else train_treats.copy()

if "target_type" not in train_treats_df.columns:

    train_treats_df["target_type"] = (
        train_treats_df["target"]
        .map(node_id_to_type)
    )

train_treats_df["source"] = (
    train_treats_df["source"]
    .astype(str)
)

train_treats_df["target"] = (
    train_treats_df["target"]
    .astype(str)
)


# ============================================================
# ALL KNOWN POSITIVES FOR FILTERED RANKING
# ============================================================

treats["source"] = (
    treats["source"]
    .astype(str)
)

treats["target"] = (
    treats["target"]
    .astype(str)
)


# ============================================================
# VALIDATION SCORING
# ============================================================

records = []

print(
    "\nScoring inner-validation cases..."
)

for _, row in validation.iterrows():

    compound_id = str(
        row["compound_id"]
    )

    true_target = str(
        row["true_target"]
    )

    target_type = str(
        row["target_type"]
    )

    # --------------------------------------------------------
    # Candidate pool
    # --------------------------------------------------------

    candidates = nodes[
        nodes["node_type"]
        == target_type
    ]["node_id"].astype(
        str
    ).tolist()

    known_positive = set(
        treats[
            treats["source"]
            == compound_id
        ]["target"]
        .astype(str)
    )

    candidates = [
        candidate
        for candidate in candidates
        if (
            candidate == true_target
            or candidate not in known_positive
        )
    ]

    # --------------------------------------------------------
    # Local IDs
    # --------------------------------------------------------

    compound_local = node_id_to_local[
        compound_id
    ]

    target_locals = [
        node_id_to_local[
            candidate
        ]
        for candidate in candidates
    ]

    compound_emb = (
        out["Compound"][
            compound_local
        ]
        .unsqueeze(0)
        .expand(
            len(candidates),
            -1
        )
    )

    target_idx = torch.tensor(
        target_locals,
        dtype=torch.long,
        device=train_rgcn.DEVICE
    )

    target_emb = out[
        target_type
    ][
        target_idx
    ]

    # --------------------------------------------------------
    # GNN DistMult scores
    # --------------------------------------------------------

    with torch.no_grad():

        gnn_scores = (
            model.decoder(
                compound_emb,
                target_emb
            )
            .detach()
            .cpu()
            .numpy()
        )

    # --------------------------------------------------------
    # Training-only popularity
    # --------------------------------------------------------

    popularity_counts = (
        train_treats_df[
            train_treats_df["target"]
            .isin(candidates)
        ]["target"]
        .value_counts()
        .to_dict()
    )

    popularity_scores = np.asarray(
        [
            float(
                popularity_counts.get(
                    candidate,
                    0
                )
            )
            for candidate in candidates
        ],
        dtype=float
    )

    # --------------------------------------------------------
    # Rank normalize
    # --------------------------------------------------------

    gnn_norm = rank_normalize(
        gnn_scores
    )

    popularity_norm = rank_normalize(
        popularity_scores
    )

    true_index = candidates.index(
        true_target
    )

    # --------------------------------------------------------
    # Alpha sweep
    # --------------------------------------------------------

    for alpha in ALPHAS:

        combined_scores = (
            (
                1.0 - alpha
            )
            *
            popularity_norm
            +
            alpha
            *
            gnn_norm
        )

        true_score = (
            combined_scores[
                true_index
            ]
        )

        rank = average_rank(
            true_score,
            list(combined_scores)
        )

        records.append(
            {
                "compound_id":
                    compound_id,

                "true_target":
                    true_target,

                "target_type":
                    target_type,

                "alpha":
                    alpha,

                "rank":
                    rank,

                "n_candidates":
                    len(candidates)
            }
        )


# ============================================================
# METRICS
# ============================================================

cases = pd.DataFrame(
    records
)

metric_rows = []

for alpha in ALPHAS:

    subset = cases[
        cases["alpha"]
        == alpha
    ]

    ranks = (
        subset["rank"]
        .to_numpy(
            dtype=float
        )
    )

    metric_rows.append(
        {
            "fold":
                FOLD_NUM,

            "alpha":
                alpha,

            "n_cases":
                len(subset),

            "MRR":
                np.mean(
                    1.0 / ranks
                ),

            "Hits@1":
                np.mean(
                    ranks <= 1
                ),

            "Hits@5":
                np.mean(
                    ranks <= 5
                ),

            "Hits@10":
                np.mean(
                    ranks <= 10
                )
        }
    )

metrics = pd.DataFrame(
    metric_rows
)


# ============================================================
# SELECT ALPHA
# ============================================================

selected = (
    metrics
    .sort_values(
        [
            "MRR",
            "Hits@10",
            "Hits@5",
            "Hits@1"
        ],
        ascending=False
    )
    .iloc[0]
)

selected_alpha = float(
    selected["alpha"]
)


# ============================================================
# DISPLAY
# ============================================================

print(
    "\n"
    + "=" * 70
)

print(
    "INNER VALIDATION ALPHA RESULTS"
)

print(
    "=" * 70
)

print(
    metrics.to_string(
        index=False,
        float_format=lambda x:
            f"{x:.4f}"
    )
)

print(
    "\nSELECTED ALPHA:",
    f"{selected_alpha:.2f}"
)


# ============================================================
# SAVE
# ============================================================

metrics_path = os.path.join(
    RESULTS_DIR,
    f"alpha_validation_metrics_fold{FOLD_NUM}.csv"
)

cases_path = os.path.join(
    RESULTS_DIR,
    f"alpha_validation_cases_fold{FOLD_NUM}.csv"
)

selection_path = os.path.join(
    RESULTS_DIR,
    f"alpha_selection_fold{FOLD_NUM}.csv"
)

metrics.to_csv(
    metrics_path,
    index=False
)

cases.to_csv(
    cases_path,
    index=False
)

pd.DataFrame(
    [
        {
            "fold":
                FOLD_NUM,

            "selected_alpha":
                selected_alpha,

            "selection_basis":
                "inner_validation_MRR",

            "best_epoch":
                best_epoch,

            "best_validation_loss":
                best_val_loss,

            "n_validation_cases":
                len(validation)
        }
    ]
).to_csv(
    selection_path,
    index=False
)

print(
    "\nSaved:"
)

print(
    metrics_path
)

print(
    cases_path
)

print(
    selection_path
)

print(
    "\nFOLD 0 ALPHA VALIDATION COMPLETE"
)