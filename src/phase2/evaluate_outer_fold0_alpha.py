import os
import sys
import pandas as pd
import numpy as np
import torch
from collections import defaultdict

# ============================================================
# PATH SETUP
# ============================================================

ROOT = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        "..",
        ".."
    )
)

sys.path.insert(0, os.path.join(ROOT, "src", "phase2"))

import train_rgcn
import build_hetero_data

# ============================================================
# CONFIG
# ============================================================

FOLD_NUM = int(sys.argv[1]) if len(sys.argv) > 1 else 0

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

EVAL_CANDIDATES_PATH = os.path.join(
    ROOT,
    "results",
    "phase2",
    f"eval_candidates_fold{FOLD_NUM}.csv"
)

ALPHA_SELECTION_PATH = os.path.join(
    ROOT,
    "results",
    "phase2",
    f"alpha_selection_fold{FOLD_NUM}.csv"
)

OUTPUT_RESULTS_PATH = None
OUTPUT_CASES_PATH = None

# ============================================================
# HELPERS
# ============================================================

def average_rank_tie_aware(
    true_score,
    candidate_scores
):
    better = sum(
        score > true_score
        for score in candidate_scores
    )

    equal = sum(
        score == true_score
        for score in candidate_scores
    )

    return (
        1
        + better
        + 0.5 * (equal - 1)
    )


def rank_metrics(ranks):
    ranks = np.asarray(
        ranks,
        dtype=float
    )

    return {
        "MRR": float(
            np.mean(1.0 / ranks)
        ),

        "Hits@1": float(
            np.mean(ranks <= 1)
        ),

        "Hits@5": float(
            np.mean(ranks <= 5)
        ),

        "Hits@10": float(
            np.mean(ranks <= 10)
        )
    }


def rank_normalize(scores):
    """
    Convert scores to [0,1] using descending
    tie-aware average rank.

    Highest score -> 1
    Lowest score -> 0
    Equal scores receive the same normalized value.
    """

    scores = np.asarray(
        scores,
        dtype=float
    )

    n = len(scores)

    if n <= 1:
        return np.ones(n)

    # Descending unique score levels
    unique_scores = np.unique(scores)[::-1]

    normalized = np.zeros(n, dtype=float)

    if len(unique_scores) == 1:
        return np.full(n, 0.5)

    for score in unique_scores:

        indices = np.where(scores == score)[0]

        # Average 1-based rank for this tied group
        better_count = np.sum(
            scores > score
        )

        tie_count = len(indices)

        average_rank = (
            1.0
            + better_count
            + (tie_count - 1) / 2.0
        )

        # Convert rank to [0,1]
        normalized_value = (
            1.0
            - (average_rank - 1.0)
            / (n - 1.0)
        )

        normalized[indices] = normalized_value

    return normalized
# ============================================================
# MAIN
# ============================================================

print("=" * 70)
print(
    f"OUTER TEST EVALUATION -- FOLD {FOLD_NUM}"
)
print("=" * 70)


# ============================================================
# LOAD DATA
# ============================================================

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

eval_candidates = pd.read_csv(
    EVAL_CANDIDATES_PATH
)

alpha_selection = pd.read_csv(
    ALPHA_SELECTION_PATH
)

print(
    "Nodes loaded          :",
    len(nodes)
)

print(
    "Edges loaded          :",
    len(edges)
)

print(
    "Folded TREATS         :",
    len(treats)
)

print(
    "Training negatives    :",
    len(train_neg_full)
)

print(
    "Evaluation candidates :",
    len(eval_candidates)
)


# ============================================================
# VERIFY SELECTED ALPHA
# ============================================================

saved_alpha = float(
    alpha_selection.iloc[0]["selected_alpha"]
)

print(
    "\nAlpha selection file:",
    saved_alpha
)
SELECTED_ALPHA = saved_alpha

print(
    "\nSelected alpha:",
    SELECTED_ALPHA
)
OUTPUT_RESULTS_PATH = os.path.join(
    ROOT,
    "results",
    "phase2",
    f"outer_test_results_fold{FOLD_NUM}_alpha{int(SELECTED_ALPHA * 100):03d}.csv"
)

OUTPUT_CASES_PATH = os.path.join(
    ROOT,
    "results",
    "phase2",
    f"outer_test_cases_fold{FOLD_NUM}_alpha{int(SELECTED_ALPHA * 100):03d}.csv"
)

# ============================================================
# BUILD OFFICIAL OUTER GRAPH
# ============================================================

print(
    f"\nBuilding official Fold {FOLD_NUM} graph..."
)

(
    data,
    node_id_to_local,
    node_id_to_type,
    train_treats,
    test_treats
) = build_hetero_data.build_fold_hetero_data(
    NODES_PATH,
    EDGES_PATH,
    FOLD_TREATS_PATH,
    FOLD_NUM,
    embed_dim=train_rgcn.EMBED_DIM
)

outer_edge_index_dict = {
    edge_type:
        data[edge_type].edge_index.clone()
    for edge_type in data.edge_types
}

print(
    "\nOuter training TREATS :",
    len(train_treats)
)

print(
    "Outer test TREATS     :",
    len(test_treats)
)


# ============================================================
# NORMALIZE IDs
# ============================================================

treats["source"] = (
    treats["source"].astype(str)
)

treats["target"] = (
    treats["target"].astype(str)
)

eval_candidates[
    "compound_id"
] = (
    eval_candidates[
        "compound_id"
    ].astype(str)
)

eval_candidates[
    "true_target"
] = (
    eval_candidates[
        "true_target"
    ].astype(str)
)

eval_candidates[
    "target_type"
] = (
    eval_candidates[
        "target_type"
    ].astype(str)
)


# ============================================================
# VERIFY OUTER TEST CASES
# ============================================================

test_pairs = set(
    (
        str(row["source"]),
        str(row["target"])
    )
    for _, row
    in test_treats.iterrows()
)

candidate_pairs = set(
    (
        str(row["compound_id"]),
        str(row["true_target"])
    )
    for _, row
    in eval_candidates.iterrows()
)

missing_test_pairs = (
    test_pairs - candidate_pairs
)

if missing_test_pairs:

    print(
        "\nWARNING: Some test pairs are missing:"
    )

    for pair in sorted(
        missing_test_pairs
    ):
        print(pair)

    raise ValueError(
        "Evaluation candidate file does not "
        "cover all outer test pairs."
    )

print(
    "\nVerified outer test cases:",
    len(candidate_pairs)
)


# ============================================================
# POPULARITY SOURCE
# ============================================================

train_treats_df = treats[
    treats["fold"] != FOLD_NUM
].copy()

print(
    "\nPopularity source edges:",
    len(train_treats_df)
)

if len(train_treats_df) != len(
    train_treats
):
    raise ValueError(
        "Popularity source does not match "
        "outer training TREATS count."
    )


# ============================================================
# POPULARITY COUNTS
# ============================================================

popularity_counts = (
    train_treats_df[
        "target"
    ]
    .value_counts()
    .to_dict()
)


# ============================================================
# KNOWN POSITIVE TARGETS
# ============================================================

known_positive = defaultdict(set)

for _, row in treats.iterrows():

    known_positive[
        str(row["source"])
    ].add(
        str(row["target"])
    )


# ============================================================
# OFFICIAL INNER VALIDATION
# ============================================================

print(
    "\nRunning official inner validation "
    "to determine final training epoch..."
)

node_counts = {
    node_type:
        data[node_type].num_nodes
    for node_type
    in data.node_types
}

(
    best_epoch,
    _,
    best_val_loss
) = train_rgcn.select_best_epoch(
    data,
    node_counts,
    outer_edge_index_dict,
    train_neg_full,
    node_id_to_local
)

print(
    "\nSelected best epoch:",
    best_epoch
)

print(
    "Best validation loss:",
    f"{best_val_loss:.6f}"
)


# ============================================================
# FINAL RETRAIN
# ============================================================

print(
    f"\nRetraining final Fold {FOLD_NUM} model "
    "on all outer-training positives..."
)

final_model = train_rgcn.final_retrain(
    data,
    node_counts,
    outer_edge_index_dict,
    train_neg_full,
    best_epoch,
    node_id_to_local
)


# ============================================================
# FINAL MODEL FORWARD PASS
# ============================================================

final_model.eval()

with torch.no_grad():

    out = final_model(
        node_counts,
        outer_edge_index_dict
    )


# ============================================================
# CANDIDATE POOLS
# ============================================================

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


# ============================================================
# SCORE OUTER TEST CASES
# ============================================================

print(
    "\nScoring untouched outer test cases..."
)

records = []

for _, row in eval_candidates.iterrows():

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
    # OFFICIAL CANDIDATE POOL
    # --------------------------------------------------------

    if target_type == "Disease":

        pool = disease_ids

    elif target_type == "Symptom":

        pool = symptom_ids

    else:

        raise ValueError(
            f"Unknown target type: {target_type}"
        )


    candidates = [
        candidate
        for candidate in pool
        if (
            candidate == true_target
            or candidate not in known_positive[
                compound_id
            ]
        )
    ]


    if true_target not in candidates:

        raise ValueError(
            "True target missing from candidate pool: "
            + true_target
        )


    # --------------------------------------------------------
    # LOCAL INDICES
    # --------------------------------------------------------

    compound_local = (
        train_rgcn.node_local_idx(
            compound_id,
            node_id_to_local
        )
    )

    candidate_local = torch.tensor(
        [
            train_rgcn.node_local_idx(
                candidate,
                node_id_to_local
            )
            for candidate in candidates
        ],
        dtype=torch.long,
        device=train_rgcn.DEVICE
    )


    # --------------------------------------------------------
    # GNN SCORES
    # --------------------------------------------------------

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

    candidate_emb = (
        out[target_type][
            candidate_local
        ]
    )

    with torch.no_grad():

        gnn_scores = (
            final_model.decoder(
                compound_emb,
                candidate_emb
            )
            .detach()
            .cpu()
            .numpy()
        )


    # --------------------------------------------------------
    # POPULARITY SCORES
    # --------------------------------------------------------

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
    # NORMALIZE RANK SIGNALS
    # --------------------------------------------------------

    popularity_norm = rank_normalize(
        popularity_scores
    )

    gnn_norm = rank_normalize(
        gnn_scores
    )


    # --------------------------------------------------------
    # COMBINED SCORE
    # --------------------------------------------------------

    combined_scores = (
        (1.0 - SELECTED_ALPHA)
        * popularity_norm
        +
        SELECTED_ALPHA
        * gnn_norm
    )


    # --------------------------------------------------------
    # TRUE TARGET INDEX
    # --------------------------------------------------------

    true_index = candidates.index(
        true_target
    )


    # --------------------------------------------------------
    # RANKS
    # --------------------------------------------------------

    popularity_rank = (
        average_rank_tie_aware(
            popularity_scores[
                true_index
            ],
            list(popularity_scores)
        )
    )

    gnn_rank = (
        average_rank_tie_aware(
            gnn_scores[
                true_index
            ],
            list(gnn_scores)
        )
    )

    combined_rank = (
        average_rank_tie_aware(
            combined_scores[
                true_index
            ],
            list(combined_scores)
        )
    )


    records.append(
        {
            "compound_id":
                compound_id,

            "true_target":
                true_target,

            "target_type":
                target_type,

            "n_candidates":
                len(candidates),

            "popularity_rank":
                popularity_rank,

            "gnn_rank":
                gnn_rank,

            "combined_rank":
                combined_rank,

            "popularity_mrr":
                1.0 / popularity_rank,

            "gnn_mrr":
                1.0 / gnn_rank,

            "combined_mrr":
                1.0 / combined_rank,

            "popularity_hit_at_1":
                int(
                    popularity_rank <= 1
                ),

            "popularity_hit_at_5":
                int(
                    popularity_rank <= 5
                ),

            "popularity_hit_at_10":
                int(
                    popularity_rank <= 10
                ),

            "gnn_hit_at_1":
                int(
                    gnn_rank <= 1
                ),

            "gnn_hit_at_5":
                int(
                    gnn_rank <= 5
                ),

            "gnn_hit_at_10":
                int(
                    gnn_rank <= 10
                ),

            "combined_hit_at_1":
                int(
                    combined_rank <= 1
                ),

            "combined_hit_at_5":
                int(
                    combined_rank <= 5
                ),

            "combined_hit_at_10":
                int(
                    combined_rank <= 10
                )
        }
    )


# ============================================================
# RESULTS DATAFRAME
# ============================================================

results_df = pd.DataFrame(
    records
)


# ============================================================
# AGGREGATE METRICS
# ============================================================

popularity_metrics = rank_metrics(
    results_df[
        "popularity_rank"
    ].values
)

gnn_metrics = rank_metrics(
    results_df[
        "gnn_rank"
    ].values
)

combined_metrics = rank_metrics(
    results_df[
        "combined_rank"
    ].values
)


# ============================================================
# PRINT RESULTS
# ============================================================

print("\n")
print("=" * 70)
print(f"FOLD {FOLD_NUM} OUTER TEST RESULTS")
print("=" * 70)

print(
    f"\nNumber of outer test cases: "
    f"{len(results_df)}"
)

print(
    f"Selected alpha: "
    f"{SELECTED_ALPHA:.2f}"
)

print(
    "\nPopularity baseline:"
)

print(
    f"  MRR     : "
    f"{popularity_metrics['MRR']:.4f}"
)

print(
    f"  Hits@1  : "
    f"{popularity_metrics['Hits@1']:.4f}"
)

print(
    f"  Hits@5  : "
    f"{popularity_metrics['Hits@5']:.4f}"
)

print(
    f"  Hits@10 : "
    f"{popularity_metrics['Hits@10']:.4f}"
)


print(
    "\nGNN:"
)

print(
    f"  MRR     : "
    f"{gnn_metrics['MRR']:.4f}"
)

print(
    f"  Hits@1  : "
    f"{gnn_metrics['Hits@1']:.4f}"
)

print(
    f"  Hits@5  : "
    f"{gnn_metrics['Hits@5']:.4f}"
)

print(
    f"  Hits@10 : "
    f"{gnn_metrics['Hits@10']:.4f}"
)


print(
    f"\nPopularity + GNN "
    f"(alpha={SELECTED_ALPHA:.2f}):"
)

print(
    f"  MRR     : "
    f"{combined_metrics['MRR']:.4f}"
)

print(
    f"  Hits@1  : "
    f"{combined_metrics['Hits@1']:.4f}"
)

print(
    f"  Hits@5  : "
    f"{combined_metrics['Hits@5']:.4f}"
)

print(
    f"  Hits@10 : "
    f"{combined_metrics['Hits@10']:.4f}"
)


# ============================================================
# SAVE RESULTS
# ============================================================

results_df.to_csv(
    OUTPUT_CASES_PATH,
    index=False
)


summary_df = pd.DataFrame(
    [
        {
            "fold": FOLD_NUM,
            "method": "Popularity",
            "alpha": 0.0,
            "n_cases": len(results_df),
            **popularity_metrics
        },

        {
            "fold": FOLD_NUM,
            "method": "GNN",
            "alpha": 1.0,
            "n_cases": len(results_df),
            **gnn_metrics
        },

        {
            "fold": FOLD_NUM,
            "method":
                "Popularity+GNN",
            "alpha":
                SELECTED_ALPHA,
            "n_cases":
                len(results_df),
            **combined_metrics
        }
    ]
)

summary_df.to_csv(
    OUTPUT_RESULTS_PATH,
    index=False
)


# ============================================================
# FINAL
# ============================================================

print("\nSaved case-level results:")
print(
    OUTPUT_CASES_PATH
)

print(
    "\nSaved summary results:"
)

print(
    OUTPUT_RESULTS_PATH
)

print("\nEvaluation complete.")
print("=" * 70)