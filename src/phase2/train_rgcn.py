
"""
AyurKOSH Phase 2
Heterogeneous GraphSAGE/R-GCN-style encoder + DistMult decoder

Training objective:
    Weighted Binary Cross-Entropy with Logits

Prediction task:
    Compound --TREATS--> Disease
    Compound --TREATS--> Symptom

Evaluation:
    Compound-held-out 5-fold filtered ranking
    MRR
    Hits@1
    Hits@5
    Hits@10

Methodology:
    1. Outer test compounds are completely held out.
    2. Inner validation TREATS edges are removed from message passing.
    3. Best epoch is selected on clean validation data.
    4. Fresh model is retrained using all outer-training data.
    5. Outer held-out compounds are evaluated only after final training.
"""

import sys
import copy
import random
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn import HeteroConv, SAGEConv


# ============================================================
# PATHS
# ============================================================

NODES_PATH = (
    sys.argv[1]
    if len(sys.argv) > 1
    else "/content/GNN_Ayurveda_project/results/phase1/nodes.csv"
)

EDGES_PATH = (
    sys.argv[2]
    if len(sys.argv) > 2
    else "/content/GNN_Ayurveda_project/results/phase1/edges.csv"
)

FOLD_TREATS_PATH = (
    sys.argv[3]
    if len(sys.argv) > 3
    else "/content/GNN_Ayurveda_project/results/phase2/treats_edges_with_fold.csv"
)

FOLD_NUM = (
    int(sys.argv[4])
    if len(sys.argv) > 4
    else 0
)

EVAL_CANDIDATES_PATH = (
    sys.argv[5]
    if len(sys.argv) > 5
    else f"/content/GNN_Ayurveda_project/results/phase2/eval_candidates_fold{FOLD_NUM}.csv"
)

TRAIN_NEGATIVES_PATH = (
    sys.argv[6]
    if len(sys.argv) > 6
    else f"/content/GNN_Ayurveda_project/results/phase2/train_hard_negatives_fold{FOLD_NUM}.csv"
)


# ============================================================
# HYPERPARAMETERS
# ============================================================

EMBED_DIM = 64
HIDDEN_DIM = 64

N_EPOCHS = 150

LR = 0.005
WEIGHT_DECAY = 1e-4

PATIENCE = 20
VAL_FRACTION = 0.15

SEED = 42

# Keep one consistent device for the official experiment.
DEVICE = torch.device("cpu")


# ============================================================
# REPRODUCIBILITY
# ============================================================

def set_seed(seed=42):

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


set_seed(SEED)


# ============================================================
# HETEROGENEOUS ENCODER
# ============================================================

class HeteroEncoder(nn.Module):

    def __init__(
        self,
        edge_types,
        hidden_dim
    ):

        super().__init__()

        self.conv1 = HeteroConv(
            {
                edge_type:
                    SAGEConv(
                        (-1, -1),
                        hidden_dim
                    )
                for edge_type in edge_types
            },
            aggr="mean"
        )

        self.conv2 = HeteroConv(
            {
                edge_type:
                    SAGEConv(
                        (-1, -1),
                        hidden_dim
                    )
                for edge_type in edge_types
            },
            aggr="mean"
        )

    def forward(
        self,
        x_dict,
        edge_index_dict
    ):

        x_dict = self.conv1(
            x_dict,
            edge_index_dict
        )

        x_dict = {
            node_type:
                F.relu(x)
            for node_type, x in x_dict.items()
        }

        x_dict = self.conv2(
            x_dict,
            edge_index_dict
        )

        return x_dict


# ============================================================
# DISTMULT DECODER
# ============================================================

class DistMultDecoder(nn.Module):

    def __init__(
        self,
        dim
    ):

        super().__init__()

        self.r = nn.Parameter(
            torch.randn(dim) * 0.1
        )

    def forward(
        self,
        h,
        t
    ):

        return (
            h *
            self.r *
            t
        ).sum(dim=-1)


# ============================================================
# COMPLETE MODEL
# ============================================================

class HeteroRGCNModel(nn.Module):

    def __init__(
        self,
        node_types,
        node_counts,
        edge_types,
        embed_dim=64,
        hidden_dim=64
    ):

        super().__init__()

        # ----------------------------------------------------
        # ONE SHARED COMPOUND TYPE EMBEDDING
        # ----------------------------------------------------

        self.shared_compound_embedding = nn.Parameter(
            torch.randn(
                1,
                embed_dim
            ) * 0.1
        )

        # ----------------------------------------------------
        # TRAINABLE EMBEDDINGS FOR OTHER NODE TYPES
        # ----------------------------------------------------

        self.node_embeddings = nn.ParameterDict()

        for node_type in node_types:

            if node_type == "Compound":
                continue

            self.node_embeddings[node_type] = nn.Parameter(
                torch.randn(
                    node_counts[node_type],
                    embed_dim
                ) * 0.1
            )

        # ----------------------------------------------------
        # HETEROGENEOUS GNN
        # ----------------------------------------------------

        self.encoder = HeteroEncoder(
            edge_types,
            hidden_dim
        )

        # ----------------------------------------------------
        # DISTMULT
        # ----------------------------------------------------

        self.decoder = DistMultDecoder(
            hidden_dim
        )

    def get_initial_x_dict(
        self,
        node_counts
    ):

        x_dict = {}

        # Every Compound initially receives
        # the SAME shared type-level vector.
        x_dict["Compound"] = (
            self.shared_compound_embedding
            .expand(
                node_counts["Compound"],
                -1
            )
        )

        for node_type, embedding in self.node_embeddings.items():

            x_dict[node_type] = embedding

        return x_dict

    def forward(
        self,
        node_counts,
        edge_index_dict
    ):

        x_dict = self.get_initial_x_dict(
            node_counts
        )

        return self.encoder(
            x_dict,
            edge_index_dict
        )


# ============================================================
# NODE INDEX
# ============================================================

def node_local_idx(
    node_id,
    node_id_to_local
):

    return node_id_to_local[node_id]


# ============================================================
# DATAFRAME -> TENSORS
# ============================================================

def dataframe_to_tensors(
    df,
    node_id_to_local
):

    compound = torch.tensor(
        [
            node_local_idx(
                c,
                node_id_to_local
            )
            for c in df["compound_id"]
        ],
        dtype=torch.long,
        device=DEVICE
    )

    positive_target = torch.tensor(
        [
            node_local_idx(
                t,
                node_id_to_local
            )
            for t in df["true_target"]
        ],
        dtype=torch.long,
        device=DEVICE
    )

    negative_target = torch.tensor(
        [
            node_local_idx(
                t,
                node_id_to_local
            )
            for t in df["hard_negative"]
        ],
        dtype=torch.long,
        device=DEVICE
    )

    target_types = (
        df["target_type"]
        .tolist()
    )

    return (
        compound,
        positive_target,
        negative_target,
        target_types
    )


# ============================================================
# BCE TRAINING LOSS
# ============================================================

def compute_bce_loss(
    out,
    compound_idx,
    positive_idx,
    negative_idx,
    target_types,
    decoder,
    pos_weight
):

    compound_emb = (
        out["Compound"][
            compound_idx
        ]
    )

    all_losses = []

    for target_type in set(target_types):

        mask = torch.tensor(
            [
                t == target_type
                for t in target_types
            ],
            dtype=torch.bool,
            device=DEVICE
        )

        if mask.sum().item() == 0:
            continue

        compound_batch = (
            compound_emb[mask]
        )

        positive_emb = (
            out[target_type][
                positive_idx[mask]
            ]
        )

        negative_emb = (
            out[target_type][
                negative_idx[mask]
            ]
        )

        # ----------------------------------------------------
        # Positive examples
        # ----------------------------------------------------

        positive_scores = decoder(
            compound_batch,
            positive_emb
        )

        positive_labels = torch.ones_like(
            positive_scores
        )

        positive_loss = F.binary_cross_entropy_with_logits(
            positive_scores,
            positive_labels
        )

        # ----------------------------------------------------
        # Negative examples
        # ----------------------------------------------------

        negative_scores = decoder(
            compound_batch,
            negative_emb
        )

        negative_labels = torch.zeros_like(
            negative_scores
        )

        negative_loss = F.binary_cross_entropy_with_logits(
            negative_scores,
            negative_labels
        )

        # ----------------------------------------------------
        # Combine
        # ----------------------------------------------------

        type_loss = (
            pos_weight * positive_loss
            + negative_loss
        ) / (
            pos_weight + 1.0
        )

        all_losses.append(
            type_loss
        )

    if len(all_losses) == 0:

        return torch.tensor(
            0.0,
            device=DEVICE,
            requires_grad=True
        )

    return torch.stack(
        all_losses
    ).mean()


# ============================================================
# REMOVE VALIDATION TREATS EDGES
# ============================================================

def mask_validation_treats_edges(
    edge_index_dict,
    val_pairs,
    node_id_to_local
):

    masked = {}

    # Convert validation pairs into local IDs.
    val_local_pairs = set()

    for (
        compound_id,
        target_id,
        target_type
    ) in val_pairs:

        compound_local = node_id_to_local[
            compound_id
        ]

        target_local = node_id_to_local[
            target_id
        ]

        val_local_pairs.add(
            (
                compound_local,
                target_local,
                target_type
            )
        )

    removed_total = 0

    for edge_type, edge_index in edge_index_dict.items():

        src_type, relation, dst_type = edge_type

        # Only TREATS needs masking.
        if relation != "TREATS":

            masked[edge_type] = (
                edge_index.clone()
            )

            continue

        keep = torch.ones(
            edge_index.shape[1],
            dtype=torch.bool,
            device=edge_index.device
        )

        for i in range(
            edge_index.shape[1]
        ):

            src_local = int(
                edge_index[0, i]
            )

            dst_local = int(
                edge_index[1, i]
            )

            if (
                src_local,
                dst_local,
                dst_type
            ) in val_local_pairs:

                keep[i] = False
                removed_total += 1

        masked[edge_type] = (
            edge_index[:, keep]
        )

    return (
        masked,
        removed_total
    )


# ============================================================
# CREATE MODEL
# ============================================================

def create_model(
    data
):

    node_types = list(
        data.node_types
    )

    node_counts = {
        node_type:
            data[node_type].num_nodes
        for node_type in node_types
    }

    edge_types = list(
        data.edge_types
    )

    model = HeteroRGCNModel(
        node_types=node_types,
        node_counts=node_counts,
        edge_types=edge_types,
        embed_dim=EMBED_DIM,
        hidden_dim=HIDDEN_DIM
    ).to(DEVICE)

    return (
        model,
        node_counts
    )


# ============================================================
# INNER VALIDATION / BEST EPOCH
# ============================================================

def select_best_epoch(
    data,
    node_counts,
    outer_edge_index_dict,
    train_neg_full,
    node_id_to_local
):

    # --------------------------------------------------------
    # Unique positive edges
    # --------------------------------------------------------

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
        SEED
    )

    n_val = max(
        1,
        int(
            VAL_FRACTION
            * len(unique_positives)
        )
    )

    val_indices = rng.choice(
        len(unique_positives),
        size=n_val,
        replace=False
    )

    val_positive_rows = (
        unique_positives
        .iloc[val_indices]
    )

    val_keys = set(
        (
            row["compound_id"],
            row["true_target"],
            row["target_type"]
        )
        for _, row
        in val_positive_rows.iterrows()
    )

    is_val = train_neg_full.apply(
        lambda row:
            (
                row["compound_id"],
                row["true_target"],
                row["target_type"]
            ) in val_keys,
        axis=1
    )

    inner_train = (
        train_neg_full[
            ~is_val
        ]
        .reset_index(drop=True)
    )

    validation = (
        train_neg_full[
            is_val
        ]
        .reset_index(drop=True)
    )

    print(
        f"Training pairs: {len(inner_train)}"
    )

    print(
        f"Validation pairs: {len(validation)}"
    )

    # --------------------------------------------------------
    # Remove validation TREATS edges
    # --------------------------------------------------------

    val_pairs = set(
        (
            row["compound_id"],
            row["true_target"],
            row["target_type"]
        )
        for _, row
        in val_positive_rows.iterrows()
    )

    validation_edge_index_dict, removed = (
        mask_validation_treats_edges(
            outer_edge_index_dict,
            val_pairs,
            node_id_to_local
        )
    )

    print(
        "Validation TREATS edges removed "
        f"from message passing: {removed}"
    )

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    set_seed(
        SEED
    )

    model, _ = create_model(
        data
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LR,
        weight_decay=WEIGHT_DECAY
    )

    train_c, train_p, train_n, train_types = (
        dataframe_to_tensors(
            inner_train,
            node_id_to_local
        )
    )

    val_c, val_p, val_n, val_types = (
        dataframe_to_tensors(
            validation,
            node_id_to_local
        )
    )

    # --------------------------------------------------------
    # Positive weighting
    #
    # Each positive has 5 negatives in the current files.
    #
    # We do not massively upweight positives; instead we use
    # a modest correction.
    # --------------------------------------------------------

    POS_WEIGHT = 2.0

    best_val_loss = float(
        "inf"
    )

    best_epoch = 0

    best_state = None

    patience_counter = 0

    print(
        "\nInner training with weighted BCE..."
    )

    for epoch in range(
        N_EPOCHS
    ):

        # ====================================================
        # TRAIN
        # ====================================================

        model.train()

        optimizer.zero_grad()

        out = model(
            node_counts,
            validation_edge_index_dict
        )

        train_loss = compute_bce_loss(
            out,
            train_c,
            train_p,
            train_n,
            train_types,
            model.decoder,
            POS_WEIGHT
        )

        train_loss.backward()

        optimizer.step()

        # ====================================================
        # VALIDATION
        # ====================================================

        model.eval()

        with torch.no_grad():

            val_out = model(
                node_counts,
                validation_edge_index_dict
            )

            val_loss = compute_bce_loss(
                val_out,
                val_c,
                val_p,
                val_n,
                val_types,
                model.decoder,
                POS_WEIGHT
            )

        val_value = (
            val_loss.item()
        )

        improved = (
            val_value
            < best_val_loss
        )

        if improved:

            best_val_loss = val_value

            best_epoch = epoch

            best_state = copy.deepcopy(
                model.state_dict()
            )

            patience_counter = 0

        else:

            patience_counter += 1

        if (
            epoch % 10 == 0
            or improved
        ):

            marker = (
                "  (best)"
                if improved
                else ""
            )

            print(
                f"  epoch {epoch:3d} "
                f"train_loss={train_loss.item():.4f} "
                f"val_loss={val_value:.4f}"
                f"{marker}"
            )

        if (
            patience_counter
            >= PATIENCE
        ):

            print(
                f"  Early stopping at epoch {epoch}"
            )

            break

    print(
        f"\nBest inner epoch: {best_epoch}"
    )

    print(
        f"Best validation loss: "
        f"{best_val_loss:.6f}"
    )

    return (
        best_epoch,
        best_state,
        best_val_loss
    )


# ============================================================
# FINAL RETRAIN
# ============================================================

def final_retrain(
    data,
    node_counts,
    outer_edge_index_dict,
    train_neg_full,
    best_epoch,
    node_id_to_local
):

    print(
        "\n" + "=" * 70
    )

    print(
        "FINAL TRAINING ON ALL OUTER-TRAINING DATA"
    )

    print(
        "=" * 70
    )

    print(
        f"Selected epoch count: "
        f"{best_epoch + 1}"
    )

    # --------------------------------------------------------
    # Fresh model
    # --------------------------------------------------------

    set_seed(
        SEED
    )

    model, _ = create_model(
        data
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LR,
        weight_decay=WEIGHT_DECAY
    )

    c, p, n, types = (
        dataframe_to_tensors(
            train_neg_full,
            node_id_to_local
        )
    )

    POS_WEIGHT = 2.0

    # --------------------------------------------------------
    # Train for selected number of epochs
    # --------------------------------------------------------

    for epoch in range(
        best_epoch + 1
    ):

        model.train()

        optimizer.zero_grad()

        out = model(
            node_counts,
            outer_edge_index_dict
        )

        loss = compute_bce_loss(
            out,
            c,
            p,
            n,
            types,
            model.decoder,
            POS_WEIGHT
        )

        loss.backward()

        optimizer.step()

        if (
            epoch % 10 == 0
            or epoch == best_epoch
        ):

            print(
                f"  final epoch {epoch:3d} "
                f"loss={loss.item():.4f}"
            )

    return model


# ============================================================
# TIE-AWARE RANK
# ============================================================

def average_rank_tie_aware(
    true_score,
    candidate_scores
):

    greater = sum(
        score > true_score
        for score in candidate_scores
    )

    equal = (
        sum(
            score == true_score
            for score in candidate_scores
        ) - 1
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
# EVALUATION
# ============================================================

def evaluate(
    model,
    node_counts,
    edge_index_dict,
    eval_candidates_path,
    edges_path,
    node_id_to_local,
    node_id_to_type
):

    print(
        "\n" + "=" * 70
    )

    print(
        "FINAL OUTER-FOLD EVALUATION"
    )

    print(
        "=" * 70
    )

    eval_cases = pd.read_csv(
        eval_candidates_path
    )

    # --------------------------------------------------------
    # Candidate pools
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # Known TREATS positives
    # --------------------------------------------------------

    all_edges = pd.read_csv(
        edges_path
    )

    treats = all_edges[
        all_edges["relation"] == "TREATS"
    ]

    known_positive = defaultdict(
        set
    )

    for _, row in treats.iterrows():

        known_positive[
            row["source"]
        ].add(
            row["target"]
        )

    # --------------------------------------------------------
    # GNN forward pass
    # --------------------------------------------------------

    model.eval()

    with torch.no_grad():

        out = model(
            node_counts,
            edge_index_dict
        )

    results = []

    # --------------------------------------------------------
    # Ranking
    # --------------------------------------------------------

    with torch.no_grad():

        for _, row in eval_cases.iterrows():

            compound_id = row[
                "compound_id"
            ]

            true_target = row[
                "true_target"
            ]

            target_type = row[
                "target_type"
            ]

            if target_type == "Disease":

                pool = disease_ids

            else:

                pool = symptom_ids

            # Filter known positives,
            # except the target being evaluated.
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

            compound_local = node_local_idx(
                compound_id,
                node_id_to_local
            )

            candidate_local = torch.tensor(
                [
                    node_local_idx(
                        candidate,
                        node_id_to_local
                    )
                    for candidate in candidates
                ],
                dtype=torch.long,
                device=DEVICE
            )

            compound_emb = (
                out["Compound"][
                    compound_local
                ]
                .unsqueeze(0)
            )

            candidate_emb = (
                out[target_type][
                    candidate_local
                ]
            )

            scores = model.decoder(
                compound_emb.expand(
                    len(candidates),
                    -1
                ),
                candidate_emb
            )

            scores = (
                scores
                .detach()
                .cpu()
                .numpy()
            )

            true_index = (
                candidates.index(
                    true_target
                )
            )

            true_score = (
                scores[
                    true_index
                ]
            )

            rank = average_rank_tie_aware(
                true_score,
                list(scores)
            )

            results.append(
                {
                    "compound_id":
                        compound_id,

                    "true_target":
                        true_target,

                    "target_type":
                        target_type,

                    "rank":
                        rank,

                    "n_candidates":
                        len(candidates),

                    "reciprocal_rank":
                        1.0 / rank,

                    "hit_at_1":
                        int(rank <= 1),

                    "hit_at_5":
                        int(rank <= 5),

                    "hit_at_10":
                        int(rank <= 10)
                }
            )

    results_df = pd.DataFrame(
        results
    )

    output_path = (
        f"gnn_results_fold{FOLD_NUM}.csv"
    )

    results_df.to_csv(
        output_path,
        index=False
    )

    # --------------------------------------------------------
    # Overall metrics
    # --------------------------------------------------------

    print(
        f"Test cases : {len(results_df)}"
    )

    print(
        f"MRR        : "
        f"{results_df['reciprocal_rank'].mean():.4f}"
    )

    print(
        f"Hits@1     : "
        f"{results_df['hit_at_1'].mean():.4f}"
    )

    print(
        f"Hits@5     : "
        f"{results_df['hit_at_5'].mean():.4f}"
    )

    print(
        f"Hits@10    : "
        f"{results_df['hit_at_10'].mean():.4f}"
    )

    print(
        "\nTarget-type breakdown:"
    )

    print(
        results_df
        .groupby("target_type")[
            [
                "reciprocal_rank",
                "hit_at_1",
                "hit_at_5",
                "hit_at_10"
            ]
        ]
        .mean()
    )

    print(
        f"\nWrote: {output_path}"
    )

    return results_df


# ============================================================
# MAIN
# ============================================================

def main():

    global GLOBAL_NODE_ID_TO_LOCAL

    print(
        "=" * 70
    )

    print(
        "AYURKOSH -- R-GCN + DistMult"
    )

    print(
        "Weighted BCE training + compound-held-out evaluation"
    )

    print(
        "=" * 70
    )

    print(
        f"Fold       : {FOLD_NUM}"
    )

    print(
        f"Device     : {DEVICE}"
    )

    print(
        f"Seed       : {SEED}"
    )

    print(
        f"Embed dim  : {EMBED_DIM}"
    )

    print(
        f"Hidden dim : {HIDDEN_DIM}"
    )

    print(
        f"Learning rate : {LR}"
    )

    print(
        f"Weight decay  : {WEIGHT_DECAY}"
    )

    # ========================================================
    # BUILD OUTER GRAPH
    # ========================================================

    from build_hetero_data import (
        build_fold_hetero_data
    )

    print(
        "\nBuilding outer-fold training graph..."
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
        embed_dim=EMBED_DIM
    )

    GLOBAL_NODE_ID_TO_LOCAL = (
        node_id_to_local
    )

    print(
        f"Training TREATS edges : "
        f"{len(train_treats)}"
    )

    print(
        f"Held-out TREATS edges : "
        f"{len(test_treats)}"
    )

    print(
        f"Node types            : "
        f"{len(data.node_types)}"
    )

    print(
        f"Edge types            : "
        f"{len(data.edge_types)}"
    )

    # ========================================================
    # OUTER EDGE GRAPH
    # ========================================================

    outer_edge_index_dict = {
        edge_type:
            data[edge_type].edge_index.clone()
        for edge_type in data.edge_types
    }

    # ========================================================
    # LOAD NEGATIVES
    # ========================================================

    train_neg_full = pd.read_csv(
        TRAIN_NEGATIVES_PATH
    )

    print(
        f"\nTraining negative pairs: "
        f"{len(train_neg_full)}"
    )

    # ========================================================
    # INNER VALIDATION
    # ========================================================

    (
        best_epoch,
        _,
        best_val_loss
    ) = select_best_epoch(
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

    # ========================================================
    # FINAL RETRAIN
    # ========================================================

    final_model = final_retrain(
        data,
        {
            node_type:
                data[node_type].num_nodes
            for node_type in data.node_types
        },
        outer_edge_index_dict,
        train_neg_full,
        best_epoch,
        node_id_to_local
    )

    # ========================================================
    # FINAL EVALUATION
    # ========================================================

    evaluate(
        final_model,
        {
            node_type:
                data[node_type].num_nodes
            for node_type in data.node_types
        },
        outer_edge_index_dict,
        EVAL_CANDIDATES_PATH,
        EDGES_PATH,
        node_id_to_local,
        node_id_to_type
    )

    print(
        "\n" + "=" * 70
    )

    print(
        "FOLD COMPLETE"
    )

    print(
        "=" * 70
    )

    print(
        f"Fold: {FOLD_NUM}"
    )

    print(
        f"Best inner epoch: {best_epoch}"
    )

    print(
        f"Best validation loss: "
        f"{best_val_loss:.6f}"
    )

    print(
        "Outer test TREATS edges remained completely held out."
    )


if __name__ == "__main__":
    main()
