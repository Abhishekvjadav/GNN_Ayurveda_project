"""
AyurKOSH Phase 2
TRUE R-GCN encoder + DistMult decoder

This file follows the same experimental protocol as the existing
GraphSAGE-style model in train_rgcn.py. The only architectural change is
the encoder: PyG RGCNConv is used on a homogeneous global node index with
relation IDs.

Prediction task:
    Compound --TREATS--> Disease
    Compound --TREATS--> Symptom

Evaluation:
    Compound-held-out 5-fold filtered ranking
    MRR, Hits@1, Hits@5, Hits@10

Important:
    - Held-out outer TREATS edges are excluded from message passing.
    - Inner validation TREATS edges are also removed from message passing.
    - Known positives are used only for filtered ranking, not as GNN edges.
    - The decoder remains DistMult so the model comparison changes the
      encoder, not the decoder or evaluation protocol.
"""

import sys
import copy
import random
import os
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn import RGCNConv


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

FOLD_NUM = int(sys.argv[4]) if len(sys.argv) > 4 else 0

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

CHECKPOINT_DIR = (
    "/content/GNN_Ayurveda_project/results/phase2/checkpoints"
)


# ============================================================
# HYPERPARAMETERS -- kept identical to GraphSAGE experiment
# ============================================================

EMBED_DIM = 64
HIDDEN_DIM = 64
N_EPOCHS = 150

LR = 0.005
WEIGHT_DECAY = 1e-4

PATIENCE = 20
VAL_FRACTION = 0.15

SEED = 42
POS_WEIGHT = 2.0

# Keep the official comparison on CPU for exact protocol consistency
# with the existing GraphSAGE results. Set to CUDA only if the whole
# benchmark is intentionally regenerated with the same device policy.
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
# DISTMULT DECODER
# ============================================================

class DistMultDecoder(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.r = nn.Parameter(torch.randn(dim) * 0.1)

    def forward(self, h, t):
        return (h * self.r * t).sum(dim=-1)


# ============================================================
# TRUE R-GCN MODEL
# ============================================================

class TrueRGCNModel(nn.Module):
    """
    A genuine relational GCN:
        x -> RGCNConv -> ReLU -> RGCNConv

    All heterogeneous node types are represented in one global node
    index. Each graph edge receives a relation ID corresponding to its
    heterogeneous edge type.

    Compound nodes use one shared trainable type-level vector, matching
    the existing GraphSAGE model's initialization design.
    """

    def __init__(
        self,
        node_types,
        node_counts,
        relation_count,
        embed_dim=64,
        hidden_dim=64,
    ):
        super().__init__()

        self.node_types_order = list(node_types)
        self.node_counts = dict(node_counts)

        self.offsets = {}
        offset = 0
        for node_type in self.node_types_order:
            self.offsets[node_type] = offset
            offset += self.node_counts[node_type]

        self.num_nodes = offset

        # Same compound initialization policy as the current model.
        self.shared_compound_embedding = nn.Parameter(
            torch.randn(1, embed_dim) * 0.1
        )

        # Trainable initial embeddings for every non-Compound type.
        self.node_embeddings = nn.ParameterDict()
        for node_type in self.node_types_order:
            if node_type == "Compound":
                continue

            self.node_embeddings[node_type] = nn.Parameter(
                torch.randn(
                    self.node_counts[node_type],
                    embed_dim,
                ) * 0.1
            )

        # TRUE R-GCN layers.
        self.conv1 = RGCNConv(
            embed_dim,
            hidden_dim,
            num_relations=relation_count,
            num_bases=None,
        )

        self.conv2 = RGCNConv(
            hidden_dim,
            hidden_dim,
            num_relations=relation_count,
            num_bases=None,
        )

        self.decoder = DistMultDecoder(hidden_dim)

    def initial_x(self):
        parts = []

        for node_type in self.node_types_order:
            if node_type == "Compound":
                x = self.shared_compound_embedding.expand(
                    self.node_counts[node_type], -1
                )
            else:
                x = self.node_embeddings[node_type]
            parts.append(x)

        return torch.cat(parts, dim=0)

    def forward(self, edge_index, edge_type):
        x = self.initial_x()

        x = self.conv1(
            x,
            edge_index,
            edge_type,
        )
        x = F.relu(x)

        x = self.conv2(
            x,
            edge_index,
            edge_type,
        )

        # Return a dictionary so the decoder/evaluation code stays
        # conceptually identical to the existing heterogeneous model.
        out = {}
        for node_type in self.node_types_order:
            start = self.offsets[node_type]
            end = start + self.node_counts[node_type]
            out[node_type] = x[start:end]

        return out


# ============================================================
# GLOBAL GRAPH CONVERSION
# ============================================================

def build_global_graph(data, edge_index_dict):
    """
    Convert PyG HeteroData edge dictionaries into:
        global edge_index [2, E]
        relation_id [E]

    Relation IDs are assigned to the exact edge_type tuples already
    present in build_hetero_data.py. Reverse relations therefore receive
    their own relation IDs.
    """

    node_types = list(data.node_types)

    node_counts = {
        nt: int(data[nt].num_nodes)
        for nt in node_types
    }

    offsets = {}
    offset = 0
    for nt in node_types:
        offsets[nt] = offset
        offset += node_counts[nt]

    all_edges = []
    all_relations = []
    relation_to_id = {}

    for rel_id, edge_type in enumerate(data.edge_types):
        relation_to_id[tuple(edge_type)] = rel_id

        src_type, _, dst_type = edge_type
        edge_index = edge_index_dict[edge_type]

        src = edge_index[0] + offsets[src_type]
        dst = edge_index[1] + offsets[dst_type]

        all_edges.append(
            torch.stack([src, dst], dim=0)
        )

        all_relations.append(
            torch.full(
                (edge_index.shape[1],),
                rel_id,
                dtype=torch.long,
                device=edge_index.device,
            )
        )

    if not all_edges:
        raise RuntimeError("No graph edges were found.")

    global_edge_index = torch.cat(all_edges, dim=1)
    global_edge_type = torch.cat(all_relations, dim=0)

    return (
        global_edge_index,
        global_edge_type,
        offsets,
        relation_to_id,
    )


# ============================================================
# DATAFRAME -> TENSORS
# ============================================================

def dataframe_to_tensors(df, node_id_to_local):
    compound = torch.tensor(
        [node_id_to_local[c] for c in df["compound_id"]],
        dtype=torch.long,
        device=DEVICE,
    )

    positive_target = torch.tensor(
        [node_id_to_local[t] for t in df["true_target"]],
        dtype=torch.long,
        device=DEVICE,
    )

    negative_target = torch.tensor(
        [node_id_to_local[t] for t in df["hard_negative"]],
        dtype=torch.long,
        device=DEVICE,
    )

    target_types = df["target_type"].tolist()

    return (
        compound,
        positive_target,
        negative_target,
        target_types,
    )


# ============================================================
# LOSS
# ============================================================

def compute_bce_loss(
    out,
    compound_idx,
    positive_idx,
    negative_idx,
    target_types,
    decoder,
    pos_weight,
):
    compound_emb = out["Compound"][compound_idx]
    all_losses = []

    for target_type in sorted(set(target_types)):
        mask = torch.tensor(
            [t == target_type for t in target_types],
            dtype=torch.bool,
            device=DEVICE,
        )

        if mask.sum().item() == 0:
            continue

        compound_batch = compound_emb[mask]

        positive_emb = out[target_type][positive_idx[mask]]
        negative_emb = out[target_type][negative_idx[mask]]

        positive_scores = decoder(
            compound_batch,
            positive_emb,
        )

        negative_scores = decoder(
            compound_batch,
            negative_emb,
        )

        positive_labels = torch.ones_like(positive_scores)
        negative_labels = torch.zeros_like(negative_scores)

        positive_loss = F.binary_cross_entropy_with_logits(
            positive_scores,
            positive_labels,
        )

        negative_loss = F.binary_cross_entropy_with_logits(
            negative_scores,
            negative_labels,
        )

        type_loss = (
            pos_weight * positive_loss + negative_loss
        ) / (pos_weight + 1.0)

        all_losses.append(type_loss)

    if not all_losses:
        return torch.tensor(
            0.0,
            device=DEVICE,
            requires_grad=True,
        )

    return torch.stack(all_losses).mean()


# ============================================================
# MASK VALIDATION TREATS EDGES
# ============================================================

def mask_validation_treats_edges(
    edge_index_dict,
    val_pairs,
    node_id_to_local,
):
    masked = {}
    val_local_pairs = set()

    for compound_id, target_id, target_type in val_pairs:
        val_local_pairs.add(
            (
                node_id_to_local[compound_id],
                node_id_to_local[target_id],
                target_type,
            )
        )

    removed_total = 0

    for edge_type, edge_index in edge_index_dict.items():
        src_type, relation, dst_type = edge_type

        if relation != "TREATS":
            masked[edge_type] = edge_index.clone()
            continue

        keep = torch.ones(
            edge_index.shape[1],
            dtype=torch.bool,
            device=edge_index.device,
        )

        for i in range(edge_index.shape[1]):
            src_local = int(edge_index[0, i])
            dst_local = int(edge_index[1, i])

            if (
                src_local,
                dst_local,
                dst_type,
            ) in val_local_pairs:
                keep[i] = False
                removed_total += 1

        masked[edge_type] = edge_index[:, keep]

    return masked, removed_total


# ============================================================
# MODEL CREATION
# ============================================================

def create_model(data, relation_count):
    node_types = list(data.node_types)

    node_counts = {
        node_type: int(data[node_type].num_nodes)
        for node_type in node_types
    }

    model = TrueRGCNModel(
        node_types=node_types,
        node_counts=node_counts,
        relation_count=relation_count,
        embed_dim=EMBED_DIM,
        hidden_dim=HIDDEN_DIM,
    ).to(DEVICE)

    return model, node_counts


# ============================================================
# INNER VALIDATION
# ============================================================

def select_best_epoch(
    data,
    node_counts,
    outer_edge_index_dict,
    train_neg_full,
    node_id_to_local,
    relation_count,
):
    unique_positives = (
        train_neg_full[
            ["compound_id", "true_target", "target_type"]
        ]
        .drop_duplicates()
        .reset_index(drop=True)
    )

    rng = np.random.RandomState(SEED)

    n_val = max(
        1,
        int(VAL_FRACTION * len(unique_positives)),
    )

    val_indices = rng.choice(
        len(unique_positives),
        size=n_val,
        replace=False,
    )

    val_positive_rows = unique_positives.iloc[val_indices]

    val_keys = {
        (
            row["compound_id"],
            row["true_target"],
            row["target_type"],
        )
        for _, row in val_positive_rows.iterrows()
    }

    is_val = train_neg_full.apply(
        lambda row: (
            row["compound_id"],
            row["true_target"],
            row["target_type"],
        ) in val_keys,
        axis=1,
    )

    inner_train = (
        train_neg_full[~is_val]
        .reset_index(drop=True)
    )

    validation = (
        train_neg_full[is_val]
        .reset_index(drop=True)
    )

    print(f"Training pairs: {len(inner_train)}")
    print(f"Validation pairs: {len(validation)}")

    val_pairs = set(
        (
            row["compound_id"],
            row["true_target"],
            row["target_type"],
        )
        for _, row in val_positive_rows.iterrows()
    )

    validation_edge_index_dict, removed = (
        mask_validation_treats_edges(
            outer_edge_index_dict,
            val_pairs,
            node_id_to_local,
        )
    )

    print(
        "Validation TREATS edges removed from "
        f"message passing: {removed}"
    )

    (
        val_global_edge_index,
        val_global_edge_type,
        _,
        _,
    ) = build_global_graph(
        data,
        validation_edge_index_dict,
    )

    set_seed(SEED)

    model, _ = create_model(
        data,
        relation_count,
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LR,
        weight_decay=WEIGHT_DECAY,
    )

    train_c, train_p, train_n, train_types = (
        dataframe_to_tensors(
            inner_train,
            node_id_to_local,
        )
    )

    val_c, val_p, val_n, val_types = (
        dataframe_to_tensors(
            validation,
            node_id_to_local,
        )
    )

    best_val_loss = float("inf")
    best_epoch = 0
    best_state = None
    patience_counter = 0

    print("\nInner TRUE R-GCN training with weighted BCE...")

    for epoch in range(N_EPOCHS):
        model.train()
        optimizer.zero_grad()

        out = model(
            val_global_edge_index,
            val_global_edge_type,
        )

        train_loss = compute_bce_loss(
            out,
            train_c,
            train_p,
            train_n,
            train_types,
            model.decoder,
            POS_WEIGHT,
        )

        train_loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            val_out = model(
                val_global_edge_index,
                val_global_edge_type,
            )

            val_loss = compute_bce_loss(
                val_out,
                val_c,
                val_p,
                val_n,
                val_types,
                model.decoder,
                POS_WEIGHT,
            )

        val_value = val_loss.item()
        improved = val_value < best_val_loss

        if improved:
            best_val_loss = val_value
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1

        if epoch % 10 == 0 or improved:
            marker = "  (best)" if improved else ""
            print(
                f"  epoch {epoch:3d} "
                f"train_loss={train_loss.item():.4f} "
                f"val_loss={val_value:.4f}{marker}"
            )

        if patience_counter >= PATIENCE:
            print(f"  Early stopping at epoch {epoch}")
            break

    print(f"\nBest inner epoch: {best_epoch}")
    print(f"Best validation loss: {best_val_loss:.6f}")

    return best_epoch, best_state, best_val_loss


# ============================================================
# FINAL RETRAIN
# ============================================================

def final_retrain(
    data,
    node_counts,
    outer_edge_index_dict,
    train_neg_full,
    best_epoch,
    node_id_to_local,
    relation_count,
):
    print("\n" + "=" * 70)
    print("FINAL TRUE R-GCN TRAINING ON ALL OUTER-TRAINING DATA")
    print("=" * 70)
    print(f"Selected epoch count: {best_epoch + 1}")

    set_seed(SEED)

    model, _ = create_model(
        data,
        relation_count,
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LR,
        weight_decay=WEIGHT_DECAY,
    )

    (
        global_edge_index,
        global_edge_type,
        _,
        _,
    ) = build_global_graph(
        data,
        outer_edge_index_dict,
    )

    c, p, n, types = dataframe_to_tensors(
        train_neg_full,
        node_id_to_local,
    )

    for epoch in range(best_epoch + 1):
        model.train()
        optimizer.zero_grad()

        out = model(
            global_edge_index,
            global_edge_type,
        )

        loss = compute_bce_loss(
            out,
            c,
            p,
            n,
            types,
            model.decoder,
            POS_WEIGHT,
        )

        loss.backward()
        optimizer.step()

        if epoch % 10 == 0 or epoch == best_epoch:
            print(
                f"  final epoch {epoch:3d} "
                f"loss={loss.item():.4f}"
            )

    return model


# ============================================================
# TIE-AWARE RANK
# ============================================================

def average_rank_tie_aware(true_score, candidate_scores):
    greater = sum(
        score > true_score
        for score in candidate_scores
    )

    equal = (
        sum(score == true_score for score in candidate_scores) - 1
    )

    equal = max(equal, 0)

    return 1 + greater + 0.5 * equal


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
    node_id_to_type,
    data,
):
    print("\n" + "=" * 70)
    print("FINAL OUTER-FOLD TRUE R-GCN EVALUATION")
    print("=" * 70)

    eval_cases = pd.read_csv(eval_candidates_path)

    disease_ids = [
        node_id
        for node_id, node_type in node_id_to_type.items()
        if node_type == "Disease"
    ]

    symptom_ids = [
        node_id
        for node_id, node_type in node_id_to_type.items()
        if node_type == "Symptom"
    ]

    all_edges = pd.read_csv(edges_path)

    treats = all_edges[
        all_edges["relation"] == "TREATS"
    ]

    known_positive = defaultdict(set)

    for _, row in treats.iterrows():
        known_positive[row["source"]].add(row["target"])

    model.eval()

    (
        global_edge_index,
        global_edge_type,
        offsets,
        _,
    ) = build_global_graph(
        data,
        edge_index_dict,
    )

    with torch.no_grad():
        out = model(
            global_edge_index,
            global_edge_type,
        )

    results = []

    with torch.no_grad():
        for _, row in eval_cases.iterrows():
            compound_id = row["compound_id"]
            true_target = row["true_target"]
            target_type = row["target_type"]

            pool = (
                disease_ids
                if target_type == "Disease"
                else symptom_ids
            )

            # Filter complete known TREATS positives for ranking only.
            # The held-out TREATS edges are NOT present in message passing.
            candidates = [
                candidate
                for candidate in pool
                if (
                    candidate == true_target
                    or candidate not in known_positive[compound_id]
                )
            ]

            compound_local = node_id_to_local[compound_id]
            target_local = [
                node_id_to_local[candidate]
                for candidate in candidates
            ]

            compound_emb = out["Compound"][compound_local].unsqueeze(0)
            candidate_emb = out[target_type][
                torch.tensor(
                    target_local,
                    dtype=torch.long,
                    device=DEVICE,
                )
            ]

            scores = model.decoder(
                compound_emb.expand(len(candidates), -1),
                candidate_emb,
            )

            scores = (
                scores.detach().cpu().numpy()
            )

            true_index = candidates.index(true_target)
            true_score = scores[true_index]

            rank = average_rank_tie_aware(
                true_score,
                list(scores),
            )

            results.append(
                {
                    "compound_id": compound_id,
                    "true_target": true_target,
                    "target_type": target_type,
                    "rank": rank,
                    "n_candidates": len(candidates),
                    "reciprocal_rank": 1.0 / rank,
                    "hit_at_1": int(rank <= 1),
                    "hit_at_5": int(rank <= 5),
                    "hit_at_10": int(rank <= 10),
                }
            )

    results_df = pd.DataFrame(results)

    output_path = f"rgcn_results_fold{FOLD_NUM}.csv"
    results_df.to_csv(output_path, index=False)

    print(f"Test cases : {len(results_df)}")
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

    print("\nTarget-type breakdown:")
    print(
        results_df.groupby("target_type")[
            [
                "reciprocal_rank",
                "hit_at_1",
                "hit_at_5",
                "hit_at_10",
            ]
        ].mean()
    )

    print(f"\nWrote: {output_path}")

    return results_df


# ============================================================
# MAIN
# ============================================================

def main():
    print("=" * 70)
    print("AYURKOSH -- TRUE R-GCN + DistMult")
    print("Weighted BCE + compound-held-out filtered ranking")
    print("=" * 70)

    print(f"Fold          : {FOLD_NUM}")
    print(f"Device        : {DEVICE}")
    print(f"Seed          : {SEED}")
    print(f"Embed dim     : {EMBED_DIM}")
    print(f"Hidden dim    : {HIDDEN_DIM}")
    print(f"Learning rate : {LR}")
    print(f"Weight decay  : {WEIGHT_DECAY}")

    from build_hetero_data import build_fold_hetero_data

    print("\nBuilding outer-fold training graph...")

    (
        data,
        node_id_to_local,
        node_id_to_type,
        train_treats,
        test_treats,
    ) = build_fold_hetero_data(
        NODES_PATH,
        EDGES_PATH,
        FOLD_TREATS_PATH,
        FOLD_NUM,
        embed_dim=EMBED_DIM,
    )

    print(f"Training TREATS edges : {len(train_treats)}")
    print(f"Held-out TREATS edges : {len(test_treats)}")
    print(f"Node types            : {len(data.node_types)}")
    print(f"Edge types            : {len(data.edge_types)}")

    outer_edge_index_dict = {
        edge_type: data[edge_type].edge_index.clone()
        for edge_type in data.edge_types
    }

    train_neg_full = pd.read_csv(
        TRAIN_NEGATIVES_PATH
    )

    print(
        f"\nTraining negative pairs: "
        f"{len(train_neg_full)}"
    )

    # The number of R-GCN relations equals the number of edge types
    # actually present in the outer graph, including explicit reverse
    # relations created by build_hetero_data.py.
    relation_count = len(data.edge_types)

    print(
        f"R-GCN relation types: {relation_count}"
    )

    node_counts = {
        node_type: int(data[node_type].num_nodes)
        for node_type in data.node_types
    }

    (
        best_epoch,
        _,
        best_val_loss,
    ) = select_best_epoch(
        data,
        node_counts,
        outer_edge_index_dict,
        train_neg_full,
        node_id_to_local,
        relation_count,
    )

    final_model = final_retrain(
        data,
        node_counts,
        outer_edge_index_dict,
        train_neg_full,
        best_epoch,
        node_id_to_local,
        relation_count,
    )

    os.makedirs(
        CHECKPOINT_DIR,
        exist_ok=True,
    )

    checkpoint_path = os.path.join(
        CHECKPOINT_DIR,
        f"rgcn_fold{FOLD_NUM}.pt",
    )

    checkpoint = {
        "model_state_dict": final_model.state_dict(),
        "fold": FOLD_NUM,
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
        "embed_dim": EMBED_DIM,
        "hidden_dim": HIDDEN_DIM,
        "learning_rate": LR,
        "weight_decay": WEIGHT_DECAY,
        "pos_weight": POS_WEIGHT,
        "seed": SEED,
        "node_counts": node_counts,
        "node_id_to_local": node_id_to_local,
        "node_id_to_type": node_id_to_type,
        "node_types": list(data.node_types),
        "edge_types": [
            tuple(edge_type)
            for edge_type in data.edge_types
        ],
        "relation_count": relation_count,
    }

    torch.save(
        checkpoint,
        checkpoint_path,
    )

    print(
        f"\nSaved trained checkpoint: "
        f"{checkpoint_path}"
    )

    evaluate(
        final_model,
        node_counts,
        outer_edge_index_dict,
        EVAL_CANDIDATES_PATH,
        EDGES_PATH,
        node_id_to_local,
        node_id_to_type,
        data,
    )

    print("\n" + "=" * 70)
    print("FOLD COMPLETE")
    print("=" * 70)
    print(f"Fold: {FOLD_NUM}")
    print(f"Best inner epoch: {best_epoch}")
    print(
        f"Best validation loss: "
        f"{best_val_loss:.6f}"
    )
    print(
        "Outer test TREATS edges remained completely "
        "held out from message passing."
    )


if __name__ == "__main__":
    main()
