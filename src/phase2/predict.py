import os
import sys
import random
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import HeteroConv, SAGEConv


PROJECT_ROOT = "/content/GNN_Ayurveda_project"

NODES_PATH = os.path.join(
    PROJECT_ROOT, "results/phase1/nodes.csv"
)

EDGES_PATH = os.path.join(
    PROJECT_ROOT, "results/phase1/edges.csv"
)

FOLD_TREATS_PATH = os.path.join(
    PROJECT_ROOT, "results/phase2/treats_edges_with_fold.csv"
)

CHECKPOINT_DIR = os.path.join(
    PROJECT_ROOT, "results/phase2/checkpoints"
)

DEVICE = torch.device("cpu")

EMBED_DIM = 64
HIDDEN_DIM = 64
SEED = 42


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


set_seed(SEED)


class HeteroEncoder(nn.Module):

    def __init__(self, edge_types, hidden_dim):
        super().__init__()

        self.conv1 = HeteroConv(
            {
                edge_type: SAGEConv(
                    (-1, -1),
                    hidden_dim
                )
                for edge_type in edge_types
            },
            aggr="mean"
        )

        self.conv2 = HeteroConv(
            {
                edge_type: SAGEConv(
                    (-1, -1),
                    hidden_dim
                )
                for edge_type in edge_types
            },
            aggr="mean"
        )

    def forward(self, x_dict, edge_index_dict):

        x_dict = self.conv1(
            x_dict,
            edge_index_dict
        )

        x_dict = {
            node_type: F.relu(x)
            for node_type, x in x_dict.items()
        }

        x_dict = self.conv2(
            x_dict,
            edge_index_dict
        )

        return x_dict


class DistMultDecoder(nn.Module):

    def __init__(self, dim):
        super().__init__()

        self.r = nn.Parameter(
            torch.randn(dim) * 0.1
        )

    def forward(self, h, t):

        return (
            h *
            self.r *
            t
        ).sum(dim=-1)


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

        self.shared_compound_embedding = nn.Parameter(
            torch.randn(
                1,
                embed_dim
            ) * 0.1
        )

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

        self.encoder = HeteroEncoder(
            edge_types,
            hidden_dim
        )

        self.decoder = DistMultDecoder(
            hidden_dim
        )

    def get_initial_x_dict(self, node_counts):

        x_dict = {}

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


def build_node_index(nodes):

    type_to_ids = defaultdict(list)

    for _, row in nodes.iterrows():

        type_to_ids[
            row["node_type"]
        ].append(
            row["node_id"]
        )

    node_id_to_local = {}
    node_id_to_type = {}

    for node_type, ids in type_to_ids.items():

        for local_idx, node_id in enumerate(ids):

            node_id_to_local[node_id] = local_idx
            node_id_to_type[node_id] = node_type

    type_counts = {
        node_type: len(ids)
        for node_type, ids in type_to_ids.items()
    }

    return (
        node_id_to_local,
        node_id_to_type,
        type_counts
    )


def build_fold_graph(
    nodes,
    edges,
    treats_folded,
    fold_num
):

    (
        node_id_to_local,
        node_id_to_type,
        type_counts
    ) = build_node_index(nodes)

    non_treats_edges = edges[
        edges["relation"] != "TREATS"
    ]

    train_treats = treats_folded[
        treats_folded["fold"] != fold_num
    ]

    data_edge_dict = {}

    NO_REVERSE = {"TREATS"}

    def add_edge_group(
        df,
        relation_name
    ):

        if df.empty:
            return

        tmp = df.copy()

        tmp["src_type"] = tmp[
            "source"
        ].map(node_id_to_type)

        tmp["dst_type"] = tmp[
            "target"
        ].map(node_id_to_type)

        tmp = tmp.dropna(
            subset=[
                "src_type",
                "dst_type"
            ]
        )

        for (
            src_type,
            dst_type
        ), grp in tmp.groupby(
            [
                "src_type",
                "dst_type"
            ]
        ):

            src_idx = torch.tensor(
                [
                    node_id_to_local[s]
                    for s in grp["source"]
                ],
                dtype=torch.long
            )

            dst_idx = torch.tensor(
                [
                    node_id_to_local[t]
                    for t in grp["target"]
                ],
                dtype=torch.long
            )

            edge_index = torch.stack(
                [
                    src_idx,
                    dst_idx
                ],
                dim=0
            )

            key = (
                src_type,
                relation_name,
                dst_type
            )

            data_edge_dict[key] = edge_index

            if relation_name not in NO_REVERSE:

                rev_key = (
                    dst_type,
                    f"rev_{relation_name}",
                    src_type
                )

                data_edge_dict[
                    rev_key
                ] = torch.stack(
                    [
                        dst_idx,
                        src_idx
                    ],
                    dim=0
                )

    for relation in non_treats_edges[
        "relation"
    ].unique():

        add_edge_group(
            non_treats_edges[
                non_treats_edges["relation"] == relation
            ],
            relation
        )

    add_edge_group(
        train_treats,
        "TREATS"
    )

    return (
        node_id_to_local,
        node_id_to_type,
        type_counts,
        data_edge_dict
    )


def load_fold_model(
    fold_num,
    node_types,
    node_counts,
    edge_types
):

    checkpoint_path = os.path.join(
        CHECKPOINT_DIR,
        f"gnn_rgcn_fold{fold_num}.pt"
    )

    checkpoint = torch.load(
        checkpoint_path,
        map_location=DEVICE,
        weights_only=False
    )

    model = HeteroRGCNModel(
        node_types=node_types,
        node_counts=node_counts,
        edge_types=edge_types,
        embed_dim=EMBED_DIM,
        hidden_dim=HIDDEN_DIM
    ).to(DEVICE)

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.eval()

    return model, checkpoint


def score_candidates(
    model,
    node_counts,
    edge_index_dict,
    compound_id,
    candidates,
    target_type,
    node_id_to_local
):

    with torch.no_grad():

        out = model(
            node_counts,
            edge_index_dict
        )

        compound_local = node_id_to_local[
            compound_id
        ]

        compound_emb = (
            out["Compound"][
                compound_local
            ]
            .unsqueeze(0)
        )

        candidate_local = torch.tensor(
            [
                node_id_to_local[candidate]
                for candidate in candidates
            ],
            dtype=torch.long,
            device=DEVICE
        )

        candidate_emb = out[
            target_type
        ][candidate_local]

        scores = model.decoder(
            compound_emb.expand(
                len(candidates),
                -1
            ),
            candidate_emb
        )

        return scores.cpu().numpy()


def predict(compound_id, top_k=10):

    print("\n" + "=" * 70)
    print("AYURKOSH-GNN INFERENCE")
    print("=" * 70)
    print(f"Compound: {compound_id}")

    nodes = pd.read_csv(NODES_PATH)
    edges = pd.read_csv(EDGES_PATH)
    treats_folded = pd.read_csv(FOLD_TREATS_PATH)

    (
        base_node_id_to_local,
        base_node_id_to_type,
        _
    ) = build_node_index(nodes)

    if compound_id not in base_node_id_to_type:
        raise ValueError(
            f"Unknown node ID: {compound_id}"
        )

    if base_node_id_to_type[
        compound_id
    ] != "Compound":

        raise ValueError(
            f"{compound_id} is not a Compound node."
        )

    node_name_map = (
        nodes
        .set_index("node_id")["canonical_name"]
        .to_dict()
    )

    node_raw_name_map = (
        nodes
        .set_index("node_id")["raw_name"]
        .to_dict()
    )

    disease_ids = [
        node_id
        for node_id, node_type
        in base_node_id_to_type.items()
        if node_type == "Disease"
    ]

    symptom_ids = [
        node_id
        for node_id, node_type
        in base_node_id_to_type.items()
        if node_type == "Symptom"
    ]

    known_positive = defaultdict(set)

    all_treats = edges[
        edges["relation"] == "TREATS"
    ]

    for _, row in all_treats.iterrows():

        known_positive[
            row["source"]
        ].add(
            row["target"]
        )

    known_targets = known_positive[
        compound_id
    ]

    fold_results = []

    for fold_num in range(5):

        print(
            f"\nLoading fold {fold_num}..."
        )

        (
            node_id_to_local,
            node_id_to_type,
            node_counts,
            edge_index_dict
        ) = build_fold_graph(
            nodes,
            edges,
            treats_folded,
            fold_num
        )

        node_types = list(
            node_counts.keys()
        )

        edge_types = list(
            edge_index_dict.keys()
        )

        model, checkpoint = load_fold_model(
            fold_num,
            node_types,
            node_counts,
            edge_types
        )

        saved_mapping = checkpoint.get(
            "node_id_to_local"
        )

        if saved_mapping is not None:

            if saved_mapping != node_id_to_local:

                raise RuntimeError(
                    f"Fold {fold_num}: "
                    "node mapping mismatch."
                )

        for target_type, pool in [
            ("Disease", disease_ids),
            ("Symptom", symptom_ids)
        ]:

            candidates = [
                candidate
                for candidate in pool
                if candidate not in known_targets
            ]

            scores = score_candidates(
                model,
                node_counts,
                edge_index_dict,
                compound_id,
                candidates,
                target_type,
                node_id_to_local
            )

            fold_df = pd.DataFrame(
                {
                    "candidate_id": candidates,
                    "target_type": target_type,
                    "score": scores,
                    "fold": fold_num
                }
            )

            fold_results.append(fold_df)

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

    ensemble["canonical_name"] = (
        ensemble["candidate_id"]
        .map(node_name_map)
    )

    ensemble["raw_name"] = (
        ensemble["candidate_id"]
        .map(node_raw_name_map)
    )

    ensemble["known_treats"] = (
        ensemble["candidate_id"]
        .isin(known_targets)
    )

    ensemble["status"] = np.where(
        ensemble["known_treats"],
        "KNOWN_TREATS",
        "NOVEL_PREDICTION"
    )

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

    output_columns = [
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
        output_columns
    ]

    symptom_ranked = symptom_ranked[
        output_columns
    ]

    print(
        "\n" + "=" * 70
    )
    print("TOP DISEASE PREDICTIONS")
    print("=" * 70)

    print(
        disease_ranked.to_string(index=False)
    )

    print(
        "\n" + "=" * 70
    )
    print("TOP SYMPTOM PREDICTIONS")
    print("=" * 70)

    print(
        symptom_ranked.to_string(index=False)
    )

    print(
        "\n" + "=" * 70
    )
    print("PREDICTION SUMMARY")
    print("=" * 70)

    print(
        f"Known TREATS targets : {len(known_targets)}"
    )

    print(
        f"Novel candidates     : {len(ensemble)}"
    )

    output_dir = os.path.join(
        PROJECT_ROOT,
        "results/phase2/predictions"
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
        f"{safe_name}_disease_predictions.csv"
    )

    symptom_path = os.path.join(
        output_dir,
        f"{safe_name}_symptom_predictions.csv"
    )

    all_path = os.path.join(
        output_dir,
        f"{safe_name}_all_predictions.csv"
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
        all_path,
        index=False
    )

    print(
        f"\nSaved disease predictions:\n{disease_path}"
    )

    print(
        f"\nSaved symptom predictions:\n{symptom_path}"
    )

    print(
        f"\nSaved complete prediction table:\n{all_path}"
    )

    return (
        disease_ranked,
        symptom_ranked,
        ensemble
    )


if __name__ == "__main__":

    if len(sys.argv) < 2:

        print(
            "Usage: python predict.py <compound_id> [top_k]"
        )

        sys.exit(1)

    compound_id = sys.argv[1]

    top_k = (
        int(sys.argv[2])
        if len(sys.argv) > 2
        else 10
    )

    predict(
        compound_id,
        top_k=top_k
    )
