"""
Phase 2 -- Build PyTorch Geometric HeteroData per fold

Design constraints carried over from everything decided so far:

  - Node features: this graph has no informative raw feature vectors
    per node (no embeddings, no numeric attributes) -- so every node
    type gets a learnable embedding table initialized here, and the
    GNN learns representations from structure + these embeddings
    jointly. This is standard for KG-style heterogeneous GNNs with no
    side features (same setup R-GCN's original paper uses for link
    prediction on knowledge graphs).

  - Fold masking: for fold i, the TRAINING graph excludes ONLY that
    fold's held-out TREATS edges. A held-out compound's CONTAINS
    edges (and everything else) remain in the training graph -- that
    describes what the compound IS, not the label being predicted.
    This must exactly match the masking already used to build
    eval_candidates_fold*.csv and train_hard_negatives_fold*.csv, or
    the model would be evaluated inconsistently with how baseline was
    evaluated.

  - Relation direction: PyG HeteroData edge types are
    (src_node_type, relation_name, dst_node_type) triples. Reverse
    edges are added automatically for every relation except TREATS
    (which is exactly what you want to predict -- adding its reverse
    would let the model trivially "look up" the answer via the
    reverse edge during message passing, i.e. genuine leakage, unlike
    the CONTAINS/TREATED_BY correlation discussed earlier which is
    legitimate signal).
"""

import sys
from collections import defaultdict

import pandas as pd
import torch
from torch_geometric.data import HeteroData

NODES_PATH = sys.argv[1] if len(sys.argv) > 1 else "..\\phase1\\nodes.csv"
EDGES_PATH = sys.argv[2] if len(sys.argv) > 2 else "..\\phase1\\edges.csv"
FOLD_TREATS_PATH = sys.argv[3] if len(sys.argv) > 3 else "treats_edges_with_fold.csv"
FOLD_NUM = int(sys.argv[4]) if len(sys.argv) > 4 else 0
EMBED_DIM = 64

# Relations that should NOT get an automatic reverse edge added.
# TREATS is the prediction target -- a reverse edge would leak the
# answer directly into message passing.
NO_REVERSE = {"TREATS"}


def build_node_index(nodes: pd.DataFrame):
    """node_id (string) -> (node_type, local_integer_index), plus the
    reverse mapping and per-type counts, which HeteroData needs."""
    type_to_ids = defaultdict(list)
    for _, row in nodes.iterrows():
        type_to_ids[row["node_type"]].append(row["node_id"])

    node_id_to_local = {}
    node_id_to_type = {}
    for ntype, ids in type_to_ids.items():
        for local_idx, node_id in enumerate(ids):
            node_id_to_local[node_id] = local_idx
            node_id_to_type[node_id] = ntype

    type_counts = {ntype: len(ids) for ntype, ids in type_to_ids.items()}
    return node_id_to_local, node_id_to_type, type_counts


def build_fold_hetero_data(nodes_path, edges_path, fold_treats_path, fold_num, embed_dim=EMBED_DIM):
    nodes = pd.read_csv(nodes_path)
    edges = pd.read_csv(edges_path)
    treats_folded = pd.read_csv(fold_treats_path)

    node_id_to_local, node_id_to_type, type_counts = build_node_index(nodes)

    # Non-TREATS edges: always included in the training graph, every fold.
    non_treats_edges = edges[edges["relation"] != "TREATS"]

    # TREATS edges for THIS fold's training set only (held-out fold excluded).
    train_treats = treats_folded[treats_folded["fold"] != fold_num]
    test_treats = treats_folded[treats_folded["fold"] == fold_num]

    print(f"Fold {fold_num}: {len(train_treats)} training TREATS edges, "
          f"{len(test_treats)} held-out (excluded from training graph)")

    data = HeteroData()

    # --- Node embeddings (learnable, structure-only setup) ---
    for ntype, count in type_counts.items():
        data[ntype].num_nodes = count
        data[ntype].x = torch.nn.Parameter(torch.randn(count, embed_dim) * 0.1)

    # --- Edges, grouped by (src_type, relation, dst_type) ---
    def add_edge_group(df, relation_name):
        if df.empty:
            return
        # Group by actual (src_type, dst_type) pairs present, since a
        # relation can connect more than one type pair (e.g. TREATED_BY
        # -> Herb, Mineral, Vehicle, Sub_Formulation, Other_Unclassified).
        tmp = df.copy()
        tmp["src_type"] = tmp["source"].map(node_id_to_type)
        tmp["dst_type"] = tmp["target"].map(node_id_to_type)
        tmp = tmp.dropna(subset=["src_type", "dst_type"])

        for (src_type, dst_type), grp in tmp.groupby(["src_type", "dst_type"]):
            src_idx = torch.tensor([node_id_to_local[s] for s in grp["source"]], dtype=torch.long)
            dst_idx = torch.tensor([node_id_to_local[t] for t in grp["target"]], dtype=torch.long)
            edge_index = torch.stack([src_idx, dst_idx], dim=0)

            key = (src_type, relation_name, dst_type)
            data[key].edge_index = edge_index

            if relation_name not in NO_REVERSE:
                rev_key = (dst_type, f"rev_{relation_name}", src_type)
                data[rev_key].edge_index = torch.stack([dst_idx, src_idx], dim=0)

    for relation in non_treats_edges["relation"].unique():
        add_edge_group(non_treats_edges[non_treats_edges["relation"] == relation], relation)

    # TREATS: only the training portion for this fold goes into the graph.
    add_edge_group(train_treats.rename(columns={"source": "source", "target": "target"}), "TREATS")

    return data, node_id_to_local, node_id_to_type, train_treats, test_treats


def main():
    data, node_id_to_local, node_id_to_type, train_treats, test_treats = build_fold_hetero_data(
        NODES_PATH, EDGES_PATH, FOLD_TREATS_PATH, FOLD_NUM
    )

    print("\n" + "=" * 60)
    print(f"HeteroData for fold {FOLD_NUM}")
    print("=" * 60)
    print(data)

    print("\nNode types and counts:")
    for ntype in data.node_types:
        print(f"  {ntype:20s}: {data[ntype].num_nodes} nodes, x shape {tuple(data[ntype].x.shape)}")

    print("\nEdge types and counts:")
    for etype in data.edge_types:
        print(f"  {str(etype):55s}: {data[etype].edge_index.shape[1]} edges")

    # Sanity check: no held-out TREATS edge should be reachable in this graph.
    # TREATS can split across multiple (src_type, dst_type) triples (here:
    # Compound->Disease AND Compound->Symptom), so sum across ALL of them --
    # checking only the first one found undercounts and gives a false
    # "mismatch" even when the graph is built correctly.
    treats_keys = [etype for etype in data.edge_types if etype[1] == "TREATS"]
    if treats_keys:
        n_treats_edges_in_graph = sum(data[k].edge_index.shape[1] for k in treats_keys)
        print(f"\nTREATS edge-type triples found: {treats_keys}")
        print(f"TREATS edges actually in the fold-{FOLD_NUM} training graph (summed): {n_treats_edges_in_graph}")
        print(f"Expected (train_treats count): {len(train_treats)}")
        assert n_treats_edges_in_graph == len(train_treats), \
            "MISMATCH -- held-out TREATS edges may have leaked into the training graph!"
        print("Confirmed: counts match, no leakage into the training graph.")

    print(f"\nHeld-out test TREATS edges for fold {FOLD_NUM} (NOT in this graph): {len(test_treats)}")


if __name__ == "__main__":
    main()