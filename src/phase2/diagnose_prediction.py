
import os
import sys
import pandas as pd

BASE_DIR = "/content/GNN_Ayurveda_project"

NODES_PATH = os.path.join(
    BASE_DIR, "results", "phase1", "nodes.csv"
)

EDGES_PATH = os.path.join(
    BASE_DIR, "results", "phase1", "edges.csv"
)

FOLD_PATH = os.path.join(
    BASE_DIR, "results", "phase2", "treats_edges_with_fold.csv"
)

CHECKPOINT_DIR = os.path.join(
    BASE_DIR, "results", "phase2", "checkpoints"
)


def load_data():

    nodes = pd.read_csv(NODES_PATH)
    edges = pd.read_csv(EDGES_PATH)

    return nodes, edges


def show_neighborhood(
    node_id,
    nodes,
    edges,
    title
):

    info = nodes[
        nodes["node_id"] == node_id
    ]

    if info.empty:
        print(f"Unknown node: {node_id}")
        return

    row = info.iloc[0]

    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)

    print(
        f"Node ID       : {node_id}"
    )

    print(
        f"Canonical name : {row['canonical_name']}"
    )

    print(
        f"Raw name       : {row['raw_name']}"
    )

    print(
        f"Node type      : {row['node_type']}"
    )

    outgoing = edges[
        edges["source"] == node_id
    ].copy()

    incoming = edges[
        edges["target"] == node_id
    ].copy()

    print(
        f"\nOutgoing edges: {len(outgoing)}"
    )

    if not outgoing.empty:

        for _, edge in outgoing.iterrows():

            target = nodes[
                nodes["node_id"] == edge["target"]
            ]

            if target.empty:
                continue

            target = target.iloc[0]

            print(
                f"  --{edge['relation']}--> "
                f"{target['canonical_name']} "
                f"[{target['node_type']}]"
            )

    print(
        f"\nIncoming edges: {len(incoming)}"
    )

    if not incoming.empty:

        for _, edge in incoming.iterrows():

            source = nodes[
                nodes["node_id"] == edge["source"]
            ]

            if source.empty:
                continue

            source = source.iloc[0]

            print(
                f"  {source['canonical_name']} "
                f"[{source['node_type']}] "
                f"--{edge['relation']}--> "
                f"{row['canonical_name']}"
            )


def main(
    compound_id,
    target_id
):

    nodes, edges = load_data()

    show_neighborhood(
        compound_id,
        nodes,
        edges,
        "COMPOUND NEIGHBORHOOD"
    )

    show_neighborhood(
        target_id,
        nodes,
        edges,
        "TARGET NEIGHBORHOOD"
    )

    # ------------------------------------------------------
    # Shared neighboring nodes
    # ------------------------------------------------------

    compound_out = set(
        edges[
            edges["source"] == compound_id
        ]["target"]
    )

    compound_in = set(
        edges[
            edges["target"] == compound_id
        ]["source"]
    )

    target_out = set(
        edges[
            edges["source"] == target_id
        ]["target"]
    )

    target_in = set(
        edges[
            edges["target"] == target_id
        ]["source"]
    )

    shared = (
        (compound_out | compound_in)
        &
        (target_out | target_in)
    )

    print(
        "\n" + "=" * 70
    )

    print(
        "SHARED ONE-HOP NEIGHBORS"
    )

    print(
        "=" * 70
    )

    print(
        f"Shared nodes: {len(shared)}"
    )

    if not shared:

        print(
            "No shared one-hop neighbors."
        )

    else:

        for node_id in sorted(shared):

            row = nodes[
                nodes["node_id"] == node_id
            ].iloc[0]

            print(
                f"{row['canonical_name']} "
                f"[{row['node_type']}]"
            )


if __name__ == "__main__":

    if len(sys.argv) != 3:

        print(
            "Usage:"
            " python diagnose_prediction.py "
            "<compound_id> <target_id>"
        )

        sys.exit(1)

    main(
        sys.argv[1],
        sys.argv[2]
    )
