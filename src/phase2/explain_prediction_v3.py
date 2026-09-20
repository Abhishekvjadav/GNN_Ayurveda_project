
import os
import sys
import pandas as pd
import networkx as nx

BASE_DIR = "/content/GNN_Ayurveda_project"

NODES_PATH = os.path.join(
    BASE_DIR, "results", "phase1", "nodes.csv"
)

EDGES_PATH = os.path.join(
    BASE_DIR, "results", "phase1", "edges.csv"
)


def load_graph():

    nodes = pd.read_csv(NODES_PATH)
    edges = pd.read_csv(EDGES_PATH)

    node_info = nodes.set_index("node_id").to_dict("index")

    graph = nx.MultiDiGraph()

    for _, row in nodes.iterrows():

        graph.add_node(
            row["node_id"],
            node_type=row["node_type"],
            canonical_name=row["canonical_name"],
            raw_name=row["raw_name"]
        )

    for _, row in edges.iterrows():

        graph.add_edge(
            row["source"],
            row["target"],
            relation=row["relation"]
        )

    return graph, node_info, edges


def get_relations(edges, source, target):

    matches = edges[
        (edges["source"] == source) &
        (edges["target"] == target)
    ]

    return matches["relation"].tolist()


def find_shared_neighbors(
    compound_id,
    target_id,
    edges,
    node_info
):

    compound_neighbors = set(
        edges[
            edges["source"] == compound_id
        ]["target"]
    ) | set(
        edges[
            edges["target"] == compound_id
        ]["source"]
    )

    target_neighbors = set(
        edges[
            edges["source"] == target_id
        ]["target"]
    ) | set(
        edges[
            edges["target"] == target_id
        ]["source"]
    )

    shared = (
        compound_neighbors &
        target_neighbors
    )

    evidence = []

    for node_id in shared:

        compound_edges = []

        target_edges = []

        # Compound -> shared node
        compound_to_shared = edges[
            (edges["source"] == compound_id) &
            (edges["target"] == node_id)
        ]

        for _, row in compound_to_shared.iterrows():

            compound_edges.append(
                (
                    row["relation"],
                    "outgoing"
                )
            )

        # Shared node -> Compound
        shared_to_compound = edges[
            (edges["source"] == node_id) &
            (edges["target"] == compound_id)
        ]

        for _, row in shared_to_compound.iterrows():

            compound_edges.append(
                (
                    row["relation"],
                    "incoming"
                )
            )

        # Target -> shared node
        target_to_shared = edges[
            (edges["source"] == target_id) &
            (edges["target"] == node_id)
        ]

        for _, row in target_to_shared.iterrows():

            target_edges.append(
                (
                    row["relation"],
                    "outgoing"
                )
            )

        # Shared node -> Target
        shared_to_target = edges[
            (edges["source"] == node_id) &
            (edges["target"] == target_id)
        ]

        for _, row in shared_to_target.iterrows():

            target_edges.append(
                (
                    row["relation"],
                    "incoming"
                )
            )

        if compound_edges and target_edges:

            evidence.append({
                "node_id": node_id,
                "node_type": node_info[node_id]["node_type"],
                "canonical_name": node_info[node_id][
                    "canonical_name"
                ],
                "compound_edges": compound_edges,
                "target_edges": target_edges
            })

    return evidence


def print_edge(
    node_name,
    relation,
    direction,
    other_name
):

    if direction == "outgoing":

        print(
            f"   {node_name} "
            f"--{relation}--> "
            f"{other_name}"
        )

    else:

        print(
            f"   {other_name} "
            f"--{relation}--> "
            f"{node_name}"
        )


def explain(compound_id, target_id):

    graph, node_info, edges = load_graph()

    if compound_id not in node_info:
        raise ValueError(
            f"Unknown compound: {compound_id}"
        )

    if target_id not in node_info:
        raise ValueError(
            f"Unknown target: {target_id}"
        )

    compound = node_info[compound_id]
    target = node_info[target_id]

    print("\n" + "=" * 70)
    print("AYURKOSH-GNN — STRUCTURAL EXPLANATION")
    print("=" * 70)

    print(
        f"Compound : {compound['canonical_name']}"
    )

    print(
        f"Target   : {target['canonical_name']}"
    )

    print(
        f"Types    : "
        f"{compound['node_type']} → "
        f"{target['node_type']}"
    )

    # ------------------------------------------------------
    # Direct relationship
    # ------------------------------------------------------

    direct = get_relations(
        edges,
        compound_id,
        target_id
    )

    if direct:

        print(
            "\nDirect relationship:"
        )

        for relation in direct:

            print(
                f"   {compound['canonical_name']} "
                f"--{relation}--> "
                f"{target['canonical_name']}"
            )

    else:

        print(
            "\nDirect relationship: NONE"
        )

    # ------------------------------------------------------
    # Shared-neighbor evidence
    # ------------------------------------------------------

    evidence = find_shared_neighbors(
        compound_id,
        target_id,
        edges,
        node_info
    )

    print(
        "\n" + "-" * 70
    )

    print(
        f"Shared structural neighbors: {len(evidence)}"
    )

    if not evidence:

        print(
            "No shared-neighbor structural evidence found."
        )

    else:

        for i, item in enumerate(
            evidence,
            start=1
        ):

            print(
                f"\nEvidence {i}"
            )

            print(
                f"   Node      : "
                f"{item['canonical_name']}"
            )

            print(
                f"   Node type : "
                f"{item['node_type']}"
            )

            print(
                "   Compound side:"
            )

            for relation, direction in item[
                "compound_edges"
            ]:

                print_edge(
                    item["canonical_name"],
                    relation,
                    direction,
                    compound["canonical_name"]
                )

            print(
                "   Target side:"
            )

            for relation, direction in item[
                "target_edges"
            ]:

                print_edge(
                    item["canonical_name"],
                    relation,
                    direction,
                    target["canonical_name"]
                )

    # ------------------------------------------------------
    # Interpretation
    # ------------------------------------------------------

    print(
        "\n" + "=" * 70
    )

    print(
        "INTERPRETATION"
    )

    print(
        "=" * 70
    )

    if direct:

        print(
            "The target has an explicit relationship "
            "with the compound in the knowledge graph."
        )

    elif evidence:

        print(
            "The target has shared-neighbor structural "
            "evidence with the compound."
        )

        print(
            "These relations provide graph context for "
            "the model prediction."
        )

    else:

        print(
            "No explicit direct or shared-neighbor "
            "evidence was found."
        )

    # ------------------------------------------------------
    # Structured return value
    # ------------------------------------------------------

    structured_evidence = []

    for item in evidence:

        structured_evidence.append({
            "node_id": item.get("node_id"),
            "canonical_name": item["canonical_name"],
            "node_type": item["node_type"],
            "compound_edges": [
                {
                    "relation": relation,
                    "direction": direction
                }
                for relation, direction
                in item["compound_edges"]
            ],
            "target_edges": [
                {
                    "relation": relation,
                    "direction": direction
                }
                for relation, direction
                in item["target_edges"]
            ]
        })

    return {
        "compound_id": compound_id,
        "compound_name": compound["canonical_name"],
        "compound_type": compound["node_type"],
        "target_id": target_id,
        "target_name": target["canonical_name"],
        "target_type": target["node_type"],
        "direct_relationships": direct,
        "has_direct_relationship": bool(direct),
        "shared_neighbor_count": len(evidence),
        "shared_neighbors": structured_evidence
    }
