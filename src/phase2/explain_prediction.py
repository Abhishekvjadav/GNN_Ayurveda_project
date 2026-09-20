
import os
import sys
import pandas as pd
import networkx as nx

BASE_DIR = "/content/GNN_Ayurveda_project"

NODES_PATH = os.path.join(
    BASE_DIR,
    "results",
    "phase1",
    "nodes.csv"
)

EDGES_PATH = os.path.join(
    BASE_DIR,
    "results",
    "phase1",
    "edges.csv"
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

    return graph, node_info


def find_explanation_paths(
    graph,
    node_info,
    compound_id,
    target_id,
    max_paths=10
):

    paths = []

    # ---------------------------------------------------------
    # Direct relationship
    # ---------------------------------------------------------

    if graph.has_edge(compound_id, target_id):

        for _, _, data in graph.out_edges(
            compound_id,
            data=True
        ):

            if _ == target_id:

                paths.append({
                    "path_type": "DIRECT",
                    "relations": [
                        data["relation"]
                    ],
                    "nodes": [
                        compound_id,
                        target_id
                    ]
                })

    # ---------------------------------------------------------
    # Two-hop paths
    # Compound -> Herb -> Target
    # ---------------------------------------------------------

    for herb in graph.successors(compound_id):

        if len(paths) >= max_paths:
            break

        if node_info.get(
            herb,
            {}
        ).get("node_type") != "Herb":

            continue

        if graph.has_edge(
            herb,
            target_id
        ):

            for _, _, data1 in graph.out_edges(
                compound_id,
                data=True
            ):

                if _ != herb:
                    continue

                for _, _, data2 in graph.out_edges(
                    herb,
                    data=True
                ):

                    if _ != target_id:
                        continue

                    paths.append({
                        "path_type": "COMPOUND_HERB_TARGET",
                        "relations": [
                            data1["relation"],
                            data2["relation"]
                        ],
                        "nodes": [
                            compound_id,
                            herb,
                            target_id
                        ]
                    })

    # ---------------------------------------------------------
    # Compound -> Herb -> Disease -> Target
    # Useful for symptom explanations
    # ---------------------------------------------------------

    for herb in graph.successors(compound_id):

        if len(paths) >= max_paths:
            break

        if node_info.get(
            herb,
            {}
        ).get("node_type") != "Herb":

            continue

        for disease in graph.successors(herb):

            if len(paths) >= max_paths:
                break

            if node_info.get(
                disease,
                {}
            ).get("node_type") != "Disease":

                continue

            if graph.has_edge(
                disease,
                target_id
            ):

                for _, _, data1 in graph.out_edges(
                    compound_id,
                    data=True
                ):

                    if _ != herb:
                        continue

                    for _, _, data2 in graph.out_edges(
                        herb,
                        data=True
                    ):

                        if _ != disease:
                            continue

                        for _, _, data3 in graph.out_edges(
                            disease,
                            data=True
                        ):

                            if _ != target_id:
                                continue

                            paths.append({
                                "path_type": "COMPOUND_HERB_DISEASE_TARGET",
                                "relations": [
                                    data1["relation"],
                                    data2["relation"],
                                    data3["relation"]
                                ],
                                "nodes": [
                                    compound_id,
                                    herb,
                                    disease,
                                    target_id
                                ]
                            })

    return paths[:max_paths]


def readable_path(
    path,
    node_info
):

    names = []

    for node_id in path["nodes"]:

        info = node_info.get(
            node_id,
            {}
        )

        names.append(
            info.get(
                "canonical_name",
                node_id
            )
        )

    return " → ".join(names)


def explain(
    compound_id,
    target_id
):

    graph, node_info = load_graph()

    if compound_id not in node_info:
        raise ValueError(
            f"Unknown compound: {compound_id}"
        )

    if target_id not in node_info:
        raise ValueError(
            f"Unknown target: {target_id}"
        )

    compound_name = node_info[
        compound_id
    ]["canonical_name"]

    target_name = node_info[
        target_id
    ]["canonical_name"]

    print("\n" + "=" * 70)
    print("AYURKOSH-GNN — GRAPH EXPLANATION")
    print("=" * 70)

    print(
        f"Compound : {compound_name}"
    )

    print(
        f"Target   : {target_name}"
    )

    paths = find_explanation_paths(
        graph,
        node_info,
        compound_id,
        target_id
    )

    print(
        f"\nSupporting paths found: {len(paths)}"
    )

    if not paths:

        print(
            "\nNo explicit supporting graph path "
            "was found between the compound and target."
        )

        print(
            "The prediction may therefore be driven "
            "primarily by learned graph representations."
        )

        return

    for i, path in enumerate(
        paths,
        start=1
    ):

        print(
            f"\nPath {i}"
        )

        print(
            f"Type      : {path['path_type']}"
        )

        print(
            f"Relations : {' → '.join(path['relations'])}"
        )

        print(
            f"Nodes     : {readable_path(path, node_info)}"
        )


if __name__ == "__main__":

    if len(sys.argv) != 3:

        print(
            "Usage:"
            " python explain_prediction.py "
            "<compound_id> <target_id>"
        )

        sys.exit(1)

    explain(
        sys.argv[1],
        sys.argv[2]
    )
