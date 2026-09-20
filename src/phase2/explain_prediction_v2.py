
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

    return graph, node_info


def get_edge_relations(graph, source, target):

    relations = []

    if graph.has_edge(source, target):

        for _, _, data in graph.out_edges(
            source,
            data=True
        ):

            if _ == target:
                relations.append(
                    data["relation"]
                )

    return relations


def find_paths(
    graph,
    node_info,
    compound_id,
    target_id,
    max_paths=20
):

    found = []

    # Search simple paths up to 4 edges.
    # This captures heterogeneous structural routes
    # without exploring the entire graph.

    try:
        all_paths = nx.all_simple_paths(
            graph,
            source=compound_id,
            target=target_id,
            cutoff=4
        )
    except nx.NetworkXNoPath:
        return []

    for path in all_paths:

        if len(found) >= max_paths:
            break

        relations = []

        valid = True

        for i in range(len(path) - 1):

            edge_relations = get_edge_relations(
                graph,
                path[i],
                path[i + 1]
            )

            if not edge_relations:

                valid = False
                break

            relations.append(
                edge_relations[0]
            )

        if not valid:
            continue

        node_types = [
            node_info[node]["node_type"]
            for node in path
        ]

        found.append({
            "nodes": path,
            "node_types": node_types,
            "relations": relations
        })

    return found


def readable_path(path, node_info):

    names = []

    for node_id in path["nodes"]:

        info = node_info[node_id]

        name = info["canonical_name"]

        node_type = info["node_type"]

        names.append(
            f"{name} [{node_type}]"
        )

    return " → ".join(names)


def explain(compound_id, target_id):

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

    compound_type = node_info[
        compound_id
    ]["node_type"]

    target_type = node_info[
        target_id
    ]["node_type"]

    print("\n" + "=" * 70)
    print("AYURKOSH-GNN — MULTI-HOP GRAPH EXPLANATION")
    print("=" * 70)

    print(f"Compound : {compound_name}")
    print(f"Target   : {target_name}")
    print(f"Types    : {compound_type} → {target_type}")

    paths = find_paths(
        graph,
        node_info,
        compound_id,
        target_id
    )

    print(
        f"\nSupporting graph paths found: {len(paths)}"
    )

    if not paths:

        print(
            "\nNo graph path of length ≤ 4 was found."
        )

        print(
            "This does NOT mean the model has no structural "
            "information about the target."
        )

        print(
            "It means this explainer did not find an explicit "
            "path within the searched depth."
        )

        return

    for i, path in enumerate(
        paths,
        start=1
    ):

        print("\n" + "-" * 70)

        print(f"Path {i}")

        print(
            f"Relations : "
            f"{' → '.join(path['relations'])}"
        )

        print(
            f"Types     : "
            f"{' → '.join(path['node_types'])}"
        )

        print(
            f"Nodes     : "
            f"{readable_path(path, node_info)}"
        )


if __name__ == "__main__":

    if len(sys.argv) != 3:

        print(
            "Usage:"
            " python explain_prediction_v2.py "
            "<compound_id> <target_id>"
        )

        sys.exit(1)

    explain(
        sys.argv[1],
        sys.argv[2]
    )
