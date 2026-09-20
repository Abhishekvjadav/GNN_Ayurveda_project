"""
Phase 2A -- Graph Integrity Audit

Run this against your frozen Phase 1 nodes.csv / edges.csv BEFORE writing
any split logic or touching a model. It answers: is the graph itself
sound, independent of classification quality (which Phase 1 already
covers)?

Checks:
  1. Node ID uniqueness
  2. Every edge's source/target actually exists in nodes.csv
  3. Self-loops (source == target)
  4. Exact duplicate edges (source, target, relation)
  5. Relation direction sanity (spot-check a few relations resolve to
     the node types you expect on each side)
  6. Connected components (how fragmented is the graph, and how big
     is the largest component -- this matters a lot for message
     passing: a node in a tiny disconnected component gets no benefit
     from the GNN at all)
  7. Degree distribution per node type (flags supernodes that could
     dominate message passing, and near-zero-degree nodes that
     contribute little)

Does NOT touch classification correctness -- that's Phase 1's job.
This only asks whether the graph structure itself is well-formed.
"""

import sys
from pathlib import Path
from collections import defaultdict, Counter

import pandas as pd
import networkx as nx

NODES_PATH = sys.argv[1] if len(sys.argv) > 1 else "nodes.csv"
EDGES_PATH = sys.argv[2] if len(sys.argv) > 2 else "edges.csv"

# Expected (source_type, relation, target_type) triples, for the
# direction-sanity check. Add/adjust to match your actual schema.
EXPECTED_TRIPLES = {
    "TREATED_BY": [("Disease", "Herb"), ("Disease", "Mineral"),
                   ("Disease", "Vehicle"), ("Disease", "Sub_Formulation"),
                   ("Disease", "Other_Unclassified")],
    "HAS_LAKSHAN": [("Disease", "Symptom"), ("Ayurvedic_Concept", "Ayurvedic_Concept")],
    "ALTERNATE_OF": [("Herb", "Herb"), ("Herb", "Mineral"), ("Herb", "Vehicle"),
                      ("Herb", "Sub_Formulation"), ("Herb", "Other_Unclassified"),
                      ("Mineral", "Herb"), ("Vehicle", "Herb")],
    "HAS_QUALITY": [("Herb", "Quality"), ("Mineral", "Quality"), ("Vehicle", "Quality")],
    "BELONGS_TO": [("Herb", "Group"), ("Mineral", "Group"), ("Vehicle", "Group")],
    "HAS_KALP": [("Herb", "Sub_Formulation"), ("Mineral", "Sub_Formulation"),
                 ("Vehicle", "Sub_Formulation")],
    "CONTAINS": [("Compound", "Herb"), ("Compound", "Mineral"),
                 ("Compound", "Vehicle"), ("Compound", "Sub_Formulation"),
                 ("Compound", "Other_Unclassified")],
    "TREATS": [("Compound", "Disease"), ("Compound", "Symptom"),
               ("Compound", "Other_Unclassified")],
}


def main():
    print("=" * 70)
    print("PHASE 2A -- GRAPH INTEGRITY AUDIT")
    print("=" * 70)

    nodes = pd.read_csv(NODES_PATH)
    edges = pd.read_csv(EDGES_PATH)
    print(f"\nLoaded {len(nodes):,} nodes, {len(edges):,} edges")

    issues = []

    # ---------- 1. Node ID uniqueness ----------
    dup_ids = nodes["node_id"].duplicated().sum()
    print(f"\n[1] Duplicate node_id rows: {dup_ids}")
    if dup_ids:
        issues.append(f"{dup_ids} duplicate node_id rows in nodes.csv")

    # ---------- 2. Dangling edges ----------
    node_id_set = set(nodes["node_id"])
    missing_src = ~edges["source"].isin(node_id_set)
    missing_tgt = ~edges["target"].isin(node_id_set)
    n_missing_src = missing_src.sum()
    n_missing_tgt = missing_tgt.sum()
    print(f"[2] Edges with source not in nodes.csv: {n_missing_src}")
    print(f"    Edges with target not in nodes.csv: {n_missing_tgt}")
    if n_missing_src or n_missing_tgt:
        issues.append(f"{n_missing_src} dangling source edges, {n_missing_tgt} dangling target edges")
        print("    Sample dangling edges:")
        print(edges[missing_src | missing_tgt].head(5).to_string())

    # ---------- 3. Self-loops ----------
    self_loops = edges[edges["source"] == edges["target"]]
    print(f"\n[3] Self-loop edges (source == target): {len(self_loops)}")
    if len(self_loops):
        issues.append(f"{len(self_loops)} self-loop edges")
        print(self_loops["relation"].value_counts().to_string())

    # ---------- 4. Exact duplicate edges ----------
    dup_edges = edges.duplicated(subset=["source", "target", "relation"]).sum()
    print(f"\n[4] Exact duplicate (source, target, relation) rows: {dup_edges}")
    if dup_edges:
        issues.append(f"{dup_edges} duplicate edges still present -- dedup step may not have run")

    # ---------- 5. Relation direction sanity ----------
    print("\n[5] Relation direction sanity (source_type -> relation -> target_type):")
    node_type_lookup = dict(zip(nodes["node_id"], nodes["node_type"]))
    edges_typed = edges.copy()
    edges_typed["source_type"] = edges_typed["source"].map(node_type_lookup)
    edges_typed["target_type"] = edges_typed["target"].map(node_type_lookup)

    for relation, expected_pairs in EXPECTED_TRIPLES.items():
        sub = edges_typed[edges_typed["relation"] == relation]
        if sub.empty:
            print(f"    {relation:15s}: NOT FOUND in edges.csv")
            issues.append(f"Expected relation '{relation}' has zero edges")
            continue
        observed_pairs = set(zip(sub["source_type"], sub["target_type"]))
        unexpected = observed_pairs - set(expected_pairs)
        status = "OK" if not unexpected else "UNEXPECTED PAIRS"
        print(f"    {relation:15s}: {len(sub):>6,} edges, {status}")
        if unexpected:
            issues.append(f"Relation '{relation}' has unexpected (source_type, target_type) pairs: {unexpected}")
            print(f"        Unexpected: {unexpected}")

    # ---------- 6. Connected components ----------
    print("\n[6] Connected components:")
    G = nx.Graph()
    G.add_nodes_from(nodes["node_id"])
    G.add_edges_from(zip(edges["source"], edges["target"]))
    components = list(nx.connected_components(G))
    components.sort(key=len, reverse=True)
    print(f"    Total components: {len(components)}")
    print(f"    Largest component: {len(components[0]):,} nodes "
          f"({100 * len(components[0]) / len(nodes):.1f}% of all nodes)")
    if len(components) > 1:
        sizes = Counter(len(c) for c in components[1:])
        print(f"    Remaining {len(components) - 1} components, size distribution (top 10):")
        for size, count in sorted(sizes.items(), reverse=True)[:10]:
            print(f"      size {size}: {count} component(s)")
        small_frac = sum(len(c) for c in components[1:]) / len(nodes)
        if small_frac > 0.02:
            issues.append(f"{100*small_frac:.1f}% of nodes sit outside the largest component -- "
                          f"these get no benefit from message passing with the main graph")

    # ---------- 7. Degree distribution per node type ----------
    print("\n[7] Degree distribution by node type:")
    degree = defaultdict(int)
    for s, t in zip(edges["source"], edges["target"]):
        degree[s] += 1
        degree[t] += 1
    nodes["degree"] = nodes["node_id"].map(degree).fillna(0).astype(int)

    for ntype, grp in nodes.groupby("node_type"):
        zero_deg = (grp["degree"] == 0).sum()
        print(f"    {ntype:20s}: n={len(grp):>6,}  mean_deg={grp['degree'].mean():6.1f}  "
              f"max_deg={grp['degree'].max():>6}  zero_deg={zero_deg}")
        if zero_deg:
            issues.append(f"{zero_deg} '{ntype}' nodes have zero degree (isolated within their own type)")

    # ---------- Summary ----------
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    if not issues:
        print("No structural issues found. Graph is well-formed.")
    else:
        print(f"{len(issues)} issue(s) found:")
        for i, issue in enumerate(issues, 1):
            print(f"  {i}. {issue}")

    nodes[["node_id", "node_type", "degree"]].to_csv("phase2a_node_degrees.csv", index=False)
    print("\nWrote phase2a_node_degrees.csv")


if __name__ == "__main__":
    main()
