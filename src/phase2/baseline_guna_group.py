"""
Phase 2 -- Rule-based Guna/Group baseline

No learning at all. For each (compound, candidate) pair, the score is
just the Jaccard overlap between:
  - the compound's herb-composition profile (union of Quality + Group
    node ids across its CONTAINS herbs)
  - the candidate disease/symptom's associated herb profile (union of
    Quality + Group node ids across herbs that TREATED_BY-link to it)

This is the mandatory baseline the project has required since Phase 0:
if the eventual GNN doesn't clear this by a real margin, that's a
legitimate, reportable finding -- not a failure to hide.

Scored using the SAME filtered-ranking protocol the GNN will use (rank
true_target among the filtered candidate pool), so baseline and GNN
numbers are directly comparable.

Run once per fold, then average the 5 folds' metrics for the final
reported baseline numbers (report variance across folds too -- 82
compounds is small enough that fold-to-fold variance matters).
"""

import sys

from collections import defaultdict

import pandas as pd

NODES_PATH = sys.argv[1] if len(sys.argv) > 1 else "..\\phase1\\nodes.csv"
EDGES_PATH = sys.argv[2] if len(sys.argv) > 2 else "..\\phase1\\edges.csv"
FOLD_NUM = int(sys.argv[3]) if len(sys.argv) > 3 else 0
EVAL_CANDIDATES_PATH = sys.argv[4] if len(sys.argv) > 4 else f"eval_candidates_fold{FOLD_NUM}.csv"



def jaccard(a, b):
    if not a and not b:
        return 0.0
    return len(a & b) / max(1, len(a | b))


def main():
    nodes = pd.read_csv(NODES_PATH)
    edges = pd.read_csv(EDGES_PATH)
    eval_cases = pd.read_csv(EVAL_CANDIDATES_PATH)

    all_disease_ids = nodes[nodes["node_type"] == "Disease"]["node_id"].tolist()
    all_symptom_ids = nodes[nodes["node_type"] == "Symptom"]["node_id"].tolist()

    contains = edges[edges["relation"] == "CONTAINS"]
    treated_by = edges[edges["relation"] == "TREATED_BY"]
    has_quality = edges[edges["relation"] == "HAS_QUALITY"]
    belongs_to = edges[edges["relation"] == "BELONGS_TO"]
    all_treats = edges[edges["relation"] == "TREATS"]

    herb_quality = defaultdict(set)
    for _, r in has_quality.iterrows():
        herb_quality[r["source"]].add(r["target"])
    herb_group = defaultdict(set)
    for _, r in belongs_to.iterrows():
        herb_group[r["source"]].add(r["target"])

    compound_herbs = defaultdict(set)
    for _, r in contains.iterrows():
        compound_herbs[r["source"]].add(r["target"])

    target_herbs = defaultdict(set)
    for _, r in treated_by.iterrows():
        target_herbs[r["target"]].add(r["source"])

    known_positive = defaultdict(set)
    for _, r in all_treats.iterrows():
        known_positive[r["source"]].add(r["target"])

    def profile(herb_ids):
        prof = set()
        for h in herb_ids:
            prof |= herb_quality.get(h, set())
            prof |= herb_group.get(h, set())
        return prof

    print("Pre-computing candidate profiles...")
    disease_profile = {d: profile(target_herbs.get(d, set())) for d in all_disease_ids}
    symptom_profile = {s: profile(target_herbs.get(s, set())) for s in all_symptom_ids}

    results = []
    for _, row in eval_cases.iterrows():
        compound_id = row["compound_id"]
        true_target = row["true_target"]
        target_type = row["target_type"]

        pool = all_disease_ids if target_type == "Disease" else all_symptom_ids
        cand_profile_map = disease_profile if target_type == "Disease" else symptom_profile

        candidates = [c for c in pool if c == true_target or c not in known_positive[compound_id]]
        comp_profile = profile(compound_herbs.get(compound_id, set()))

        scored = [(c, jaccard(comp_profile, cand_profile_map.get(c, set()))) for c in candidates]

        # ---------------------------------------------------------
        # Tie-aware ranking
        # ---------------------------------------------------------
        true_score = next(score for c, score in scored if c == true_target)

        # Number of candidates strictly better than the true target
        better = sum(score > true_score for c, score in scored if c != true_target)

        # Number of candidates tied with the true target
        equal = sum(
        score == true_score
        for c, score in scored
        if c != true_target
        )

        # Average rank among tied candidates
        rank = 1 + better + (0.5 * equal)

        results.append({
            "compound_id": compound_id,
            "true_target": true_target,
            "target_type": target_type,
            "rank": rank,
            "n_candidates": len(candidates),
            "reciprocal_rank": 1.0 / rank,
            "hit_at_1": int(rank <= 1),
            "hit_at_5": int(rank <= 5),
            "hit_at_10": int(rank <= 10),
        })

    results_df = pd.DataFrame(results)
    out_path = f"baseline_results_fold{FOLD_NUM}.csv"
    results_df.to_csv(out_path, index=False)

    print("\n" + "=" * 60)
    print(f"BASELINE RESULTS -- fold {FOLD_NUM}")
    print("=" * 60)
    print(f"Test cases: {len(results_df)}")
    print(f"MRR      : {results_df['reciprocal_rank'].mean():.4f}")
    print(f"Hits@1   : {results_df['hit_at_1'].mean():.4f}")
    print(f"Hits@5   : {results_df['hit_at_5'].mean():.4f}")
    print(f"Hits@10  : {results_df['hit_at_10'].mean():.4f}")
    print("\nBroken down by target_type:")
    print(results_df.groupby("target_type")[["reciprocal_rank", "hit_at_1", "hit_at_5", "hit_at_10"]].mean())
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()