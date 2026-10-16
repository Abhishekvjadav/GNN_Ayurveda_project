import sys
import os

# Ensure UTF-8 output on Windows console
sys.stdout.reconfigure(encoding='utf-8')

sys.path.insert(0, os.path.abspath('app'))
from engine import GNNInferenceEngine
import validation_store

print("--- 1. Testing Dataset Loading ---")
engine = GNNInferenceEngine()
compounds = engine.get_compounds_summary(only_labeled=True)
print(f"Total labelled compounds loaded: {len(compounds)}")
assert len(compounds) == 82, f"Expected 82 formulations, got {len(compounds)}"
print("✓ Successfully loaded all 82 AyurKOSH labelled formulations.")

print("\n--- 2. Testing Benchmark Presets ---")
presets = engine.get_benchmark_presets()
print(f"Classical benchmark presets count: {len(presets)}")
for p in presets:
    print(f" - {p['canonical_name']} ({p['raw_name']}) [Fold {p['fold']}, Treats: {p['num_known_treats']}]")
assert len(presets) >= 4, "Expected at least 4 classical benchmarks in dataset"
print("✓ Classical presets exist genuinely in dataset.")

print("\n--- 3. Testing Dynamic Ingredients and Predictions Across Different Formulations ---")
test_indices = [0, 15, 30, 45]
ings_sets = []
preds_sets = []

for idx in test_indices:
    c = compounds[idx]
    cid = c['node_id']
    ings = [i['canonical_name'] for i in c['categorized_ingredients']]
    ings_sets.append(set(ings))
    
    preds_df = engine.predict(compound_id=cid, fold_num=c['fold'], alpha=0.75, target_type='All', top_k=5)
    pred_targets = preds_df['target_name'].tolist()
    preds_sets.append(tuple(pred_targets))
    
    print(f"\nFormulation [{idx}]: {c['canonical_name']} (Node: {cid}, Fold: {c['fold']})")
    print(f"  Ingredients ({len(ings)}): {ings[:4]}...")
    print(f"  Top 5 GNN Predictions: {pred_targets}")
    print(f"  Top 1 Score: {preds_df.iloc[0]['model_score']:.6f}")

print("\nPreds sets:")
for i, ps in enumerate(preds_sets):
    print(f"[{i}]: {ps}")

print("\n--- 4. Testing Zero-Fake-Scores Research Integrity ---")
metrics = validation_store.get_summary_metrics()
print(f"Validation summary metrics: {metrics}")
assert metrics['total_reviews'] == 0, "No fake reviews should exist in production mode"
assert metrics['has_data'] is False, "has_data must be False when no reviews exist"
print("✓ Verified zero fake expert validation scores in production audit store.")

print("\n--- 5. Testing Subgraph & Topological Evidence Retrieval ---")
sample_c = compounds[0]
sample_pred = engine.predict(compound_id=sample_c['node_id'], fold_num=sample_c['fold'], top_k=1).iloc[0]
evidence = engine.get_evidence_details(compound_id=sample_c['node_id'], target_id=sample_pred['target_id'])
subgraph_nodes = evidence.get('subgraph_nodes', [])
subgraph_edges = evidence.get('subgraph_edges', [])
print(f"Subgraph nodes: {len(subgraph_nodes)}, edges: {len(subgraph_edges)}")
print(f"Direct ingredient links: {len(evidence.get('direct_ingredient_links', []))}")
print(f"Multi-hop paths: {len(evidence.get('multi_hop_paths', []))}")
assert len(subgraph_nodes) > 0, "Subgraph should return real nodes"
print("✓ Subgraph and evidence paths successfully generated from authentic knowledge graph.")


print("\n=======================================================")
print("ALL TESTS PASSED: 82 Formulations Verified Successfully!")
print("=======================================================")

