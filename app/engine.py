"""
AYURKOSH-GNN: Unified Knowledge Graph Inference & Evidence Retrieval Engine
============================================================================
Provides transparent, auditable inference, candidate ranking, ingredient classification,
and topological evidence retrieval for the AYURKOSH Expert Validation Console.

Loads the full 82-formulation labelled AyurKOSH dataset from results/phase2/compound_fold_assignment.csv.
"""

import os
import sys
import random
from collections import defaultdict
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

# Path resolution
APP_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(APP_DIR, ".."))
SRC_PHASE2 = os.path.join(ROOT_DIR, "src", "phase2")
if SRC_PHASE2 not in sys.path:
    sys.path.insert(0, SRC_PHASE2)

import train_rgcn
from build_hetero_data import build_fold_hetero_data

def _resolve_data_path(rel_primary, rel_fallback=None):
    p = os.path.join(ROOT_DIR, rel_primary)
    if os.path.exists(p):
        return p
    if rel_fallback:
        fb = os.path.join(ROOT_DIR, rel_fallback)
        if os.path.exists(fb):
            return fb
    return p

NODES_PATH = _resolve_data_path("results/phase1/nodes.csv", "data/nodes.csv")
EDGES_PATH = _resolve_data_path("results/phase1/edges.csv", "data/edges.csv")
FOLD_TREATS_PATH = _resolve_data_path("results/phase2/treats_edges_with_fold.csv", "treats_edges_with_fold.csv")
COMP_FOLDS_PATH = _resolve_data_path("results/phase2/compound_fold_assignment.csv", "compound_fold_assignment.csv")
MODELS_DIR = os.path.join(ROOT_DIR, "models")

def map_node_category(raw_type):
    """Maps raw node types to standard Ayurvedic research categories."""
    if raw_type == "Herb":
        return "Herb"
    elif raw_type == "Mineral":
        return "Mineral"
    elif raw_type == "Vehicle":
        return "Vehicle"
    elif raw_type == "Sub_Formulation":
        return "Sub-formulation"
    else:
        return "Unresolved"

class GNNInferenceEngine:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(GNNInferenceEngine, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        
        self.device = torch.device("cpu")
        print("[Engine] Loading AyurKOSH Knowledge Graph tables...")
        self.nodes_df = pd.read_csv(NODES_PATH)
        self.edges_df = pd.read_csv(EDGES_PATH)
        self.treats_df = pd.read_csv(FOLD_TREATS_PATH)
        
        # Load the official 82 labelled compounds with their cross-validation fold assignments
        self.labeled_compounds_df = pd.read_csv(COMP_FOLDS_PATH)
        self.labeled_compound_ids = self.labeled_compounds_df["compound_node_id"].tolist()
        self.compound_fold_map = dict(zip(self.labeled_compounds_df["compound_node_id"], self.labeled_compounds_df["fold"]))
        self.compound_degree_map = dict(zip(self.labeled_compounds_df["compound_node_id"], self.labeled_compounds_df["treats_degree"]))
        self.compound_bucket_map = dict(zip(self.labeled_compounds_df["compound_node_id"], self.labeled_compounds_df["degree_bucket"]))
        
        # Build quick lookups
        self.node_info = self.nodes_df.set_index("node_id").to_dict("index")
        self.node_id_to_type = {nid: info["node_type"] for nid, info in self.node_info.items()}
        self.node_id_to_canonical = {nid: info["canonical_name"] for nid, info in self.node_info.items()}
        self.node_id_to_raw = {nid: info.get("raw_name", info["canonical_name"]) for nid, info in self.node_info.items()}
        
        # Categorized ID lists
        self.disease_ids = self.nodes_df[self.nodes_df["node_type"] == "Disease"]["node_id"].tolist()
        self.symptom_ids = self.nodes_df[self.nodes_df["node_type"] == "Symptom"]["node_id"].tolist()
        self.herb_ids = self.nodes_df[self.nodes_df["node_type"] == "Herb"]["node_id"].tolist()
        self.mineral_ids = self.nodes_df[self.nodes_df["node_type"] == "Mineral"]["node_id"].tolist()
        
        # Pre-compute Compound Ingredients via CONTAINS relation
        self.contains_df = self.edges_df[self.edges_df["relation"] == "CONTAINS"]
        self.compound_ingredients = defaultdict(list)
        for _, r in self.contains_df.iterrows():
            self.compound_ingredients[r["source"]].append(r["target"])
            
        # Pre-compute Known Ground Truth Treatments
        self.known_treats = defaultdict(set)
        for _, r in self.treats_df.iterrows():
            self.known_treats[r["source"]].add(r["target"])
            
        # Pre-compute Herb -> Disease associations (TREATED_BY)
        self.treated_by_df = self.edges_df[self.edges_df["relation"] == "TREATED_BY"]
        self.disease_to_herbs = defaultdict(set)
        self.herb_to_diseases = defaultdict(set)
        for _, r in self.treated_by_df.iterrows():
            self.disease_to_herbs[r["source"]].add(r["target"])
            self.herb_to_diseases[r["target"]].add(r["source"])
                
        # Pre-compute Herb Qualities & Groups
        self.has_quality_df = self.edges_df[self.edges_df["relation"] == "HAS_QUALITY"]
        self.herb_to_qualities = defaultdict(list)
        for _, r in self.has_quality_df.iterrows():
            self.herb_to_qualities[r["source"]].append(r["target"])

        self.belongs_to_df = self.edges_df[self.edges_df["relation"] == "BELONGS_TO"]
        self.herb_to_groups = defaultdict(list)
        for _, r in self.belongs_to_df.iterrows():
            self.herb_to_groups[r["source"]].append(r["target"])

        self.has_lakshan_df = self.edges_df[self.edges_df["relation"] == "HAS_LAKSHAN"]
        self.disease_to_symptoms = defaultdict(set)
        self.symptom_to_diseases = defaultdict(set)
        for _, r in self.has_lakshan_df.iterrows():
            self.disease_to_symptoms[r["source"]].add(r["target"])
            self.symptom_to_diseases[r["target"]].add(r["source"])
            
        # Model cache per fold
        self.models = {}
        self.graphs = {}
        self.node_id_to_local_maps = {}
        self.node_id_to_type_maps = {}
        self.fold_embeddings = {}
        self.popularity_priors = {}
        
        self._load_fold(0)
        self._initialized = True
        print(f"[Engine] AYURKOSH GNN inference engine ready. Loaded {len(self.labeled_compound_ids)} labelled formulations.")

    def _load_fold(self, fold_num):
        if fold_num in self.models:
            return
            
        ckpt_path = os.path.join(MODELS_DIR, f"gnn_rgcn_fold{fold_num}.pt")
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(f"Model checkpoint not found: {ckpt_path}")
            
        ckpt = torch.load(ckpt_path, map_location="cpu")
        data, id_to_local, id_to_type, tr, te = build_fold_hetero_data(
            NODES_PATH, EDGES_PATH, FOLD_TREATS_PATH, fold_num, embed_dim=64
        )
        
        node_counts = {ntype: data[ntype].num_nodes for ntype in data.node_types}
        edge_types = list(data.edge_types)
        
        model = train_rgcn.HeteroRGCNModel(
            node_types=list(data.node_types),
            node_counts=node_counts,
            edge_types=edge_types,
            embed_dim=64,
            hidden_dim=64
        )
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()
        
        outer_edge_index_dict = {
            et: data[et].edge_index.clone() for et in data.edge_types
        }
        
        with torch.no_grad():
            node_embeddings = model(node_counts, outer_edge_index_dict)
            
        train_treats = self.treats_df[self.treats_df["fold"] != fold_num]
        pop_counts = train_treats["target"].value_counts().to_dict()
        
        self.models[fold_num] = model
        self.graphs[fold_num] = data
        self.node_id_to_local_maps[fold_num] = id_to_local
        self.node_id_to_type_maps[fold_num] = id_to_type
        self.fold_embeddings[fold_num] = node_embeddings
        self.popularity_priors[fold_num] = pop_counts

    def get_compounds_summary(self, only_labeled=True):
        """
        Returns structured summaries for all 82 labelled compounds in the AyurKOSH study.
        Alphabetically sorted by canonical name.
        """
        target_ids = self.labeled_compound_ids if only_labeled else self.nodes_df[self.nodes_df["node_type"] == "Compound"]["node_id"].tolist()
        compounds = []
        
        for cid in target_ids:
            cname = self.node_id_to_canonical.get(cid, cid)
            raw = self.node_id_to_raw.get(cid, cname)
            ingredients = self.compound_ingredients.get(cid, [])
            fold = self.compound_fold_map.get(cid, 0)
            treats_deg = self.compound_degree_map.get(cid, len(self.known_treats.get(cid, [])))
            degree_bkt = self.compound_bucket_map.get(cid, "Unknown")
            
            # Categorize actual ingredients
            cat_list = []
            cat_counts = defaultdict(int)
            for i in ingredients:
                raw_type = self.node_id_to_type.get(i, "Other_Unclassified")
                category = map_node_category(raw_type)
                cat_counts[category] += 1
                cat_list.append({
                    "node_id": i,
                    "canonical_name": self.node_id_to_canonical.get(i, i),
                    "category": category,
                    "raw_type": raw_type
                })
                
            treats = list(self.known_treats.get(cid, []))
            known_targets = []
            for t in treats:
                known_targets.append({
                    "target_id": t,
                    "canonical_name": self.node_id_to_canonical.get(t, t),
                    "target_type": self.node_id_to_type.get(t, "Disease")
                })

            compounds.append({
                "node_id": cid,
                "canonical_name": cname,
                "raw_name": raw,
                "fold": fold,
                "treats_degree": treats_deg,
                "degree_bucket": degree_bkt,
                "num_ingredients": len(ingredients),
                "categorized_ingredients": cat_list,
                "category_counts": dict(cat_counts),
                "num_known_treats": len(treats),
                "known_targets": known_targets
            })
            
        return sorted(compounds, key=lambda x: str(x["canonical_name"]))

    def get_benchmark_presets(self):
        """
        Returns only genuine classical benchmark formulations that actually exist in the 82 AyurKOSH corpus.
        """
        all_compounds = self.get_compounds_summary(only_labeled=True)
        # Canonical benchmark candidates present in the 82 labelled compounds
        benchmark_keywords = [
            "सितोपलादि",   # Sitopaladi Churna
            "हिंग्वाष्टक",  # Hingwashtak Churna
            "त्रिफळा",     # Sukshma Triphala
            "अगस्तीप्राश", # Agastya Prasha
            "संजीवनी",    # Sanjeevani Guti
            "श्वास कुठार",  # Shwasa Kuthara
            "अभयारिष्ट",   # Abhayarishta
            "सुवर्णसूतशेखर",# Suvarnasutshekhara
            "सुवर्णपर्पटी", # Suvarnaparpati
            "शतावरी घृत"   # Shatavari Ghrita
        ]
        
        benchmarks = []
        for comp in all_compounds:
            for kw in benchmark_keywords:
                if kw in comp["canonical_name"]:
                    benchmarks.append(comp)
                    break
        return benchmarks

    def predict(self, compound_id=None, custom_ingredient_ids=None, fold_num=None, alpha=0.75, target_type="All", top_k=10):
        """
        Executes link prediction over candidate universe using the formulation's held-out test fold model.
        Returns research-grade DataFrame with rank, model_score, known_in_source, and graph_evidence_count.
        """
        # Determine appropriate fold: use formulation's held-out test fold if available
        if fold_num is None:
            if compound_id and compound_id in self.compound_fold_map:
                fold_num = self.compound_fold_map[compound_id]
            else:
                fold_num = 0
                
        self._load_fold(fold_num)
        model = self.models[fold_num]
        embeddings = self.fold_embeddings[fold_num]
        id_to_local = self.node_id_to_local_maps[fold_num]
        pop_counts = self.popularity_priors[fold_num]
        
        # 1. Compound representation
        if compound_id and compound_id in id_to_local:
            comp_local = id_to_local[compound_id]
            h = embeddings["Compound"][comp_local]
            active_ingredients = self.compound_ingredients.get(compound_id, [])
        elif custom_ingredient_ids:
            herb_vecs = []
            active_ingredients = custom_ingredient_ids
            for hid in custom_ingredient_ids:
                if hid in id_to_local:
                    h_local = id_to_local[hid]
                    h_type = self.node_id_to_type.get(hid, "Herb")
                    if h_type in embeddings and h_local < embeddings[h_type].size(0):
                        herb_vecs.append(embeddings[h_type][h_local])
            if herb_vecs:
                h = torch.stack(herb_vecs).mean(dim=0)
            else:
                h = model.shared_compound_embedding[0]
        else:
            h = model.shared_compound_embedding[0]
            active_ingredients = []
            
        # 2. Candidate pool
        if target_type == "Disease":
            candidates = [(tid, "Disease") for tid in self.disease_ids]
        elif target_type == "Symptom":
            candidates = [(tid, "Symptom") for tid in self.symptom_ids]
        else:
            candidates = [(tid, "Disease") for tid in self.disease_ids] + [(tid, "Symptom") for tid in self.symptom_ids]
        
        # 3. Model Scoring
        raw_gnn_scores = []
        raw_pop_scores = []
        
        with torch.no_grad():
            r = model.decoder.r
            for tid, ttype in candidates:
                if tid in id_to_local:
                    t_local = id_to_local[tid]
                    if ttype in embeddings and t_local < embeddings[ttype].size(0):
                        t = embeddings[ttype][t_local]
                        score_gnn = float((h * r * t).sum().item())
                    else:
                        score_gnn = 0.0
                else:
                    score_gnn = 0.0
                raw_gnn_scores.append(score_gnn)
                raw_pop_scores.append(float(pop_counts.get(tid, 0)))
                
        # Rank-normalization to [0, 1]
        def _normalize(scores):
            scores = np.array(scores, dtype=float)
            n = len(scores)
            if n <= 1:
                return np.ones(n)
            unique_s = np.unique(scores)[::-1]
            if len(unique_s) == 1:
                return np.full(n, 0.5)
            norm = np.zeros(n, dtype=float)
            for sc in unique_s:
                idx = np.where(scores == sc)[0]
                better = np.sum(scores > sc)
                avg_rk = 1.0 + better + (len(idx) - 1) / 2.0
                norm[idx] = 1.0 - (avg_rk - 1.0) / (n - 1.0)
            return norm

        norm_gnn = _normalize(raw_gnn_scores)
        norm_pop = _normalize(raw_pop_scores)
        combined_scores = alpha * norm_gnn + (1.0 - alpha) * norm_pop
        
        records = []
        for idx, (tid, ttype) in enumerate(candidates):
            cname = self.node_id_to_canonical.get(tid, tid)
            is_known = False
            if compound_id:
                is_known = (tid in self.known_treats.get(compound_id, set()))
                
            # Evidence link count
            evi_count = 0
            if ttype == "Disease":
                d_herbs = self.disease_to_herbs.get(tid, set())
                evi_count = len(set(active_ingredients) & d_herbs)
            elif ttype == "Symptom":
                d_sources = self.symptom_to_diseases.get(tid, set())
                for d in d_sources:
                    evi_count += len(set(active_ingredients) & self.disease_to_herbs.get(d, set()))

            records.append({
                "target_id": tid,
                "target_name": cname,
                "target_type": ttype,
                "model_score": round(float(combined_scores[idx]), 4),
                "gnn_raw_score": round(float(raw_gnn_scores[idx]), 4),
                "known_in_source": is_known,
                "evidence_count": evi_count,
                "eval_fold": fold_num
            })
            
        df = pd.DataFrame(records)
        df = df.sort_values(by="model_score", ascending=False).reset_index(drop=True)
        df["rank"] = df.index + 1
        return df.head(top_k)

    def get_evidence_details(self, compound_id=None, custom_ingredient_ids=None, target_id=None):
        """
        Retrieves authentic, strictly grounded topological evidence from the knowledge graph.
        Never fabricates textual or clinical citations.
        """
        if not target_id:
            return {}
            
        if compound_id:
            active_ingredients = self.compound_ingredients.get(compound_id, [])
            formulation_name = self.node_id_to_canonical.get(compound_id, compound_id)
        elif custom_ingredient_ids:
            active_ingredients = custom_ingredient_ids
            formulation_name = "Custom Formulation"
        else:
            active_ingredients = []
            formulation_name = "Unknown"
            
        target_name = self.node_id_to_canonical.get(target_id, target_id)
        target_type = self.node_id_to_type.get(target_id, "Target")
        
        direct_ingredient_links = []
        multi_hop_paths = []
        shared_qualities = []
        shared_groups = []
        
        subgraph_nodes = [
            {"id": "FORMULATION", "label": formulation_name, "group": "formulation", "color": "#c29b38"}
        ]
        subgraph_edges = []
        added_nodes = set(["FORMULATION"])
        
        # Check active ingredients
        for ing in active_ingredients:
            ing_name = self.node_id_to_canonical.get(ing, ing)
            ing_cat = map_node_category(self.node_id_to_type.get(ing, "Herb"))
            
            if ing not in added_nodes:
                subgraph_nodes.append({
                    "id": ing,
                    "label": f"{ing_name}\n({ing_cat})",
                    "group": "ingredient",
                    "color": "#285238" if ing_cat == "Herb" else "#4a6b57"
                })
                added_nodes.add(ing)
            
            subgraph_edges.append({
                "from": "FORMULATION",
                "to": ing,
                "label": "CONTAINS",
                "title": f"{formulation_name} ──CONTAINS──> {ing_name}"
            })
            
            # Qualities & Groups
            quals = self.herb_to_qualities.get(ing, [])
            if quals:
                q_names = [self.node_id_to_canonical.get(q, q) for q in quals]
                shared_qualities.append({
                    "ingredient": ing_name,
                    "qualities": q_names[:4]
                })
                for q in quals[:2]:
                    q_name = self.node_id_to_canonical.get(q, q)
                    if q not in added_nodes:
                        subgraph_nodes.append({
                            "id": q,
                            "label": q_name,
                            "group": "quality",
                            "color": "#8c6b2d"
                        })
                        added_nodes.add(q)
                    subgraph_edges.append({
                        "from": ing,
                        "to": q,
                        "label": "HAS_QUALITY",
                        "title": f"{ing_name} ──HAS_QUALITY──> {q_name}"
                    })

            grps = self.herb_to_groups.get(ing, [])
            if grps:
                g_names = [self.node_id_to_canonical.get(g, g) for g in grps]
                shared_groups.append({
                    "ingredient": ing_name,
                    "groups": g_names[:3]
                })
            
            # Check TREATED_BY connection
            if target_type == "Disease":
                if ing in self.disease_to_herbs.get(target_id, set()):
                    direct_ingredient_links.append({
                        "ingredient": ing_name,
                        "category": ing_cat,
                        "relation": "TREATED_BY",
                        "target": target_name,
                        "description": f"AyurKOSH encodes: Disease '{target_name}' is treated by '{ing_name}'"
                    })
                    if target_id not in added_nodes:
                        subgraph_nodes.append({
                            "id": target_id,
                            "label": f"{target_name}\n(Disease)",
                            "group": "target",
                            "color": "#1e3a8a"
                        })
                        added_nodes.add(target_id)
                    subgraph_edges.append({
                        "from": target_id,
                        "to": ing,
                        "label": "TREATED_BY",
                        "title": f"{target_name} ──TREATED_BY──> {ing_name}"
                    })
                    
        # If Symptom: check 2-hop via Disease
        if target_type == "Symptom":
            connected_diseases = self.symptom_to_diseases.get(target_id, set())
            for d in connected_diseases:
                d_herbs = self.disease_to_herbs.get(d, set())
                d_name = self.node_id_to_canonical.get(d, d)
                shared_ings = set(active_ingredients) & d_herbs
                for ing in shared_ings:
                    ing_name = self.node_id_to_canonical.get(ing, ing)
                    multi_hop_paths.append({
                        "path": f"{formulation_name} ──[CONTAINS]──> {ing_name} <──[TREATED_BY]── {d_name} (Disease) ──[HAS_LAKSHAN]──> {target_name} (Symptom)",
                        "intermediate": d_name,
                        "relation_1": "TREATED_BY",
                        "relation_2": "HAS_LAKSHAN"
                    })
                    if d not in added_nodes:
                        subgraph_nodes.append({
                            "id": d,
                            "label": f"{d_name}\n(Disease)",
                            "group": "intermediate",
                            "color": "#475569"
                        })
                        added_nodes.add(d)
                    if target_id not in added_nodes:
                        subgraph_nodes.append({
                            "id": target_id,
                            "label": f"{target_name}\n(Symptom)",
                            "group": "target",
                            "color": "#7c2d12"
                        })
                        added_nodes.add(target_id)
                    subgraph_edges.append({
                        "from": d,
                        "to": ing,
                        "label": "TREATED_BY",
                        "title": f"{d_name} ──TREATED_BY──> {ing_name}"
                    })
                    subgraph_edges.append({
                        "from": d,
                        "to": target_id,
                        "label": "HAS_LAKSHAN",
                        "title": f"{d_name} ──HAS_LAKSHAN──> {target_name}"
                    })

        if target_id not in added_nodes:
            subgraph_nodes.append({
                "id": target_id,
                "label": f"{target_name}\n({target_type})",
                "group": "target",
                "color": "#1e3a8a" if target_type == "Disease" else "#7c2d12"
            })

        is_known = False
        if compound_id:
            is_known = (target_id in self.known_treats.get(compound_id, set()))

        return {
            "target_id": target_id,
            "target_name": target_name,
            "target_type": target_type,
            "is_known_in_source_data": is_known,
            "direct_ingredient_links": direct_ingredient_links,
            "multi_hop_paths": multi_hop_paths[:6],
            "shared_qualities": shared_qualities[:6],
            "shared_groups": shared_groups[:4],
            "classical_citation_notice": "No explicit textual citation is encoded in the source dataset for this edge. Graph evidence is derived from topological message passing across 22,329 relational edges.",
            "subgraph_nodes": subgraph_nodes,
            "subgraph_edges": subgraph_edges
        }

    def generate_validation_sample(self, sample_type="Top-5", batch_size=15, target_type="All"):
        """
        Samples a structured validation batch across the 82 formulations to support standardized review.
        """
        compounds = self.get_compounds_summary(only_labeled=True)
        sample_records = []
        
        rng = random.Random(42)
        shuffled_compounds = list(compounds)
        rng.shuffle(shuffled_compounds)

        for comp in shuffled_compounds:
            if len(sample_records) >= batch_size:
                break
            preds = self.predict(compound_id=comp["node_id"], target_type=target_type, top_k=10)
            if preds.empty:
                continue

            if sample_type == "Top-1":
                candidates = preds.head(1)
            elif sample_type == "Top-5":
                candidates = preds.head(5)
            elif sample_type == "Top-10":
                candidates = preds.head(10)
            elif sample_type == "Stratified":
                idx_pool = [0]
                if len(preds) > 3:
                    idx_pool.append(len(preds) // 2)
                if len(preds) > 8:
                    idx_pool.append(len(preds) - 1)
                candidates = preds.iloc[idx_pool]
            else:
                candidates = preds.sample(n=min(2, len(preds)), random_state=42)

            for _, row in candidates.iterrows():
                if len(sample_records) >= batch_size:
                    break
                sample_records.append({
                    "formulation_id": comp["node_id"],
                    "formulation_name": comp["canonical_name"],
                    "target_id": row["target_id"],
                    "target_name": row["target_name"],
                    "target_type": row["target_type"],
                    "model_rank": row["rank"],
                    "model_score": row["model_score"],
                    "known_in_source": row["known_in_source"],
                    "evidence_count": row["evidence_count"]
                })

        return pd.DataFrame(sample_records)
