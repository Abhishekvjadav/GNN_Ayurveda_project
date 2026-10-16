"""
AYURKOSH Expert Validation Store
================================
Maintains an immutable, transparent, and auditable repository of expert domain reviews.
Strictly adheres to research integrity:
- Zero fabrication of clinical evidence, classical citations, or fake agreement statistics.
- Distinguishes single-reviewer assessments from multi-reviewer inter-rater agreement.
- Records negative/disagreement decisions as first-class validation evidence.
"""

import os
import json
import uuid
import datetime
import pandas as pd
from collections import defaultdict

APP_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(APP_DIR, ".."))
VALIDATION_DIR = os.path.join(ROOT_DIR, "results", "clinical_validations")

# Separate stores for Experts and Students
JSON_PATH          = os.path.join(VALIDATION_DIR, "expert_validations.json")
CSV_PATH           = os.path.join(VALIDATION_DIR, "expert_validations.csv")
JSON_EXPERT        = os.path.join(VALIDATION_DIR, "expert_validations.json")
CSV_EXPERT         = os.path.join(VALIDATION_DIR, "expert_validations.csv")
JSON_PRACTITIONER  = os.path.join(VALIDATION_DIR, "practitioner_validations.json")
CSV_PRACTITIONER   = os.path.join(VALIDATION_DIR, "practitioner_validations.csv")
JSON_STUDENT       = os.path.join(VALIDATION_DIR, "student_validations.json")
CSV_STUDENT        = os.path.join(VALIDATION_DIR, "student_validations.csv")

CATEGORY_STUDENT      = "Student"
CATEGORY_EXPERT       = "Expert"
CATEGORY_PRACTITIONER = "Practitioner"

os.makedirs(VALIDATION_DIR, exist_ok=True)

def _load_json(path):
    """Safe JSON load — returns [] if file missing or corrupt."""
    if not os.path.exists(path):
        return []
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
            return data if isinstance(data, list) else []
    except Exception as e:
        print(f"[ValidationStore] Error reading {path}: {e}")
        return []

def _save_json_csv(path_json, path_csv, records):
    """Persist records to JSON and CSV."""
    with open(path_json, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2, ensure_ascii=False)
    try:
        pd.DataFrame(records).to_csv(path_csv, index=False, encoding="utf-8")
    except Exception as e:
        print(f"[ValidationStore] Warning saving CSV {path_csv}: {e}")

def load_all_records():
    """Loads ALL validation records (practitioner + student combined)."""
    p = _load_json(JSON_PRACTITIONER)
    s = _load_json(JSON_STUDENT)
    # Also include legacy records that pre-date the split
    legacy = _load_json(JSON_PATH)
    seen_ids = set(r["record_id"] for r in p + s if "record_id" in r)
    legacy_only = [r for r in legacy if r.get("record_id") not in seen_ids]
    return p + s + legacy_only

def load_records_by_category(category):
    """
    Load only Expert or Student records.
    category: 'Expert' | 'Practitioner' | 'Student'
    """
    if category in (CATEGORY_EXPERT, CATEGORY_PRACTITIONER):
        records = _load_json(JSON_EXPERT)
        if not records and os.path.exists(JSON_PRACTITIONER):
            records = _load_json(JSON_PRACTITIONER)
        return records
    elif category == CATEGORY_STUDENT:
        return _load_json(JSON_STUDENT)
    return load_all_records()

def save_assessment_record(
    reviewer_profile,
    formulation_id,
    formulation_name,
    target_id,
    target_name,
    target_type,
    model_rank,
    model_score,
    known_in_source,
    evidence_summary,
    q1_relevance,
    q2_evidence_support,
    q3_classification,
    q4_further_research,
    q5_comments,
    review_mode="Standard"
):
    """
    Appends a new expert assessment record with complete provenance and timestamp.
    Routes to the correct store based on reviewer_category (Student / Practitioner).
    """
    now = datetime.datetime.now()
    now_str = now.strftime("%Y-%m-%d %H:%M:%S")
    date_code = now.strftime("%Y%m%d")
    reviewer_category = reviewer_profile.get("reviewer_category", reviewer_profile.get("role", CATEGORY_EXPERT))
    if "student" in str(reviewer_category).lower():
        reviewer_category = CATEGORY_STUDENT
    else:
        reviewer_category = CATEGORY_EXPERT
    prefix = "STU" if reviewer_category == CATEGORY_STUDENT else "EXP"
    record_id = f"{prefix}-{date_code}-{uuid.uuid4().hex[:6].upper()}"

    record = {
        "record_id": record_id,
        "timestamp": now_str,
        "reviewer_category": reviewer_category,
        "reviewer_id": reviewer_profile.get("reviewer_id", "EXP-ANONYMOUS"),
        "reviewer_name": reviewer_profile.get("name", "Anonymous Reviewer"),
        "qualification": reviewer_profile.get("qualification", ""),
        "specialization": reviewer_profile.get("specialization", ""),
        "years_of_practice": reviewer_profile.get("years_of_practice", 0),
        "institution": reviewer_profile.get("institution", ""),
        "role": reviewer_profile.get("role", "Ayurvedic Expert"),
        "date_of_review": reviewer_profile.get("date_of_review", now.strftime("%Y-%m-%d")),
        "registration_id": reviewer_profile.get("registration_id", "Not provided"),
        "formulation_id": formulation_id,
        "formulation_name": formulation_name,
        "target_id": target_id,
        "target_name": target_name,
        "target_type": target_type,
        "model_rank": int(model_rank) if model_rank is not None else None,
        "model_score": float(model_score) if model_score is not None else None,
        "known_in_source_data": bool(known_in_source),
        "evidence_summary": evidence_summary,
        "q1_relevance": int(q1_relevance),
        "q2_evidence_support": int(q2_evidence_support),
        "q3_classification": str(q3_classification),
        "q4_further_research": str(q4_further_research),
        "q5_comments": str(q5_comments).strip(),
        "review_mode": str(review_mode)
    }

    # Route to the correct category store
    if reviewer_category == CATEGORY_STUDENT:
        records = _load_json(JSON_STUDENT)
        records.append(record)
        _save_json_csv(JSON_STUDENT, CSV_STUDENT, records)
    else:
        records = _load_json(JSON_EXPERT)
        records.append(record)
        _save_json_csv(JSON_EXPERT, CSV_EXPERT, records)
        _save_json_csv(JSON_PRACTITIONER, CSV_PRACTITIONER, records)

    # Also write to combined legacy store for backward compatibility
    all_records = load_all_records()
    _save_json_csv(JSON_PATH, CSV_PATH, all_records)

    return record_id

def get_summary_metrics(category=None):
    """
    category: None (all), 'Practitioner', or 'Student'
    """
    """
    Computes rigorous summary metrics over recorded assessments.
    Strictly reports zero or None when no data exists.
    Distinguishes single-reviewer from multi-reviewer inter-rater agreement.
    """
    records = load_records_by_category(category) if category else load_all_records()
    if not records:
        return {
            "has_data": False,
            "total_reviews": 0,
            "unique_experts": 0,
            "unique_formulations": 0,
            "unique_associations": 0,
            "mean_relevance": None,
            "mean_evidence_support": None,
            "classification_distribution": {},
            "classification_percentages": {},
            "further_research_counts": {},
            "disease_vs_symptom": {},
            "rank_tier_breakdown": {},
            "agreement_status": "No expert assessments recorded yet."
        }

    total_reviews = len(records)
    expert_ids = set(r.get("reviewer_id") for r in records if r.get("reviewer_id"))
    expert_names = set(r.get("reviewer_name") for r in records if r.get("reviewer_name"))
    unique_experts = max(len(expert_ids), len(expert_names))
    unique_formulations = len(set(r.get("formulation_id", r.get("formulation_name")) for r in records))
    
    # Association key = (formulation, target)
    assoc_keys = set((r.get("formulation_name"), r.get("target_name")) for r in records)
    unique_associations = len(assoc_keys)

    rel_scores = [r["q1_relevance"] for r in records if "q1_relevance" in r and r["q1_relevance"] is not None]
    evi_scores = [r["q2_evidence_support"] for r in records if "q2_evidence_support" in r and r["q2_evidence_support"] is not None]
    
    mean_rel = sum(rel_scores) / len(rel_scores) if rel_scores else 0.0
    mean_evi = sum(evi_scores) / len(evi_scores) if evi_scores else 0.0

    # Classifications
    class_counts = defaultdict(int)
    for r in records:
        c = r.get("q3_classification", "Unclassified")
        class_counts[c] += 1

    class_pct = {k: round((v / total_reviews) * 100.0, 1) for k, v in class_counts.items()}

    # Further research
    research_counts = defaultdict(int)
    for r in records:
        c = r.get("q4_further_research", "Unspecified")
        research_counts[c] += 1

    # Disease vs Symptom breakdown
    target_type_stats = defaultdict(lambda: {"count": 0, "rel_scores": [], "evi_scores": []})
    for r in records:
        ttype = r.get("target_type", "Unknown")
        target_type_stats[ttype]["count"] += 1
        if "q1_relevance" in r and r["q1_relevance"]:
            target_type_stats[ttype]["rel_scores"].append(r["q1_relevance"])
        if "q2_evidence_support" in r and r["q2_evidence_support"]:
            target_type_stats[ttype]["evi_scores"].append(r["q2_evidence_support"])

    disease_vs_symptom = {}
    for ttype, data in target_type_stats.items():
        disease_vs_symptom[ttype] = {
            "count": data["count"],
            "pct": round((data["count"] / total_reviews) * 100.0, 1),
            "mean_relevance": round(sum(data["rel_scores"]) / len(data["rel_scores"]), 2) if data["rel_scores"] else None,
            "mean_evidence": round(sum(data["evi_scores"]) / len(data["evi_scores"]), 2) if data["evi_scores"] else None
        }

    # Rank Tier breakdown
    rank_tiers = {"Top-1": [], "Top-2..5": [], "Top-6..10": [], "Lower (>10)": []}
    for r in records:
        rk = r.get("model_rank")
        if rk == 1:
            rank_tiers["Top-1"].append(r)
        elif rk and 2 <= rk <= 5:
            rank_tiers["Top-2..5"].append(r)
        elif rk and 6 <= rk <= 10:
            rank_tiers["Top-6..10"].append(r)
        else:
            rank_tiers["Lower (>10)"].append(r)

    rank_tier_breakdown = {}
    for tier, t_records in rank_tiers.items():
        if t_records:
            t_rel = [r["q1_relevance"] for r in t_records if "q1_relevance" in r]
            t_evi = [r["q2_evidence_support"] for r in t_records if "q2_evidence_support" in r]
            rank_tier_breakdown[tier] = {
                "count": len(t_records),
                "pct": round((len(t_records) / total_reviews) * 100.0, 1),
                "mean_relevance": round(sum(t_rel) / len(t_rel), 2) if t_rel else None,
                "mean_evidence": round(sum(t_evi) / len(t_evi), 2) if t_evi else None
            }

    # Inter-rater agreement logic
    if unique_experts <= 1:
        agreement_status = "Single-reviewer assessment (Multiple independent expert reviews required to compute inter-rater agreement statistics)."
    else:
        # Check if multiple experts evaluated the same (formulation, target) pair
        pair_reviews = defaultdict(list)
        for r in records:
            pair_reviews[(r.get("formulation_id"), r.get("target_id"))].append(r)
        
        shared_pairs = [pair for pair, revs in pair_reviews.items() if len(revs) > 1]
        if shared_pairs:
            # Simple concordance percentage across shared pairs
            concordant = 0
            for pair in shared_pairs:
                revs = pair_reviews[pair]
                classifs = [rv.get("q3_classification") for rv in revs]
                if len(set(classifs)) == 1:
                    concordant += 1
            rate = round((concordant / len(shared_pairs)) * 100.0, 1)
            agreement_status = f"Multi-reviewer concordance on overlapping associations: {rate}% across {len(shared_pairs)} shared evaluations."
        else:
            agreement_status = f"Multiple reviewers ({unique_experts} experts) recorded, but evaluations were on disjoint association sets."

    return {
        "has_data": True,
        "total_reviews": total_reviews,
        "unique_experts": unique_experts,
        "unique_formulations": unique_formulations,
        "unique_associations": unique_associations,
        "mean_relevance": round(mean_rel, 2),
        "mean_evidence_support": round(mean_evi, 2),
        "classification_distribution": dict(class_counts),
        "classification_percentages": class_pct,
        "further_research_counts": dict(research_counts),
        "disease_vs_symptom": disease_vs_symptom,
        "rank_tier_breakdown": rank_tier_breakdown,
        "agreement_status": agreement_status
    }

def _build_audit_md(records, metrics, category_label):
    """Shared helper that builds the audit markdown body."""
    now_str = datetime.datetime.now().strftime("%d %B %Y, %H:%M:%S")
    md = []
    md.append(f"# AYURKOSH-GNN: {category_label} Domain Validation Report")
    md.append("### Structured Ayurvedic Expert Assessment & Evidence Review")
    md.append(f"**Date Generated**: {now_str}")
    md.append(f"**Project**: AyurKOSH-GNN (Heterogeneous Knowledge Graph Learning for Ayurvedic Formulation–Disease and Symptom Association Retrieval)")
    md.append(f"**Project Period**: 2026–2027 (KSCST 50th Series Student Project Proposal)")
    md.append(f"**Reviewer Category**: {category_label}")
    md.append(f"**Validation Methodology**: Structured domain-expert assessment of model-retrieved formulation–disease/symptom associations.")
    md.append("\n> **Scope Statement**: This validation module supports transparent, structured domain-expert evidence for the KSCST student project evaluation. It does not establish clinical efficacy, clinical trials, or medical treatment recommendations.\n")
    md.append("---\n")

    if not metrics["has_data"]:
        md.append("## 1. Summary of Assessments")
        md.append(f"_No {category_label.lower()} assessments recorded yet in this evaluation store._")
        return "\n".join(md)

    md.append("## 1. Executive Assessment Metrics")
    md.append(f"- **Total Independent Assessments Recorded**: {metrics['total_reviews']}")
    md.append(f"- **Participating Reviewers**: {metrics['unique_experts']}")
    md.append(f"- **Distinct Formulations Evaluated**: {metrics['unique_formulations']}")
    md.append(f"- **Unique Associations Reviewed**: {metrics['unique_associations']}")
    md.append(f"- **Mean Ayurvedic Relevance**: {metrics['mean_relevance']} / 5.0")
    md.append(f"- **Mean Evidence Support**: {metrics['mean_evidence_support']} / 5.0")
    md.append(f"- **Agreement Status**: {metrics['agreement_status']}")
    md.append("\n### Classification Distribution:")
    for cls_name, count in metrics["classification_distribution"].items():
        pct = metrics["classification_percentages"].get(cls_name, 0.0)
        md.append(f"- **{cls_name}**: {count} ({pct}%)")

    md.append("\n---\n")
    md.append("## 2. Itemized Verification Records\n")
    for r in records:
        cat_tag = f" [{r.get('reviewer_category', 'Expert')}]" if r.get('reviewer_category') else ""
        md.append(f"### Record `{r['record_id']}`{cat_tag} • {r['formulation_name']} ──> {r['target_name']} ({r['target_type']})")
        md.append(f"- **Reviewer ID**: `{r['reviewer_id']}` ({r.get('role', 'Expert')}, {r.get('institution', 'Independent')})")
        md.append(f"- **Timestamp**: {r['timestamp']} | **Review Mode**: {r.get('review_mode', 'Standard')}")
        md.append(f"- **Model Retrieval Output**: Rank #{r.get('model_rank', 'N/A')}, Score = `{r.get('model_score', 'N/A')}`")
        md.append(f"- **Known in Source KG?**: {'Yes' if r.get('known_in_source_data') else 'No (Novel Graph Association)'}")
        md.append(f"- **Q1. Ayurvedic Relevance**: {r['q1_relevance']} / 5.0")
        md.append(f"- **Q2. Evidence Support**: {r['q2_evidence_support']} / 5.0")
        md.append(f"- **Q3. Expert Classification**: **{r['q3_classification']}**")
        md.append(f"- **Q4. Recommended for Further Research**: {r['q4_further_research']}")
        if r.get('q5_comments'):
            md.append(f"- **Reviewer Comments**: _{r['q5_comments']}_")
        md.append("")
    return "\n".join(md)

def generate_kscst_audit_report_markdown(category=None):
    """
    Generates official formatted KSCST audit markdown.
    category: None (all), 'Practitioner', or 'Student'
    """
    if category == CATEGORY_PRACTITIONER:
        records = _load_json(JSON_PRACTITIONER)
        metrics = get_summary_metrics(CATEGORY_PRACTITIONER)
        label = "Practitioner"
    elif category == CATEGORY_STUDENT:
        records = _load_json(JSON_STUDENT)
        metrics = get_summary_metrics(CATEGORY_STUDENT)
        label = "Student"
    else:
        records = load_all_records()
        metrics = get_summary_metrics()
        label = "Combined Expert"
    return _build_audit_md(records, metrics, label)
