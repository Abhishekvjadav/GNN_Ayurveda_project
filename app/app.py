"""
AYURKOSH-GNN: Expert Validation Console
=======================================
Structured Ayurvedic Expert Assessment & Evidence Review
An academic research validation workstation for independent domain evaluation
of model-retrieved formulation–disease/symptom associations.

KSCST 50th Series Student Project Proposal (2026–2027)
"""

import os
import sys
import streamlit as st
import pandas as pd

# Page Configuration
st.set_page_config(
    page_title="AYURKOSH Expert Validation Console",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Research-Grade Aesthetic
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Cinzel:wght@600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        color: #1a202c;
        background-color: #fcfbf9;
    }
    
    h1, h2, h3, .brand-title {
        font-family: 'Cinzel', Georgia, serif;
        color: #1b382b;
        letter-spacing: 0.3px;
    }
    
    /* Academic Card styling */
    .research-card {
        background-color: #ffffff;
        border: 1px solid #e2dfd7;
        border-radius: 8px;
        padding: 18px 22px;
        margin-bottom: 16px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.03);
    }
    
    /* Metrics styling */
    [data-testid="stMetricValue"] {
        font-size: 22px;
        font-weight: 700;
        color: #1b382b;
    }
    
    [data-testid="stMetricLabel"] {
        font-size: 12.5px;
        color: #4a5568;
        font-weight: 500;
    }
    
    /* Buttons */
    .stButton > button {
        border-radius: 6px;
        font-weight: 600;
        letter-spacing: 0.2px;
        transition: all 0.15s ease-in-out;
    }
    
    /* Tables */
    .dataframe {
        font-size: 13px;
    }
</style>
""", unsafe_allow_html=True)

# Imports
from expert_view import render_expert_view
from engine import GNNInferenceEngine
import validation_store
import auth

APP_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(APP_DIR, ".."))

def _render_records_tab(records, category_label, key_prefix):
    """Reusable records table + download block for one category."""
    if not records:
        st.info(f"No {category_label.lower()} assessments recorded yet.", icon="ℹ️")
        return

    df = pd.DataFrame(records)

    # Ensure reviewer_category column exists (backwards compat)
    if "reviewer_category" not in df.columns:
        df["reviewer_category"] = category_label

    c1, c2, c3 = st.columns([1, 1, 1])
    with c1:
        rev_filter = st.selectbox(f"Filter by Reviewer ID:", ["All Reviewers"] + list(df["reviewer_id"].unique()), key=f"{key_prefix}_rev")
    with c2:
        class_filter = st.selectbox("Filter by Classification:", ["All Classifications"] + list(df["q3_classification"].unique()), key=f"{key_prefix}_cls")
    with c3:
        type_filter = st.selectbox("Filter by Target Type:", ["All Types"] + list(df["target_type"].unique()), key=f"{key_prefix}_ttype")

    filtered = df.copy()
    if rev_filter != "All Reviewers":
        filtered = filtered[filtered["reviewer_id"] == rev_filter]
    if class_filter != "All Classifications":
        filtered = filtered[filtered["q3_classification"] == class_filter]
    if type_filter != "All Types":
        filtered = filtered[filtered["target_type"] == type_filter]

    st.markdown(f"##### {category_label} — {len(filtered)} of {len(df)} records:")

    display_cols = [c for c in [
        "record_id", "timestamp", "reviewer_id", "reviewer_category",
        "formulation_name", "target_name", "target_type",
        "model_rank", "model_score", "q1_relevance", "q2_evidence_support",
        "q3_classification", "q4_further_research", "q5_comments"
    ] if c in filtered.columns]
    st.dataframe(filtered[display_cols], use_container_width=True, hide_index=True)

    d1, d2 = st.columns(2)
    with d1:
        st.download_button(
            f"📥 Download {category_label} Records (CSV)",
            data=filtered.to_csv(index=False).encode("utf-8"),
            file_name=f"AYURKOSH_{category_label}_Validations.csv",
            mime="text/csv",
            use_container_width=True,
            key=f"{key_prefix}_csv"
        )
    with d2:
        cat_code = "Practitioner" if category_label == "Practitioner" else ("Student" if category_label == "Student" else None)
        rep_md = validation_store.generate_kscst_audit_report_markdown(category=cat_code)
        st.download_button(
            f"📥 Export {category_label} KSCST Audit Report (MD)",
            data=rep_md,
            file_name=f"AYURKOSH_{category_label}_Audit_Report.md",
            mime="text/markdown",
            use_container_width=True,
            key=f"{key_prefix}_md"
        )

def render_previous_reviews_view():
    if not auth.is_admin():
        st.error("🔒 **Access Restricted**: Assessment records and audit reports can only be viewed and downloaded by the Project Administrator.")
        st.info("If you are the Project Lead, open the **🔒 Admin Portal** in the sidebar to sign in via Google OAuth or Admin Passkey.")
        return

    st.html("""
        <div style="background:#ffffff;padding:18px 24px;border-radius:8px;border:1px solid #e2dfd7;border-left:5px solid #1b382b;margin-bottom:20px;">
            <h2 style="color:#1b382b;margin:0;font-size:22px;">Assessment Records</h2>
            <p style="color:#64748b;margin:4px 0 0 0;font-size:13px;">
                Expert and Student reviews are stored separately for audit integrity.
            </p>
        </div>
    """)

    p_records = validation_store.load_records_by_category("Expert")
    s_records = validation_store.load_records_by_category("Student")
    all_records = validation_store.load_all_records()

    # Summary badges
    b1, b2, b3 = st.columns(3)
    b1.metric("🩺 Expert Records", len(p_records))
    b2.metric("🎓 Student Records", len(s_records))
    b3.metric("📋 Total Records", len(all_records))

    st.markdown("---")

    tab_p, tab_s, tab_all = st.tabs([
        f"🩺 Expert ({len(p_records)})",
        f"🎓 Student ({len(s_records)})",
        f"📋 All Combined ({len(all_records)})"
    ])

    with tab_p:
        _render_records_tab(p_records, "Expert", "exp")

    with tab_s:
        _render_records_tab(s_records, "Student", "stu")

    with tab_all:
        _render_records_tab(all_records, "All", "all")


def render_validation_summary_view():
    if not auth.is_admin():
        st.error("🔒 **Access Restricted**: Validation summary metrics can only be viewed by the Project Administrator.")
        st.info("If you are the Project Lead, open the **🔒 Admin Portal** in the sidebar to authenticate.")
        return

    st.markdown("""
        <div style="background: #ffffff; padding: 18px 24px; border-radius: 8px; border: 1px solid #e2dfd7; border-left: 5px solid #1b382b; margin-bottom: 20px;">
            <h2 style="color: #1b382b; margin: 0; font-size: 22px;">Validation Summary & Metrics</h2>
            <p style="color: #64748b; margin: 4px 0 0 0; font-size: 13px;">
                Quantitative analysis of expert concordance, evidence support, and category breakdowns.
            </p>
        </div>
    """, unsafe_allow_html=True)

    metrics = validation_store.get_summary_metrics()
    
    if not metrics["has_data"]:
        st.info("No expert assessments recorded yet. Once Ayurvedic practitioners evaluate associations, comprehensive statistical distributions will appear here.", icon="ℹ️")
        return

    # Core Headline Metrics
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Total Reviewers", metrics["unique_experts"])
    m2.metric("Formulations", metrics["unique_formulations"])
    m3.metric("Associations", metrics["total_reviews"])
    m4.metric("Mean Relevance", f"{metrics['mean_relevance']} / 5.0")
    m5.metric("Mean Evidence", f"{metrics['mean_evidence_support']} / 5.0")

    st.markdown(f"**Inter-Rater Agreement Status**: `{metrics['agreement_status']}`")

    # Classification Breakdown
    st.markdown("---")
    st.markdown("#### Expert Classification Distribution")
    dist_cols = st.columns(len(metrics["classification_distribution"]))
    idx = 0
    for cls_name, count in metrics["classification_distribution"].items():
        pct = metrics["classification_percentages"].get(cls_name, 0.0)
        with dist_cols[idx % len(dist_cols)]:
            st.metric(cls_name, f"{count} ({pct}%)")
        idx += 1

    # Target Type Breakdown (Disease vs Symptom)
    st.markdown("---")
    t1, t2 = st.columns([1, 1])
    with t1:
        st.markdown("#### Disease vs Symptom Breakdown")
        ds_data = []
        for ttype, dt in metrics["disease_vs_symptom"].items():
            ds_data.append({
                "Target Category": ttype,
                "Count": dt["count"],
                "Proportion": f"{dt['pct']}%",
                "Mean Relevance": f"{dt['mean_relevance']} / 5.0" if dt['mean_relevance'] else "—",
                "Mean Evidence Support": f"{dt['mean_evidence']} / 5.0" if dt['mean_evidence'] else "—"
            })
        st.table(pd.DataFrame(ds_data))

    with t2:
        st.markdown("#### Retrieval Rank Tier Breakdown")
        tier_data = []
        for tier, dt in metrics["rank_tier_breakdown"].items():
            tier_data.append({
                "Rank Tier": tier,
                "Assessments": dt["count"],
                "Proportion": f"{dt['pct']}%",
                "Mean Relevance": f"{dt['mean_relevance']} / 5.0" if dt['mean_relevance'] else "—",
                "Mean Evidence": f"{dt['mean_evidence']} / 5.0" if dt['mean_evidence'] else "—"
            })
        st.table(pd.DataFrame(tier_data))

    # Research Sampling Utility (Section 14)
    st.markdown("---")
    st.markdown("#### 🔬 Research Validation Batch Generator")
    st.caption("Generate a structured sample of associations across the 82 formulations to provide an expert for systematic batch review.")
    
    sb_c1, sb_c2, sb_c3, sb_c4 = st.columns([1, 1, 1, 1])
    with sb_c1:
        samp_strategy = st.selectbox("Sampling Strategy:", ["Top-1 Predictions", "Top-5 Predictions", "Top-10 Predictions", "Stratified (Top/Mid/Lower)"])
    with sb_c2:
        samp_size = st.number_input("Batch Size (Total Associations):", min_value=5, max_value=50, value=15, step=5)
    with sb_c3:
        samp_type = st.selectbox("Target Condition:", ["All Types", "Disease Only", "Symptom Only"])
    with sb_c4:
        engine = GNNInferenceEngine()
        gen_btn = st.button("Generate Validation Batch", use_container_width=True)

    if gen_btn:
        s_code = "Top-1" if "Top-1" in samp_strategy else ("Top-5" if "Top-5" in samp_strategy else ("Top-10" if "Top-10" in samp_strategy else "Stratified"))
        t_code = "Disease" if "Disease" in samp_type else ("Symptom" if "Symptom" in samp_type else "All")
        batch_df = engine.generate_validation_sample(sample_type=s_code, batch_size=samp_size, target_type=t_code)
        st.markdown(f"##### Generated Research Batch ({len(batch_df)} items):")
        st.dataframe(batch_df[["formulation_name", "target_name", "target_type", "model_rank", "model_score", "known_in_source", "evidence_count"]], use_container_width=True)
        csv_batch = batch_df.to_csv(index=False).encode('utf-8')
        st.download_button("📥 Download Batch Template (CSV)", data=csv_batch, file_name="AYURKOSH_Validation_Batch_Template.csv", mime="text/csv")

def main():
    # Sidebar
    with st.sidebar:
        st.markdown("""
            <div style="text-align: center; padding: 10px 0 14px 0;">
                <span style="font-size: 38px;">🌿</span>
                <h3 class="brand-title" style="margin: 4px 0 0 0; color: #1b382b; font-size: 19px;">AYURKOSH</h3>
                <span style="color: #64748b; font-size: 12px; font-weight: 600; letter-spacing: 0.5px;">EXPERT VALIDATION CONSOLE</span>
            </div>
        """, unsafe_allow_html=True)

        if auth.is_admin():
            nav_options = [
                "🌿 Expert Review Workstation",
                "📋 Previous Assessment Records",
                "📊 Validation Summary & Metrics"
            ]
        else:
            nav_options = ["🌿 Expert Review Workstation"]

        selected_page = st.radio(
            "Navigation:",
            nav_options,
            index=0
        )

        # Admin login / status controls
        auth.render_admin_sidebar_controls()

        st.markdown("---")
        st.markdown("##### 🏛️ KSCST Project Evidence")
        st.caption("""
            • **Project**: AyurKOSH-GNN  
            • **Validation Type**: Ayurvedic Domain Expert Review  
            • **Programme**: KSCST 50th Series (2026–2027)  
            • **Methodology**: Structured 5-question multi-hop topological evidence review  
            • **Scope**: Non-clinical research assessment
        """)
        st.markdown("<small style='color:#718096; font-size:11px;'><em>Note: This validation module is designed to provide structured domain-expert evidence for the KSCST student project evaluation.</em></small>", unsafe_allow_html=True)

        st.markdown("---")
        st.markdown("##### ⚡ Graph Statistics")
        st.caption("• **Nodes**: 13,584 entities (11 types)\n• **Edges**: 22,329 relations (8 types)\n• **Labeled Compounds**: 82 formulations\n• **Evaluation**: 5-Fold Stratified Hold-Out")

    # Route view
    if selected_page == "🌿 Expert Review Workstation":
        render_expert_view()
    elif selected_page == "📋 Previous Assessment Records" and auth.is_admin():
        render_previous_reviews_view()
    elif selected_page == "📊 Validation Summary & Metrics" and auth.is_admin():
        render_validation_summary_view()
    else:
        render_expert_view()

if __name__ == "__main__":
    main()
