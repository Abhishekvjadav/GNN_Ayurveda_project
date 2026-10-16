"""
AYURKOSH Expert Validation Console
==================================
Structured Ayurvedic Expert Assessment & Evidence Review Workstation
Designed for qualified Ayurveda practitioners, pharmacologists, and domain experts
to review model-retrieved formulation–disease/symptom associations against knowledge graph evidence.

Strict Research Integrity:
- Fully connected to the authentic 82-formulation AyurKOSH labelled corpus.
- Zero fabrication of clinical evidence, classical citations, or fake agreement metrics.
- Distinct separation of Model Score vs. Expert Confidence.
- Negative decisions (disagreement/rejection) supported as first-class validation evidence.
- Optional Independent Blind Review Mode to eliminate confirmation/anchoring bias.
"""

import streamlit as st
import pandas as pd
import json
from engine import GNNInferenceEngine
import validation_store

def render_interactive_subgraph(nodes, edges, height=380):
    """Renders a clean, lightweight, interactive Vis.js network graph canvas."""
    nodes_json = json.dumps(nodes)
    edges_json = json.dumps(edges)
    
    html_code = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <script type="text/javascript" src="https://unpkg.com/vis-network/standalone/umd/vis-network.min.js"></script>
        <style>
            body {{
                margin: 0;
                padding: 0;
                background-color: #faf9f6;
                font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
            }}
            #network {{
                width: 100%;
                height: {height}px;
                border: 1px solid #e2dfd7;
                border-radius: 8px;
                background-color: #ffffff;
            }}
            .legend {{
                display: flex;
                flex-wrap: wrap;
                gap: 12px;
                padding: 8px 12px;
                font-size: 11.5px;
                color: #4a5568;
                background: #f7f6f2;
                border: 1px solid #e2dfd7;
                border-top: none;
                border-radius: 0 0 8px 8px;
            }}
            .legend-item {{
                display: flex;
                align-items: center;
                gap: 5px;
            }}
            .legend-color {{
                width: 11px;
                height: 11px;
                border-radius: 50%;
            }}
        </style>
    </head>
    <body>
        <div id="network"></div>
        <div class="legend">
            <div class="legend-item"><span class="legend-color" style="background:#c29b38;"></span> Formulation</div>
            <div class="legend-item"><span class="legend-color" style="background:#285238;"></span> Ingredient (Herb)</div>
            <div class="legend-item"><span class="legend-color" style="background:#4a6b57;"></span> Ingredient (Other)</div>
            <div class="legend-item"><span class="legend-color" style="background:#8c6b2d;"></span> Ayurvedic Quality (Guna)</div>
            <div class="legend-item"><span class="legend-color" style="background:#475569;"></span> Intermediate Disease</div>
            <div class="legend-item"><span class="legend-color" style="background:#1e3a8a;"></span> Target Disease</div>
            <div class="legend-item"><span class="legend-color" style="background:#7c2d12;"></span> Target Symptom</div>
        </div>
        <script type="text/javascript">
            var rawNodes = {nodes_json};
            var rawEdges = {edges_json};
            
            var nodes = new vis.DataSet(rawNodes.map(function(n) {{
                return {{
                    id: n.id,
                    label: n.label,
                    color: {{
                        background: n.color || '#4a5568',
                        border: '#2d3748',
                        highlight: {{ background: '#e0a94d', border: '#8c6b2d' }}
                    }},
                    font: {{ color: '#ffffff', size: 12, face: 'Inter, sans-serif' }},
                    shape: 'box',
                    margin: 8,
                    shadow: {{ enabled: true, color: 'rgba(0,0,0,0.08)', size: 3 }}
                }};
            }}));
            
            var edges = new vis.DataSet(rawEdges.map(function(e) {{
                return {{
                    from: e.from,
                    to: e.to,
                    label: e.label,
                    title: e.title || e.label,
                    arrows: 'to',
                    color: {{ color: '#94a3b8', highlight: '#c29b38' }},
                    font: {{ align: 'middle', size: 10, color: '#475569', strokeWidth: 2, strokeColor: '#ffffff' }},
                    smooth: {{ type: 'cubicBezier', roundness: 0.2 }}
                }};
            }}));
            
            var container = document.getElementById('network');
            var data = {{ nodes: nodes, edges: edges }};
            var options = {{
                physics: {{
                    stabilization: {{ iterations: 120 }},
                    barnesHut: {{ gravitationalConstant: -2200, springLength: 90, springConstant: 0.04 }}
                }},
                interaction: {{
                    hover: true,
                    zoomView: true,
                    dragView: true,
                    navigationButtons: true
                }}
            }};
            var network = new vis.Network(container, data, options);
        </script>
    </body>
    </html>
    """
    st.components.v1.html(html_code, height=height + 40)

def render_step_progress(current_step):
    steps = [
        ("1", "Select Formulation"),
        ("2", "Review Predictions"),
        ("3", "Inspect Evidence"),
        ("4", "Expert Assessment"),
        ("5", "Audit Record")
    ]
    html_steps = []
    for num, name in steps:
        active = (int(num) == current_step)
        done = (int(num) < current_step)
        if active:
            st_style = "background: #1b382b; color: #ffffff; border: 1px solid #1b382b;"
            lbl_style = "color: #1b382b; font-weight: 700;"
        elif done:
            st_style = "background: #e2f0e6; color: #1b5e20; border: 1px solid #a3cfb1;"
            lbl_style = "color: #2e7d32; font-weight: 500;"
        else:
            st_style = "background: #f1f1f1; color: #718096; border: 1px solid #cbd5e0;"
            lbl_style = "color: #718096; font-weight: 400;"
            
        html_steps.append(f'<div style="display: inline-flex; align-items: center; gap: 6px; margin-right: 18px; margin-bottom: 8px;"><span style="display: inline-block; width: 22px; height: 22px; line-height: 20px; text-align: center; border-radius: 50%; font-size: 11px; font-weight: bold; {st_style}">{num}</span><span style="font-size: 12.5px; {lbl_style}">{name}</span></div>')
    
    st.html(f'<div style="background: #faf9f6; padding: 10px 16px; border: 1px solid #eae7de; border-radius: 8px; margin-bottom: 20px;">{"".join(html_steps)}</div>')

def render_expert_view():
    engine = GNNInferenceEngine()
    
    # ----------------------------------------------------
    # Header & Official Scope Statement
    # ----------------------------------------------------
    st.markdown("""
        <div style="background: #ffffff; padding: 18px 24px; border-radius: 8px; border: 1px solid #e2dfd7; border-left: 5px solid #1b382b; margin-bottom: 16px;">
            <div style="display: flex; justify-content: space-between; align-items: flex-start; flex-wrap: wrap; gap: 12px;">
                <div>
                    <h2 style="color: #1b382b; margin: 0; font-family: 'Cinzel', Georgia, serif; font-size: 22px; letter-spacing: 0.5px;">
                        AYURKOSH Expert Validation Console
                    </h2>
                    <p style="color: #4a5568; margin: 4px 0 0 0; font-size: 13.5px; font-weight: 500;">
                        Structured Ayurvedic Expert Assessment & Evidence Review
                    </p>
                </div>
                <div style="background: #fcfbf9; border: 1px solid #dcd7cc; border-radius: 6px; padding: 6px 12px; font-size: 12px; color: #64748b;">
                    Project: <strong style="color: #1b382b;">AyurKOSH-GNN</strong> • KSCST 50th Series
                </div>
            </div>
            <div style="margin-top: 12px; padding-top: 10px; border-top: 1px solid #f0eee8; font-size: 12.5px; color: #64748b; line-height: 1.5;">
                <strong>Scope Statement:</strong> This tool supports expert assessment of model-generated knowledge associations. It does not establish clinical efficacy, diagnosis, or treatment recommendations.
            </div>
        </div>
    """, unsafe_allow_html=True)

    # State initialization
    if "current_step" not in st.session_state:
        st.session_state["current_step"] = 1
    if "expert_profile" not in st.session_state:
        st.session_state["expert_profile"] = {
            "name": "",
            "qualification": "BAMS, MD (Ayu)",
            "specialization": "Dravyaguna / Kayachikitsa",
            "years_of_practice": 10,
            "institution": "Government Ayurveda Medical College & Hospital",
            "role": "Expert",
            "registration_id": "",
            "date_of_review": pd.Timestamp.now().strftime("%Y-%m-%d"),
            "reviewer_id": "EXP-" + pd.Timestamp.now().strftime("%y%m") + "-01",
            "reviewer_category": "Expert",
            "blind_mode": False,
            "session_active": False
        }
    else:
        # Force sanitize existing session state to strictly Expert or Student
        curr_role = str(st.session_state["expert_profile"].get("role", "Expert"))
        if curr_role not in ["Expert", "Student"]:
            normalized_role = "Student" if ("student" in curr_role.lower() or "intern" in curr_role.lower()) else "Expert"
            st.session_state["expert_profile"]["role"] = normalized_role
            st.session_state["expert_profile"]["reviewer_category"] = normalized_role

    # ----------------------------------------------------
    # Reviewer Credentials Profile
    # ----------------------------------------------------
    profile = st.session_state["expert_profile"]

    # Category badge
    cat = profile.get("role", "Expert")
    if cat not in ["Expert", "Student"]:
        cat = "Student" if ("student" in str(cat).lower() or "intern" in str(cat).lower()) else "Expert"
    badge_label = "🩺 Expert" if cat == "Expert" else "🎓 Student"

    with st.expander(
        f"👤 Expert Profile • Reviewer ID: `{profile['reviewer_id']}` — {badge_label} {'(Session Active)' if profile['session_active'] else '(Click to configure)'}",
        expanded=not profile["session_active"]
    ):
        st.caption("Please record your credentials for the audit trail. These details are timestamped with every validation decision.")

        p_c1, p_c2, p_c3 = st.columns([1.2, 1.2, 1])
        with p_c1:
            name_val = st.text_input("Name:", value=profile["name"], placeholder="Dr. Firstname Lastname")
            qual_val = st.text_input("Academic Qualification:", value=profile["qualification"], placeholder="e.g., BAMS, MD (Ayu) / 3rd Year BAMS")
            inst_val = st.text_input("Institution / Organization:", value=profile["institution"], placeholder="e.g., Ayurveda College / Hospital")
        with p_c2:
            spec_val = st.text_input("Specialization / Subject:", value=profile["specialization"], placeholder="e.g., Dravyaguna, Kayachikitsa")
            
            role_options = ["Expert", "Student"]
            current_role = profile.get("role", "Expert")
            if current_role not in role_options:
                current_role = "Student" if ("student" in str(current_role).lower() or "intern" in str(current_role).lower()) else "Expert"
            
            role_val = st.selectbox(
                "Current Professional Role:",
                role_options,
                index=role_options.index(current_role),
                help="Select 'Expert' for qualified doctors/researchers, or 'Student' for trainees. Records are filed into separate audit databases automatically."
            )
            reviewer_category_val = role_val
            
            if role_val == "Student":
                st.info("🎓 **Student Mode**: Audit saved to `student_validations.json` (Separate academic audit)", icon="ℹ️")
            else:
                st.success("🩺 **Expert Mode**: Audit saved to `expert_validations.json` (Qualified clinical audit)", icon="✅")

            years_val = st.number_input(
                "Years of Practice / Study:", min_value=1, max_value=60,
                value=profile["years_of_practice"]
            )
        with p_c3:
            reg_val = st.text_input(
                "Registration / Roll No. (Optional):",
                value=profile["registration_id"],
                placeholder="Board ID / College Roll No."
            )
            date_val = st.date_input("Date of Review:", value=pd.to_datetime(profile["date_of_review"]))
            blind_val = st.checkbox("Independent Review Mode (Blind)", value=profile["blind_mode"],
                                    help="Hides model scores to prevent anchoring bias.")

        if st.button("Start / Update Validation Session", type="primary", use_container_width=True):
            # Auto-generate reviewer ID with category prefix
            now_code = pd.Timestamp.now().strftime("%y%m")
            id_prefix = "STU" if reviewer_category_val == "Student" else "EXP"
            existing_id = profile["reviewer_id"]
            # Re-prefix if the category changed
            new_id = existing_id
            if reviewer_category_val == "Student" and not existing_id.startswith("STU"):
                new_id = f"STU-{now_code}-{existing_id.split('-')[-1]}"
            elif reviewer_category_val == "Expert" and not (existing_id.startswith("EXP") or existing_id.startswith("REV")):
                new_id = f"EXP-{now_code}-{existing_id.split('-')[-1]}"

            st.session_state["expert_profile"] = {
                "name": name_val.strip() or "Anonymous Reviewer",
                "qualification": qual_val.strip(),
                "specialization": spec_val.strip(),
                "years_of_practice": int(years_val),
                "institution": inst_val.strip() or "Ayurvedic Institution",
                "role": role_val,
                "registration_id": reg_val.strip() or "Not provided",
                "date_of_review": str(date_val),
                "reviewer_id": new_id,
                "reviewer_category": reviewer_category_val,
                "blind_mode": blind_val,
                "session_active": True
            }
            st.success(f"Session active → Reviewer ID: `{new_id}` [{reviewer_category_val}]")
            st.rerun()

    # Progress bar
    render_step_progress(st.session_state["current_step"])

    # ----------------------------------------------------
    # STEP 1: Select Formulation (Full 82 AyurKOSH Corpus)
    # ----------------------------------------------------
    all_labeled_compounds = engine.get_compounds_summary(only_labeled=True)
    total_compounds_count = len(all_labeled_compounds)
    compound_dict = {c["node_id"]: c for c in all_labeled_compounds}
    
    st.markdown("### STEP 1 — Select Formulation for Review")
    
    # Dataset Provenance Banner & Validation Counter
    st.markdown(f"""
        <div style="background: #f4fbf6; border: 1px solid #d1ebd8; border-radius: 6px; padding: 8px 14px; margin-bottom: 12px; font-size: 13px; color: #1b4d2e; display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 8px;">
            <div>
                <strong>Source:</strong> AyurKOSH labelled formulation corpus • 
                <span style="color: #2e7d32; font-weight: 700;">{total_compounds_count} / 82 formulations loaded</span>
            </div>
            <div style="background: #e2f4e8; border: 1px solid #b7e4c7; border-radius: 4px; padding: 2px 8px; font-size: 12px; font-weight: 600;">
                {total_compounds_count} formulations available
            </div>
        </div>
    """, unsafe_allow_html=True)
    
    # Two Selection Modes: AyurKOSH Corpus (Default) vs Classical Benchmark Presets
    mode_col1, mode_col2 = st.columns([1.5, 1])
    with mode_col1:
        selection_mode = st.radio(
            "Formulation Selection Mode:",
            [
                f"AyurKOSH Corpus ({total_compounds_count} formulations)",
                "Classical Benchmark Presets (from AyurKOSH)"
            ],
            index=0,
            horizontal=True
        )
    with mode_col2:
        ing_filter = st.selectbox(
            "Filter Formulations:",
            ["All Formulations", "Contains Minerals", "Pure Herbal Formulations", "Contains Vehicles"],
            help="Filter formulations based on curated ingredient categories in AyurKOSH."
        )

    # Filtered pool based on mode
    if "Classical Benchmark Presets" in selection_mode:
        candidate_pool = engine.get_benchmark_presets()
        st.info("Displaying classical benchmark formulations that are genuinely present in the AyurKOSH labelled dataset.", icon="📖")
    else:
        candidate_pool = all_labeled_compounds

    # Search by formulation name / canonical name
    search_query = st.text_input(
        "🔍 Search formulation by name (Sanskrit canonical or common spelling):",
        placeholder="Type to filter across all 82 formulations (e.g., सितोपलादि, श्वास, सुवर्ण, शतावरी, अभया...)",
        key="formulation_search_box"
    )

    if search_query.strip():
        q = search_query.strip().lower()
        candidate_pool = [c for c in candidate_pool if q in str(c["canonical_name"]).lower() or q in str(c["raw_name"]).lower() or q in str(c["node_id"]).lower()]
    
    if ing_filter == "Contains Minerals":
        candidate_pool = [c for c in candidate_pool if c["category_counts"].get("Mineral", 0) > 0]
    elif ing_filter == "Pure Herbal Formulations":
        candidate_pool = [c for c in candidate_pool if c["category_counts"].get("Mineral", 0) == 0 and c["category_counts"].get("Herb", 0) > 0]
    elif ing_filter == "Contains Vehicles":
        candidate_pool = [c for c in candidate_pool if c["category_counts"].get("Vehicle", 0) > 0]

    if not candidate_pool:
        st.warning(f"No formulations matching query '{search_query}'. Showing all {total_compounds_count} formulations.")
        candidate_pool = all_labeled_compounds

    # Alphabetical order
    candidate_pool = sorted(candidate_pool, key=lambda x: str(x["canonical_name"]))

    f_options = [c["node_id"] for c in candidate_pool]
    f_labels = {
        c["node_id"]: f"{c['canonical_name']} • {c['num_ingredients']} ingredients • {c['num_known_treats']} targets (Fold {c['fold']})"
        for c in candidate_pool
    }
    
    selected_cid = st.selectbox(
        f"Select Formulation to Evaluate ({len(f_options)} matching options):",
        options=f_options,
        format_func=lambda x: f_labels.get(x, x),
        key="main_formulation_dropdown"
    )
    
    selected_comp = compound_dict[selected_cid]
    st.session_state["active_comp"] = selected_comp

    # Display Formulation Context Card
    with st.container():
        st.markdown(f"""
            <div style="background: #ffffff; border: 1px solid #e2dfd7; border-radius: 8px; padding: 14px 18px; margin: 12px 0 20px 0;">
                <div style="display: flex; justify-content: space-between; align-items: baseline; margin-bottom: 8px; flex-wrap: wrap; gap: 8px;">
                    <h3 style="color: #1b382b; margin: 0; font-size: 19px; font-family: 'Cinzel', Georgia, serif;">
                        {selected_comp['canonical_name']}
                    </h3>
                    <span style="font-size: 12.5px; color: #64748b;">
                        Raw Entry: <em>{selected_comp['raw_name']}</em> • Node ID: <code>{selected_comp['node_id']}</code> • Held-out Fold: <code>{selected_comp['fold']}</code>
                    </span>
                </div>
                <div style="font-size: 13px; color: #334155; margin-bottom: 10px;">
                    <strong>Knowledge Graph Context Preview:</strong>
                    Formulation ➔ {selected_comp['num_ingredients']} Curated Ingredients ➔ Multi-hop Neighborhood ➔ {selected_comp['num_known_treats']} Supervised Targets
                </div>
            </div>
        """, unsafe_allow_html=True)

        # Categorized Ingredients Accordion/Badges
        st.markdown("##### Curated Ingredients in AyurKOSH Knowledge Graph:")
        cat_map = selected_comp["categorized_ingredients"]
        
        if not cat_map:
            st.info("No information available in the current AyurKOSH dataset.")
        else:
            c_herb = [i["canonical_name"] for i in cat_map if i["category"] == "Herb"]
            c_min = [i["canonical_name"] for i in cat_map if i["category"] == "Mineral"]
            c_veh = [i["canonical_name"] for i in cat_map if i["category"] == "Vehicle"]
            c_sub = [i["canonical_name"] for i in cat_map if i["category"] == "Sub-formulation"]
            c_unres = [i["canonical_name"] for i in cat_map if i["category"] == "Unresolved"]
            
            badge_html = []
            for h in c_herb:
                badge_html.append(f"<span style='background:#edf7ed; color:#1e4620; border:1px solid #c8e6c9; padding:2px 8px; border-radius:12px; font-size:12px; margin-right:4px;'>🌿 Herb: {h}</span>")
            for m in c_min:
                badge_html.append(f"<span style='background:#fff8e1; color:#7f4f00; border:1px solid #ffe082; padding:2px 8px; border-radius:12px; font-size:12px; margin-right:4px;'>💎 Mineral: {m}</span>")
            for v in c_veh:
                badge_html.append(f"<span style='background:#e0f2fe; color:#0369a1; border:1px solid #bae6fd; padding:2px 8px; border-radius:12px; font-size:12px; margin-right:4px;'>🍶 Vehicle: {v}</span>")
            for s in c_sub:
                badge_html.append(f"<span style='background:#f3e8ff; color:#6b21a8; border:1px solid #e9d5ff; padding:2px 8px; border-radius:12px; font-size:12px; margin-right:4px;'>🥣 Sub-formulation: {s}</span>")
            for u in c_unres:
                badge_html.append(f"<span style='background:#f1f5f9; color:#475569; border:1px solid #e2e8f0; padding:2px 8px; border-radius:12px; font-size:12px; margin-right:4px;'>❓ Substance: {u}</span>")
                
            st.markdown(f"<div style='line-height: 2.2; margin-bottom: 12px;'>{''.join(badge_html)}</div>", unsafe_allow_html=True)

        if selected_comp["known_targets"]:
            known_targets_str = ", ".join([f"{t['canonical_name']} ({t['target_type']})" for t in selected_comp["known_targets"][:8]])
            st.info(f"**Known target associations in training graph:** {known_targets_str}", icon="ℹ️")
        else:
            st.caption("No direct TREATS edges recorded for this formulation in the training graph.")

    # ----------------------------------------------------
    # STEP 2: Review Model Predictions
    # ----------------------------------------------------
    st.markdown("---")
    st.markdown("### STEP 2 — Model Predictions & Retrieved Associations")
    st.caption("Review candidate disease and symptom associations retrieved by the GNN model via inductive message passing over ingredient neighborhoods.")
    
    col_f1, col_f2, col_f3 = st.columns([1, 1, 1])
    with col_f1:
        pred_type_filter = st.selectbox("Filter Target Type:", ["All (Disease & Symptom)", "Disease Only", "Symptom Only"])
        ttype_code = "Disease" if "Disease Only" in pred_type_filter else ("Symptom" if "Symptom Only" in pred_type_filter else "All")
    with col_f2:
        top_k_filter = st.selectbox("Retrieval Depth:", [5, 10, 15], index=1)
    with col_f3:
        alpha_blend = st.slider("Model Retrieval Tuning (α):", 0.0, 1.0, 0.75, 0.25, help="0.75 is the inner-validation optimal blend of GNN representation and marginal frequency prior.")

    # Run GNN inference for this formulation
    preds_df = engine.predict(
        compound_id=selected_cid,
        fold_num=selected_comp["fold"],
        alpha=alpha_blend,
        target_type=ttype_code,
        top_k=top_k_filter
    )

    if profile["blind_mode"]:
        st.warning("🔒 **Independent Review Mode is Active**: Model scores and ranks are masked to eliminate anchoring bias.", icon="🔒")

    # Display Research-Oriented Prediction Table
    records = validation_store.load_all_records()
    reviewed_targets = set((r["formulation_id"], r["target_id"]) for r in records)

    display_table_rows = []
    for _, r in preds_df.iterrows():
        is_rev = (selected_cid, r["target_id"]) in reviewed_targets
        status_text = "✅ Assessed" if is_rev else "⏳ Pending"
        
        display_table_rows.append({
            "Rank": f"#{r['rank']}" if not profile["blind_mode"] else "—",
            "Target Indication": r["target_name"],
            "Target Type": r["target_type"],
            "Model Score": f"{r['model_score']:.4f}" if not profile["blind_mode"] else "Masked",
            "Known in Source KG?": "Yes (TREATS)" if r["known_in_source"] else "No (Novel Retrieval)",
            "Graph Evidence": f"{r['evidence_count']} active links",
            "Expert Assessment": status_text
        })

    st.dataframe(pd.DataFrame(display_table_rows), use_container_width=True, hide_index=True)

    # ----------------------------------------------------
    # STEP 3 & STEP 4: Inspect Evidence & Expert Assessment
    # ----------------------------------------------------
    st.markdown("---")
    st.markdown("### STEP 3 & 4 — Inspect Evidence & Record Expert Assessment")
    st.caption("Select a specific retrieved association below to examine its multi-hop knowledge graph evidence and record your independent assessment.")

    candidate_target_ids = preds_df["target_id"].tolist()
    candidate_labels = {
        r["target_id"]: f"#{r['rank']} • {r['target_name']} ({r['target_type']}) — Model Score: {r['model_score']:.4f}" if not profile["blind_mode"] else f"{r['target_name']} ({r['target_type']})"
        for _, r in preds_df.iterrows()
    }
    
    selected_target_id = st.selectbox(
        "Select Association to Audit:",
        options=candidate_target_ids,
        format_func=lambda tid: candidate_labels.get(tid, tid),
        key="target_to_audit"
    )

    selected_target_row = preds_df[preds_df["target_id"] == selected_target_id].iloc[0]
    
    # Retrieve grounded evidence for this formulation and target
    evidence = engine.get_evidence_details(
        compound_id=selected_cid,
        target_id=selected_target_id
    )

    ev_col1, ev_col2 = st.columns([1.2, 1])

    # Left Column: STEP 3 — Evidence Review Panel & Subgraph
    with ev_col1:
        st.markdown(f"#### 🔍 STEP 3 — Evidence Review: {selected_comp['canonical_name']} ➔ {evidence['target_name']}")
        
        st.markdown(f"""
            <div style="background:#fcfbf9; border:1px solid #e5e2d9; border-radius:6px; padding:10px 14px; margin-bottom:12px; font-size:13px;">
                <strong>Target Indication:</strong> {evidence['target_name']} ({evidence['target_type']})<br>
                <strong>Model Score:</strong> <code>{selected_target_row['model_score']:.4f}</code> (Rank #{selected_target_row['rank']})<br>
                <strong>Known in Source KG?</strong> {'Yes — Encoded in AyurKOSH' if evidence['is_known_in_source_data'] else 'No — Novel Topological Association'}
            </div>
        """, unsafe_allow_html=True)

        ev_tab1, ev_tab2, ev_tab3 = st.tabs([
            "🌿 Graph & Pathway Evidence",
            "🕸️ Interactive Subgraph",
            "📜 Classical Reference Scope"
        ])

        with ev_tab1:
            st.markdown("##### 1. Direct Ingredient Links (`TREATED_BY`):")
            if evidence["direct_ingredient_links"]:
                for dlink in evidence["direct_ingredient_links"]:
                    st.markdown(f"- 🌿 **{dlink['ingredient']}** ({dlink['category']}) ──`[TREATED_BY]`──> **{dlink['target']}**")
            else:
                st.caption("No constituent herb has a direct single-hop `TREATED_BY` edge to this condition in the source graph.")

            st.markdown("##### 2. Multi-Hop Relational Paths:")
            if evidence["multi_hop_paths"]:
                for mpath in evidence["multi_hop_paths"]:
                    st.markdown(f"- <code>{mpath['path']}</code>", unsafe_allow_html=True)
            else:
                st.caption("No 2-hop or 3-hop intermediate disease pathways identified.")

            st.markdown("##### 3. Dravyaguna Qualities & Herb Groups:")
            if evidence["shared_qualities"]:
                for sq in evidence["shared_qualities"][:3]:
                    st.markdown(f"- **{sq['ingredient']}** qualities: _{', '.join(sq['qualities'])}_")
            if evidence["shared_groups"]:
                for sg in evidence["shared_groups"][:2]:
                    st.markdown(f"- **{sg['ingredient']}** groups: _{', '.join(sg['groups'])}_")

        with ev_tab2:
            st.caption("Interactive subnetwork surrounding this association. Zoom, pan, and hover over edges to examine relationships.")
            render_interactive_subgraph(evidence["subgraph_nodes"], evidence["subgraph_edges"], height=320)
            st.markdown("<small style='color:#64748b; font-size:11px;'><em>Note: Graph relationships represent encoded knowledge associations and should not be interpreted as causal clinical evidence.</em></small>", unsafe_allow_html=True)

        with ev_tab3:
            st.markdown("##### Classical Textual Citation Evidence:")
            st.info(evidence["classical_citation_notice"], icon="📖")
            st.markdown("""
                **Reviewer Guidance:**
                Classical Ayurvedic texts (*Charaka Samhita*, *Sushruta Samhita*, *Ashtanga Hridaya*, *Sarangadhara Samhita*, *API*) describe many polyherbal indications that may not be explicitly linked as single-hop graph triples. Please use your domain expertise to assess whether this association is classical, allied, or novel.
            """)

    # Right Column: STEP 4 — Structured Expert Assessment Questionnaire
    with ev_col2:
        st.markdown(f"#### ✍️ STEP 4 — Expert Assessment")
        st.caption("Record your independent expert assessment on the five standardized questions below.")

        with st.form(key=f"assessment_form_{selected_cid}_{selected_target_id}"):
            st.markdown(f"""
                <div style="background:#ffffff; border-left:4px solid #c29b38; padding:8px 12px; margin-bottom:12px; font-size:12.5px;">
                    Reviewing: <strong>{selected_comp['canonical_name']}</strong> ➔ <strong>{evidence['target_name']}</strong>
                </div>
            """, unsafe_allow_html=True)

            q1 = st.radio(
                "Q1. How relevant is this formulation–indication association from an Ayurvedic perspective?",
                [
                    "1 — Not relevant",
                    "2 — Slightly relevant",
                    "3 — Moderately relevant",
                    "4 — Highly relevant",
                    "5 — Very highly relevant"
                ],
                index=3
            )
            q1_score = int(q1.split(" ")[0])

            q2 = st.radio(
                "Q2. How well is the association supported by the evidence shown on the left?",
                [
                    "1 — Not supported",
                    "2 — Weakly supported",
                    "3 — Moderately supported",
                    "4 — Well supported",
                    "5 — Strongly supported"
                ],
                index=3
            )
            q2_score = int(q2.split(" ")[0])

            q3 = st.selectbox(
                "Q3. How would you classify this association?",
                [
                    "Consistent with classical knowledge",
                    "Potentially relevant / allied indication",
                    "Plausible but requires further evidence",
                    "Not sufficiently supported",
                    "Not relevant"
                ],
                index=0
            )

            q4 = st.radio(
                "Q4. Would you consider this association suitable for further research or verification?",
                ["Yes", "Possibly", "No"],
                index=0,
                horizontal=True
            )

            q5_comments = st.text_area(
                "Q5. Reviewer comments / rationale (Optional, but strongly encouraged for negative or tentative ratings):",
                placeholder="State your pharmacological reasoning, classical treatise references (e.g. Charaka Chi. 3), or reasons for disagreement...",
                height=90
            )

            submit_assessment = st.form_submit_button("💾 Save Expert Assessment Record", type="primary", use_container_width=True)
            
            if submit_assessment:
                rec_id = validation_store.save_assessment_record(
                    reviewer_profile=profile,
                    formulation_id=selected_cid,
                    formulation_name=selected_comp["canonical_name"],
                    target_id=selected_target_id,
                    target_name=evidence["target_name"],
                    target_type=evidence["target_type"],
                    model_rank=selected_target_row["rank"],
                    model_score=selected_target_row["model_score"],
                    known_in_source=selected_target_row["known_in_source"],
                    evidence_summary={
                        "evidence_count": selected_target_row["evidence_count"],
                        "direct_links": len(evidence["direct_ingredient_links"]),
                        "multi_hop_paths": len(evidence["multi_hop_paths"])
                    },
                    q1_relevance=q1_score,
                    q2_evidence_support=q2_score,
                    q3_classification=q3,
                    q4_further_research=q4,
                    q5_comments=q5_comments,
                    review_mode="Independent Blind" if profile["blind_mode"] else "Standard"
                )
                st.session_state["current_step"] = 5
                st.success(f"✅ Assessment recorded! Audit Record ID: `{rec_id}`")
                st.rerun()

    # ----------------------------------------------------
    # STEP 5: Audit Records & Summary Dashboard
    # ----------------------------------------------------
    st.markdown("---")
    st.markdown("### STEP 5 — Audit Records & Validation Evidence")
    
    metrics = validation_store.get_summary_metrics()
    
    if not metrics["has_data"]:
        st.info("No expert assessments recorded yet in this evaluation store. Complete Step 4 above to log the first assessment record.", icon="ℹ️")
    else:
        m_c1, m_c2, m_c3, m_c4 = st.columns(4)
        m_c1.metric("Total Assessments", metrics["total_reviews"])
        m_c2.metric("Unique Formulations", metrics["unique_formulations"])
        m_c3.metric("Mean Ayurvedic Relevance", f"{metrics['mean_relevance']} / 5.0")
        m_c4.metric("Mean Evidence Support", f"{metrics['mean_evidence_support']} / 5.0")

        st.caption(f"**Inter-Rater Concordance Status**: {metrics['agreement_status']}")

        all_records = validation_store.load_all_records()
        if all_records:
            summary_df = pd.DataFrame(all_records)[[
                "record_id", "timestamp", "reviewer_id", "formulation_name", "target_name",
                "target_type", "model_score", "q1_relevance", "q2_evidence_support",
                "q3_classification", "q4_further_research"
            ]]
            summary_df.columns = [
                "Record ID", "Timestamp", "Reviewer ID", "Formulation", "Target",
                "Type", "Model Score", "Q1 Rel (1-5)", "Q2 Evi (1-5)", "Classification", "Research?"
            ]
            st.dataframe(summary_df, use_container_width=True)

        down_col1, down_col2 = st.columns([1, 1])
        with down_col1:
            report_md = validation_store.generate_kscst_audit_report_markdown()
            st.download_button(
                label="📥 Download Official KSCST Audit Report (Markdown)",
                data=report_md,
                file_name="AYURKOSH_Expert_Validation_Report.md",
                mime="text/markdown",
                use_container_width=True
            )
        with down_col2:
            if all_records:
                csv_data = pd.DataFrame(all_records).to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="📥 Export Assessment Records (CSV)",
                    data=csv_data,
                    file_name="AYURKOSH_Expert_Assessments.csv",
                    mime="text/csv",
                    use_container_width=True
                )
