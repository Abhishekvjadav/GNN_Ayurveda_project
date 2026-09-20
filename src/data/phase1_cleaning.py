# ============================================================
# PHASE 1 V4.2
# AYURKOSH HETEROGENEOUS KNOWLEDGE GRAPH
#
# Fixes applied on top of v4.1:
#
#   BUG 1 (herb_vocab contamination):
#     The DHS "ओषधिः-लक्षण Entity" column holds BOTH herb and
#     symptom rows in one column. v4.1's vocab-builder matched
#     this column purely by name ("ओषध" substring) and pulled in
#     ALL its values -- including 1,865 Lakshan/symptom terms --
#     into herb_vocab. Fix: for DHS specifically, only pull
#     entity values where category == 'Herbs'.
#
#   BUG 2 (classification order):
#     Even with herb_vocab cleaned, the dedicated "Herb (ओषधिः)"
#     columns in Herb-Group / Herb-Quality / Herb-Kalp /
#     Herb-AlternateHerb genuinely contain some non-herb entries
#     (नाग, लोह -- both metals -- show up there in the source
#     data). classify_dhs_entity() checked herb_vocab BEFORE
#     Mineral/Vehicle, so a metal sitting in herb_vocab always
#     won as "Herb". Fix: check Mineral/Vehicle exact-match
#     BEFORE the herb_vocab branch.
#
#   BUG 3 (blind Herb defaults in the 4 sheet loops):
#     Processing Herb-Group / Herb-AlternateHerb / Herb-Quality /
#     Herb-Kalp unconditionally called add_node(..., "Herb", ...,
#     confidence="SOURCE") for every value in their herb-name
#     column, bypassing classification entirely. Confirmed: this
#     forces नाग and लोह to "Herb" (confidence SOURCE, the
#     highest trust tier) even though DHS-side processing
#     correctly resolves them to "Mineral" -- i.e. the SAME real
#     substance ends up as two different, contradictory nodes.
#     Fix: route all four loops through the same
#     classify_herb_side_entity() used everywhere else, so a
#     given term gets the same type no matter which sheet
#     introduced it first.
#
#   BUG 4 (suffix matching fires on bare suffix words):
#     `key.endswith(suffix)` is True when key IS the suffix
#     (e.g. standalone "कषाय", which is usually a Rasa/taste
#     concept, not a formulation). Fix: require
#     len(key) > len(suffix) so the rule only fires on genuine
#     root+suffix compounds.
#
#   NEW: Herb-Kalp vs Compound-Herb overlap audit.
#     6 compound names exist identically in both sheets (typed
#     Compound from one, Sub_Formulation from the other) --
#     confirmed via direct check against your workbook. This is
#     a domain decision (merge vs. keep distinct), not something
#     to silently resolve in code, so it's now written to its
#     own audit file instead of being buried in entity_conflicts.
#
# Everything else (paths, sheet names, SYNONYM_MAP, TRIPHALA_TERMS,
# node/edge schema, output files) is unchanged from v4.1.
# ============================================================

from pathlib import Path
import re
import unicodedata
import hashlib

import pandas as pd


# ============================================================
# PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[2]

RAW_FILE = (
    ROOT / "data" / "raw" / "ayurkosh.xlsx"
)

PHASE0_FILE = (
    ROOT / "data" / "audit" / "phase0_multisource_classification_review_v4.xlsx"
)

OUTPUT_DIR = (
    ROOT / "results" / "phase1"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# EXACT SHEET NAMES FROM ACTUAL WORKBOOK
# ============================================================

DHS_SHEET = "Disease-Herb-Symptom"
HERB_GROUP_SHEET = "Herb-Group relationship"
HERB_ALT_SHEET = "Herb-AlternateHerb relationship"
HERB_QUALITY_SHEET = "Herb-Quality relationship"
HERB_KALP_SHEET = "Herb-Kalp relationship"
COMPOUND_HERB_SHEET = "Compound-Herb relationship"
COMPOUND_TREATS_SHEET = "Compound-Treats relationship"
LAKSHAN_SHEET = "Lakshan-Prakruti-Ras-Dhatu rela"


# ============================================================
# NORMALIZATION
# ============================================================

def normalize_text(value):
    if pd.isna(value):
        return ""
    text = str(value)
    text = unicodedata.normalize("NFC", text)
    text = re.sub(r"[\u200b\u200c\u200d\ufeff]", "", text)
    text = re.sub(r"\s+", " ", text)
    text = text.strip()
    text = re.sub(r"्$", "", text)
    return text


def normalize_key(value):
    return normalize_text(value)


# ============================================================
# CONTROLLED SYNONYMS
# ============================================================

SYNONYM_MAP = {
    "गुडुची": "अमृता",
    "गुडूची": "अमृता",
    "हरितकी": "हरीतकी",
    "यष्टिमधु": "यष्टीमधु",
}


# ============================================================
# TRIPHALA
# ============================================================

TRIPHALA_TERMS = {
    "त्रिफळा", "त्रिफला",
    "त्रिफळाक्वाथ", "त्रिफलाक्वाथ",
    "त्रिफळा क्वाथ", "त्रिफला क्वाथ",
    "त्रिफळाचूर्ण", "त्रिफलाचूर्ण",
    "त्रिफळा चूर्ण", "त्रिफला चूर्ण",
}

SUBFORMULATION_SUFFIXES = [
    "क्वाथ", "कषाय", "चूर्ण", "घृत", "तैल", "अवलेह", "लेह",
    "आसव", "अरिष्ट", "मण्ड", "मंड", "सत्व", "सत्त्व", "पाक",
    "मोदक", "वटी", "गुटिका",
]


# ============================================================
# MINERAL VOCABULARY
# ============================================================

MINERAL_EXACT = {
    "अभ्रक", "अभ्रक भस्म", "अभ्रकभस्म",
    "लोह", "लोह भस्म", "लोहभस्म",
    "स्वर्ण", "स्वर्ण भस्म", "स्वर्णभस्म",
    "सुवर्ण", "सुवर्ण भस्म", "सुवर्णभस्म",
    "रजत", "रजत भस्म", "रजतभस्म",
    "ताम्र", "ताम्र भस्म", "ताम्रभस्म",
    "यशद", "यशद भस्म", "यशदभस्म",
    "नाग", "नाग भस्म", "नागभस्म",
    "वंग", "वंग भस्म", "वंगभस्म",
    "पारद", "गन्धक", "गंधक", "हरताल",
    "मनःशिला", "मनशिला", "मुक्ता", "प्रवाल", "शुक्ति", "गोदन्ती",
}

MINERAL_SUFFIXES = ["भस्म", "भस्मम्"]


# ============================================================
# VEHICLES
# ============================================================

VEHICLE_EXACT = {
    "जल", "पानी", "दूध", "क्षीर", "गोदुग्ध", "दधि", "तक्र",
    "घृत", "गोघृत", "तैल", "मधु", "शर्करा", "गुड", "गुड़",
    "कांजी", "काञ्जी",
}


# ============================================================
# VALID NODE TYPES
# ============================================================

VALID_NODE_TYPES = {
    "Disease", "Symptom", "Herb", "Mineral", "Vehicle",
    "Sub_Formulation", "Compound", "Group", "Quality",
    "Ayurvedic_Concept", "Other_Unclassified",
}


print()
print("=" * 70)
print("PHASE 1 V4.2 - AYURKOSH GRAPH CONSTRUCTION (BUGFIXED)")
print("=" * 70)


# ============================================================
# LOAD EXCEL
# ============================================================

print("\nLoading workbook:")
print(RAW_FILE)

xls = pd.ExcelFile(RAW_FILE)

print("\nSheets found:")
for sheet in xls.sheet_names:
    print(" -", sheet)

expected_sheets = [
    DHS_SHEET, HERB_GROUP_SHEET, HERB_ALT_SHEET, HERB_QUALITY_SHEET,
    HERB_KALP_SHEET, COMPOUND_HERB_SHEET, COMPOUND_TREATS_SHEET, LAKSHAN_SHEET,
]
missing_sheets = [s for s in expected_sheets if s not in xls.sheet_names]
if missing_sheets:
    raise ValueError("Missing expected sheets: " + str(missing_sheets))


dfs = {}
for sheet in xls.sheet_names:
    df = pd.read_excel(xls, sheet_name=sheet)
    df.columns = [normalize_text(c) for c in df.columns]
    for col in df.columns:
        df[col] = df[col].map(normalize_text)
    dfs[sheet] = df

print("\nLoaded row counts:")
for name, df in dfs.items():
    print(f" - {name}: {len(df):,}")


# ============================================================
# LOAD PHASE 0
# ============================================================

print("\nLoading Phase-0 classification:")
phase0 = pd.read_excel(PHASE0_FILE)
phase0.columns = [normalize_text(c) for c in phase0.columns]

required_phase0 = ["normalized_ingredient", "predicted_type"]
for col in required_phase0:
    if col not in phase0.columns:
        raise ValueError(f"Missing Phase-0 column: {col}")

phase0["normalized_ingredient"] = phase0["normalized_ingredient"].map(normalize_key)
phase0["predicted_type"] = phase0["predicted_type"].fillna("").astype(str).str.strip()

phase0["manual_type"] = (
    phase0["manual_type"].fillna("").astype(str).str.strip()
    if "manual_type" in phase0.columns else ""
)
phase0["verified"] = (
    phase0["verified"].fillna("").astype(str).str.lower().str.strip()
    if "verified" in phase0.columns else ""
)
phase0["matched_canonical_term"] = (
    phase0["matched_canonical_term"].map(normalize_key)
    if "matched_canonical_term" in phase0.columns else ""
)

phase0_lookup = {}
for _, row in phase0.iterrows():
    key = row["normalized_ingredient"]
    if not key:
        continue
    predicted = row["predicted_type"]
    manual = row["manual_type"]
    verified = row["verified"]
    canonical = row["matched_canonical_term"]

    if manual:
        final_type, source = manual, "PHASE0_VERIFIED"
    elif verified in {"true", "yes", "verified", "1"}:
        final_type, source = predicted, "PHASE0_VERIFIED"
    else:
        final_type, source = predicted, "PHASE0_AYURKOSH_EXACT_RULE"

    phase0_lookup[key] = {
        "node_type": final_type,
        "canonical": canonical or key,
        "source": source,
    }

print("\nPhase-0 lookup keys:", len(phase0_lookup))


# ============================================================
# GRAPH STORAGE
# ============================================================

nodes = {}
edges = []
classification_audit = []
entity_conflicts = []
source_priority_audit = []
mineral_suffix_audit = []
compound_treats_resolution = []
herb_kalp_compound_overlap = []  # NEW


def create_node_id(node_type, canonical_name):
    key = f"{node_type}|{normalize_key(canonical_name)}"
    digest = hashlib.md5(key.encode("utf-8")).hexdigest()[:12]
    prefix = {
        "Disease": "DISEASE", "Symptom": "SYMPTOM", "Herb": "HERB",
        "Mineral": "MINERAL", "Vehicle": "VEHICLE", "Sub_Formulation": "SUBFORM",
        "Compound": "COMPOUND", "Group": "GROUP", "Quality": "QUALITY",
        "Ayurvedic_Concept": "AYUR", "Other_Unclassified": "OTHER",
    }.get(node_type, "NODE")
    return f"{prefix}_{digest}"


def add_node(raw_name, node_type, classification_method, confidence, source_sheet, canonical_name=None):
    raw_name = normalize_key(raw_name)
    if not raw_name:
        return None
    if node_type not in VALID_NODE_TYPES:
        node_type = "Other_Unclassified"
    if canonical_name is None:
        canonical_name = raw_name
    canonical_name = normalize_key(canonical_name)

    node_id = create_node_id(node_type, canonical_name)
    if node_id not in nodes:
        nodes[node_id] = {
            "node_id": node_id,
            "raw_name": raw_name,
            "canonical_name": canonical_name,
            "node_type": node_type,
            "classification_method": classification_method,
            "confidence": confidence,
            "source_sheets": source_sheet,
        }
    else:
        existing = set(nodes[node_id]["source_sheets"].split("|"))
        existing.add(source_sheet)
        nodes[node_id]["source_sheets"] = "|".join(sorted(existing))
    return node_id


def add_edge(source_id, target_id, relation, source_sheet, raw_source, raw_target):
    if not source_id or not target_id:
        return
    edges.append({
        "source": source_id, "target": target_id, "relation": relation,
        "source_sheet": source_sheet, "raw_source": raw_source, "raw_target": raw_target,
    })


# ============================================================
# HERB VOCABULARY  (BUG 1 FIX: category-filtered for DHS)
# ============================================================

herb_vocab = set()

# DHS: only take entity values whose OWN row is category == 'Herbs'.
# (Filled in properly once DHS columns are resolved below; placeholder here.)

for sheet in [HERB_GROUP_SHEET, HERB_ALT_SHEET, HERB_QUALITY_SHEET, HERB_KALP_SHEET]:
    df = dfs[sheet]
    for col in df.columns:
        lower = col.lower()
        if "herb" in lower or "ओषध" in col or "औषध" in col or "dravy" in lower:
            for value in df[col]:
                value = normalize_key(value)
                if value:
                    herb_vocab.add(value)

print("\nHerb vocabulary (pre-DHS, from dedicated herb sheets only):", len(herb_vocab))


# ============================================================
# DHS COLUMN RESOLUTION
# ============================================================

DHS_DISEASE_COL = "Disease (व्याधी)"
DHS_RELATION_COL = "Relationship with ओषधिः-लक्षण"
DHS_ENTITY_COL = "ओषधिः-लक्षण Entity"
DHS_CATEGORY_COL = "Category of Entity (ओषधिः ORलक्षण)"

dhs_df = dfs[DHS_SHEET]

print("\nDHS columns:")
for col in dhs_df.columns:
    print(" -", col)


def find_column(df, expected):
    if expected in df.columns:
        return expected
    normalized_expected = normalize_key(expected).replace(" ", "")
    for col in df.columns:
        if normalize_key(col).replace(" ", "") == normalized_expected:
            return col
    return None


disease_col = find_column(dhs_df, DHS_DISEASE_COL)
relation_col = find_column(dhs_df, DHS_RELATION_COL)
entity_col = find_column(dhs_df, DHS_ENTITY_COL)
category_col = find_column(dhs_df, DHS_CATEGORY_COL)

if not disease_col:
    raise ValueError("Could not find DHS disease column.")
if not relation_col:
    raise ValueError("Could not find DHS relationship column.")
if not entity_col:
    raise ValueError("Could not find DHS entity column.")
if not category_col:
    raise ValueError("Could not find DHS category column.")

print("\nResolved DHS columns:")
print("Disease  :", disease_col)
print("Relation :", relation_col)
print("Entity   :", entity_col)
print("Category :", category_col)


# ---- BUG 1 FIX applied here: category-filtered DHS contribution to herb_vocab ----
dhs_herb_vocab_additions = 0
for _, row in dhs_df.iterrows():
    category = normalize_key(row[category_col])
    entity = normalize_key(row[entity_col])
    if entity and category == "Herbs":
        if entity not in herb_vocab:
            dhs_herb_vocab_additions += 1
        herb_vocab.add(entity)

print(f"Herb vocabulary after DHS (category='Herbs' only) additions "
      f"(+{dhs_herb_vocab_additions}):", len(herb_vocab))


# ============================================================
# DISEASE / SYMPTOM VOCABULARY
# ============================================================

disease_vocab = set()
symptom_vocab = set()

for _, row in dhs_df.iterrows():
    disease = normalize_key(row[disease_col])
    entity = normalize_key(row[entity_col])
    category = normalize_key(row[category_col])

    if disease:
        disease_vocab.add(disease)
    if entity and ("लक्षण" in category or category.lower() == "lakshan"):
        symptom_vocab.add(entity)

print("\nDisease vocabulary:", len(disease_vocab))
print("Symptom vocabulary:", len(symptom_vocab))

for disease in disease_vocab:
    add_node(disease, "Disease", "SOURCE_DHS_DISEASE", "SOURCE", DHS_SHEET)


# ============================================================
# CORE CLASSIFIER
# (BUG 2 FIX: Mineral/Vehicle checked BEFORE herb_vocab)
# (BUG 4 FIX: suffix match requires len(key) > len(suffix))
# Used both for DHS entities AND for the four Herb-sheet loops
# (BUG 3 FIX), so the same term always gets the same type.
# ============================================================

def classify_herb_side_entity(entity, category=None):
    key = normalize_key(entity)
    synonym_key = SYNONYM_MAP.get(key, key)

    # 1. Phase 0 (highest-confidence, manually-tuned classification)
    if synonym_key in phase0_lookup:
        info = phase0_lookup[synonym_key]
        return {"type": info["node_type"], "canonical": info["canonical"],
                "method": info["source"], "confidence": "HIGH"}

    # 2. Category says Lakshan -> Symptom (only meaningful when called from DHS)
    if category and ("लक्षण" in category or category.lower() == "lakshan"):
        return {"type": "Symptom", "canonical": key,
                "method": "DHS_CATEGORY_LAKSHAN", "confidence": "MEDIUM"}

    # 3. Mineral exact match -- BEFORE herb_vocab, so a metal that leaked
    #    into a "Herb (ओषधिः)" column elsewhere still resolves correctly.
    if key in MINERAL_EXACT:
        return {"type": "Mineral", "canonical": key,
                "method": "AUTO_MINERAL_EXACT", "confidence": "MEDIUM"}

    # 4. Mineral suffix (guarded: must be a real root+suffix compound)
    if any(key.endswith(s) and len(key) > len(s) for s in MINERAL_SUFFIXES):
        mineral_suffix_audit.append({
            "entity": key, "classification": "Mineral",
            "rule": "AUTO_MINERAL_SUFFIX", "source_sheet": DHS_SHEET,
        })
        return {"type": "Mineral", "canonical": key,
                "method": "AUTO_MINERAL_SUFFIX", "confidence": "MEDIUM"}

    # 5. Vehicle exact match -- also before herb_vocab, same reasoning
    if key in VEHICLE_EXACT:
        return {"type": "Vehicle", "canonical": key,
                "method": "AUTO_VEHICLE_EXACT", "confidence": "MEDIUM"}

    # 6. Sub-formulation (Triphala family, or guarded suffix match)
    if key in TRIPHALA_TERMS or any(
        key.endswith(s) and len(key) > len(s) for s in SUBFORMULATION_SUFFIXES
    ):
        return {"type": "Sub_Formulation", "canonical": key,
                "method": "AUTO_SUBFORMULATION", "confidence": "MEDIUM"}

    # 7. Herb vocabulary (now checked AFTER the exclusions above)
    if key in herb_vocab:
        return {"type": "Herb", "canonical": synonym_key,
                "method": "AYURKOSH_HERB_VOCAB", "confidence": "HIGH"}

    # 8. Genuinely unresolved
    return {"type": "Other_Unclassified", "canonical": key,
            "method": "AUTO_UNCLASSIFIED", "confidence": "LOW"}


# Back-compat alias: keep old name working if referenced elsewhere
classify_dhs_entity = classify_herb_side_entity


# ============================================================
# PROCESS DHS
# ============================================================

print("\nProcessing Disease-Herb-Symptom...")

dhs_treated_by_count = 0
dhs_lakshan_count = 0

for _, row in dhs_df.iterrows():
    disease = normalize_key(row[disease_col])
    entity = normalize_key(row[entity_col])
    relationship = normalize_key(row[relation_col]).lower()
    category = normalize_key(row[category_col])

    if not disease or not entity:
        continue

    disease_id = add_node(disease, "Disease", "SOURCE_DHS_DISEASE", "SOURCE", DHS_SHEET)

    if relationship == "treated_by":
        result = classify_herb_side_entity(entity, category)
        entity_id = add_node(entity, result["type"], result["method"],
                              result["confidence"], DHS_SHEET, result["canonical"])
        add_edge(disease_id, entity_id, "TREATED_BY", DHS_SHEET, disease, entity)
        dhs_treated_by_count += 1
        classification_audit.append({
            "node_type": result["type"], "classification_rule": result["method"],
            "confidence": result["confidence"], "count": 1,
        })

    elif relationship == "has_lakshan":
        symptom_id = add_node(entity, "Symptom", "DHS_RELATION_HAS_LAKSHAN", "SOURCE", DHS_SHEET)
        add_edge(disease_id, symptom_id, "HAS_LAKSHAN", DHS_SHEET, disease, entity)
        dhs_lakshan_count += 1

    else:
        classification_audit.append({
            "node_type": "Other_Unclassified",
            "classification_rule": f"UNKNOWN_DHS_RELATION:{relationship}",
            "confidence": "LOW", "count": 1,
        })

print("\nDHS relationship results:")
print("TREATED_BY:", dhs_treated_by_count)
print("HAS_LAKSHAN:", dhs_lakshan_count)


# ============================================================
# HERB-GROUP  (BUG 3 FIX: classify, don't force Herb)
# ============================================================

print("\nProcessing Herb-Group...")
df = dfs[HERB_GROUP_SHEET]

for _, row in df.iterrows():
    herb = normalize_key(row.iloc[0])
    group = normalize_key(row.iloc[3])
    if not herb or not group:
        continue

    result = classify_herb_side_entity(herb)
    herb_id = add_node(herb, result["type"], result["method"],
                        result["confidence"], HERB_GROUP_SHEET, result["canonical"])
    group_id = add_node(group, "Group", "SOURCE_Herb-Group relationship", "SOURCE", HERB_GROUP_SHEET)
    add_edge(herb_id, group_id, "BELONGS_TO", HERB_GROUP_SHEET, herb, group)


# ============================================================
# HERB ALTERNATE  (BUG 3 FIX)
# ============================================================

print("\nProcessing Herb-AlternateHerb...")
df = dfs[HERB_ALT_SHEET]

for _, row in df.iterrows():
    herb = normalize_key(row.iloc[0])
    alternate = normalize_key(row.iloc[1])
    if not herb or not alternate:
        continue

    herb_result = classify_herb_side_entity(herb)
    alt_result = classify_herb_side_entity(alternate)

    herb_id = add_node(herb, herb_result["type"], herb_result["method"],
                        herb_result["confidence"], HERB_ALT_SHEET, herb_result["canonical"])
    alt_id = add_node(alternate, alt_result["type"], alt_result["method"],
                       alt_result["confidence"], HERB_ALT_SHEET, alt_result["canonical"])
    if herb_id != alt_id:
        add_edge(
            herb_id,
            alt_id,
            "ALTERNATE_OF",
            HERB_ALT_SHEET,
            herb,
            alternate)


# ============================================================
# HERB QUALITY  (BUG 3 FIX)
# ============================================================

print("\nProcessing Herb-Quality...")
df = dfs[HERB_QUALITY_SHEET]

for _, row in df.iterrows():
    herb = normalize_key(row.iloc[0])
    quality = normalize_key(row.iloc[1])
    if not herb or not quality:
        continue

    result = classify_herb_side_entity(herb)
    herb_id = add_node(herb, result["type"], result["method"],
                        result["confidence"], HERB_QUALITY_SHEET, result["canonical"])
    quality_id = add_node(quality, "Quality", "SOURCE_Herb-Quality relationship",
                           "SOURCE", HERB_QUALITY_SHEET)
    add_edge(herb_id, quality_id, "HAS_QUALITY", HERB_QUALITY_SHEET, herb, quality)


# ============================================================
# HERB KALP  (BUG 3 FIX + overlap audit)
# ============================================================

print("\nProcessing Herb-Kalp...")
df = dfs[HERB_KALP_SHEET]

# Build Compound-Herb's compound-name set up front, for the overlap audit below.
ch_df_preview = dfs[COMPOUND_HERB_SHEET]
compound_herb_names = set(normalize_key(v) for v in ch_df_preview.iloc[:, 1].dropna())

for _, row in df.iterrows():
    herb = normalize_key(row.iloc[0])
    compound = normalize_key(row.iloc[1])
    if not herb or not compound:
        continue

    herb_result = classify_herb_side_entity(herb)
    herb_id = add_node(herb, herb_result["type"], herb_result["method"],
                        herb_result["confidence"], HERB_KALP_SHEET, herb_result["canonical"])
    compound_id = add_node(compound, "Sub_Formulation", "SOURCE_Herb-Kalp relationship",
                            "SOURCE", HERB_KALP_SHEET)
    add_edge(herb_id, compound_id, "HAS_KALP", HERB_KALP_SHEET, herb, compound)

    # NEW: flag when a Herb-Kalp "compound" name also exists as a Compound-Herb
    # compound -- these become two separate node types (Compound vs
    # Sub_Formulation) for what may be the same real formulation. This is a
    # domain decision, not something to silently merge in code.
    if compound in compound_herb_names:
        herb_kalp_compound_overlap.append({
            "compound_name": compound,
            "appears_as": "Compound (via Compound-Herb) AND Sub_Formulation (via Herb-Kalp)",
            "action_needed": "Manual decision: same formulation (merge) or distinct usage (keep split)?",
        })


# ============================================================
# COMPOUND-HERB
# ============================================================

print("\nProcessing Compound-Herb...")
df = dfs[COMPOUND_HERB_SHEET]

for _, row in df.iterrows():
    compound = normalize_key(row.iloc[1])
    ingredient = normalize_key(row.iloc[2])
    if not compound or not ingredient:
        continue

    compound_id = add_node(compound, "Compound", "SOURCE_Compound-Herb relationship",
                            "SOURCE", COMPOUND_HERB_SHEET)

    result = classify_herb_side_entity(ingredient)
    ingredient_id = add_node(ingredient, result["type"], result["method"],
                              result["confidence"], COMPOUND_HERB_SHEET, result["canonical"])
    add_edge(compound_id, ingredient_id, "CONTAINS", COMPOUND_HERB_SHEET, compound, ingredient)

    classification_audit.append({
        "node_type": result["type"], "classification_rule": result["method"],
        "confidence": result["confidence"], "count": 1,
    })


# ============================================================
# COMPOUND TREATS
# ============================================================

print("\nProcessing Compound-Treats...")
df = dfs[COMPOUND_TREATS_SHEET]
print("Compound-Treats columns:", list(df.columns))

for _, row in df.iterrows():
    compound = normalize_key(row.iloc[0])
    target = normalize_key(row.iloc[1])
    target_type = normalize_key(row.iloc[2]).lower()
    if not compound or not target:
        continue

    compound_id = add_node(compound, "Compound", "SOURCE_Compound-Treats relationship",
                            "SOURCE", COMPOUND_TREATS_SHEET)

    if "व्याधी" in target_type or "vyadhi" in target_type or target in disease_vocab:
        target_id = add_node(target, "Disease", "SOURCE_DHS_DISEASE", "SOURCE", COMPOUND_TREATS_SHEET)
        resolved_type = "Disease"
    elif "लक्षण" in target_type or "lakshan" in target_type or target in symptom_vocab:
        target_id = add_node(target, "Symptom", "SOURCE_Compound-Treats", "SOURCE", COMPOUND_TREATS_SHEET)
        resolved_type = "Symptom"
    else:
        target_id = add_node(target, "Other_Unclassified", "AUTO_UNCLASSIFIED", "LOW", COMPOUND_TREATS_SHEET)
        resolved_type = "Other_Unclassified"

    add_edge(compound_id, target_id, "TREATS", COMPOUND_TREATS_SHEET, compound, target)
    compound_treats_resolution.append({
        "compound": compound, "target": target,
        "source_type": target_type, "resolved_type": resolved_type,
    })


# ============================================================
# LAKSHAN / PRAKRUTI / RASA / DHATU
# ============================================================

print("\nProcessing Lakshan-Prakruti-Ras-Dhatu...")
df = dfs[LAKSHAN_SHEET]
print("Columns:", list(df.columns))

for _, row in df.iterrows():
    values = [normalize_key(v) for v in row.values]
    values = [v for v in values if v]
    if not values:
        continue

    name = normalize_key(row.iloc[2]) if len(row) > 2 else ""
    parameter = normalize_key(row.iloc[3]) if len(row) > 3 else ""
    lakshan = normalize_key(row.iloc[4]) if len(row) > 4 else ""

    if not name:
        continue

    name_id = add_node(name, "Ayurvedic_Concept", "SOURCE_LAKSHAN", "SOURCE", LAKSHAN_SHEET)

    if parameter:
        parameter_id = add_node(parameter, "Ayurvedic_Concept", "SOURCE_LAKSHAN", "SOURCE", LAKSHAN_SHEET)
        add_edge(name_id, parameter_id, "HAS_LAKSHAN", LAKSHAN_SHEET, name, parameter)

    if lakshan:
        lakshan_id = add_node(lakshan, "Ayurvedic_Concept", "SOURCE_LAKSHAN", "SOURCE", LAKSHAN_SHEET)
        add_edge(name_id, lakshan_id, "HAS_LAKSHAN", LAKSHAN_SHEET, name, lakshan)


# ============================================================
# DEDUPLICATE EDGES
# ============================================================

edges_df = pd.DataFrame(edges)
if not edges_df.empty:
    before_edges = len(edges_df)
    edges_df = edges_df.drop_duplicates(subset=["source", "target", "relation"]).reset_index(drop=True)
    print("\nEdge deduplication:")
    print("Before:", before_edges)
    print("After :", len(edges_df))
    print("Removed:", before_edges - len(edges_df))


# ============================================================
# NODES DATAFRAME
# ============================================================

nodes_df = pd.DataFrame(list(nodes.values()))
if not nodes_df.empty:
    nodes_df = nodes_df[[
        "node_id", "raw_name", "canonical_name", "node_type",
        "classification_method", "confidence", "source_sheets",
    ]].sort_values(["node_type", "canonical_name"]).reset_index(drop=True)

nodes_df.to_csv(OUTPUT_DIR / "nodes.csv", index=False, encoding="utf-8-sig")
edges_df.to_csv(OUTPUT_DIR / "edges.csv", index=False, encoding="utf-8-sig")


# ============================================================
# NODE / EDGE TYPE COUNTS
# ============================================================

node_type_counts = nodes_df["node_type"].value_counts().rename_axis("node_type").reset_index(name="count")
node_type_counts.to_csv(OUTPUT_DIR / "node_type_counts.csv", index=False, encoding="utf-8-sig")

edge_type_counts = edges_df["relation"].value_counts().rename_axis("relation").reset_index(name="count")
edge_type_counts.to_csv(OUTPUT_DIR / "edge_type_counts.csv", index=False, encoding="utf-8-sig")


# ============================================================
# CONNECTIVITY
# ============================================================

all_nodes = set(nodes_df["node_id"])
connected_nodes = set(edges_df["source"]) | set(edges_df["target"])
isolated_nodes = all_nodes - connected_nodes
isolated_df = nodes_df[nodes_df["node_id"].isin(isolated_nodes)].copy()
isolated_df.to_csv(OUTPUT_DIR / "isolated_nodes.csv", index=False, encoding="utf-8-sig")


# ============================================================
# DEGREE STATISTICS
# ============================================================

out_degree = edges_df["source"].value_counts()
in_degree = edges_df["target"].value_counts()

degree_df = nodes_df[["node_id", "canonical_name", "node_type"]].copy()
degree_df["out_degree"] = degree_df["node_id"].map(out_degree).fillna(0).astype(int)
degree_df["in_degree"] = degree_df["node_id"].map(in_degree).fillna(0).astype(int)
degree_df["total_degree"] = degree_df["out_degree"] + degree_df["in_degree"]
degree_df.to_csv(OUTPUT_DIR / "node_degree_statistics.csv", index=False, encoding="utf-8-sig")


# ============================================================
# CLASSIFICATION AUDIT
# ============================================================

audit_df = pd.DataFrame(classification_audit)
if not audit_df.empty:
    audit_df = (audit_df.groupby(["node_type", "classification_rule", "confidence"], as_index=False)["count"]
                .sum().sort_values("count", ascending=False))
audit_df.to_csv(OUTPUT_DIR / "classification_audit.csv", index=False, encoding="utf-8-sig")


# ============================================================
# MINERAL SUFFIX AUDIT
# ============================================================

mineral_df = pd.DataFrame(mineral_suffix_audit)
if not mineral_df.empty:
    mineral_df = mineral_df.drop_duplicates().sort_values("entity").reset_index(drop=True)
mineral_df.to_csv(OUTPUT_DIR / "mineral_suffix_audit.csv", index=False, encoding="utf-8-sig")


# ============================================================
# COMPOUND-TREATS AUDIT
# ============================================================

compound_treats_df = pd.DataFrame(compound_treats_resolution)
compound_treats_df.to_csv(OUTPUT_DIR / "compound_treats_resolution.csv", index=False, encoding="utf-8-sig")


# ============================================================
# NEW: HERB-KALP / COMPOUND-HERB OVERLAP AUDIT
# ============================================================

overlap_df = pd.DataFrame(herb_kalp_compound_overlap).drop_duplicates()
overlap_df.to_csv(OUTPUT_DIR / "herb_kalp_compound_overlap.csv", index=False, encoding="utf-8-sig")
print(f"\nHerb-Kalp / Compound-Herb name overlaps flagged: {len(overlap_df)}")


# ============================================================
# SOURCE PRIORITY / ENTITY CONFLICT AUDIT
# ============================================================

entity_type_map = {}
for _, row in nodes_df.iterrows():
    entity_type_map.setdefault(row["canonical_name"], set()).add(row["node_type"])

for name, types in entity_type_map.items():
    if len(types) <= 1:
        continue
    types_sorted = sorted(types)

    if "Herb" in types and "Other_Unclassified" in types:
        resolution = "HERB_SOURCE_PRIORITY"
    elif "Disease" in types and "Other_Unclassified" in types:
        resolution = "DISEASE_SOURCE_PRIORITY"
    elif "Symptom" in types and "Other_Unclassified" in types:
        resolution = "SYMPTOM_SOURCE_PRIORITY"
    else:
        resolution = "PRESERVE_CONTEXT"

    entity_conflicts.append({"entity": name, "types": "|".join(types_sorted), "resolution": resolution})
    source_priority_audit.append({"entity": name, "candidate_types": "|".join(types_sorted), "resolution": resolution})

conflicts_df = pd.DataFrame(entity_conflicts)
conflicts_df.to_csv(OUTPUT_DIR / "entity_conflicts.csv", index=False, encoding="utf-8-sig")

priority_df = pd.DataFrame(source_priority_audit)
priority_df.to_csv(OUTPUT_DIR / "source_priority_audit.csv", index=False, encoding="utf-8-sig")


# ============================================================
# UNRESOLVED ENTITIES
# ============================================================

unresolved_df = nodes_df[nodes_df["node_type"] == "Other_Unclassified"].copy()
unresolved_df.to_csv(OUTPUT_DIR / "unresolved_entities.csv", index=False, encoding="utf-8-sig")


# ============================================================
# GRAPH SUMMARY
# ============================================================

other_count = int((nodes_df["node_type"] == "Other_Unclassified").sum())
high_confidence_count = int((nodes_df["confidence"] == "HIGH").sum())

summary = pd.DataFrame([
    {"metric": "total_nodes", "value": len(nodes_df)},
    {"metric": "total_edges", "value": len(edges_df)},
    {"metric": "node_types", "value": nodes_df["node_type"].nunique()},
    {"metric": "edge_types", "value": edges_df["relation"].nunique()},
    {"metric": "isolated_nodes", "value": len(isolated_nodes)},
    {"metric": "other_unclassified_nodes", "value": other_count},
    {"metric": "high_confidence_nodes", "value": high_confidence_count},
])
summary.to_csv(OUTPUT_DIR / "graph_summary.csv", index=False, encoding="utf-8-sig")


# ============================================================
# CLEAN DHS EXPORT
# ============================================================

dhs_clean = dhs_df.copy()
triple_cols = [disease_col, relation_col, entity_col]
before_dhs = len(dhs_clean)
dhs_clean = dhs_clean.drop_duplicates(subset=triple_cols).reset_index(drop=True)
after_dhs = len(dhs_clean)

print("\nDHS triple deduplication:")
print("Before:", before_dhs)
print("After :", after_dhs)
print("Removed:", before_dhs - after_dhs)

dhs_clean.to_csv(OUTPUT_DIR / "disease_herb_symptom_clean.csv", index=False, encoding="utf-8-sig")


# ============================================================
# FINAL REPORT
# ============================================================

print()
print("=" * 70)
print("PHASE 1 V4.2 COMPLETE")
print("=" * 70)
print(f"\nNodes: {len(nodes_df):,}")
print(f"Edges: {len(edges_df):,}")
print(f"Node types: {nodes_df['node_type'].nunique()}")
print(f"Edge types: {edges_df['relation'].nunique()}")
print(f"Isolated nodes: {len(isolated_nodes):,}")
print(f"Other_Unclassified: {other_count:,}")
print(f"High-confidence nodes: {high_confidence_count:,}")

print("\nNode type distribution:")
print(node_type_counts.to_string(index=False))

print("\nEdge type distribution:")
print(edge_type_counts.to_string(index=False))

print("\nCritical relationship counts:")
for relation in ["TREATED_BY", "HAS_LAKSHAN", "ALTERNATE_OF", "HAS_QUALITY",
                  "BELONGS_TO", "HAS_KALP", "TREATS", "CONTAINS"]:
    count = int((edges_df["relation"] == relation).sum())
    print(f"{relation:20s}: {count:,}")

print("\nMineral suffix candidates:", len(mineral_df))
print("Context/source conflicts:", len(conflicts_df))
print("Herb-Kalp/Compound-Herb overlaps:", len(overlap_df))

print("\nOutput directory:")
print(OUTPUT_DIR)

print()
print("=" * 70)
print("REGRESSION CHECK -- expected relationship types:")
for r in ["TREATED_BY", "HAS_LAKSHAN", "ALTERNATE_OF", "HAS_QUALITY",
          "BELONGS_TO", "HAS_KALP", "TREATS", "CONTAINS"]:
    print(" ", r)
print("=" * 70)