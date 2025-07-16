# app.py

import streamlit as st
import pandas as pd
from pyvis.network import Network
from streamlit.components.v1 import html
import os
import joblib

# ----------------------------
# App Config
# ----------------------------
st.set_page_config(page_title="TNBC KG & ML", layout="wide")

KG_DATA_PATH = "data/tnbc_kg_triplets.csv"
TRIAL_DATA_PATH = "data/tnbc_trials_labeled.csv"

# ----------------------------
# Load Data
# ----------------------------
@st.cache_data
def load_kg_data():
    if os.path.exists(KG_DATA_PATH):
        return pd.read_csv(KG_DATA_PATH)
    else:
        st.error("❌ KG data file not found.")
        return pd.DataFrame()

@st.cache_data
def load_trial_data():
    if os.path.exists(TRIAL_DATA_PATH):
        return pd.read_csv(TRIAL_DATA_PATH)
    else:
        st.warning("⚠️ Trial data file not found.")
        return pd.DataFrame()

kg_df = load_kg_data()
trial_df = load_trial_data()

# ----------------------------
# Header
# ----------------------------
st.markdown("<h1>🔬 TNBC Clinical Trial Knowledge Graph</h1><hr>", unsafe_allow_html=True)

# ----------------------------
# Sidebar Filters
# ----------------------------
st.sidebar.header("🎯 Filters")
query = st.sidebar.text_input("🔍 Search")
relations = ["All"] + sorted(kg_df["relation"].unique())
selected_relations = st.sidebar.multiselect("Relations", relations, default=["All"])

kg_filtered = kg_df if "All" in selected_relations else kg_df[kg_df["relation"].isin(selected_relations)]

if query:
    kg_filtered = kg_filtered[
        kg_filtered["head"].str.contains(query, case=False, na=False) |
        kg_filtered["tail"].str.contains(query, case=False, na=False)
    ]

# ----------------------------
# Show Triplets Table
# ----------------------------
st.subheader("📄 Triplets Table")
st.dataframe(kg_filtered.head(200), use_container_width=True)

# ----------------------------
# Node Inference + Color
# ----------------------------
def infer_type(node):
    node = str(node).strip()
    if node.startswith("NCT"):
        return "TRIAL"
    elif node.upper() in ["PD-1", "PD-L1", "BRCA1", "BRCA2", "VEGF", "HER2", "EGFR", "TP53", "AKT1", "PIK3CA"]:
        return "GENE"
    elif "university" in node.lower() or "center" in node.lower() or "institute" in node.lower():
        return "SPONSOR"
    elif any(k in node.lower() for k in ["umab", "limab", "tinib", "drug", "ol", "inib"]):
        return "DRUG"
    elif len(node) < 12 and node.isupper():
        return "BIOMARKER"
    else:
        return "OTHER"

color_map = {
    "GENE": "#FF6B6B",
    "DRUG": "#89CFF0",
    "TRIAL": "#B0E57C",
    "SPONSOR": "#FFD700",
    "BIOMARKER": "#FFA07A",
    "OTHER": "#D3D3D3"
}

# ----------------------------
# Draw Knowledge Graph
# ----------------------------
def draw_network(df, limit=150):
    df = df.dropna(subset=["head", "tail", "relation"]).copy().head(limit)
    net = Network(height="600px", width="100%", directed=True)

    nodes = set(df["head"]).union(set(df["tail"]))
    for node in nodes:
        ntype = infer_type(node)
        net.add_node(str(node), label=str(node), color=color_map.get(ntype, "#D3D3D3"), title=f"{ntype}: {node}")

    for _, row in df.iterrows():
        net.add_edge(str(row["head"]), str(row["tail"]), label=row["relation"], title=f"{row['head']} → {row['tail']}")

    return net

# ----------------------------
# Show Graph
# ----------------------------
st.subheader("🧠 Knowledge Graph")
if kg_filtered.empty:
    st.warning("⚠️ No data to visualize.")
else:
    net = draw_network(kg_filtered)
    net.save_graph("kg.html")
    with open("kg.html", "r", encoding="utf-8") as f:
        html(f.read(), height=600, scrolling=True)

# ----------------------------
# ML Prediction Section
# ----------------------------
st.markdown("---")
st.header("🔮 Predict Trial Outcome")

if not trial_df.empty:
    selected_title = st.selectbox("🎓 Select Trial Title", trial_df["title"].dropna().unique())

    trial = trial_df[trial_df["title"] == selected_title].iloc[0]

    st.markdown(f"""
    **Trial ID**: {trial['trial_id']}  
    **Phase**: {trial['phase']}  
    **Sponsor**: {trial['sponsor']}  
    **Intervention Type**: {trial['intervention_type']}  
    """)

    biomarkers = ["BRCA1", "BRCA2", "PD-1", "PD-L1", "HER2", "EGFR", "TP53", "AKT1", "PIK3CA"]
    selected_biomarkers = st.multiselect("🧬 Select Biomarkers", biomarkers)

    try:
        model = joblib.load("models/trial_outcome_model.pkl")
        feature_names = joblib.load("models/feature_names.pkl")

        # Build feature vector
        features = {
            f"phase_{trial['phase']}": 1,
            f"sponsor_{trial['sponsor']}": 1,
            f"intervention_type_{trial['intervention_type']}": 1,
            **{bm: 1 for bm in selected_biomarkers}
        }

        user_df = pd.DataFrame([features])
        for col in feature_names:
            if col not in user_df:
                user_df[col] = 0
        user_df = user_df[feature_names]

        pred = model.predict(user_df)[0]
        prob = model.predict_proba(user_df)[0][1]

        st.success(f"📈 Prediction: {'✅ Success Likely' if pred == 1 else '❌ Likely to Fail'}")
        st.write(f"📊 Confidence Score: **{prob:.2f}**")

    except Exception as e:
        st.error(f"⚠️ Model prediction failed: {e}")
else:
    st.warning("⚠️ Trial data not available for prediction.")

# ----------------------------
# Footer
# ----------------------------
st.markdown("---")
st.caption("🚀 Built with ❤️ for TNBC Research • GitHub: Tamil1818/tnbc-kg-app • Streamlit • PyVis • ClinicalTrials.gov")
