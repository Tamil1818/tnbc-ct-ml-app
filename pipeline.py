# pipeline.py

import requests
import pandas as pd
from utils import (
    enrich_with_gene_targets,
    normalize_tnbc_targets,
    classify_trial_type,
)

# -----------------------------
# STEP 1: Fetch Clinical Trials
# -----------------------------
def fetch_all_trials(search_term="triple-negative breast cancer", max_trials=2000, page_size=100):
    base = "https://beta.clinicaltrials.gov/api/v2/studies"
    params = {"query.term": search_term, "pageSize": page_size}
    all_trials, token = [], ""

    while len(all_trials) < max_trials:
        if token:
            params["pageToken"] = token
        res = requests.get(base, params=params)
        if res.status_code != 200:
            print("❌ Failed to fetch trials.")
            break
        data = res.json()
        all_trials.extend(data.get("studies", []))
        token = data.get("nextPageToken", "")
        if not token:
            break

    print(f"✅ Fetched {len(all_trials)} clinical trials.")
    return all_trials

# -----------------------------
# STEP 2: Normalize Raw Trial Data
# -----------------------------
def normalize_trials(raw_trials):
    rows = []
    for trial in raw_trials:
        ps = trial.get("protocolSection", {})
        arms = ps.get("armsInterventionsModule", {}).get("interventions", [])
        ids = ps.get("identificationModule", {})
        rows.append({
            "trial_id": ids.get("nctId", ""),
            "title": ids.get("officialTitle", ""),
            "intervention_name": "; ".join([a.get("name", "") for a in arms]),
            "intervention_type": "; ".join([a.get("type", "") for a in arms]),
            "phase": ps.get("designModule", {}).get("phaseList", {}).get("phases", [""])[0],
            "status": ps.get("statusModule", {}).get("overallStatus", ""),
            "summary": ps.get("descriptionModule", {}).get("briefSummary", ""),
            "sponsor": ps.get("sponsorCollaboratorsModule", {}).get("leadSponsor", {}).get("name", "")
        })
    return pd.DataFrame(rows)

# -----------------------------
# STEP 3: Generate Triplets for KG
# -----------------------------
def generate_triplets(df):
    triplets = []
    for _, r in df.iterrows():
        tid = r["trial_id"]
        triplets += [
            {"head": tid, "relation": "has_title", "tail": r["title"][:100]},
            {"head": tid, "relation": "has_status", "tail": r["status"]},
            {"head": tid, "relation": "has_phase", "tail": r["phase"]},
            {"head": tid, "relation": "sponsored_by", "tail": r["sponsor"]}
        ]
        for drug, dtype in zip(r["intervention_name"].split("; "), r["intervention_type"].split("; ")):
            if drug.strip():
                triplets.append({"head": tid, "relation": "tests", "tail": drug.strip()})
                triplets.append({"head": drug.strip(), "relation": "has_type", "tail": dtype.strip()})
    return pd.DataFrame(triplets)

# -----------------------------
# STEP 4: Label Outcomes (for ML)
# -----------------------------
def label_outcome(status):
    if pd.isna(status):
        return None
    status = status.strip().upper()
    if status in ["COMPLETED", "ACTIVE_NOT_RECRUITING"]:
        return 1
    elif status in ["TERMINATED", "WITHDRAWN", "SUSPENDED"]:
        return 0
    return None

# -----------------------------
# STEP 5: Run the Full Pipeline
# -----------------------------
def run_pipeline():
    raw = fetch_all_trials()
    df = normalize_trials(raw)

    # Label outcomes for ML training
    df["outcome"] = df["status"].apply(label_outcome)
    df.dropna(subset=["outcome"], inplace=True)
    df.to_csv("data/tnbc_trials_labeled.csv", index=False)

    # Generate triplets and enrich
    triplets = generate_triplets(df)
    triplets = normalize_tnbc_targets(triplets)
    triplets = enrich_with_gene_targets(triplets)
    triplets = classify_trial_type(triplets)

    # Save final triplet file
    triplets.to_csv("data/tnbc_kg_triplets.csv", index=False)
    print(f"💾 Saved enriched triplets to: data/tnbc_kg_triplets.csv")

# -----------------------------
# Entry Point for Script
# -----------------------------
if __name__ == "__main__":
    run_pipeline()
