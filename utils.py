import re
import pandas as pd
from collections import defaultdict

# -----------------------------
# Static Drug → Gene Targets
# -----------------------------
drug_gene_targets = {
    "Pembrolizumab": ["PD-1"],
    "Nivolumab": ["PD-1"],
    "Atezolizumab": ["PD-L1"],
    "Olaparib": ["BRCA1", "BRCA2"]
}

# -----------------------------
# Normalize TNBC Mentions
# -----------------------------
def normalize_tnbc_targets(df):
    synonyms = {"triple negative breast cancer", "tnbc", "triple-negative breast cancer"}
    df["tail"] = df["tail"].apply(
        lambda x: "TNBC" if str(x).strip().lower() in synonyms else x
    )
    return df

# -----------------------------
# Enrich KG with Drug → Gene Links
# -----------------------------
def enrich_with_gene_targets(df):
    new_edges = [
        {"head": drug, "relation": "targets_gene", "tail": gene}
        for drug, genes in drug_gene_targets.items()
        for gene in genes
    ]
    return pd.concat([df, pd.DataFrame(new_edges)], ignore_index=True)

# -----------------------------
# Classify Trial Type by Keywords
# -----------------------------
def classify_trial_type(df):
    keywords = {
        "IMMUNO": ["pembrolizumab", "nivolumab", "atezolizumab", "pd-1", "pd-l1"],
        "CHEMO": ["paclitaxel", "carboplatin", "cisplatin", "doxorubicin"],
        "TARGETED": ["olaparib", "trastuzumab", "bevacizumab", "brca", "her2"]
    }

    trial_types = defaultdict(set)

    for _, row in df.iterrows():
        tail = str(row["tail"]).lower()
        for trial_type, match_words in keywords.items():
            if any(keyword in tail for keyword in match_words):
                trial_types[row["head"]].add(trial_type)

    extra = [
        {"head": head, "relation": "has_trial_type", "tail": ", ".join(sorted(types))}
        for head, types in trial_types.items()
    ]
    return pd.concat([df, pd.DataFrame(extra)], ignore_index=True)

# -----------------------------
# Extract Gene Mentions from Descriptions
# -----------------------------
def extract_genes_from_text(trial_descs):
    genes = ["BRCA1", "BRCA2", "PD-1", "PD-L1", "HER2", "EGFR", "TP53", "AKT1", "PIK3CA", "CDK4", "CDK6"]
    pattern = re.compile(r"\b(" + "|".join(genes) + r")\b", flags=re.IGNORECASE)

    mentions = []
    for trial in trial_descs:
        nct_id = trial.get("nctId", "")
        desc = trial.get("description", {}).get("text", "")
        for gene in set(re.findall(pattern, desc)):
            mentions.append({
                "head": nct_id,
                "relation": "mentions_gene",
                "tail": gene.upper()
            })

    return pd.DataFrame(mentions)
