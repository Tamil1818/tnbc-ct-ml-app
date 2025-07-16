import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer

def extract_ml_features(df):
    df = df.copy()
    df["biomarkers"] = df["title"].str.extractall(
        r'\b(BRCA1|BRCA2|PD-1|PD-L1|HER2|EGFR|TP53|AKT1|PIK3CA)\b'
    ).groupby(level=0)[0].apply(lambda x: list(set(x)))
    df["biomarkers"] = df["biomarkers"].apply(lambda x: x if isinstance(x, list) else [])

    df["phase"] = df["phase"].fillna("Not Reported").str.upper()
    df["sponsor"] = df["sponsor"].fillna("Unknown")
    df["intervention_type"] = df["intervention_type"].fillna("Unknown")

    features = df[["phase", "sponsor", "intervention_type", "biomarkers"]]
    y = df["outcome"]

    encoded = pd.get_dummies(features[["phase", "sponsor", "intervention_type"]])
    mlb = MultiLabelBinarizer()
    bio_df = pd.DataFrame(mlb.fit_transform(features["biomarkers"]), columns=mlb.classes_)

    encoded = pd.concat([encoded, bio_df], axis=1)
    return encoded, y, encoded.columns.tolist()
