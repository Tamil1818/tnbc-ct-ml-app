# ml/predict_trial.py
import pandas as pd
import joblib
from ml_features import extract_ml_features

def predict_trial(trial_id):
    df = pd.read_csv("data/tnbc_trials_labeled.csv")
    
    if trial_id not in df["trial_id"].values:
        print(f"❌ Trial ID '{trial_id}' not found.")
        return

    X, y, feature_names = extract_ml_features(df)
    model = joblib.load("models/trial_outcome_model.pkl")

    index = df[df["trial_id"] == trial_id].index[0]
    trial_features = X.iloc[[index]]

    prediction = model.predict(trial_features)[0]
    confidence = model.predict_proba(trial_features)[0][prediction]

    label = "Likely to Succeed ✅" if prediction == 1 else "Likely to Fail ❌"
    print(f"🔍 Prediction for {trial_id}: {label} (confidence: {confidence:.2f})")

# Example usage
if __name__ == "__main__":
    trial_id = input("Enter trial ID (e.g., NCT01234567): ")
    predict_trial(trial_id)
