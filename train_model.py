import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from ml_features import extract_ml_features

df = pd.read_csv("data/tnbc_trials_labeled.csv")
X, y, feature_names = extract_ml_features(df)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

joblib.dump(model, "models/trial_outcome_model.pkl")
joblib.dump(feature_names, "models/feature_names.pkl")

print(f"✅ Model trained. Accuracy on test: {model.score(X_test, y_test):.2f}")
print("✅ Model and features saved.")
