"""
train.py
Trains XGBoost, RandomForest, and SVM on the Telco Customer Churn dataset.
Saves each model and the reference test set for drift detection.
"""

import os
import pandas as pd
import joblib
import yaml
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# ── Load config ────────────────────────────────────────────────
with open("configs/test_config.yaml") as f:
    config = yaml.safe_load(f)

# ── Load & preprocess data ─────────────────────────────────────
print("Loading dataset...")
df = pd.read_csv(config["data"]["path"])
df = df.drop(columns=["customerID"], errors="ignore")
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
df = df.dropna()

# Encode all categorical columns
label_encoders = {}
for col in df.select_dtypes(include="object").columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

target = config["data"]["target_column"]
X = df.drop(columns=[target])
y = df[target]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ── Save feature names ─────────────────────────────────────────
os.makedirs("models", exist_ok=True)
joblib.dump(list(X.columns), "models/feature_names.pkl")
print(f"Feature names saved: {list(X.columns)}")

# ── Train & save each model ────────────────────────────────────
models = {
    "xgboost": XGBClassifier(
        use_label_encoder=False,
        eval_metric="logloss",
        random_state=42,
        n_estimators=100,
        learning_rate=0.1,
        max_depth=5
    ),
    "random_forest": RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        max_depth=10
    ),
    "svm": Pipeline([
    ("scaler", StandardScaler()),
    ("svm", SVC(
        kernel="rbf",
        probability=True,
        random_state=42,
        C=1.0,
        class_weight="balanced"
    ))
]),
}

for name, model in models.items():
    print(f"Training {name}...")
    model.fit(X_train, y_train)
    joblib.dump(model, f"models/{name}.pkl")
    print(f"  ✅ Saved to models/{name}.pkl")

# ── Save reference test set for drift detection ────────────────
os.makedirs("data", exist_ok=True)
X_test.to_csv("data/X_test_reference.csv", index=False)
print("✅ Reference test set saved to data/X_test_reference.csv")
print("\n✅ All models trained successfully.")
