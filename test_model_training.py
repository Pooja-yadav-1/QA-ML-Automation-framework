"""
tests/test_model_training.py
Validates accuracy, F1 score, and reproducibility for all configured models.
Uses @pytest.mark.parametrize to run each test against XGBoost, RandomForest, SVM.
"""

import pytest
import joblib
import yaml
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

with open("configs/test_config.yaml") as f:
    config = yaml.safe_load(f)


@pytest.fixture(scope="module")
def test_data():
    """Load and preprocess data, return test split."""
    df = pd.read_csv(config["data"]["path"])
    df = df.drop(columns=["customerID"], errors="ignore")
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df = df.dropna()
    for col in df.select_dtypes(include="object").columns:
        df[col] = LabelEncoder().fit_transform(df[col])
    target = config["data"]["target_column"]
    X = df.drop(columns=[target])
    y = df[target]
    _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_test, y_test


@pytest.mark.parametrize("model_cfg", config["models"])
def test_model_file_exists(model_cfg):
    """Model PKL file must exist on disk."""
    import os
    assert os.path.exists(model_cfg["path"]), \
        f"Model file not found: {model_cfg['path']}. Run python scripts/train.py first."


@pytest.mark.parametrize("model_cfg", config["models"])
def test_accuracy_above_threshold(model_cfg, test_data):
    """Each model must meet its configured accuracy threshold."""
    X_test, y_test = test_data
    model = joblib.load(model_cfg["path"])
    acc = accuracy_score(y_test, model.predict(X_test))
    print(f"\n{model_cfg['name']} Accuracy: {acc:.4f} (threshold: {model_cfg['accuracy_threshold']})")
    assert acc >= model_cfg["accuracy_threshold"], \
        f"{model_cfg['name']} accuracy {acc:.4f} < threshold {model_cfg['accuracy_threshold']}"


@pytest.mark.parametrize("model_cfg", config["models"])
def test_f1_above_threshold(model_cfg, test_data):
    """Each model must meet its configured F1 score threshold."""
    X_test, y_test = test_data
    model = joblib.load(model_cfg["path"])
    f1 = f1_score(y_test, model.predict(X_test))
    print(f"\n{model_cfg['name']} F1: {f1:.4f} (threshold: {model_cfg['f1_threshold']})")
    assert f1 >= model_cfg["f1_threshold"], \
        f"{model_cfg['name']} F1 {f1:.4f} < threshold {model_cfg['f1_threshold']}"


@pytest.mark.parametrize("model_cfg", config["models"])
def test_reproducibility(model_cfg, test_data):
    """Same input must produce identical predictions on repeated calls."""
    X_test, _ = test_data
    model = joblib.load(model_cfg["path"])
    sample = X_test.iloc[:20]
    pred1 = list(model.predict(sample))
    pred2 = list(model.predict(sample))
    assert pred1 == pred2, \
        f"{model_cfg['name']} predictions are not reproducible"


@pytest.mark.parametrize("model_cfg", config["models"])
def test_output_shape(model_cfg, test_data):
    """Model output must have same length as input."""
    X_test, _ = test_data
    model = joblib.load(model_cfg["path"])
    preds = model.predict(X_test)
    assert len(preds) == len(X_test), \
        f"{model_cfg['name']}: output length {len(preds)} != input length {len(X_test)}"


@pytest.mark.parametrize("model_cfg", config["models"])
def test_predict_proba_available(model_cfg, test_data):
    """Models must support probability output for dashboard use."""
    X_test, _ = test_data
    model = joblib.load(model_cfg["path"])
    proba = model.predict_proba(X_test[:5])
    assert proba.shape == (5, 2), \
        f"{model_cfg['name']} predict_proba shape unexpected: {proba.shape}"
    assert all(0 <= p <= 1 for row in proba for p in row), \
        f"{model_cfg['name']} predict_proba returned values outside [0, 1]"
