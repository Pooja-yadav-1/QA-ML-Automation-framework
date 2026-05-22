"""
tests/test_predictions.py
Validates prediction outputs — correct types, valid class labels, and
that probabilities sum to 1.
"""

import os
import pytest
import joblib
import numpy as np
import pandas as pd
import yaml

with open("configs/test_config.yaml") as f:
    config = yaml.safe_load(f)


@pytest.fixture(scope="module")
def feature_names():
    if not os.path.exists("models/feature_names.pkl"):
        pytest.skip("Run python scripts/train.py first")
    return joblib.load("models/feature_names.pkl")


def make_sample(feature_names, n=10):
    """Create a random but valid input DataFrame."""
    return pd.DataFrame(
        np.random.randint(0, 3, size=(n, len(feature_names))),
        columns=feature_names
    )


@pytest.mark.parametrize("model_cfg", config["models"])
def test_prediction_classes_valid(model_cfg, feature_names):
    """Predictions must only contain 0 or 1."""
    model = joblib.load(model_cfg["path"])
    preds = model.predict(make_sample(feature_names))

    invalid = [p for p in preds if p not in [0, 1]]

    assert len(invalid) == 0, \
        f"{model_cfg['name']} produced invalid class labels: {invalid}"


@pytest.mark.parametrize("model_cfg", config["models"])
def test_proba_sums_to_one(model_cfg, feature_names):
    """Probability rows must sum to ~1.0."""
    model = joblib.load(model_cfg["path"])
    proba = model.predict_proba(make_sample(feature_names))

    row_sums = proba.sum(axis=1)

    for i, s in enumerate(row_sums):
        assert abs(s - 1.0) < 1e-5, \
            f"{model_cfg['name']} row {i} probabilities sum to {s}"


@pytest.mark.parametrize("model_cfg", config["models"])
def test_batch_prediction(model_cfg, feature_names):
    """Model must handle batch of 100 samples without error."""
    model = joblib.load(model_cfg["path"])

    preds = model.predict(make_sample(feature_names, 100))

    assert len(preds) == 100


@pytest.mark.parametrize("model_cfg", config["models"])
def test_single_sample_prediction(model_cfg, feature_names):
    """Model must handle a single-row DataFrame."""
    model = joblib.load(model_cfg["path"])

    preds = model.predict(make_sample(feature_names, 1))

    assert len(preds) == 1
