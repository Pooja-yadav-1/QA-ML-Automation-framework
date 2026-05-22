"""
tests/test_robustness.py
Tests model stability under adversarial or edge-case inputs:
noisy data, all-zero inputs, extreme values, and single-feature variance.
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


@pytest.mark.parametrize("model_cfg", config["models"])
def test_handles_all_zeros(model_cfg, feature_names):
    """Model must not crash or return NaN on all-zero input."""

    n_features = len(feature_names)

    model = joblib.load(model_cfg["path"])

    zero_input = pd.DataFrame(
        np.zeros((5, n_features)),
        columns=feature_names
    )

    preds = model.predict(zero_input)

    assert len(preds) == 5

    assert not any(np.isnan(preds.astype(float))), \
        f"{model_cfg['name']} returned NaN on zero input"


@pytest.mark.parametrize("model_cfg", config["models"])
def test_handles_random_noise(model_cfg, feature_names):
    """Model must handle random float noise without crashing."""

    n_features = len(feature_names)

    model = joblib.load(model_cfg["path"])

    noisy = pd.DataFrame(
        np.random.rand(10, n_features),
        columns=feature_names
    )

    preds = model.predict(noisy)

    assert len(preds) == 10


@pytest.mark.parametrize("model_cfg", config["models"])
def test_handles_extreme_values(model_cfg, feature_names):
    """Model should not crash on extreme values."""

    n_features = len(feature_names)

    model = joblib.load(model_cfg["path"])

    extreme = pd.DataFrame(
        np.full((5, n_features), 9999.0),
        columns=feature_names
    )

    try:
        preds = model.predict(extreme)
        assert len(preds) == 5

    except Exception as e:
        pytest.fail(f"{model_cfg['name']} crashed on extreme values: {e}")


@pytest.mark.parametrize("model_cfg", config["models"])
def test_prediction_determinism_across_calls(model_cfg, feature_names):
    """
    Freshly loaded models should produce identical predictions.
    """

    n_features = len(feature_names)

    sample = pd.DataFrame(
        np.ones((5, n_features)),
        columns=feature_names
    )

    model1 = joblib.load(model_cfg["path"])
    model2 = joblib.load(model_cfg["path"])

    pred1 = list(model1.predict(sample))
    pred2 = list(model2.predict(sample))

    assert pred1 == pred2, \
        f"{model_cfg['name']} gives different predictions"


@pytest.mark.parametrize("model_cfg", config["models"])
def test_feature_count_mismatch_raises(model_cfg, feature_names):
    """Model must raise error on incorrect feature count."""

    n_features = len(feature_names)

    model = joblib.load(model_cfg["path"])

    wrong_input = pd.DataFrame(
        np.ones((3, n_features - 1))
    )

    with pytest.raises(Exception):
        model.predict(wrong_input)
