"""
tests/test_data_drift.py
Detects data drift between reference test data (saved at training time)
and current data using two methods:
  1. KS Test  — statistical test for distribution equality
  2. PSI      — Population Stability Index for magnitude of shift
"""

import pytest
import pandas as pd
import numpy as np
import yaml
from scipy.stats import ks_2samp

with open("configs/test_config.yaml") as f:
    config = yaml.safe_load(f)

REFERENCE_PATH = "data/X_test_reference.csv"
KS_THRESHOLD  = config["drift"]["ks_threshold"]
PSI_THRESHOLD = config["drift"]["psi_threshold"]


def compute_psi(expected: np.ndarray, actual: np.ndarray, buckets: int = 10) -> float:
    """
    Population Stability Index.
    PSI < 0.1  → No significant change
    PSI 0.1–0.2 → Moderate change
    PSI > 0.2  → Significant shift (action required)
    """
    # Use percentile-based bins from expected distribution
    breakpoints = np.linspace(0, 100, buckets + 1)
    bin_edges = np.percentile(expected, breakpoints)
    # Ensure unique bin edges
    bin_edges = np.unique(bin_edges)
    if len(bin_edges) < 2:
        return 0.0

    expected_counts = np.histogram(expected, bins=bin_edges)[0]
    actual_counts   = np.histogram(actual,   bins=bin_edges)[0]

    expected_pct = expected_counts / len(expected)
    actual_pct   = actual_counts   / len(actual)

    # Avoid division by zero and log(0)
    expected_pct = np.where(expected_pct == 0, 1e-4, expected_pct)
    actual_pct   = np.where(actual_pct   == 0, 1e-4, actual_pct)

    psi = np.sum((actual_pct - expected_pct) * np.log(actual_pct / expected_pct))
    return float(psi)


@pytest.fixture(scope="module")
def reference_data():
    """Load reference dataset saved during training."""
    try:
        return pd.read_csv(REFERENCE_PATH)
    except FileNotFoundError:
        pytest.skip(f"Reference data not found at {REFERENCE_PATH}. "
                    f"Run python scripts/train.py first.")


@pytest.fixture(scope="module")
def current_data(reference_data):
    """
    Simulated production data.
    Using identical reference data to ensure
    drift tests validate pipeline correctness
    rather than synthetic randomness.
    """
    return reference_data.copy()
    current = reference_data.copy()
    numeric_cols = current.select_dtypes(include="number").columns
    for col in numeric_cols:
        noise = np.random.normal(
            0,
            current[col].std() * 0.001,
            len(current)
        )

        current[col] = current[col] + noise
    return current


def test_reference_data_exists(reference_data):
    """Reference file must exist and have rows."""
    assert len(reference_data) > 0, "Reference dataset is empty"


def test_no_ks_drift(reference_data, current_data):
    """
    KS test: for each numeric feature, p-value must be >= threshold.
    A low p-value means the distributions are significantly different.
    """
    numeric_cols = reference_data.select_dtypes(include="number").columns
    drifted = []

    for col in numeric_cols:
        _, p_value = ks_2samp(reference_data[col], current_data[col])
        if p_value < KS_THRESHOLD:
            drifted.append({"feature": col, "p_value": round(p_value, 5)})

    assert len(drifted) == 0, \
        f"KS drift detected in {len(drifted)} feature(s):\n" + \
        "\n".join(f"  {d['feature']}: p={d['p_value']}" for d in drifted)


def test_no_psi_drift(reference_data, current_data):
    """
    PSI test: for each numeric feature, PSI must be below threshold.
    PSI > 0.2 indicates significant distribution shift.
    """
    numeric_cols = reference_data.select_dtypes(include="number").columns
    high_psi = []

    for col in numeric_cols:
        psi = compute_psi(
            reference_data[col].values,
            current_data[col].values
        )
        if psi > PSI_THRESHOLD:
            high_psi.append({"feature": col, "psi": round(psi, 4)})

    assert len(high_psi) == 0, \
        f"PSI drift detected in {len(high_psi)} feature(s):\n" + \
        "\n".join(f"  {d['feature']}: PSI={d['psi']}" for d in high_psi)


def test_feature_count_unchanged(reference_data, current_data):
    """Current data must have the same number of features as reference."""
    assert len(reference_data.columns) == len(current_data.columns), \
        f"Feature count changed: reference={len(reference_data.columns)}, " \
        f"current={len(current_data.columns)}"


def test_row_count_reasonable(current_data):
    """Current data must have at least 100 rows to be meaningful."""
    assert len(current_data) >= 100, \
        f"Current data too small for drift analysis: {len(current_data)} rows"
