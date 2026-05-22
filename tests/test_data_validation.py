"""
tests/test_data_validation.py
Validates the dataset structure, types, and integrity before training.
"""

import pytest
import pandas as pd
import yaml

with open("configs/test_config.yaml") as f:
    config = yaml.safe_load(f)


@pytest.fixture(scope="module")
def df():
    data = pd.read_csv(config["data"]["path"])
    data["TotalCharges"] = pd.to_numeric(data["TotalCharges"], errors="coerce")
    return data


def test_dataset_loads(df):
    """Dataset must load and have rows."""
    assert df is not None
    assert len(df) > 0, "Dataset is empty"


def test_required_columns_present(df):
    """All required columns from config must exist."""
    missing = [col for col in config["data"]["required_columns"]
               if col not in df.columns]
    assert len(missing) == 0, f"Missing columns: {missing}"


def test_no_excessive_nulls(df):
    """No column should have more than 10% null values."""
    null_pct = df.isnull().mean()
    bad_cols = null_pct[null_pct > 0.1].index.tolist()
    assert len(bad_cols) == 0, f"Columns with >10% nulls: {bad_cols}"


def test_target_column_binary(df):
    """Target column must be binary (2 unique values)."""
    target = config["data"]["target_column"]
    unique_vals = df[target].nunique()
    assert unique_vals == 2, \
        f"Target column should have 2 unique values, got {unique_vals}"


def test_numeric_columns_reasonable(df):
    """Numeric columns should not have negative values where unexpected."""
    assert (df["tenure"] >= 0).all(), "tenure contains negative values"
    assert (df["MonthlyCharges"] >= 0).all(), "MonthlyCharges contains negative values"


def test_minimum_dataset_size(df):
    """Dataset must have at least 1000 rows for meaningful training."""
    assert len(df) >= 1000, f"Dataset too small: {len(df)} rows"


def test_target_class_balance(df):
    """Neither class should be more than 90% of the dataset (extreme imbalance check)."""
    target = config["data"]["target_column"]
    counts = df[target].value_counts(normalize=True)
    for cls, pct in counts.items():
        assert pct < 0.90, \
            f"Class {cls} is {pct:.1%} of dataset — extreme imbalance"
