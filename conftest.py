"""
conftest.py
Shared pytest fixtures and configuration.
"""

import pytest
import pandas as pd
import yaml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

with open("configs/test_config.yaml") as f:
    config = yaml.safe_load(f)


@pytest.fixture(scope="session")
def full_dataframe():
    """Load and return the full cleaned dataset."""
    df = pd.read_csv(config["data"]["path"])
    df = df.drop(columns=["customerID"], errors="ignore")
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df = df.dropna()
    for col in df.select_dtypes(include="object").columns:
        df[col] = LabelEncoder().fit_transform(df[col])
    return df


@pytest.fixture(scope="session")
def train_test_split_data(full_dataframe):
    """Return X_train, X_test, y_train, y_test splits."""
    target = config["data"]["target_column"]
    X = full_dataframe.drop(columns=[target])
    y = full_dataframe[target]
    return train_test_split(X, y, test_size=0.2, random_state=42)
