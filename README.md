# ML Model QA Automation Framework

A production-grade automated testing framework that benchmarks **XGBoost**, **RandomForest**, and **SVM** models across accuracy, F1, robustness, and data drift metrics. Integrated with GitHub Actions CI/CD and a Streamlit dashboard for real-time model health monitoring.

---

## 🏗️ Project Structure

```
qa-ml-automation-framework/
├── configs/
│   └── test_config.yaml          # Thresholds, model paths, drift settings
├── data/
│   ├── Telco_Customer_Churn.csv  # Dataset (download separately)
│   └── X_test_reference.csv     # Auto-generated reference for drift detection
├── models/
│   ├── xgboost.pkl               # Trained XGBoost model
│   ├── random_forest.pkl         # Trained RandomForest model
│   ├── svm.pkl                   # Trained SVM model
│   └── feature_names.pkl         # Feature alignment reference
├── scripts/
│   ├── train.py                  # Trains all 3 models
│   └── predict.py                # Single inference script
├── tests/
│   ├── conftest.py               # Shared fixtures
│   ├── test_data_validation.py   # Dataset integrity tests
│   ├── test_model_training.py    # Accuracy, F1, reproducibility
│   ├── test_predictions.py       # Output validation
│   ├── test_robustness.py        # Edge case / noise tests
│   └── test_data_drift.py        # KS test + PSI drift detection
├── .github/workflows/
│   └── ci.yml                    # GitHub Actions pipeline
├── app.py                        # Streamlit dashboard
└── requirements.txt
```

---

## 🚀 Quick Start

### 1. Clone & install
```bash
git clone https://github.com/DarSahran/qa-ml-automation-framework.git
cd qa-ml-automation-framework
pip install -r requirements.txt
```

### 2. Add dataset
Download [Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) and place it at `data/Telco_Customer_Churn.csv`.

### 3. Train all models
```bash
python scripts/train.py
```
This trains XGBoost, RandomForest, and SVM, saves their PKL files, and creates the drift reference baseline.

### 4. Run tests
```bash
pytest tests/ -v
```

### 5. Launch dashboard
```bash
streamlit run app.py
```

---

## 🧪 Test Categories

| Test File | What It Tests |
|-----------|--------------|
| `test_data_validation.py` | Required columns, null %, class balance, data types |
| `test_model_training.py`  | Accuracy threshold, F1 threshold, reproducibility, output shape |
| `test_predictions.py`     | Valid class labels, proba sums to 1, batch & single inference |
| `test_robustness.py`      | Zero inputs, random noise, extreme values, feature mismatch |
| `test_data_drift.py`      | KS test per feature, PSI per feature, feature count stability |

---

## 📊 Model Thresholds

| Model        | Accuracy | F1 Score |
|-------------|----------|----------|
| XGBoost     | ≥ 0.78   | ≥ 0.55   |
| RandomForest| ≥ 0.76   | ≥ 0.52   |
| SVM         | ≥ 0.74   | ≥ 0.50   |

---

## 🌊 Drift Detection

Two methods are used:

- **KS Test** — Kolmogorov-Smirnov test compares distributions; p-value < 0.05 = drift
- **PSI** — Population Stability Index; PSI > 0.2 = significant distribution shift

---

## 🔧 Tech Stack

- **Python 3.10**
- **pytest** — test framework with parametrize for multi-model testing
- **XGBoost, scikit-learn** — ML models
- **scipy** — KS test for drift detection
- **Streamlit + Plotly** — monitoring dashboard
- **GitHub Actions** — CI/CD pipeline
- **YAML** — configuration-driven thresholds

---

## 📄 License
MIT
