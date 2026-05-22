# ML Model QA Automation Framework

A production-grade automated testing framework that benchmarks **XGBoost**, **RandomForest**, and **SVM** models across accuracy, F1, robustness, and data drift metrics. Integrated with GitHub Actions CI/CD and a Streamlit dashboard for real-time model health monitoring.

---

## 🏗️ Project Structure

```
qa-ml-automation-framework/
├── configs/
│   └── test_config.yaml          # Thresholds, model paths, drift settings
├── data/
│   ├── Telco_Customer_Churn.csv  # Dataset 
│   └── X_test_reference.csv    
├── models/
│   ├── xgboost.pkl               # Trained XGBoost model
│   ├── random_forest.pkl         # Trained RandomForest model
│   ├── svm.pkl                   # Trained SVM model
│   └── feature_names.pkl         # Feature alignment reference
├── scripts/
│   ├── train.py                  
│   └── predict.py               
├── tests/
│   ├── conftest.py              
│   ├── test_data_validation.py  
│   ├── test_model_training.py    
│   ├── test_predictions.py       
│   ├── test_robustness.py        
│   └── test_data_drift.py        
├── .github/workflows/
│   └── ci.yml                    
├── app.py
└── requirements.txt
```

---

## 🚀 Quick Start

### 1. Install Necessary requirements



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
| XGBoost     | ≥ 0.78   | ≥ 0.54   |
| RandomForest| ≥ 0.76   | ≥ 0.52   |
| SVM         | ≥ 0.73   | ≥ 0.50   |

---

## 🌊 Drift Detection

Two methods are used:

- **KS Test** — Kolmogorov-Smirnov test compares distributions; p-value < 0.05 = drift
- **PSI** — Population Stability Index; PSI > 0.2 = significant distribution shift

---
### 📈 Launch Dashboard
py -m streamlit run app.py

Open in browser:
http://localhost:8501
<img width="1920" height="1200" alt="image" src="https://github.com/user-attachments/assets/6f6e7e1e-c88c-4ddd-87c1-17fa709883e6" />


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
