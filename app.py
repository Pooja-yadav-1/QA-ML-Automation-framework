"""
app.py
Streamlit dashboard for real-time ML model health monitoring.
Tabs: Prediction | Model Comparison | Data Drift
Run: streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score
from scipy.stats import ks_2samp

# ── Page Config ────────────────────────────────────────────────
st.set_page_config(
    page_title="ML Model QA Dashboard",
    page_icon="🧪",
    layout="wide"
)

# ── Custom CSS ─────────────────────────────────────────────────
st.markdown("""
<style>
    .metric-card {
        background: #1e1e2e;
        border-radius: 10px;
        padding: 16px;
        border-left: 4px solid #7c3aed;
        margin-bottom: 10px;
    }
    .pass-badge { color: #22c55e; font-weight: bold; }
    .fail-badge { color: #ef4444; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

# ── Model Registry ─────────────────────────────────────────────
MODEL_CONFIGS = {
    "XGBoost": {
        "path": "models/xgboost.pkl",
        "threshold_acc": 0.78,
        "threshold_f1":  0.55,
        "color": "#7c3aed"
    },
    "RandomForest": {
        "path": "models/random_forest.pkl",
        "threshold_acc": 0.76,
        "threshold_f1":  0.52,
        "color": "#2563eb"
    },
    "SVM": {
        "path": "models/svm.pkl",
        "threshold_acc": 0.74,
        "threshold_f1":  0.50,
        "color": "#0891b2"
    },
}

# ── Helpers ────────────────────────────────────────────────────
def encode_inputs(df):
    df = df.copy()
    for col in df.select_dtypes(include="object").columns:
        df[col] = LabelEncoder().fit_transform(df[col])
    return df


@st.cache_resource
def load_model(path):
    if os.path.exists(path):
        return joblib.load(path)
    return None


@st.cache_resource
def load_feature_names():
    if os.path.exists("models/feature_names.pkl"):
        return joblib.load("models/feature_names.pkl")
    return None


@st.cache_data
def load_eval_data():
    from sklearn.model_selection import train_test_split
    import yaml
    with open("configs/test_config.yaml") as f:
        config = yaml.safe_load(f)
    if not os.path.exists(config["data"]["path"]):
        return None, None
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


def compute_psi(expected, actual, buckets=10):
    breakpoints = np.linspace(0, 100, buckets + 1)
    bin_edges = np.unique(np.percentile(expected, breakpoints))
    if len(bin_edges) < 2:
        return 0.0
    exp_c = np.histogram(expected, bins=bin_edges)[0] / len(expected)
    act_c = np.histogram(actual,   bins=bin_edges)[0] / len(actual)
    exp_c = np.where(exp_c == 0, 1e-4, exp_c)
    act_c = np.where(act_c == 0, 1e-4, act_c)
    return float(np.sum((act_c - exp_c) * np.log(act_c / exp_c)))


def prepare_input(df, features):
    encoded = encode_inputs(df)
    if features:
        for col in features:
            if col not in encoded.columns:
                encoded[col] = 0
        encoded = encoded[features]
    return encoded


# ── Sidebar ────────────────────────────────────────────────────
st.sidebar.header("🧾 Customer Profile")
st.sidebar.markdown("---")

selected_model_name = st.sidebar.selectbox(
    "🤖 Select model for prediction",
    list(MODEL_CONFIGS.keys())
)

st.sidebar.markdown("---")
gender            = st.sidebar.selectbox("Gender", ["Male", "Female"])
senior            = st.sidebar.selectbox("Senior Citizen", [0, 1])
partner           = st.sidebar.selectbox("Has Partner", ["Yes", "No"])
dependents        = st.sidebar.selectbox("Has Dependents", ["Yes", "No"])
tenure            = st.sidebar.slider("Tenure (Months)", 0, 72, 12)
phone_service     = st.sidebar.selectbox("Phone Service", ["Yes", "No"])
multiple_lines    = st.sidebar.selectbox("Multiple Lines", ["Yes", "No", "No phone service"])
internet_service  = st.sidebar.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
online_security   = st.sidebar.selectbox("Online Security", ["Yes", "No", "No internet service"])
online_backup     = st.sidebar.selectbox("Online Backup", ["Yes", "No", "No internet service"])
device_protection = st.sidebar.selectbox("Device Protection", ["Yes", "No", "No internet service"])
tech_support      = st.sidebar.selectbox("Tech Support", ["Yes", "No", "No internet service"])
streaming_tv      = st.sidebar.selectbox("Streaming TV", ["Yes", "No", "No internet service"])
streaming_movies  = st.sidebar.selectbox("Streaming Movies", ["Yes", "No", "No internet service"])
contract          = st.sidebar.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
paperless_billing = st.sidebar.selectbox("Paperless Billing", ["Yes", "No"])
monthly_charges   = st.sidebar.number_input("Monthly Charges ($)", 0.0, 200.0, 70.0)
total_charges     = st.sidebar.number_input("Total Charges ($)", 0.0, 10000.0, 3000.0)

input_df = pd.DataFrame([{
    "gender": gender, "SeniorCitizen": senior,
    "Partner": partner, "Dependents": dependents,
    "tenure": tenure, "PhoneService": phone_service,
    "MultipleLines": multiple_lines, "InternetService": internet_service,
    "OnlineSecurity": online_security, "OnlineBackup": online_backup,
    "DeviceProtection": device_protection, "TechSupport": tech_support,
    "StreamingTV": streaming_tv, "StreamingMovies": streaming_movies,
    "Contract": contract, "PaperlessBilling": paperless_billing,
    "MonthlyCharges": monthly_charges, "TotalCharges": total_charges
}])

feature_names = load_feature_names()

# ── Title ──────────────────────────────────────────────────────
st.title("🧪 ML Model QA Automation Dashboard")
st.markdown("Real-time model health monitoring · **XGBoost** · **RandomForest** · **SVM**")
st.markdown("---")

# ── Tabs ───────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["🔮 Prediction", "📊 Model Comparison", "🌊 Data Drift"])

# ───────────── TAB 1: PREDICTION ──────────────────────────────
with tab1:
    st.subheader(f"Prediction using **{selected_model_name}**")

    cfg   = MODEL_CONFIGS[selected_model_name]
    model = load_model(cfg["path"])

    if model is None:
        st.warning(f"⚠️ {selected_model_name} model not found. "
                   f"Run `python scripts/train.py` first.")
    else:
        enc_input  = prepare_input(input_df, feature_names)
        prediction = model.predict(enc_input)[0]
        prob       = model.predict_proba(enc_input)[0][1]

        c1, c2, c3 = st.columns(3)
        c1.metric("Model",            selected_model_name)
        c2.metric("Churn Probability", f"{prob:.2%}")
        c3.metric("Verdict",          "🔴 Will Churn" if prediction == 1 else "🟢 Will Stay")

        st.progress(int(prob * 100))

        if prediction == 1:
            st.error("⚠️ This customer is **likely to churn**. Consider a retention offer.")
        else:
            st.success("✅ This customer is **not likely to churn**.")

        st.markdown("#### Compare All Models on This Customer")
        cols = st.columns(3)
        for i, (name, c) in enumerate(MODEL_CONFIGS.items()):
            m = load_model(c["path"])
            with cols[i]:
                if m:
                    enc  = prepare_input(input_df, feature_names)
                    p    = m.predict_proba(enc)[0][1]
                    pred = m.predict(enc)[0]
                    st.metric(
                        label=name,
                        value=f"{p:.2%}",
                        delta="↑ Churn Risk" if pred == 1 else "↓ Stable"
                    )
                else:
                    st.metric(name, "Not loaded", delta="train first")

# ───────────── TAB 2: MODEL COMPARISON ────────────────────────
with tab2:
    st.subheader("📊 Model Health & QA Metrics")

    X_test, y_test = load_eval_data()

    if X_test is None:
        st.warning("Dataset not found. Place `Telco_Customer_Churn.csv` in `data/`.")
    else:
        results = []
        for name, cfg in MODEL_CONFIGS.items():
            m = load_model(cfg["path"])
            if m:
                preds    = m.predict(X_test)
                acc      = accuracy_score(y_test, preds)
                f1       = f1_score(y_test, preds)
                acc_pass = acc >= cfg["threshold_acc"]
                f1_pass  = f1  >= cfg["threshold_f1"]
                results.append({
                    "Model":         name,
                    "Accuracy":      round(acc, 4),
                    "F1 Score":      round(f1, 4),
                    "Acc Threshold": cfg["threshold_acc"],
                    "F1 Threshold":  cfg["threshold_f1"],
                    "Acc Status":    "✅ PASS" if acc_pass else "❌ FAIL",
                    "F1 Status":     "✅ PASS" if f1_pass  else "❌ FAIL",
                    "Overall":       "✅ PASS" if (acc_pass and f1_pass) else "❌ FAIL"
                })
            else:
                results.append({
                    "Model": name, "Accuracy": "—", "F1 Score": "—",
                    "Acc Threshold": cfg["threshold_acc"],
                    "F1 Threshold": cfg["threshold_f1"],
                    "Acc Status": "⚠️ Not loaded",
                    "F1 Status": "⚠️ Not loaded",
                    "Overall": "⚠️ Not loaded"
                })

        st.dataframe(pd.DataFrame(results), use_container_width=True)

        # Bar chart
        loaded = [r for r in results if r["Accuracy"] != "—"]
        if loaded:
            try:
                import plotly.graph_objects as go
                fig = go.Figure()
                names = [r["Model"]    for r in loaded]
                accs  = [r["Accuracy"] for r in loaded]
                f1s   = [r["F1 Score"] for r in loaded]
                fig.add_trace(go.Bar(name="Accuracy", x=names, y=accs,
                                     marker_color="#7c3aed"))
                fig.add_trace(go.Bar(name="F1 Score",  x=names, y=f1s,
                                     marker_color="#2563eb"))
                fig.update_layout(
                    barmode="group",
                    title="Accuracy vs F1 Score by Model",
                    yaxis=dict(range=[0.4, 1.0]),
                    plot_bgcolor="#0e1117",
                    paper_bgcolor="#0e1117",
                    font_color="white"
                )
                st.plotly_chart(fig, use_container_width=True)
            except ImportError:
                st.info("Install plotly for the chart: pip install plotly")

# ───────────── TAB 3: DATA DRIFT ──────────────────────────────
with tab3:
    st.subheader("🌊 Data Drift Detection")
    st.markdown("Compares reference distribution (saved at training) against current data.")

    ref_path = "data/X_test_reference.csv"

    if not os.path.exists(ref_path):
        st.warning("Reference data not found. Run `python scripts/train.py` first.")
    else:
        ref     = pd.read_csv(ref_path)
        current = ref.copy()
        # Simulate slight noise — replace with real live data in production
        num_cols = current.select_dtypes(include="number").columns
        for col in num_cols:
            current[col] = current[col] + np.random.normal(
                0, current[col].std() * 0.02, len(current)
            )

        ks_results  = []
        psi_results = []

        for col in num_cols:
            _, p = ks_2samp(ref[col], current[col])
            psi  = compute_psi(ref[col].values, current[col].values)
            ks_results.append({
                "Feature": col,
                "KS p-value": round(p, 5),
                "Drift?": "🔴 YES" if p < 0.05 else "🟢 No"
            })
            psi_results.append({
                "Feature": col,
                "PSI": round(psi, 4),
                "Status": "🔴 HIGH" if psi > 0.2 else ("🟡 MODERATE" if psi > 0.1 else "🟢 Stable")
            })

        drifted_ks  = [r for r in ks_results  if "YES" in r["Drift?"]]
        drifted_psi = [r for r in psi_results if "HIGH" in r["Status"]]

        c1, c2, c3 = st.columns(3)
        c1.metric("KS Drift Status",
                  "🔴 Drifted" if drifted_ks  else "🟢 Stable",
                  f"{len(drifted_ks)} feature(s)")
        c2.metric("PSI High Drift",
                  "🔴 Drifted" if drifted_psi else "🟢 Stable",
                  f"{len(drifted_psi)} feature(s)")
        c3.metric("Features Monitored", len(num_cols))

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**KS Test Results** (p < 0.05 = drift)")
            st.dataframe(pd.DataFrame(ks_results), use_container_width=True)
        with col2:
            st.markdown("**PSI Results** (> 0.2 = significant shift)")
            st.dataframe(pd.DataFrame(psi_results), use_container_width=True)

        with st.expander("ℹ️ How drift detection works"):
            st.markdown("""
            **KS Test (Kolmogorov-Smirnov):**
            Compares the cumulative distribution of each feature between reference and current data.
            A p-value < 0.05 means the distributions are statistically different.

            **PSI (Population Stability Index):**
            Measures the magnitude of distribution shift.
            - PSI < 0.1 → No significant change
            - PSI 0.1–0.2 → Moderate change, monitor
            - PSI > 0.2 → Significant drift, investigate

            **In production:** Replace the `current` data source above with live inference logs
            or a database query to detect real-world drift.
            """)

# ── Footer ─────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<center><small>ML Model QA Automation Framework · "
    "pytest · XGBoost · RandomForest · SVM · GitHub Actions · Streamlit</small></center>",
    unsafe_allow_html=True
)
