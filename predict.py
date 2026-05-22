"""
predict.py
Run a single prediction using a trained model.
Usage: python scripts/predict.py --model xgboost
"""

import argparse
import joblib
import numpy as np
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument("--model", default="xgboost",
                    choices=["xgboost", "random_forest", "svm"])
args = parser.parse_args()

model = joblib.load(f"models/{args.model}.pkl")
feature_names = joblib.load("models/feature_names.pkl")

# Sample input (all zeros — replace with real values)
sample = pd.DataFrame([np.zeros(len(feature_names))], columns=feature_names)

prediction = model.predict(sample)[0]
probability = model.predict_proba(sample)[0][1]

print(f"Model     : {args.model}")
print(f"Prediction: {'Churn' if prediction == 1 else 'No Churn'}")
print(f"Probability: {probability:.4f}")
