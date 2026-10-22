"""Score customers with the saved churn model.

Usage:
    python predict.py                              # built-in sample customer
    python predict.py customers.csv                # score every row of a CSV
    python predict.py customers.csv -o scored.csv  # ...and save the result

The CSV needs the raw columns of the Telco dataset (customerID is optional,
Churn is ignored). Keep churn_utils.py next to this script: the saved model
refers to it by name.
"""
import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

MODEL_PATH = Path(__file__).parent / "customer_churn_model.pkl"
LOW_RISK_CUTOFF = 0.20

SAMPLE_CUSTOMER = {
    "gender": "Female",
    "SeniorCitizen": 0,
    "Partner": "No",
    "Dependents": "No",
    "tenure": 2,
    "PhoneService": "Yes",
    "MultipleLines": "No",
    "InternetService": "Fiber optic",
    "OnlineSecurity": "No",
    "OnlineBackup": "No",
    "DeviceProtection": "No",
    "TechSupport": "No",
    "StreamingTV": "Yes",
    "StreamingMovies": "Yes",
    "Contract": "Month-to-month",
    "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check",
    "MonthlyCharges": 95.7,
    "TotalCharges": 190.4,
}


def load_model(path=MODEL_PATH):
    saved = joblib.load(path)
    return saved["model"], saved["threshold"], saved["features"]


def prepare(customers, features):
    missing = [col for col in features if col not in customers.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    data = customers[features].copy()
    data["TotalCharges"] = pd.to_numeric(data["TotalCharges"], errors="coerce")
    data.loc[data["TotalCharges"].isna() & (data["tenure"] == 0), "TotalCharges"] = 0

    if data["TotalCharges"].isna().any():
        raise ValueError("TotalCharges is missing for customers with tenure > 0.")
    return data


def score(customers, model, threshold, features):
    proba = model.predict_proba(prepare(customers, features))[:, 1]

    result = pd.DataFrame(
        {
            "Churn_Probability": proba.round(4),
            "Predicted_Churn": (proba >= threshold).astype(int),
            "Risk_Group": pd.cut(
                proba,
                bins=[-np.inf, LOW_RISK_CUTOFF, threshold, np.inf],
                labels=["Low Risk", "Medium Risk", "High Risk"],
            ),
        },
        index=customers.index,
    )
    if "customerID" in customers.columns:
        result.insert(0, "customerID", customers["customerID"])
    return result


def main():
    parser = argparse.ArgumentParser(description="Predict customer churn.")
    parser.add_argument("input_csv", nargs="?", help="CSV file with customer rows")
    parser.add_argument("-o", "--output", help="Where to save the scored CSV")
    args = parser.parse_args()

    model, threshold, features = load_model()

    if args.input_csv:
        customers = pd.read_csv(args.input_csv)
    else:
        customers = pd.DataFrame([SAMPLE_CUSTOMER])

    result = score(customers, model, threshold, features)
    print(f"Decision threshold: {threshold}\n")
    print(result.to_string(index=False))

    if args.output:
        result.to_csv(args.output, index=False)
        print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()
