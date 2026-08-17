"""
risk_model.py
-------------
STAGE: "Risk-Based Patient Prioritization" (challenge requirement #2)

Trains a model that predicts probability of AD progression using ONLY
data available at Stage 1 (cognitive screening + baseline clinical
data + co-morbidities) — because that's what's actually available for
the whole screened population before anyone gets blood work or MRI.

We use Gradient Boosting (interpretable via SHAP, strong on tabular
clinical data, handles nonlinearity + missingness patterns well).
"""

import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score, classification_report, brier_score_loss
from sklearn.calibration import CalibratedClassifierCV

# Features available at STAGE 1 (cognitive screening) — the input for
# the very first prioritization pass across the whole population.
STAGE1_FEATURES = [
    "age", "sex_male", "education_years",
    "moca_score", "mmse_score",
    "hypertension", "diabetes", "depression",
    "family_history_ad", "apoe4_carrier",
]

TARGET = "future_ad_diagnosis"


def train_stage1_model(processed_csv: str, model_out: str):
    df = pd.read_csv(processed_csv)
    X = df[STAGE1_FEATURES]
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    base_model = GradientBoostingClassifier(
        n_estimators=150, max_depth=3, learning_rate=0.05, random_state=42
    )

    # Calibrate probabilities so "risk score" is a genuine probability,
    # not just a ranking — this matters for clinical trust & thresholds.
    model = CalibratedClassifierCV(base_model, method="isotonic", cv=5)
    model.fit(X_train, y_train)

    probs = model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, probs)
    brier = brier_score_loss(y_test, probs)
    preds = (probs >= 0.5).astype(int)

    print(f"Stage-1 model — AUC: {auc:.3f} | Brier score: {brier:.3f}")
    print(classification_report(y_test, preds, target_names=["No Progression", "Progression"]))

    joblib.dump({"model": model, "features": STAGE1_FEATURES}, model_out)
    print(f"Saved model to {model_out}")
    return model, auc


def risk_to_tier(prob: float) -> str:
    """Turns a continuous probability into a clinical priority tier."""
    if prob >= 0.66:
        return "High"
    elif prob >= 0.33:
        return "Medium"
    else:
        return "Low"

if __name__ == "__main__":
    import os
    processed_csv = os.path.join(os.path.dirname(__file__), "..", "data", "processed", "patients_clean.csv")
    models_dir = os.path.join(os.path.dirname(__file__), "..", "models")
    os.makedirs(models_dir, exist_ok=True)
    model_out = os.path.join(models_dir, "stage1_risk_model.joblib")
    train_stage1_model(processed_csv=processed_csv, model_out=model_out)