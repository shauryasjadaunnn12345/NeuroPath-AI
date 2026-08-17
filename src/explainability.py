"""
explainability.py
------------------
STAGE: "Explainability and Transparency" (requirement #5)

Uses SHAP to explain WHY the Stage-1 model gave a patient their risk
score, in terms clinicians can read (e.g. "low MoCA score increased
this patient's risk by 0.12").
"""

import joblib
import pandas as pd
import shap

from risk_model import STAGE1_FEATURES


def explain_patient(patient_row: pd.Series, model_path: str, top_n: int = 5):
    bundle = joblib.load(model_path)
    model = bundle["model"]

    X = patient_row[STAGE1_FEATURES].to_frame().T

    # CalibratedClassifierCV wraps multiple base estimators (one per CV
    # fold); use the first calibrated classifier's underlying estimator
    # for a representative SHAP explanation.
    base_estimator = model.calibrated_classifiers_[0].estimator
    explainer = shap.TreeExplainer(base_estimator)
    shap_values = explainer.shap_values(X)

    # shap_values shape: (1, n_features) for binary classification w/ GBM
    values = shap_values[0] if shap_values.ndim == 2 else shap_values[0, :, 1]

    contributions = pd.DataFrame({
        "feature": STAGE1_FEATURES,
        "patient_value": X.iloc[0].values,
        "shap_contribution": values,
    }).sort_values("shap_contribution", key=abs, ascending=False)

    top = contributions.head(top_n)

    readable = []
    for _, row in top.iterrows():
        direction = "increased" if row.shap_contribution > 0 else "decreased"
        readable.append(
            f"{row.feature} = {row.patient_value} ({direction} risk by {abs(row.shap_contribution):.3f})"
        )

    return top, readable


if __name__ == "__main__":
    import os
    processed_csv = os.path.join(os.path.dirname(__file__), "..", "data", "processed", "patients_clean.csv")
    model_path = os.path.join(os.path.dirname(__file__), "..", "models", "stage1_risk_model.joblib")
    df = pd.read_csv(processed_csv)
    sample_patient = df.iloc[0]
    top, readable = explain_patient(sample_patient, model_path=model_path)
    print(f"Explanation for {sample_patient['patient_id']}:")
    for line in readable:
        print(" -", line)