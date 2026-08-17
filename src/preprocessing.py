"""
preprocessing.py
-----------------
STAGE: "Data Ingestion and Integration" (challenge requirement #1)

Takes raw patient-level CSVs (cognitive scores, co-morbidities, blood
work, MRI) and produces a single, clean, patient-centric table ready
for modeling.

If you swap in real ADNI/OASIS data, edit COLUMN_MAP below so your
source column names map onto the internal names this pipeline expects.
Nothing downstream needs to change.
"""

import pandas as pd
import numpy as np

# ---------------------------------------------------------------------
# Map your real dataset's column names -> internal names used everywhere
# else in this project. Edit the RIGHT-hand values to match your file.
# ---------------------------------------------------------------------
COLUMN_MAP = {
    "patient_id": "patient_id",
    "age": "age",
    "sex": "sex",
    "education_years": "education_years",
    "moca_score": "moca_score",
    "mmse_score": "mmse_score",
    "hypertension": "hypertension",
    "diabetes": "diabetes",
    "depression": "depression",
    "family_history_ad": "family_history_ad",
    "apoe4_carrier": "apoe4_carrier",
    "plasma_ptau181": "plasma_ptau181",
    "abeta_42_40_ratio": "abeta_42_40_ratio",
    "nfl_level": "nfl_level",
    "hippocampal_volume_mm3": "hippocampal_volume_mm3",
    "whole_brain_volume_cm3": "whole_brain_volume_cm3",
    "ventricular_volume_mm3": "ventricular_volume_mm3",
    "amyloid_suvr": "amyloid_suvr",
    "future_ad_diagnosis": "future_ad_diagnosis",  # only present in training data
}

NUMERIC_COLS = [
    "age", "education_years", "moca_score", "mmse_score",
    "plasma_ptau181", "abeta_42_40_ratio", "nfl_level",
    "hippocampal_volume_mm3", "whole_brain_volume_cm3",
    "ventricular_volume_mm3", "amyloid_suvr",
]
BINARY_COLS = ["hypertension", "diabetes", "depression", "family_history_ad", "apoe4_carrier"]


def load_raw(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df.rename(columns={v: k for k, v in COLUMN_MAP.items() if v in df.columns})
    return df


def clean_and_standardize(df: pd.DataFrame) -> pd.DataFrame:
    """De-identify, validate ranges, impute missing values."""
    df = df.copy()

    # 1. De-identification: drop any direct identifiers if present.
    for col in ["name", "mrn", "dob", "address", "phone", "email"]:
        if col in df.columns:
            df = df.drop(columns=[col])

    # 2. Type coercion
    for col in NUMERIC_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    for col in BINARY_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)

    # 3. Range sanity checks (flag, don't silently drop)
    if "moca_score" in df.columns:
        df.loc[(df.moca_score < 0) | (df.moca_score > 30), "moca_score"] = np.nan
    if "mmse_score" in df.columns:
        df.loc[(df.mmse_score < 0) | (df.mmse_score > 30), "mmse_score"] = np.nan

    # 4. Missing-value imputation: median for numeric (robust to outliers).
    #    Track which columns were imputed for each patient (transparency).
    missing_flags = pd.DataFrame(index=df.index)
    for col in NUMERIC_COLS:
        if col in df.columns:
            missing_flags[f"{col}_was_missing"] = df[col].isna().astype(int)
            df[col] = df[col].fillna(df[col].median())

    df = pd.concat([df, missing_flags], axis=1)

    # 5. Encode sex
    if "sex" in df.columns:
        df["sex_male"] = (df["sex"].astype(str).str.upper() == "M").astype(int)

    return df


def build_patient_centric_table(raw_path: str, processed_path: str) -> pd.DataFrame:
    df = load_raw(raw_path)
    df = clean_and_standardize(df)
    df.to_csv(processed_path, index=False)
    return df

import os
if __name__ == "__main__":
    import os
    raw_path = os.path.join(os.path.dirname(__file__), "..", "data", "raw", "synthetic_cohort.csv")
    processed_dir = os.path.join(os.path.dirname(__file__), "..", "data", "processed")
    os.makedirs(processed_dir, exist_ok=True)
    processed_path = os.path.join(processed_dir, "patients_clean.csv")
    out = build_patient_centric_table(raw_path=raw_path, processed_path=processed_path)
    print(f"Processed {len(out)} patients -> data/processed/patients_clean.csv")
    print(out.columns.tolist())