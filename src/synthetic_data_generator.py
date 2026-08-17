"""
synthetic_data_generator.py
----------------------------
Generates a synthetic patient dataset that MIMICS the structure of the
OASIS / ADNI cohorts, so you can build and test the entire pipeline
before you get access to the real data.

Real data notes (read this before you swap files):
- OASIS (oasis-brains.org): OASIS-1 is a free, no-application-needed
  cross-sectional CSV + MRI set. OASIS-2 is longitudinal. Download the
  CSV directly from https://www.oasis-brains.org/ (you must do this
  from your own machine/browser since this sandbox can't reach that
  domain).
- ADNI (adni.loni.usc.edu): requires a data-use application (free, but
  approval takes a few days). Once approved you can export a CSV of
  the ADNIMERGE table, which has similar columns to what's below.

Once you have the real CSV, you only need to make sure the column
names line up with COLUMN MAP in preprocessing.py (or rename them to
match this synthetic schema) — nothing else in the pipeline changes.
"""

import numpy as np
import pandas as pd

RNG = np.random.default_rng(42)


def generate_synthetic_cohort(n_patients: int = 800) -> pd.DataFrame:
    """Creates one row per patient with baseline screening data."""

    patient_id = [f"P{str(i).zfill(5)}" for i in range(1, n_patients + 1)]

    age = RNG.normal(72, 8, n_patients).clip(50, 95).round(0)
    sex = RNG.choice(["M", "F"], n_patients)
    education_years = RNG.normal(14, 3, n_patients).clip(4, 22).round(0)

    # Underlying (hidden) risk propensity - drives everything else.
    # In real life you would NOT have this; it's just how we simulate
    # a realistic, internally-consistent population.
    risk_propensity = RNG.beta(2, 5, n_patients)  # skewed toward low risk

    # --- Stage 1: Cognitive screening ---
    moca = (26 - risk_propensity * 14 + RNG.normal(0, 2, n_patients)).clip(0, 30).round(0)
    mmse = (28 - risk_propensity * 12 + RNG.normal(0, 1.5, n_patients)).clip(0, 30).round(0)

    # --- Co-morbidities (binary flags) ---
    hypertension = RNG.binomial(1, 0.25 + 0.2 * risk_propensity)
    diabetes = RNG.binomial(1, 0.15 + 0.15 * risk_propensity)
    depression = RNG.binomial(1, 0.10 + 0.25 * risk_propensity)
    family_history_ad = RNG.binomial(1, 0.10 + 0.35 * risk_propensity)
    apoe4_carrier = RNG.binomial(1, 0.15 + 0.45 * risk_propensity)  # genetic risk factor

    # --- Stage 2: Blood-based biomarkers ---
    # Plasma p-tau181 and Abeta42/40 ratio are real emerging AD blood biomarkers.
    plasma_ptau181 = (1.0 + risk_propensity * 3.5 + RNG.normal(0, 0.4, n_patients)).clip(0.2, 8)
    abeta_42_40_ratio = (0.12 - risk_propensity * 0.05 + RNG.normal(0, 0.01, n_patients)).clip(0.03, 0.15)
    nfl_level = (15 + risk_propensity * 40 + RNG.normal(0, 5, n_patients)).clip(5, 90)  # neurofilament light

    # --- Stage 3: MRI-derived structural measures ---
    hippocampal_volume_mm3 = (3800 - risk_propensity * 1400 + RNG.normal(0, 200, n_patients)).clip(1500, 4500)
    whole_brain_volume_cm3 = (1100 - risk_propensity * 120 + RNG.normal(0, 40, n_patients)).clip(850, 1300)
    ventricular_volume_mm3 = (25000 + risk_propensity * 15000 + RNG.normal(0, 3000, n_patients)).clip(10000, 70000)

    # --- Stage 4: PET (amyloid) — only meaningful/simulated for reference;
    # in the real pipeline this is NOT collected upfront, only recommended.
    amyloid_suvr = (1.05 + risk_propensity * 0.6 + RNG.normal(0, 0.08, n_patients)).clip(0.8, 2.2)

    df = pd.DataFrame({
        "patient_id": patient_id,
        "age": age,
        "sex": sex,
        "education_years": education_years,
        "moca_score": moca,
        "mmse_score": mmse,
        "hypertension": hypertension,
        "diabetes": diabetes,
        "depression": depression,
        "family_history_ad": family_history_ad,
        "apoe4_carrier": apoe4_carrier,
        "plasma_ptau181": plasma_ptau181.round(3),
        "abeta_42_40_ratio": abeta_42_40_ratio.round(4),
        "nfl_level": nfl_level.round(2),
        "hippocampal_volume_mm3": hippocampal_volume_mm3.round(1),
        "whole_brain_volume_cm3": whole_brain_volume_cm3.round(1),
        "ventricular_volume_mm3": ventricular_volume_mm3.round(1),
        "amyloid_suvr": amyloid_suvr.round(3),
    })

    # --- Ground-truth label for MODEL TRAINING ONLY ---
    # In real datasets this comes from clinical diagnosis (CDR / final
    # diagnosis field in ADNI, or CDR in OASIS). We simulate it here
    # from the same risk_propensity so the model has real signal to learn.
    diagnosis_prob = risk_propensity
    diagnosis = RNG.binomial(1, diagnosis_prob)
    df["future_ad_diagnosis"] = diagnosis  # 1 = progressed to AD within follow-up window

    return df

import os
if __name__ == "__main__":
    df = generate_synthetic_cohort(800)
    out_dir = os.path.join(os.path.dirname(__file__), "..", "data", "raw")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "synthetic_cohort.csv")
    df.to_csv(out_path, index=False)
    print(f"Wrote {len(df)} patients to {out_path}")
    print(df.head())
