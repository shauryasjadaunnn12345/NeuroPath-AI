"""
pathway_engine.py
------------------
STAGE: "Progressive Diagnostic Pathway Recommendation" (requirement #3)

Implements the 4-stage escalation funnel described in the brief:
  Stage 1: Cognitive Screening      -> initial risk tier (whole population)
  Stage 2: Blood-Based Biomarkers   -> refine risk for Medium/High only
  Stage 3: MRI Evaluation           -> narrow down high-risk candidates
  Stage 4: PET Scan Prioritization  -> flag for advanced confirmation

Design principle: each stage only "spends" an expensive test on
patients the previous stage flagged as Medium or High risk. Low-risk
patients stop after Stage 1 (re-screened routinely instead). This is
what makes the system resource-efficient, per the challenge goal.

This module is DECISION SUPPORT ONLY:
- it outputs priority tiers + recommended next step + reasoning
- it never outputs a diagnosis or treatment recommendation
"""

import pandas as pd
import numpy as np
import joblib

from risk_model import STAGE1_FEATURES, risk_to_tier


def score_stage1(df: pd.DataFrame, model_path: str) -> pd.DataFrame:
    bundle = joblib.load(model_path)
    model, features = bundle["model"], bundle["features"]

    df = df.copy()
    df["stage1_risk_prob"] = model.predict_proba(df[features])[:, 1]
    df["stage1_tier"] = df["stage1_risk_prob"].apply(risk_to_tier)
    df["recommended_next_step"] = np.where(
        df["stage1_tier"] == "Low",
        "Routine re-screening in 12 months",
        "Proceed to Stage 2: Blood-based biomarkers",
    )
    df["current_stage"] = "Stage 1 complete"
    return df


def score_stage2(df: pd.DataFrame) -> pd.DataFrame:
    """
    Refines risk using blood biomarkers, but ONLY for patients who
    passed Stage 1 as Medium/High (mirrors real-world resource use).
    Uses simple, transparent clinical thresholds (rule-based, not a
    black box) so it stays interpretable, per requirement #5.
    """
    df = df.copy()
    eligible = df["stage1_tier"].isin(["Medium", "High"])

    def biomarker_flag(row):
        flags = []
        if row["plasma_ptau181"] > 2.5:
            flags.append("elevated plasma p-tau181")
        if row["abeta_42_40_ratio"] < 0.08:
            flags.append("low Abeta42/40 ratio")
        if row["nfl_level"] > 40:
            flags.append("elevated NfL")
        return flags

    df["stage2_biomarker_flags"] = df.apply(
        lambda r: biomarker_flag(r) if eligible[r.name] else [], axis=1
    )
    df["stage2_flag_count"] = df["stage2_biomarker_flags"].apply(len)

    # Refine tier: boost tier if 2+ biomarker flags present; de-escalate
    # if a "Medium" patient has zero flags.
    def refine(row):
        if not eligible[row.name]:
            return row["stage1_tier"], row["recommended_next_step"], "Not eligible for Stage 2 (Stage 1 = Low)"
        if row["stage2_flag_count"] >= 2:
            tier = "High"
            nxt = "Proceed to Stage 3: MRI evaluation"
        elif row["stage2_flag_count"] == 1:
            tier = row["stage1_tier"]
            nxt = "Proceed to Stage 3: MRI evaluation" if tier == "High" else "Monitor; consider MRI at clinician discretion"
        else:
            tier = "Medium" if row["stage1_tier"] == "High" else "Low"
            nxt = "Routine re-screening in 12 months" if tier == "Low" else "Monitor; repeat biomarkers in 6 months"
        reasoning = f"{row['stage2_flag_count']} biomarker flag(s): {', '.join(row['stage2_biomarker_flags']) or 'none'}"
        return tier, nxt, reasoning

    refined = df.apply(refine, axis=1, result_type="expand")
    df["stage2_tier"], df["recommended_next_step"], df["stage2_reasoning"] = refined[0], refined[1], refined[2]
    df["current_stage"] = np.where(eligible, "Stage 2 complete", df["current_stage"])
    return df


def score_stage3(df: pd.DataFrame) -> pd.DataFrame:
    """MRI structural evaluation — only for Stage 2 'High' patients."""
    df = df.copy()
    eligible = df["stage2_tier"] == "High"

    def mri_flag(row):
        flags = []
        if row["hippocampal_volume_mm3"] < 3000:
            flags.append("hippocampal atrophy")
        if row["ventricular_volume_mm3"] > 40000:
            flags.append("ventricular enlargement")
        if row["whole_brain_volume_cm3"] < 1000:
            flags.append("reduced whole-brain volume")
        return flags

    df["stage3_mri_flags"] = df.apply(lambda r: mri_flag(r) if eligible[r.name] else [], axis=1)
    df["stage3_flag_count"] = df["stage3_mri_flags"].apply(len)

    def refine(row):
        if not eligible[row.name]:
            return row["stage2_tier"], row["recommended_next_step"], "Not eligible for Stage 3 (Stage 2 != High)"
        if row["stage3_flag_count"] >= 2:
            tier = "High"
            nxt = "Proceed to Stage 4: PET scan prioritization"
        elif row["stage3_flag_count"] == 1:
            tier = "High"
            nxt = "Clinician review; consider PET scan"
        else:
            tier = "Medium"
            nxt = "Monitor; repeat MRI in 6-12 months"
        reasoning = f"{row['stage3_flag_count']} structural flag(s): {', '.join(row['stage3_mri_flags']) or 'none'}"
        return tier, nxt, reasoning

    refined = df.apply(refine, axis=1, result_type="expand")
    df["stage3_tier"], df["recommended_next_step"], df["stage3_reasoning"] = refined[0], refined[1], refined[2]
    df["current_stage"] = np.where(eligible, "Stage 3 complete", df["current_stage"])
    return df


def score_stage4(df: pd.DataFrame) -> pd.DataFrame:
    """Final PET scan prioritization flag for advanced confirmation."""
    df = df.copy()
    eligible = (df["stage3_tier"] == "High") & (df["current_stage"] == "Stage 3 complete")

    df["pet_scan_recommended"] = eligible
    df["current_stage"] = np.where(eligible, "Stage 4: PET scan recommended", df["current_stage"])
    df["recommended_next_step"] = np.where(
        eligible, "PET scan for amyloid confirmation; refer to specialist for evaluation", df["recommended_next_step"]
    )
    return df


def run_full_pathway(processed_csv: str, model_path: str, out_csv: str) -> pd.DataFrame:
    df = pd.read_csv(processed_csv)
    df = score_stage1(df, model_path)
    df = score_stage2(df)
    df = score_stage3(df)
    df = score_stage4(df)

    # Final priority tier = the most refined tier the patient reached
    df["final_priority_tier"] = df["stage3_tier"].where(
        df["current_stage"].isin(["Stage 3 complete", "Stage 4: PET scan recommended"]),
        df["stage2_tier"].where(df["current_stage"] == "Stage 2 complete", df["stage1_tier"]),
    )

    cols = [
        "patient_id", "age", "sex", "moca_score", "mmse_score",
        "stage1_risk_prob", "stage1_tier",
        "stage2_flag_count", "stage2_tier",
        "stage3_flag_count", "stage3_tier",
        "pet_scan_recommended", "final_priority_tier",
        "current_stage", "recommended_next_step",
    ]
    df[cols].sort_values("stage1_risk_prob", ascending=False).to_csv(out_csv, index=False)
    return df


if __name__ == "__main__":
    import os
    processed_csv = os.path.join(os.path.dirname(__file__), "..", "data", "processed", "patients_clean.csv")
    model_path = os.path.join(os.path.dirname(__file__), "..", "models", "stage1_risk_model.joblib")
    out_dir = os.path.join(os.path.dirname(__file__), "..", "outputs")
    os.makedirs(out_dir, exist_ok=True)
    out_csv = os.path.join(out_dir, "prioritized_patients.csv")
    result = run_full_pathway(processed_csv=processed_csv, model_path=model_path, out_csv=out_csv)
    print(result["final_priority_tier"].value_counts())
    print(f"PET recommended for {result['pet_scan_recommended'].sum()} patients")