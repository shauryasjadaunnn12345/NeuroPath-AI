"""
app.py
------
STAGE: "Clinician-Focused Interface" (requirement #4)

Run with:  streamlit run app.py
(run from inside the app/ folder, or adjust the sys.path line below)

Shows:
- Prioritized list of patients (sortable/filterable)
- Suggested next diagnostic step per patient
- Patient progression through the 4 diagnostic stages
- SHAP-based reasoning for any selected patient
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

import streamlit as st
import pandas as pd
import joblib

from pathway_engine import run_full_pathway
from explainability import explain_patient

PROJECT_ROOT = os.path.join(os.path.dirname(__file__), "..")
PROCESSED_CSV = os.path.join(PROJECT_ROOT, "data", "processed", "patients_clean.csv")
MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "stage1_risk_model.joblib")
OUT_CSV = os.path.join(PROJECT_ROOT, "outputs", "prioritized_patients.csv")

st.set_page_config(page_title="Alzheimer's Diagnostic Prioritization", layout="wide")

st.title("Early Alzheimer's Diagnostic Pathway — Prioritization Dashboard")
st.caption(
    "Clinical DECISION-SUPPORT tool only. This system does not provide a medical "
    "diagnosis or treatment recommendation. All decisions must be made by a qualified clinician."
)

@st.cache_data
def load_data():
    df = run_full_pathway(PROCESSED_CSV, MODEL_PATH, OUT_CSV)
    return df

df = load_data()

# --- Sidebar filters ---
st.sidebar.header("Filters")
tier_filter = st.sidebar.multiselect(
    "Priority tier", options=["High", "Medium", "Low"], default=["High", "Medium"]
)
stage_filter = st.sidebar.multiselect(
    "Current stage", options=sorted(df["current_stage"].unique()), default=sorted(df["current_stage"].unique())
)

filtered = df[
    df["final_priority_tier"].isin(tier_filter) & df["current_stage"].isin(stage_filter)
].sort_values("stage1_risk_prob", ascending=False)

# --- Summary metrics ---
col1, col2, col3, col4 = st.columns(4)
col1.metric("Total screened", len(df))
col2.metric("High priority", int((df.final_priority_tier == "High").sum()))
col3.metric("Medium priority", int((df.final_priority_tier == "Medium").sum()))
col4.metric("PET scans recommended", int(df.pet_scan_recommended.sum()))

st.divider()

# --- Prioritized patient list ---
st.subheader("Prioritized Patient List")
display_cols = [
    "patient_id", "age", "sex", "moca_score", "mmse_score",
    "stage1_risk_prob", "final_priority_tier", "current_stage", "recommended_next_step",
]
st.dataframe(
    filtered[display_cols].rename(columns={
        "stage1_risk_prob": "Risk score",
        "final_priority_tier": "Priority tier",
        "current_stage": "Stage reached",
        "recommended_next_step": "Suggested next step",
    }),
    use_container_width=True,
    height=400,
)

st.divider()

# --- Patient detail / reasoning ---
st.subheader("Patient Detail & Reasoning")
selected_id = st.selectbox("Select a patient", options=filtered["patient_id"].tolist())

if selected_id:
    raw_df = pd.read_csv(PROCESSED_CSV)
    patient_row = raw_df[raw_df.patient_id == selected_id].iloc[0]
    pathway_row = df[df.patient_id == selected_id].iloc[0]

    c1, c2 = st.columns([1, 1])
    with c1:
        st.markdown(f"**Patient:** {selected_id}  |  **Age:** {patient_row.age}  |  **Sex:** {patient_row.sex}")
        st.markdown(f"**Priority tier:** {pathway_row.final_priority_tier}")
        st.markdown(f"**Stage reached:** {pathway_row.current_stage}")
        st.markdown(f"**Suggested next step:** {pathway_row.recommended_next_step}")

        st.markdown("**Progression through stages:**")
        stages = ["Stage 1 complete", "Stage 2 complete", "Stage 3 complete", "Stage 4: PET scan recommended"]
        reached = stages.index(pathway_row.current_stage) if pathway_row.current_stage in stages else 0
        st.progress((reached + 1) / len(stages))

    with c2:
        st.markdown("**Why this risk score? (top contributing factors)**")
        _, readable = explain_patient(patient_row, MODEL_PATH)
        for line in readable:
            st.markdown(f"- {line}")

st.divider()
st.caption(
    "⚠️ Limitations: model trained on synthetic/demo data for prototyping purposes. "
    "Before clinical use, retrain on validated cohorts (ADNI/OASIS or institutional data "
    "with IRB approval), and validate performance with a clinical research team."
)
