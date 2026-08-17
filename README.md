# AI-Driven Prioritization System for Early Alzheimer's Diagnostic Pathways
# NeuroPath-AI
NeuroPath AI is an AI decision-support tool that prioritizes patients for early Alzheimer's diagnosis. It analyzes cognitive, blood, and MRI data to risk-stratify populations and recommend a staged testing pathway — supporting clinicians, not diagnosing.
### Complete build guide — assumes zero prior setup

This guide walks you through everything: what each file does, how to run it,
how to swap in real ADNI/OASIS data, and how it maps back to the challenge
requirements. Read top to bottom the first time; after that use it as a reference.

---

## 0. What you're building (mental model)

```
Raw patient data (CSV)
        │
        ▼
 preprocessing.py        →  clean, patient-centric table
        │
        ▼
 risk_model.py            →  trains a Stage-1 ML risk model
        │
        ▼
 pathway_engine.py         →  runs all 4 stages, outputs prioritized list
        │
        ▼
 explainability.py          →  SHAP reasoning per patient
        │
        ▼
 app/app.py (Streamlit)      →  clinician-facing dashboard
```

Every box is a separate, runnable Python file. You can run each one alone
(useful for debugging) or run them all in sequence (what the app does
automatically).

---

## 1. Environment setup

You need Python 3.10+ installed. Then, from the project root:

```bash
python3 -m venv venv
source venv/bin/activate        # on Windows: venv\Scripts\activate
pip install -r requirements.txt
```

That installs: `pandas`, `numpy`, `scikit-learn`, `shap`, `joblib`,
`streamlit`, `matplotlib`.

---

## 2. Project structure

```

├── data/
│   ├── raw/                  # original CSVs land here (synthetic or real)
│   └── processed/            # cleaned, patient-centric table
├── models/                   # trained model files (.joblib)
├── outputs/                  # final prioritized patient list (.csv)
├── src/
│   ├── synthetic_data_generator.py   # makes fake data to develop/test with
│   ├── preprocessing.py              # ingestion + cleaning (req. #1)
│   ├── risk_model.py                 # ML risk scoring (req. #2)
│   ├── pathway_engine.py             # 4-stage escalation logic (req. #3)
│   └── explainability.py             # SHAP reasoning (req. #5)
├── app/
│   └── app.py                        # Streamlit clinician UI (req. #4)
├── requirements.txt
└── GUIDE.md                          # this file
```

---

## 3. Run the whole pipeline end-to-end (fastest path)

From the project root, run each script in order:

```bash
cd src

# Step 1 — generate a fake dataset shaped like OASIS/ADNI (skip this once you have real data)
python3 synthetic_data_generator.py

# Step 2 — clean & standardize into one patient-centric table
python3 preprocessing.py

# Step 3 — train the Stage-1 risk model
python3 risk_model.py

# Step 4 — run the full 4-stage prioritization pathway
python3 pathway_engine.py

# Step 5 — sanity-check explainability on one patient
python3 explainability.py
```

Then launch the dashboard:

```bash
cd ../app
streamlit run app.py
```

Open the URL Streamlit prints (usually `http://localhost:8501`).

---

## 4. What each file does, in clinical terms

### `synthetic_data_generator.py` — stand-in for real data
Creates 800 fake patients with realistic-looking cognitive scores,
co-morbidities, blood biomarkers, and MRI measures, all internally
consistent (sicker patients have worse scores across the board — like
real disease progression would). This lets you build and demo the
whole system before you get real data access. **Delete/ignore this
once you have a real dataset.**

### `preprocessing.py` — Data Ingestion & Integration (requirement #1)
- Renames columns from your source file to a standard internal schema
  (`COLUMN_MAP` — edit this dict when you plug in real ADNI/OASIS data)
- Strips any identifying fields (name, DOB, MRN, etc.) — never load
  these in the first place if you can avoid it
- Validates value ranges (e.g. MoCA/MMSE must be 0–30)
- Imputes missing values with the column median, and adds a
  `*_was_missing` flag column so the model and clinician can see what
  was imputed vs. observed
- Outputs one row per patient into `data/processed/patients_clean.csv`

### `risk_model.py` — Risk-Based Patient Prioritization (requirement #2)
- Trains a **Gradient Boosting Classifier**, wrapped in
  `CalibratedClassifierCV` so the output is a genuine probability
  (0–1), not just a ranking score
- Trained **only on Stage-1 features** (age, sex, education, MoCA,
  MMSE, co-morbidities, APOE4 status, family history) — because that's
  all that's available for the *entire* screened population before
  anyone gets expensive testing
- `risk_to_tier()` converts probability → High / Medium / Low
  (thresholds: ≥0.66 High, ≥0.33 Medium, else Low — tune these to your
  population's base rate and clinical review)
- Reports AUC and Brier score (calibration quality) on a held-out test
  set every time you retrain

### `pathway_engine.py` — Progressive Diagnostic Pathway (requirement #3)
Implements the exact 4-stage funnel from the brief, and — importantly —
each stage **only evaluates patients the previous stage flagged**, so
expensive tests aren't wasted on low-risk patients:

| Stage | Input | Logic | Output |
|---|---|---|---|
| 1. Cognitive Screening | MoCA/MMSE + baseline clinical data | ML model | Initial tier (all patients) |
| 2. Blood Biomarkers | p-tau181, Abeta42/40, NfL | Rule-based thresholds | Refined tier (Medium/High only) |
| 3. MRI Evaluation | Hippocampal volume, ventricular volume, whole-brain volume | Rule-based thresholds | Narrowed high-risk list |
| 4. PET Prioritization | (flag only) | Stage-3 = High | Final PET recommendation |

Stages 2 and 3 use **transparent clinical thresholds** rather than a
second black-box model — this keeps the whole pathway interpretable
end-to-end (requirement #5) and easy for a clinical team to audit and
adjust. If you'd rather have Stage 2/3 also be ML-driven, you'd train
separate models the same way `risk_model.py` does, just with those
stages' features and swap the rule-based functions for
`model.predict_proba(...)` calls.

### `explainability.py` — Explainability & Transparency (requirement #5)
Uses **SHAP** (SHapley Additive exPlanations) to show, per patient,
which factors pushed their risk score up or down and by how much —
e.g. *"MoCA score = 19 (increased risk by 0.18)"*. This is what the
clinician sees in the dashboard's reasoning panel.

### `app/app.py` — Clinician-Focused Interface (requirement #4)
A Streamlit dashboard with:
- Summary metrics (total screened, high/medium priority counts, PET
  scans recommended)
- A filterable, sortable prioritized patient table
- A patient detail panel showing stage progression and SHAP reasoning
- A persistent disclaimer that this is decision support only

---

## 5. Getting real data (ADNI / OASIS)

**OASIS** (easiest — no application needed for OASIS-1):
1. Go to https://www.oasis-brains.org/
2. Download the OASIS-1 cross-sectional CSV (demographics + MMSE + CDR
   + MRI-derived volumes are included)
3. Rename/remap columns to match `COLUMN_MAP` in `preprocessing.py`

**ADNI** (richer data, requires free application):
1. Apply for data access at https://adni.loni.usc.edu/ (takes a few
   days for approval)
2. Once approved, export the `ADNIMERGE` table (has MoCA/MMSE,
   co-morbidities, plasma biomarkers, MRI volumetrics, amyloid PET
   SUVR, and diagnosis labels — nearly a 1:1 match to this pipeline's
   schema)
3. Save as CSV into `data/raw/`, update `COLUMN_MAP`, and rerun the
   pipeline from `preprocessing.py` onward

**Important:** both are governed by data-use agreements — don't
redistribute the raw files, and cite them appropriately in any
write-up or publication (both sites specify required citations).

---

## 6. Retraining on real data — what to change

1. Put your CSV in `data/raw/`.
2. Edit `COLUMN_MAP` in `preprocessing.py` so it matches your file's
   real column names.
3. If your data doesn't have a `future_ad_diagnosis`-equivalent label,
   derive one from the CDR (Clinical Dementia Rating) or clinical
   diagnosis field — e.g. `label = 1 if CDR > 0 else 0`, or based on
   conversion status at follow-up if it's longitudinal data.
4. Rerun: `preprocessing.py` → `risk_model.py` → `pathway_engine.py`.
5. Check the printed AUC/Brier score. With real, larger cohorts you
   should see AUC meaningfully above the ~0.65–0.70 you'll get on the
   synthetic demo data.
6. Re-tune the tier thresholds in `risk_to_tier()` (in `risk_model.py`)
   and the biomarker/MRI thresholds in `pathway_engine.py` — ideally
   with a clinician reviewing where the cut points should sit for your
   population.

---

## 7. Mapping to the evaluation criteria

| Criterion | Where it's addressed |
|---|---|
| Clinical Relevance (20%) | Staged funnel mirrors real diagnostic workflow; risk model uses established AD risk factors (APOE4, hippocampal atrophy, p-tau181, etc.) |
| Data Handling (20%) | `preprocessing.py` — unified schema, de-identification, imputation with transparency flags |
| Prioritization Logic (20%) | `risk_model.py` (calibrated ML) + `pathway_engine.py` (transparent stage rules) |
| User Experience (15%) | `app/app.py` — filterable dashboard, one-click patient drill-down |
| Ethics & Safety (10%) | De-identification, synthetic/public data only, explicit "decision support only" disclaimers throughout, documented limitations |
| Innovation & Impact (15%) | Resource-efficient funnel (only ~3% of screened population reaches PET-recommended stage in the demo run — 23 of 800) |

---

## 8. Known limitations 

- The synthetic data generator creates data with by-construction signal;
  a real cohort will look noisier and require more careful feature
  engineering and possibly a larger, deeper model.
- The Stage 2/3 thresholds are illustrative, not clinically validated —
  they need review from a neurologist or biomarker specialist before
  any real use.


---

