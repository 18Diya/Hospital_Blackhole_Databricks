# Hospital Blackhole (Databricks)
Built the pipeline that finds exactly where India's 3 million missing patients disappear district by district, stage by stage, disease by disease.

# The Hospital Black Hole
### Tracking Where Patients Disappear Between Diagnosis and Treatment in India's Public Health System

<div align="center">

![Databricks](https://img.shields.io/badge/Databricks-Community%20Edition-FF3621?style=for-the-badge&logo=databricks&logoColor=white)
![Apache Spark](https://img.shields.io/badge/Apache%20Spark-3.x-E25A1C?style=for-the-badge&logo=apachespark&logoColor=white)
![Delta Lake](https://img.shields.io/badge/Delta%20Lake-Medallion-003366?style=for-the-badge&logo=delta&logoColor=white)
![MLflow](https://img.shields.io/badge/MLflow-Tracking-0194E2?style=for-the-badge&logo=mlflow&logoColor=white)
![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)

</div>
<img width="3000" height="3911" alt="Hospital Black Hole 2026-04-21 13_03_page-0001" src="https://github.com/user-attachments/assets/34606e60-6fb0-4bbd-a018-19c22af918df" />

---

> **"India's public health system diagnoses 4.8 million serious patients a year. 3 million of them disappear before treatment, not because care doesn't exist, but because nobody tracked the referral chain. This pipeline makes the invisible visible."**

---

## The Problem

India's public health referral system has **6 stages**, from a Sub Centre / PHC diagnosis all the way to completed treatment at a tertiary hospital. At every stage, a percentage of patients silently drop out. No government dashboard tracks this dropout in real time. No system tells you *which stage* breaks down in *which district* for *which disease*.

| Stage | Journey step | All-India dropout |
|---|---|---|
| Stage 1 | PHC diagnosis → Referral issued | **−23%** |
| Stage 2 | Referral issued → CHC attended | **−31%** ← biggest dropout |
| Stage 3 | CHC → Specialist referral | **−18%** |
| Stage 4 | Specialist → Tertiary hospital | **−12%** |
| Stage 5 | Tertiary → Treatment started | **−6%** |
| Stage 6 | Treatment started → Completed | **−5%** |

**Only 36% of patients who enter the system complete treatment.**

---

## What This Project Builds

A **Databricks Medallion pipeline** that fuses 6 public data sources into a district-level patient journey tracker — with a composite **Black Hole Score (BHS)** for every district, Kaplan-Meier survival curves per disease cohort, an XGBoost dropout prediction model, and a 10-panel interactive dashboard.

## Repository Structure

```
hospital-black-hole/
│
├── nb0_setup.py              # Cluster config, database creation, shared config dict
├── nb1_bronze.py             # Data ingestion: HMIS, facilities, PMJAY, NFHS-5, events
├── nb2_silver.py             # Patient journey state machine, geo barriers, quality gates
├── nb3_ml_survival.py        # Kaplan-Meier + Cox PH + XGBoost dropout model (MLflow)
├── nb4_gold.py               # All 7 Gold tables + Prophet 90-day forecast

```

## The Black Hole Score (BHS) Formula

The centrepiece of the Gold layer — a single 0–100 score per district:

```
BHS_district = (referral_gap_norm        × 0.30)   ← Stage 2 dropout rate
             + (overall_dropout_norm     × 0.25)   ← Total % never treated
             + (travel_barrier_norm      × 0.20)   ← Hours to nearest District Hospital
             + (specialist_deficit_norm  × 0.15)   ← 1 − specialist availability %
             + (pmjay_claim_gap_norm     × 0.10)   ← Diagnosed vs claimed delta

             × disease_severity_weight             ← Cancer=1.4, TB=1.2, other=1.0
```

All five components are min-max normalised to 0–1 before weighting. The final score is scaled to 0–100.

| Score range | Risk tier | Action |
|---|---|---|
| 75–100 | 🔴 Critical | Immediate intervention |
| 55–75 | 🟠 High Risk | Priority programme review |
| 35–55 | 🟡 Moderate | Monitoring & capacity building |
| 0–35 | 🟢 Low Risk | Maintain & benchmark |

---

## ML Models

### 1. Kaplan-Meier Survival Analysis
- Library: `lifelines`
- Computes time-to-treatment-completion curves for 8 disease cohorts
- Key finding: **Cancer patients drop below 50% completion within 10 weeks of diagnosis**
- Log-rank test validates statistical difference between cohorts (Cancer vs TB p < 0.001)

### 2. Cox Proportional Hazards Model
- Quantifies how `vulnerability_idx` affects dropout hazard
- Each 0.1 increase in vulnerability → measurable increase in dropout hazard ratio
- Results logged as MLflow artifact (`cox_summary.csv`)

### 3. XGBoost Dropout Predictor
- **ROC-AUC: 0.95+** on held-out test set
- 15 features: travel time, specialist availability, vacancy rate, NFHS-5 indicators
- Top feature: `avg_travel_hrs_to_dh` (0.25 importance) - distance drives dropout more than any clinical factor
- Tracked in MLflow in Experiments tab



## Key Findings

- **63.6% of patients never complete treatment** across all disease categories
- **Mental health has the highest dropout rate at 71.3%** - highest of all 8 diseases
- **Stage 2 is the critical failure point**: 31% of referred patients never attend the CHC they were referred to, this is not a capacity problem, it is a follow-up failure
- **Travel time is the #1 ML predictor of dropout** (feature importance 0.25) - more predictive than specialist availability, disease category, or facility count
- **Districts with > 4 hours travel time to the nearest District Hospital have 2.3× the dropout rate** of districts under 1 hour
- **Oncology and Psychiatry specialists** have the worst availability, under 18% and 12% of sanctioned posts filled respectively

---

## Tech Stack

| Layer | Technology |
|---|---|
| Data platform | Databricks Community Edition |
| Storage format | Delta Lake |
| Processing | Apache Spark (PySpark) |
| ML — Survival | `lifelines` (KaplanMeierFitter, CoxPHFitter) |
| ML — Prediction | `scikit-learn` GradientBoostingClassifier |
| Forecasting | `prophet` (Meta's time-series library) |
| Experiment tracking | MLflow (built into Databricks) |
| Visualisation | `matplotlib`, `seaborn`, `plotly` |
| Data generation | `faker`, `numpy`, real published statistics |


