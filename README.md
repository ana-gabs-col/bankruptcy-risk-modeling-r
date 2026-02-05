# Bankruptcy Risk Modeling (R)
Reproducible classification pipeline to predict next-year corporate bankruptcy from financial ratios, using imbalance-aware evaluation (AUC-ROC) and interpretable risk drivers.

## Problem
Bankruptcy is rare but costly. When events are infrequent, accuracy is misleading; the goal is to **rank firms by risk** using AUC-ROC and clearly documented steps.

## Data
- **Observations:** 1,850 firm-year records (Shanghai Stock Exchange, 1999–2009)
- **Features:** 63 financial ratios (reduced to 52 after correlation screening)
- **Target:** Bankruptcy (binary; prevalence ~3%)
- **Missing values:** none

> Note: Raw data may be excluded or anonymized. This repo focuses on the reproducible workflow, documentation, and outputs.

## Method (Reproducible Pipeline)
1) EDA + diagnostics  
2) Feature screening (remove highly correlated predictors, |r| > 0.90)  
3) Robust preprocessing: winsorization (1st/99th percentiles) + robust scaling (median/IQR)  
4) Train/validation split (80/20)  
5) Imbalance handling: **SMOTE** on training set (50/50 balance)  
6) Models compared: Logistic Regression, SVM (RBF), Random Forest, Gradient Boosting  
7) Primary metric: **AUC-ROC** (+ confusion matrix at a selected threshold)

## Results (Validation)
**Best model:** tuned Gradient Boosting — **AUC ~0.89**  
Key drivers include **Retained Earnings / Total Assets** and leverage/solvency ratios.

## Repository Structure
- `src/` pipeline code (preprocessing, training, evaluation)
- `notebooks/` EDA and experiments
- `figures/` ROC curve, confusion matrix, feature importance
- `docs/` methodology notes + QA checklist
- `data_dictionary.md` variable definitions + schema

## How to Run
**Option A (recommended):** use `renv` for reproducibility  
1) `renv::restore()`  
2) Run `src/run_pipeline.R`

**Option B:** install packages listed in `requirements.R`

## Deliverables
- AUC-ROC evaluation table
- Model comparison notes
- Data dictionary + QA checklist
- Figures for stakeholder-ready reporting
