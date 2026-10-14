# 04_hgbr

## Question / hypothesis

Does a HistGradientBoostingRegressor — which handles missing values natively and
treats missingness as a signal — outperform the Ridge baseline under the same
5-fold GroupKFold CV?

## Motivation

- **Sourcing strategy:** user
- **Source(s):**
  - User request: "Replace the Ridge pipeline with a HistGradientBoostingRegressor
    (set random_state=0). Remove the SimpleImputer, as HGBR handles missing values
    natively and missingness is a clinical signal here."
- **Why this matters:** Ridge needs explicit imputation that discards the information
  that a value is *absent*. HGBR's histogram-binning algorithm natively routes NaN
  rows to the best child node during training — missingness pattern becomes a learnable
  split. Given that `ledd` (37 %), `time_since_intake_on` (46 %), `time_since_intake_off`
  (79 %), `on` (30 %), and `off` (42 %) are all heavily missing, preserving those
  signals should improve over imputed Ridge. The skore internal HGB baseline already
  showed RMSE 7.66 under grouped CV (vs Ridge 10.60), so there is ~3 RMSE points of
  headroom to recover.

## Method

- **Files touched:** `src/parkinsons/pipeline.py` (add `build_hgbr_learner`),
  `experiments/04_hgbr.py`, `tests/smoke/test_04_hgbr.py`
- **Change versus 03_ridge_cv:** Estimator changes from
  `Pipeline([drop_strings, SimpleImputer, Ridge])` to a bare
  `HistGradientBoostingRegressor(random_state=0)`. No imputer — HGBR handles NaN.
  String columns are still dropped (HGBR in sklearn 1.x cannot encode arbitrary
  strings; categorical encoding of `gene`, `cohort`, `rater_id` is out of scope here).
- **CV:** Same 5-fold GroupKFold on `patient_id`, same `load_groups()` helper — folds
  are identical to 03_ridge_cv for a fair comparison.
- **Out of scope:** hyperparameter tuning, categorical encoding, feature engineering.

## Risks / things that could invalidate the result

- HGBR can overfit on small training folds if `max_iter` is too high; defaults
  include early stopping (`early_stopping='auto'`) which mitigates this.
- String columns are still dropped — HGBR could benefit from `cohort`, `gene`,
  `rater_id` if they were properly encoded. Leaving them out is conservative.
- The `on` / `off` leakage question remains open (see EDA). If those columns are
  derived from `target`, HGBR will exploit the signal aggressively; Ridge did too
  but less so due to linearity.

## Status

- **State:** done
- **Approved by user on:** 2026-10-07
- **Headline result:** RMSE 7.76 ± 0.15, MAE 6.10 ± 0.10, R² 0.779 ± 0.004 (5-fold GroupKFold by patient_id)
- **Implication for next iteration:** HGBR cuts Ridge RMSE from 10.60 → 7.76 (−27 %) and R²
  from 0.587 → 0.779. All skore checks pass (no overfitting, no underfitting, stable across
  folds). skore confirms HGBR is on par with its internal HGB baseline (RMSE 7.65). Remaining
  tips: (1) `cohort`, `gene`, `rater_id`, `sexM`, `time_since_intake_off` have near-zero
  permutation importance — consider dropping; (2) `learning_rate` and `max_leaf_nodes` are
  worth tuning. Next steps: hyperparameter search, or add categorical encoding for `gene`/`rater_id`.
