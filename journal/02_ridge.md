# 02_ridge

## Question / hypothesis

Does a Ridge regression with median imputation outperform the always-predict-mean
floor established by 01_dummy, and by how much?

## Motivation

- **Sourcing strategy:** user
- **Source(s):**
  - User request: "Create a machine learning pipeline with a median SimpleImputer
    and a Ridge regressor. Evaluate it against the dummy model using skore.evaluate
    and push the comparison report to the hub with the key '02_ridge'."
- **Why this matters:** Ridge is a regularised linear model that can extract signal
  from the continuous features (age, ledd, on, off) even in the presence of heavy
  missingness. Median imputation is a robust first-pass strategy that doesn't require
  assumptions about the missing-data mechanism. The comparison report lets us see
  directly how much linear signal the features carry above the mean-prediction floor.

## Method

- **Files touched:** `src/parkinsons/pipeline.py`, `experiments/02_ridge.py`,
  `tests/smoke/test_02_ridge.py`
- **Change versus 01_dummy:** The estimator is replaced with a
  `Pipeline([SimpleImputer(strategy='median'), Ridge()])`. The imputer handles the
  heavy missingness in `gene`, `ledd`, `time_since_intake_on/off`, `on`, `off`.
  Categorical columns (`patient_id`, `cohort`, `gene`, `rater_id`) are dropped before
  fitting because Ridge expects numeric input.
- **Comparison:** The experiment calls `skore.evaluate` on both the Ridge pipeline
  and the pre-fitted dummy report, producing a `ComparisonReport` stored under key
  `'02_ridge'`.
- **Cross-validation:** `splitter=0.2` (80/20 random row holdout) — same as 01_dummy,
  per the lab guide.
- **Out of scope for this experiment:** grouped CV, feature engineering, hyperparameter
  tuning, encoding of categorical columns.

## Risks / things that could invalidate the result

- The 80/20 split ignores patient grouping — the same caveat as 01_dummy applies.
  The comparison is fair (both use the same split), but the absolute metrics are
  optimistic relative to a proper patient holdout.
- `on` and `off` have high Pearson r with `target` (0.67 and 0.87). If those columns
  are derived from `target`, including them would inflate Ridge's score artificially.
  Keeping them in for now; the leakage question is flagged in the EDA open questions.
- Dropping string columns means Ridge sees only 8 of 12 features. A future experiment
  can add encoding for `gene`, `cohort`, `rater_id`.

## Status

- **State:** done
- **Approved by user on:** 2026-10-07
- **Headline result:** RMSE 10.49, MAE 8.44, R² 0.595 (80/20 random row holdout)
- **Implication for next iteration:** Ridge cuts dummy RMSE from 16.48 → 10.49 (−36 %) and
  R² from ≈0 → 0.60. Linear signal is real. skore flags: (1) `age` ↔ `age_at_diagnosis`
  correlated ≥ 0.9 — drop or engineer `disease_duration`; (2) several features useless
  by permutation importance; (3) Ridge alpha untouned. HistGradientBoosting still beats
  it (RMSE 7.18). Next natural step: patient-grouped CV to get honest holdout metrics, or
  a tree-based model.
