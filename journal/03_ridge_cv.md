# 03_ridge_cv

## Question / hypothesis

What is the honest RMSE of the Ridge + median-imputer pipeline when patients are
never shared between train and validation folds — and how does it compare to the
dummy baseline under the same grouped CV?

## Motivation

- **Sourcing strategy:** user
- **Source(s):**
  - User request: "Update the evaluation to use a 5-fold GroupKFold based on the
    'patient_id' column. We must ensure the same patient never appears in both the
    train and validation splits."
- **Why this matters:** The 80/20 random holdout in 02_ridge (RMSE 10.49, R² 0.60)
  is optimistic because the same patient can appear in both train and test. A model
  that memorises a patient's baseline severity will "predict" them well just by
  recognising their typical score — not by learning generalizable features. Grouped
  CV enforces a strict patient holdout: every fold's validation set contains patients
  never seen during training in that fold. The resulting metrics reflect how well the
  model generalises to **new patients**, which is the clinically relevant question.

## Method

- **Files touched:** `src/parkinsons/data.py` (add `load_groups` helper),
  `experiments/03_ridge_cv.py`, `tests/smoke/test_03_ridge_cv.py`
- **Change versus 02_ridge:** The splitter changes from `splitter=0.2` to
  `GroupKFold(n_splits=5)` with `groups=patient_id`. Both the dummy and Ridge
  pipelines are evaluated under this same CV scheme so the comparison is fair.
  The Ridge pipeline itself is unchanged (same `build_ridge_learner()`).
- **CV details:** `GroupKFold(n_splits=5)` — 5 folds, each fold's validation set
  contains ~20 % of patients not seen in training for that fold. Groups are the
  `patient_id` array extracted from `X_train`.
- **Out of scope for this experiment:** feature engineering, hyperparameter tuning,
  encoding of categorical columns, changing the pipeline architecture.

## Risks / things that could invalidate the result

- RMSE will very likely be higher than the optimistic 02_ridge number (10.49),
  because the patient-leakage shortcut is removed. This is the correct direction —
  higher CV RMSE here is a more honest estimate, not a regression.
- 5 folds on ~5,576 patients gives ~1,115 unique patients per fold. Some small
  cohorts may have fewer patients, introducing fold-to-fold variance. If variance
  is high, `RepeatedGroupKFold` would be the next step.
- The `on` and `off` columns' near-perfect correlation with `target` (r=0.87 / 0.67)
  still raises leakage concerns. They are included here as in 02_ridge.

## Status

- **State:** done
- **Approved by user on:** 2026-10-07
- **Headline result:** RMSE 10.60 ± 0.27, MAE 8.53 ± 0.18, R² 0.587 ± 0.016 (5-fold GroupKFold by patient_id)
- **Implication for next iteration:** Removing patient leakage costs only ~0.1 RMSE vs 02_ridge
  (10.60 vs 10.49) — the random holdout was not dramatically over-optimistic. Low fold variance
  (σ=0.27) confirms the model is stable. skore still flags `age` ↔ `age_at_diagnosis` collinearity
  (SKD008) and several zero-importance features (SKD012). HistGradientBoosting grouped-CV baseline
  is RMSE 7.66 — there is roughly 3 RMSE points left to gain with a tree-based model or better
  feature engineering.
