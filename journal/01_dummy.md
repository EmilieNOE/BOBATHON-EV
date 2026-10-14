# 01_dummy

## Question / hypothesis

What RMSE does a model that always predicts the training mean achieve,
establishing the lower-bound performance any real model must beat?

## Motivation

- **Sourcing strategy:** user
- **Source(s):**
  - User request: "Run a dummy regressor from sklearn to act as my baseline model."
- **Why this matters:** Before investing in any real model, we need to know what
  "just guess the average" looks like in terms of RMSE. That number anchors every
  subsequent experiment — a model is only useful if it beats this floor.
  This is also the standard Kaggle starting point to sanity-check the pipeline
  end-to-end (data loading, hub push, Kaggle submission format) before the science begins.

## Method

- **Files touched:** `src/parkinsons/data.py`, `src/parkinsons/pipeline.py`,
  `src/parkinsons/evaluate.py`, `experiments/01_dummy.py`
- **Change versus baseline:** First experiment — establishes the workspace.
  The estimator is `DummyRegressor(strategy='mean')`, which ignores all features
  and predicts the training-set mean for every row. No preprocessing or feature
  engineering. Evaluated with a simple 80/20 random row holdout (`splitter=0.2`),
  as specified in the lab guide. Patient-grouped CV will come in a later experiment.
- **Cross-validation:** `splitter=0.2` (random 80/20 holdout) — fixed for this
  experiment. The patient-grouping leak is intentional here; it is the point of
  comparison for the later grouped-CV experiment.
- **Out of scope for this experiment:** feature engineering, imputation, grouped CV,
  hyperparameter tuning.

## Risks / things that could invalidate the result

- The 80/20 split ignores patient grouping: the same patient can appear in both
  train and test, so the RMSE is a lower bound for any real holdout. This is the
  known limitation and will be addressed in a later grouped-CV experiment.
- DummyRegressor ignores all features; the RMSE here is purely a function of
  target variance (std ≈ 16.5). There is no feature-related signal to leak.

## Status

- **State:** done
- **Approved by user on:** 2026-10-07
- **Headline result:** RMSE 16.48 ± n/a, MAE 13.48 (80/20 random row holdout)
- **Implication for next iteration:** The mean-prediction floor is set. skore's HistGradientBoosting
  internal baseline achieves RMSE 7.18, MAE 5.66, R² 0.81 — any real model should aim in that
  direction. Next step: a Ridge regression on the same 80/20 holdout (02_ridge) to see whether
  simple linear features beat the mean.
