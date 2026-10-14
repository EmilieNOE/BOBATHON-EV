# %% [markdown]
# # Experiment: 12_ensemble
#
# **Date:** 2026-10-08
# **Goal:** Average predictions from several HGBR seeds (random_state 0, 1, 2, 3, 4)
# trained with the best tuned hyperparameters from 11_tuned, optionally blended
# with a Ridge on the engineered feature set. The ensemble is evaluated with
# grouped CV. Hub key: 12_ensemble.
#
# Design rationale: bagging HGBR seeds reduces the variance of any single
# random initialization while keeping the same bias. Ridge adds a globally
# smooth predictor that complements HGBR's localised splits.

# %%
import numpy as np
import pandas as pd
import skore
from skore import login
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer

from parkinsons import PROJECT_ROOT
from parkinsons.data import load_dataset, load_groups, load_test_dataset
from parkinsons.hub import load_skore_credentials
from parkinsons.pipeline import _apply_patient_features, build_hgbr_tuned

# %% [markdown]
# ## Hub login

# %%
cfg = load_skore_credentials()
login(mode="hub")
project = skore.Project(
    name="bobathon-esilv",
    mode="hub",
    workspace=cfg["workspace"],
)

# %% [markdown]
# ## Data and groups

# %%
DATA_DIR = PROJECT_ROOT / "data"
X, y = load_dataset(data_dir=DATA_DIR)
visits = X[["patient_id", "age"]].copy()
groups = load_groups(data_dir=DATA_DIR)

# %% [markdown]
# ## Best hyperparameters from 11_tuned (hard-coded for reproducibility)
#
# Paste the best_params printed by 11_tuned.py here so the ensemble does not
# re-run the search. Update these values after running 11_tuned.py.

# %%
# Best params from 11_tuned RandomizedSearchCV run.
BEST_PARAMS = {
    "learning_rate": 0.02,
    "max_iter": 1000,
    "max_leaf_nodes": 31,
    "min_samples_leaf": 20,
    "l2_regularization": 0.0,
}

SEEDS = [0, 1, 2, 3, 4]

# %% [markdown]
# ## Precompute 5-fold GroupKFold splits

# %%
cv_splits = list(GroupKFold(n_splits=5).split(X, y, groups=visits["patient_id"]))

# %% [markdown]
# ## Collect OOF predictions from each seed

# %%
oof_hgbr = np.zeros((len(X), len(SEEDS)))

for s_idx, seed in enumerate(SEEDS):
    oof_seed = np.full(len(X), np.nan)
    for train_idx, val_idx in cv_splits:
        m = build_hgbr_tuned(**BEST_PARAMS, random_state=seed)
        m.fit(X.iloc[train_idx], y.iloc[train_idx])
        oof_seed[val_idx] = m.predict(X.iloc[val_idx])
    oof_hgbr[:, s_idx] = oof_seed
    print(f"Seed {seed}: OOF RMSE = {np.sqrt(np.mean((oof_seed - y.values)**2)):.4f}")

# Mean of HGBR seeds
oof_ensemble_mean = oof_hgbr.mean(axis=1)
ensemble_mean_rmse = np.sqrt(np.mean((oof_ensemble_mean - y.values) ** 2))
print(f"\nHGBR seed-mean OOF RMSE: {ensemble_mean_rmse:.4f}")

# %% [markdown]
# ## OOF Ridge predictions (on engineered features, no target)

# %%
ridge_pipe = Pipeline(
    steps=[
        ("engineer", FunctionTransformer(_apply_patient_features, validate=False)),
        ("ridge", Ridge()),
    ]
)

oof_ridge = np.full(len(X), np.nan)
for train_idx, val_idx in cv_splits:
    ridge_pipe.fit(X.iloc[train_idx], y.iloc[train_idx])
    oof_ridge[val_idx] = ridge_pipe.predict(X.iloc[val_idx])

ridge_rmse = np.sqrt(np.mean((oof_ridge - y.values) ** 2))
print(f"Ridge OOF RMSE: {ridge_rmse:.4f}")

# %% [markdown]
# ## Grid search over blend weights: HGBR-mean vs Ridge

# %%
ALPHAS = [0.0, 0.05, 0.1, 0.15, 0.2]

best_alpha = 0.0
best_blend_rmse = ensemble_mean_rmse

print("\nBlend search (α = weight on Ridge):")
for alpha in ALPHAS:
    blended = (1 - alpha) * oof_ensemble_mean + alpha * oof_ridge
    rmse = np.sqrt(np.mean((blended - y.values) ** 2))
    print(f"  α={alpha:.2f}  RMSE={rmse:.4f}")
    if rmse < best_blend_rmse:
        best_blend_rmse = rmse
        best_alpha = alpha

print(f"\nBest α={best_alpha}  →  OOF RMSE {best_blend_rmse:.4f}")

# %% [markdown]
# ## Fold-wise RMSE for the best blended ensemble
#
# We re-compute per-fold so we can report mean ± std.

# %%
fold_rmses_ens = []
for f_idx, (train_idx, val_idx) in enumerate(cv_splits):
    # HGBR-mean on this fold
    hgbr_preds_fold = np.zeros((len(val_idx), len(SEEDS)))
    for s_idx, seed in enumerate(SEEDS):
        hgbr_preds_fold[:, s_idx] = oof_hgbr[val_idx, s_idx]
    hgbr_mean_fold = hgbr_preds_fold.mean(axis=1)

    ridge_fold = oof_ridge[val_idx]
    blended_fold = (1 - best_alpha) * hgbr_mean_fold + best_alpha * ridge_fold
    rmse_fold = np.sqrt(np.mean((blended_fold - y.values[val_idx]) ** 2))
    fold_rmses_ens.append(rmse_fold)

fold_arr = np.array(fold_rmses_ens)
print(f"12_ensemble — RMSE per fold: {fold_arr.round(4).tolist()}")
print(f"  Mean: {fold_arr.mean():.4f}  Std: {fold_arr.std():.4f}")

# %% [markdown]
# ## Build a skore report using the best single seed (for hub compatibility)
#
# skore.evaluate wraps one estimator; we use the single best seed to get a
# formal CrossValidationReport. The ensemble OOF numbers above are the true
# metric.

# %%
report_12 = skore.evaluate(
    build_hgbr_tuned(**BEST_PARAMS, random_state=0), X, y, splitter=cv_splits
)

# %% [markdown]
# ## Push to hub

# %%
project.put("12_ensemble", report_12)
print(
    f"12_ensemble hub report pushed (seed-0 base; full-ensemble OOF RMSE "
    f"{fold_arr.mean():.4f} ± {fold_arr.std():.4f})"
)

# %% [markdown]
# ## Kaggle submission — refit ensemble on full training set

# %%
submissions_dir = PROJECT_ROOT / "submissions"
submissions_dir.mkdir(exist_ok=True)

X_test = load_test_dataset(data_dir=DATA_DIR)

# HGBR-seed ensemble
test_hgbr_preds = np.zeros((len(X_test), len(SEEDS)))
for s_idx, seed in enumerate(SEEDS):
    m = build_hgbr_tuned(**BEST_PARAMS, random_state=seed)
    m.fit(X, y)
    test_hgbr_preds[:, s_idx] = m.predict(X_test)

test_hgbr_mean = test_hgbr_preds.mean(axis=1)

# Ridge on full training data
ridge_pipe.fit(X, y)
test_ridge_preds = ridge_pipe.predict(X_test)

preds_12 = (1 - best_alpha) * test_hgbr_mean + best_alpha * test_ridge_preds

submission_12 = pd.DataFrame({"target": preds_12}, index=X_test.index)
submission_12.index.name = "Index"
submission_12.to_csv(submissions_dir / "12_ensemble.csv")
print(f"Submission written: {submissions_dir / '12_ensemble.csv'}")

# %% [markdown]
# ## Best hub report URL
# %%
print("\n=== BEST REPORT ===")
print("Compare all experiments at:")
print(f"https://app.skore.ai/{cfg['workspace']}/project/bobathon-esilv")
print("\nThe best individual report key to paste in the Kaggle submission description")
print("will be whichever of 11_tuned or 12_ensemble has the lower mean grouped-CV RMSE.")
