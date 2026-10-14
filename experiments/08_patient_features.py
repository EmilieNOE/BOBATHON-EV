# %% [markdown]
# # Experiment: 08_patient_features
#
# **Date:** 2026-10-08
# **Goal:** Add patient-level temporal features to the HGBR baseline (04_hgbr /
# 07_best_model). New features: time_since_diagnosis, on_minus_off, n_visits,
# years_since_first_visit, and per-patient mean/prev/next for on, off, ledd.
# Same 5-fold GroupKFold. Hub key: 08_patient_features.

# %%
import numpy as np
import pandas as pd
import skore
from skore import login
from sklearn.model_selection import GroupKFold

from parkinsons import PROJECT_ROOT
from parkinsons.data import load_dataset, load_groups, load_test_dataset
from parkinsons.hub import load_skore_credentials
from parkinsons.pipeline import build_hgbr_patient_features

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
visits = X[["patient_id", "age"]].copy()   # retained for feature building
groups = load_groups(data_dir=DATA_DIR)

# %% [markdown]
# ## Precompute 5-fold GroupKFold splits (same rule as 04_hgbr / 07_best_model)

# %%
cv_splits = list(GroupKFold(n_splits=5).split(X, y, groups=visits["patient_id"]))

print(f"5-fold GroupKFold — {len(cv_splits)} folds")
for i, (tr, te) in enumerate(cv_splits):
    print(
        f"  Fold {i}: train {len(tr)} rows / {groups.iloc[tr].nunique()} patients, "
        f"val {len(te)} rows / {groups.iloc[te].nunique()} patients"
    )

# %% [markdown]
# ## Evaluate HGBR + patient features

# %%
report_08 = skore.evaluate(build_hgbr_patient_features(), X, y, splitter=cv_splits)

# %% [markdown]
# ## Report: RMSE summary

# %%
rmse_agg = report_08.metrics.rmse()
rmse_mean = rmse_agg.iloc[0, 0]   # (RMSE, mean)
rmse_std  = rmse_agg.iloc[0, 1]   # (RMSE, std)
rmse_per_fold = report_08.metrics.rmse(aggregate=None).iloc[0].values
print(f"08_patient_features — RMSE per fold: {rmse_per_fold.round(4).tolist()}")
print(f"  Mean: {rmse_mean:.4f}  Std: {rmse_std:.4f}")

# %% [markdown]
# ## Push to hub

# %%
project.put("08_patient_features", report_08)

# %% [markdown]
# ## Kaggle submission

# %%
submissions_dir = PROJECT_ROOT / "submissions"
submissions_dir.mkdir(exist_ok=True)

X_test = load_test_dataset(data_dir=DATA_DIR)

final_08 = build_hgbr_patient_features()
final_08.fit(X, y)
preds_08 = final_08.predict(X_test)

submission_08 = pd.DataFrame({"target": preds_08}, index=X_test.index)
submission_08.index.name = "Index"
submission_08.to_csv(submissions_dir / "08_patient_features.csv")
print(f"Submission written: {submissions_dir / '08_patient_features.csv'}")
