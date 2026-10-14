# %% [markdown]
# # Experiment: 09_rater_cohort
#
# **Date:** 2026-10-08
# **Goal:** Extend experiment 08 by adding rater_id, cohort, and gene as native
# categorical features for HGBR. HGBR (sklearn ≥ 1.4) can split on category
# dtype columns directly with ``categorical_features="from_dtype"``, avoiding
# the high-cardinality OHE explosion while still learning rater bias.
# Hub key: 09_rater_cohort.

# %%
import numpy as np
import pandas as pd
import skore
from skore import login
from sklearn.model_selection import GroupKFold

from parkinsons import PROJECT_ROOT
from parkinsons.data import load_dataset, load_groups, load_test_dataset
from parkinsons.hub import load_skore_credentials
from parkinsons.pipeline import build_hgbr_with_categoricals

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
# ## Cardinality check: rater_id, cohort, gene

# %%
for col in ("rater_id", "cohort", "gene"):
    n = X[col].nunique(dropna=False)
    print(f"  {col}: {n} unique values  (NaN={X[col].isna().sum()})")

# %% [markdown]
# ## Precompute 5-fold GroupKFold splits

# %%
cv_splits = list(GroupKFold(n_splits=5).split(X, y, groups=visits["patient_id"]))

# %% [markdown]
# ## Evaluate HGBR + patient features + categoricals

# %%
report_09 = skore.evaluate(build_hgbr_with_categoricals(), X, y, splitter=cv_splits)

# %% [markdown]
# ## Report: RMSE summary

# %%
rmse_agg = report_09.metrics.rmse()
rmse_mean = rmse_agg.iloc[0, 0]
rmse_std  = rmse_agg.iloc[0, 1]
rmse_per_fold = report_09.metrics.rmse(aggregate=None).iloc[0].values
print(f"09_rater_cohort — RMSE per fold: {rmse_per_fold.round(4).tolist()}")
print(f"  Mean: {rmse_mean:.4f}  Std: {rmse_std:.4f}")

# %% [markdown]
# ## Push to hub

# %%
project.put("09_rater_cohort", report_09)

# %% [markdown]
# ## Kaggle submission

# %%
submissions_dir = PROJECT_ROOT / "submissions"
submissions_dir.mkdir(exist_ok=True)

X_test = load_test_dataset(data_dir=DATA_DIR)

final_09 = build_hgbr_with_categoricals()
final_09.fit(X, y)
preds_09 = final_09.predict(X_test)

submission_09 = pd.DataFrame({"target": preds_09}, index=X_test.index)
submission_09.index.name = "Index"
submission_09.to_csv(submissions_dir / "09_rater_cohort.csv")
print(f"Submission written: {submissions_dir / '09_rater_cohort.csv'}")
