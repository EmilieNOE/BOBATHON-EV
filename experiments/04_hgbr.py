# %% [markdown]
# # Experiment: 04_hgbr
#
# **Date:** 2026-10-07
# **Goal:** Replace the Ridge + imputer pipeline with HistGradientBoostingRegressor
# (random_state=0, no imputer). Evaluate under the same 5-fold GroupKFold on
# patient_id. HGBR handles NaN natively, preserving missingness as a learnable signal.
# **Result:** filled in after the run.

# %%
import pandas as pd
import skore
from skore import login
from sklearn.model_selection import GroupKFold

from parkinsons import PROJECT_ROOT
from parkinsons.data import load_dataset, load_groups, load_test_dataset
from parkinsons.hub import load_skore_credentials
from parkinsons.pipeline import build_hgbr_learner, build_learner

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
groups = load_groups(data_dir=DATA_DIR)

# %% [markdown]
# ## Build the same 5-fold GroupKFold splits as 03_ridge_cv
#
# Pre-materialise the splits so both the dummy and HGBR evaluations use
# identical fold boundaries — a requirement for a fair comparison.

# %%
cv = GroupKFold(n_splits=5)
splits = list(cv.split(X, y, groups))

print(f"5-fold GroupKFold — {len(splits)} folds")
for i, (tr, te) in enumerate(splits):
    print(
        f"  Fold {i}: train {len(tr)} rows / {groups.iloc[tr].nunique()} patients, "
        f"val {len(te)} rows / {groups.iloc[te].nunique()} patients"
    )

# %% [markdown]
# ## Evaluate HGBR and dummy under GroupKFold

# %%
report_dummy_cv = skore.evaluate(build_learner(), X, y, splitter=splits)
report_hgbr_cv = skore.evaluate(build_hgbr_learner(), X, y, splitter=splits)

# %% [markdown]
# ## In-memory comparison

# %%
comparison = skore.compare(
    {"DummyRegressor": report_dummy_cv, "HGBR": report_hgbr_cv}
)
comparison

# %% [markdown]
# ## Persist HGBR CrossValidationReport to hub

# %%
project.put("04_hgbr", report_hgbr_cv)

# %% [markdown]
# ## Kaggle submission
#
# Refit on all training rows, predict on X_test.

# %%
submissions_dir = PROJECT_ROOT / "submissions"
submissions_dir.mkdir(exist_ok=True)

X_test = load_test_dataset(data_dir=DATA_DIR)

final = build_hgbr_learner()
final.fit(X, y)
predictions = final.predict(X_test)

submission = pd.DataFrame({"target": predictions}, index=X_test.index)
submission.index.name = "Index"
submission.to_csv(submissions_dir / "04_hgbr.csv")
print(f"Submission written: {submissions_dir / '04_hgbr.csv'}")
