# %% [markdown]
# # Experiment: 03_ridge_cv
#
# **Date:** 2026-10-07
# **Goal:** Re-evaluate Ridge (median imputer + Ridge) and DummyRegressor with a
# strict 5-fold GroupKFold on patient_id. No patient appears in both the train
# and validation sets of any fold. Compare results against the dummy baseline.
# **Result:** filled in after the run.

# %%
import pandas as pd
import skore
from skore import login
from sklearn.model_selection import GroupKFold

from parkinsons import PROJECT_ROOT
from parkinsons.data import load_dataset, load_groups, load_test_dataset
from parkinsons.hub import load_skore_credentials
from parkinsons.pipeline import build_learner, build_ridge_learner

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
#
# `load_groups` returns the `patient_id` column with the same index as `X`.
# This is passed to `GroupKFold.split` so that no patient crosses fold boundaries.

# %%
DATA_DIR = PROJECT_ROOT / "data"
X, y = load_dataset(data_dir=DATA_DIR)
groups = load_groups(data_dir=DATA_DIR)

# %% [markdown]
# ## Build grouped CV splits
#
# Pre-materialise the 5 GroupKFold splits so the same (train, val) index arrays
# can be used for both the dummy and the Ridge evaluation. This guarantees
# identical folds across estimators — a requirement for a fair comparison.
# `splitter` accepts a list of (train_idx, test_idx) arrays directly.

# %%
cv = GroupKFold(n_splits=5)
splits = list(cv.split(X, y, groups))

print(f"5-fold GroupKFold — {len(splits)} folds")
for i, (tr, te) in enumerate(splits):
    n_train_patients = groups.iloc[tr].nunique()
    n_val_patients = groups.iloc[te].nunique()
    print(
        f"  Fold {i}: train {len(tr)} rows / {n_train_patients} patients, "
        f"val {len(te)} rows / {n_val_patients} patients"
    )

# %% [markdown]
# ## Evaluate both estimators under GroupKFold
#
# Each call receives the same pre-generated `splits` list, so the train/val
# boundaries are identical for both estimators — the comparison is fair.

# %%
report_dummy_cv = skore.evaluate(build_learner(), X, y, splitter=splits)
report_ridge_cv = skore.evaluate(build_ridge_learner(), X, y, splitter=splits)

# %% [markdown]
# ## In-memory comparison
#
# Side-by-side metrics under the same patient-grouped CV.

# %%
comparison = skore.compare(
    {"DummyRegressor": report_dummy_cv, "Ridge": report_ridge_cv}
)
comparison

# %% [markdown]
# ## Persist Ridge CV report to hub
#
# `project.put` accepts EstimatorReport and CrossValidationReport only.
# Push the Ridge CrossValidationReport under key "03_ridge_cv".

# %%
project.put("03_ridge_cv", report_ridge_cv)

# %% [markdown]
# ## Kaggle submission
#
# Refit on **all** training rows, predict on `X_test`, save to
# `submissions/03_ridge_cv.csv`.

# %%
submissions_dir = PROJECT_ROOT / "submissions"
submissions_dir.mkdir(exist_ok=True)

X_test = load_test_dataset(data_dir=DATA_DIR)

final = build_ridge_learner()
final.fit(X, y)
predictions = final.predict(X_test)

submission = pd.DataFrame({"target": predictions}, index=X_test.index)
submission.index.name = "Index"
submission.to_csv(submissions_dir / "03_ridge_cv.csv")
print(f"Submission written: {submissions_dir / '03_ridge_cv.csv'}")
