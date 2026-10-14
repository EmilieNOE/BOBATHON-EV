# %% [markdown]
# # Experiment: 05_skrub_tabular
#
# **Date:** 2026-10-07
# **Goal:** Leverage skrub's TableVectorizer to encode *all* columns — including
# categorical strings (`gene`, `cohort`, `rater_id`) — without manual preprocessing.
# Pipeline: TableVectorizer() → HistGradientBoostingRegressor(random_state=0).
# Evaluated under the same 5-fold patient-GroupKFold used in prior steps so
# results are directly comparable.
# **Result:** filled in after the run.

# %%
import pandas as pd
import skore
from skore import login
from sklearn.model_selection import GroupKFold

from parkinsons import PROJECT_ROOT
from parkinsons.data import load_groups, load_test_dataset
from parkinsons.hub import load_skore_credentials
from parkinsons.pipeline import build_skrub_learner

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
# ## Build `visits` — the full training table
#
# Merge `X_train` and `y_train` into a single `visits` dataframe.
# `X_full` drops `patient_id` (ID column, not a feature) and `target`
# (the label). The `Index` column is already the DataFrame index so there
# is nothing to drop explicitly for it; if it were a regular column we drop
# it here too for safety.

# %%
DATA_DIR = PROJECT_ROOT / "data"

visits = pd.read_csv(DATA_DIR / "X_train.csv", index_col="Index").join(
    pd.read_csv(DATA_DIR / "y_train.csv", index_col="Index")
)

y = visits["target"]
X_full = visits.drop(columns=["patient_id", "target"])

print(f"X_full shape: {X_full.shape}")
print(f"Columns: {list(X_full.columns)}")

# %% [markdown]
# ## Patient-grouped CV splits
#
# Reuse the same 5-fold GroupKFold strategy as experiments 03 and 04 so the
# fold boundaries are consistent and cross-experiment comparison is valid.

# %%
groups = load_groups(data_dir=DATA_DIR)

cv = GroupKFold(n_splits=5)
cv_splits = list(cv.split(X_full, y, groups))

print(f"5-fold GroupKFold — {len(cv_splits)} folds")
for i, (tr, te) in enumerate(cv_splits):
    print(
        f"  Fold {i}: train {len(tr)} rows / {groups.iloc[tr].nunique()} patients, "
        f"val {len(te)} rows / {groups.iloc[te].nunique()} patients"
    )

# %% [markdown]
# ## Build and evaluate the skrub pipeline
#
# `skore.evaluate` runs cross-validation with the pre-materialised splits,
# ensuring patient-level isolation across folds.

# %%
model = build_skrub_learner()
report = skore.evaluate(model, X_full, y, splitter=cv_splits)
report

# %% [markdown]
# ## Push the report to the Skore hub

# %%
project.put("05_skrub_tabular", report)
print("Report pushed to hub under key '05_skrub_tabular'")

# %% [markdown]
# ## Kaggle submission
#
# Refit the pipeline on all training rows, predict on `X_test`.
# Drop `patient_id` from `X_test` so the feature set matches `X_full`.

# %%
submissions_dir = PROJECT_ROOT / "submissions"
submissions_dir.mkdir(exist_ok=True)

X_test_raw = load_test_dataset(data_dir=DATA_DIR)
X_test = X_test_raw.drop(columns=["patient_id"])

final = build_skrub_learner()
final.fit(X_full, y)
predictions = final.predict(X_test)

submission = pd.DataFrame({"target": predictions}, index=X_test.index)
submission.index.name = "Index"
submission.to_csv(submissions_dir / "05_skrub_tabular.csv")
print(f"Submission written: {submissions_dir / '05_skrub_tabular.csv'}")
