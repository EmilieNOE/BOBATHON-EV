# %% [markdown]
# # Experiment: 07_best_model
#
# **Date:** 2026-10-12
# **Goal:** Step 7 — identify the best model across Steps 1–6 based on
# patient-grouped 5-fold CrossValidation RMSE and submit it.
#
# ## Hub summary (all steps, lowest RMSE wins)
#
# | Key              | Type            | Learner                   | CV RMSE  | Std   |
# |------------------|-----------------|---------------------------|----------|-------|
# | 01_dummy         | estimator 80/20 | DummyRegressor            | 16.478   | —     |
# | 02_ridge         | estimator 80/20 | Ridge                     | 10.488   | —     |
# | 03_ridge_cv      | cross-validation| Ridge                     | 10.597   | 0.267 |
# | 04_hgbr          | cross-validation| HGBR                      |  7.756   | 0.153 |
# | 05_skrub_tabular | cross-validation| TableVectorizer + HGBR    |  7.527   | 0.137 | ← BEST
# | 06_skrub_dataops | cross-validation| TableVectorizer + HGBR    |  7.645   | 0.142 |
#   (06 includes patient_id as a feature; that hurts generalisation slightly)
#
# **Winner: 05_skrub_tabular** — `TableVectorizer() → HGBR(random_state=0)`,
# trained on X_full (visits minus patient_id and target), mean CV RMSE 7.527.
#
# This script:
#   1. Re-evaluates the winning pipeline under the same 5-fold GroupKFold to
#      produce a fresh CrossValidationReport for the hub.
#   2. Fits the pipeline on **all** 44 590 training rows.
#   3. Saves the submission CSV to `submissions/07_best_model.csv`.
#   4. Publishes the report under key `07_best_model`.
# **Result:** filled in after the run.

# %%
import pandas as pd
import skore
from sklearn.model_selection import GroupKFold

from parkinsons import PROJECT_ROOT
from parkinsons.data import load_groups, load_test_dataset
from parkinsons.hub import load_skore_credentials
from parkinsons.pipeline import build_skrub_learner

# %% [markdown]
# ## Hub login

# %%
cfg = load_skore_credentials()
print(f"Workspace : {cfg['workspace']}  (id={cfg['workspace_id']})")
print(f"Hub URL   : {cfg['hub_url']}")

project = skore.Project(name="bobathon-esilv", mode="hub", workspace=cfg["workspace"])
print(f"Connected : {project}")

# %% [markdown]
# ## Build visits and X_full / y
#
# Same construction as Step 5: join X_train + y_train; drop `patient_id`
# (ID column, not a feature) and `target` (label) to produce `X_full`.
# The DataFrame index `Index` is never a column, so nothing extra is dropped.

# %%
DATA_DIR = PROJECT_ROOT / "data"

visits = pd.read_csv(DATA_DIR / "X_train.csv", index_col="Index").join(
    pd.read_csv(DATA_DIR / "y_train.csv", index_col="Index")
)

y = visits["target"]
X_full = visits.drop(columns=["patient_id", "target"])

print(f"X_full shape : {X_full.shape}")
print(f"Columns      : {list(X_full.columns)}")

# %% [markdown]
# ## Patient-grouped 5-fold CV splits
#
# Identical construction to Steps 03–05 so the fold boundaries are comparable.

# %%
groups = load_groups(data_dir=DATA_DIR)

cv = GroupKFold(n_splits=5)
cv_splits = list(cv.split(X_full, y, groups))

print(f"\n5-fold GroupKFold — {len(cv_splits)} folds")
for i, (tr, te) in enumerate(cv_splits):
    print(
        f"  Fold {i}: train {len(tr):>5} rows / {groups.iloc[tr].nunique()} patients, "
        f"val {len(te):>5} rows / {groups.iloc[te].nunique()} patients"
    )

# %% [markdown]
# ## Evaluate the winning pipeline
#
# `build_skrub_learner()` returns `Pipeline([TableVectorizer(), HGBR(random_state=0)])`.
# `skore.evaluate` with the pre-materialised GroupKFold splits produces a
# CrossValidationReport that is identical in methodology to Step 5.

# %%
model = build_skrub_learner()
print(f"\nPipeline : {model}")
print("\nRunning skore.evaluate (5-fold GroupKFold) …")
report = skore.evaluate(model, X_full, y, splitter=cv_splits)
print(report)

# %% [markdown]
# ## Save submission CSV
#
# Refit on all 44 590 training rows, predict on X_test.
# Drop `patient_id` from X_test so the feature set matches X_full exactly.

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
out_path = submissions_dir / "07_best_model.csv"
submission.to_csv(out_path)
print(f"\nSubmission written : {out_path}")
print(f"Rows in CSV        : {len(submission)}")

# %% [markdown]
# ## Publish to hub
#
# Key: `07_best_model`. project.put() prints the direct report URL.

# %%
project.put("07_best_model", report)
