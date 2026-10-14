# %% [markdown]
# # Experiment: 02_ridge
#
# **Date:** 2026-10-07
# **Goal:** Compare a Ridge pipeline (median imputation + Ridge) against the dummy
# baseline on the same 80/20 holdout to measure how much linear signal the features
# carry above always-predict-mean.
# **Result:** filled in after the run.

# %%
import pandas as pd
import skore
from skore import login

from parkinsons import PROJECT_ROOT
from parkinsons.data import load_dataset, load_test_dataset
from parkinsons.evaluate import splitter
from parkinsons.hub import load_skore_credentials
from parkinsons.pipeline import build_learner, build_ridge_learner

# %% [markdown]
# ## Hub login
#
# Credentials are read from `.skore`. Must call `load_skore_credentials()`
# before `login(mode="hub")` to export SKORE_HUB_URI and SKORE_HUB_API_KEY.

# %%
cfg = load_skore_credentials()
login(mode="hub")
project = skore.Project(
    name="bobathon-esilv",
    mode="hub",
    workspace=cfg["workspace"],
)

# %% [markdown]
# ## Data
#
# Full training set: 44,590 rows × 12 features + target.

# %%
DATA_DIR = PROJECT_ROOT / "data"
X, y = load_dataset(data_dir=DATA_DIR)

# %% [markdown]
# ## Evaluate both estimators on the same split
#
# Evaluate DummyRegressor and the Ridge pipeline on the **same** 80/20
# random holdout so the metrics are directly comparable. Each `evaluate`
# call with a float splitter fixes `random_state=0` internally, guaranteeing
# the same train/test split.

# %%
report_dummy = skore.evaluate(build_learner(), X, y, splitter=splitter)
report_ridge = skore.evaluate(build_ridge_learner(), X, y, splitter=splitter)

# %% [markdown]
# ## Comparison report (in-memory)
#
# `skore.compare` consolidates both reports into a side-by-side
# `ComparisonReport`. `project.put` only accepts individual EstimatorReport /
# CrossValidationReport objects, so the comparison is produced here for
# inspection; the Ridge report is pushed separately under key "02_ridge".

# %%
comparison = skore.compare(
    {"DummyRegressor": report_dummy, "Ridge": report_ridge}
)
comparison

# %% [markdown]
# ## Persist to hub
#
# Push the Ridge report under key "02_ridge". The DummyRegressor report is
# already on the hub as "01_dummy"; pushing the Ridge report here lets the
# hub show both side-by-side.

# %%
project.put("02_ridge", report_ridge)

# %% [markdown]
# ## Kaggle submission
#
# Refit the Ridge pipeline on **all** training rows (no CV split), predict on
# the test visits, write to `submissions/02_ridge.csv` with header `Index,target`.

# %%
submissions_dir = PROJECT_ROOT / "submissions"
submissions_dir.mkdir(exist_ok=True)

X_test = load_test_dataset(data_dir=DATA_DIR)

final = build_ridge_learner()
final.fit(X, y)
predictions = final.predict(X_test)

submission = pd.DataFrame({"target": predictions}, index=X_test.index)
submission.index.name = "Index"
submission.to_csv(submissions_dir / "02_ridge.csv")
print(f"Submission written: {submissions_dir / '02_ridge.csv'}")
