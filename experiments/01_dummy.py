# %% [markdown]
# # Experiment: 01_dummy
#
# **Date:** 2026-10-07
# **Goal:** Establish a performance floor using DummyRegressor(strategy='mean'),
# which always predicts the training-set mean. Every real model must beat this.
# **Result:** filled in after the run.

# %%
import skore
from skore import login

from parkinsons import PROJECT_ROOT
from parkinsons.data import load_dataset, load_test_dataset
from parkinsons.evaluate import splitter
from parkinsons.hub import load_skore_credentials
from parkinsons.pipeline import build_learner

# %% [markdown]
# ## Hub login
#
# Credentials are read from `.skore` (written by `scripts/install_skore.py`).
# `load_skore_credentials()` exports SKORE_HUB_URI and SKORE_HUB_API_KEY so
# `login()` does not open a browser. Always call it before `login(mode="hub")`.

# %%
cfg = load_skore_credentials()
login(mode="hub")
project = skore.Project(
    name="bobathon-esilv",
    mode="hub",
    workspace=cfg["workspace"],
)

# %% [markdown]
# ## Data and learner
#
# Load the full training set and build an unfit DummyRegressor.

# %%
DATA_DIR = PROJECT_ROOT / "data"
X, y = load_dataset(data_dir=DATA_DIR)
learner = build_learner()

# %% [markdown]
# ## Evaluate
#
# splitter=0.2 means an 80/20 random row holdout → returns an EstimatorReport.
# skore picks regression metrics automatically (RMSE, MAE, R²).
# sklearn-style (X, y) form because DummyRegressor has a standard fit(X, y).

# %%
report = skore.evaluate(learner, X, y, splitter=splitter)
report

# %% [markdown]
# ## Persist to hub
#
# Key = "01_dummy". Reusing this key in a future run overwrites the stored report.

# %%
project.put("01_dummy", report)

# %% [markdown]
# ## Kaggle submission
#
# Refit on all training rows (no CV split), predict the test visits,
# write to submissions/01_dummy.csv with header Index,target.

# %%
submissions_dir = PROJECT_ROOT / "submissions"
submissions_dir.mkdir(exist_ok=True)

X_test = load_test_dataset(data_dir=DATA_DIR)

final = build_learner()
final.fit(X, y)
predictions = final.predict(X_test)

import pandas as pd  # noqa: E402

submission = pd.DataFrame({"target": predictions}, index=X_test.index)
submission.index.name = "Index"
submission.to_csv(submissions_dir / "01_dummy.csv")
print(f"Submission written: {submissions_dir / '01_dummy.csv'}")
