# %% [markdown]
# # Audit — 01_dummy: DummyRegressor mean-prediction baseline
#
# Read-only review of the stored report below: its checks and metrics.

# %%
import skore
from skore import login

from parkinsons import PROJECT_ROOT
from parkinsons.hub import load_skore_credentials

# %% [markdown]
# ## Open the project
#
# Open the same project the experiment wrote to. The init block below
# matches `experiments/01_dummy.py` exactly.

# %%
cfg = load_skore_credentials()
login(mode="hub")
project = skore.Project(
    name="bobathon-esilv",
    mode="hub",
    workspace=cfg["workspace"],
)
project

# %% [markdown]
# ## List the available reports
#
# `project.summarize()` provides an overview of all reports in this
# Project — useful to confirm the experiment's report landed and to
# spot duplicate keys from accidental re-runs.

# %%
summary = project.summarize()
summary

# %% [markdown]
# ## Load the report
#
# URL printed by `project.put("01_dummy", ...)`:
#   https://skore.probabl.ai/evdgreenteam/bobathon-esilv/estimators/46163
#
# Type segment is `estimators` (plural) → id uses singular: `estimator`.

# %%
REPORT_ID = "skore:report:estimator:46163"

report = project.get(REPORT_ID)
report

# %% [markdown]
# ## Checks summary
#
# `report.checks.summarize().frame()` returns a DataFrame with
# `code`, `severity` (passed / issue / tip), and `documentation_url`.
# Follow any `issue` or `tip` URL for the recommended mitigation.

# %%
report.checks.summarize().frame()

# %% [markdown]
# ## Metrics summary
#
# `report.metrics.summarize().frame()` shows task-appropriate defaults
# for regression: RMSE, MAE, R², and fit/predict timings.

# %%
report.metrics.summarize().frame()

# %% [markdown]
# ## End of audit
#
# This file is the durable record of how the experiment's report was
# reviewed. Re-run it any time to refresh the checks and metrics above.
