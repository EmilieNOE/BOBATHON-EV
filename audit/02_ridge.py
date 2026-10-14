# %% [markdown]
# # Audit — 02_ridge: Ridge regression with median imputation
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
# Open the same project the experiment wrote to.

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

# %%
summary = project.summarize()
summary

# %% [markdown]
# ## Load the report
#
# URL printed by `project.put("02_ridge", ...)`:
#   https://skore.probabl.ai/evdgreenteam/bobathon-esilv/estimators/47002
#
# Type segment is `estimators` → id uses singular: `estimator`.

# %%
REPORT_ID = "skore:report:estimator:47002"

report = project.get(REPORT_ID)
report

# %% [markdown]
# ## Checks summary

# %%
report.checks.summarize().frame()

# %% [markdown]
# ## Metrics summary

# %%
report.metrics.summarize().frame()

# %% [markdown]
# ## End of audit
