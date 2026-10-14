# %% [markdown]
# # Audit — 03_ridge_cv: Ridge with 5-fold GroupKFold by patient_id
#
# Read-only review of the stored CrossValidationReport.

# %%
import skore
from skore import login

from parkinsons import PROJECT_ROOT
from parkinsons.hub import load_skore_credentials

# %%
cfg = load_skore_credentials()
login(mode="hub")
project = skore.Project(
    name="bobathon-esilv",
    mode="hub",
    workspace=cfg["workspace"],
)
project

# %%
summary = project.summarize()
summary

# %% [markdown]
# ## Load the report
#
# URL printed by `project.put("03_ridge_cv", ...)`:
#   https://skore.probabl.ai/evdgreenteam/bobathon-esilv/cross-validations/47351
#
# Type segment is `cross-validations` → id uses singular: `cross-validation`.

# %%
REPORT_ID = "skore:report:cross-validation:47351"

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
