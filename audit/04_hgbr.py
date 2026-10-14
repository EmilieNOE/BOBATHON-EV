# %% [markdown]
# # Audit — 04_hgbr: HistGradientBoostingRegressor with 5-fold GroupKFold
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
# URL: https://skore.probabl.ai/evdgreenteam/bobathon-esilv/cross-validations/47467
# `cross-validations` → id uses singular `cross-validation`.

# %%
REPORT_ID = "skore:report:cross-validation:47467"
report = project.get(REPORT_ID)
report

# %%
report.checks.summarize().frame()

# %%
report.metrics.summarize().frame()

# %% [markdown]
# ## End of audit
