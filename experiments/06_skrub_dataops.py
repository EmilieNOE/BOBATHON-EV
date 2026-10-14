# %% [markdown]
# # Experiment: 06_skrub_dataops
#
# **Date:** 2026-10-07
# **Goal:** Replace the manual sklearn Pipeline + GroupKFold setup with skrub's
# DataOps graph. The CV strategy (5-fold GroupKFold on patient_id) is embedded
# directly in the graph via `mark_as_X`, so there is no separate `cv.split()` call.
#
# Key differences from 05_skrub_tabular:
# - `visits` is the entire merged training table (X_train + y_train).
# - `X_op` drops only `target`; `patient_id` is intentionally kept as a feature —
#   TableVectorizer encodes it. The DataFrame index (`Index`) is never a column,
#   so there is nothing to drop for it.
# - The groups column (`patient_id`) is referenced as a DataOp node
#   (`data["patient_id"]`) and passed to `split_kwargs` so GroupKFold
#   respects patient-level boundaries during CV.
# - `skore.evaluate(learner, data={"visits": visits})` is the integration point;
#   it consumes the SkrubLearner and returns a skore CrossValidationReport.
# **Result:** filled in after the run.

# %%
import pandas as pd
import skore
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import GroupKFold

import skrub
from parkinsons import PROJECT_ROOT
from parkinsons.data import load_test_dataset
from parkinsons.hub import load_skore_credentials

# %% [markdown]
# ## Hub login

# %%
cfg = load_skore_credentials()
print(f"Workspace : {cfg['workspace']}  (id={cfg['workspace_id']})")
print(f"Hub URL   : {cfg['hub_url']}")

project = skore.Project(name="bobathon-esilv", mode="hub", workspace=cfg["workspace"])
print(f"Connected : {project}")

# %% [markdown]
# ## Build `visits` — the full training table
#
# Same merge as Step 5: X_train joined with y_train. The DataFrame index is
# `Index` (the Kaggle row identifier); it is never a column so it does not
# appear in X_op at all.

# %%
DATA_DIR = PROJECT_ROOT / "data"

visits = pd.read_csv(DATA_DIR / "X_train.csv", index_col="Index").join(
    pd.read_csv(DATA_DIR / "y_train.csv", index_col="Index")
)

print(f"visits shape : {visits.shape}")
print(f"visits columns : {list(visits.columns)}")

# %% [markdown]
# ## Build the DataOps graph
#
# `data["patient_id"]` is a DataOp node referencing the `patient_id` column.
# It is passed as the groups source to `split_kwargs` so GroupKFold sees the
# correct patient labels at CV time.
#
# `X_op` drops **only** `target`. `patient_id` stays in X intentionally —
# TableVectorizer encodes it as a high-cardinality categorical (GapEncoder).
# The `Index` column does not exist as a column (it is the DataFrame index),
# so nothing extra needs to be dropped for it.

# %%
data = skrub.var("visits", visits)

groups_op = data["patient_id"]

X_op = data.drop(columns=["target"]).skb.mark_as_X(
    cv=GroupKFold(n_splits=5),
    split_kwargs={"groups": groups_op},
)
y_op = data["target"].skb.mark_as_y()

# Confirm the columns seen by X_op at preview time (full dataset, no split)
x_preview = X_op.skb.preview()
print(f"\nX_op columns ({len(x_preview.columns)}) : {list(x_preview.columns)}")
print(f"'Index' in X_op columns     : {'Index' in x_preview.columns}")
print(f"'patient_id' in X_op columns: {'patient_id' in x_preview.columns}")

# %% [markdown]
# ## Apply TableVectorizer → HGBR

# %%
pred = X_op.skb.apply(skrub.TableVectorizer()).skb.apply(
    HistGradientBoostingRegressor(random_state=0),
    y=y_op,
)

# %% [markdown]
# ## Evaluate with skore
#
# `skore.evaluate(learner, data={"visits": visits})` runs the CV embedded in
# the graph (5-fold GroupKFold on patient_id) and returns a CrossValidationReport.

# %%
learner = pred.skb.make_learner()
print(f"\nLearner : {learner}")

print("\nRunning skore.evaluate (5-fold GroupKFold) …")
report = skore.evaluate(learner, data={"visits": visits})
print(report)

# %% [markdown]
# ## Publish to hub

# %%
project.put("06_skrub_dataops", report)
# project.put() prints the report URL via rich console

# %% [markdown]
# ## Feature importances
#
# Fit the full pipeline once on all training data to inspect HGBR feature
# importances. The TableVectorizer will emit named columns; HGBR does not
# expose `feature_importances_` but does expose `feature_names_in_` after fit.
# We also check the fitted TableVectorizer's output columns.

# %%
from sklearn.pipeline import Pipeline as SKPipeline

# Re-use the full visits X/y
X_full = visits.drop(columns=["target"])
y_full = visits["target"]

# Build a standalone sklearn pipeline identical to what the DataOps graph uses
standalone = SKPipeline([
    ("tv", skrub.TableVectorizer()),
    ("hgbr", HistGradientBoostingRegressor(random_state=0)),
])
standalone.fit(X_full, y_full)

tv_fitted = standalone.named_steps["tv"]
hgbr_fitted = standalone.named_steps["hgbr"]

encoded_cols = tv_fitted.get_feature_names_out()
print(f"\nTableVectorizer output columns ({len(encoded_cols)}):")
print(list(encoded_cols))

print(f"\n'Index' in encoded cols     : {any('Index' in c for c in encoded_cols)}")
print(f"'patient_id' in encoded cols: {any('patient_id' in c for c in encoded_cols)}")

if hasattr(hgbr_fitted, 'feature_importances_'):
    import numpy as np
    importances = hgbr_fitted.feature_importances_
    top_idx = np.argsort(importances)[::-1][:15]
    print("\nTop 15 HGBR feature importances:")
    for i in top_idx:
        print(f"  {encoded_cols[i]:50s}  {importances[i]:.4f}")
else:
    print("\nHGBR does not expose feature_importances_ (expected for HGBR).")
    print("Fitted estimator type:", type(hgbr_fitted))
