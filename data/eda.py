# %% [markdown]
# # EDA: Parkinson's Disease UPDRS Score Prediction
#
# Exploratory data analysis of the Parkinson's disease dataset,
# run before designing a model.
#
# - **Raw data** is read-only — loaded from `data/X_train.csv`,
#   `data/y_train.csv`, `data/X_test.csv`. This file never cleans
#   or modifies the raw data.
# - **Outputs** go under `EDA_DIR` (the repo's `data/`): an
#   `eda_<table>.html` report per table, summarized in `eda.md`.

# %%
import json
from pathlib import Path

import pandas as pd
import skrub

# This file lives in data/, so parents[1] is the repo root.
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# EDA outputs always land here (created if missing); raw data is in data/ too.
EDA_DIR = PROJECT_ROOT / "data"
EDA_DIR.mkdir(parents=True, exist_ok=True)

# %% [markdown]
# ## Load the raw data
#
# Load X_train and y_train (merged into a single training table), and X_test.

# %%
X_train = pd.read_csv(PROJECT_ROOT / "data" / "X_train.csv", index_col="Index")
y_train = pd.read_csv(PROJECT_ROOT / "data" / "y_train.csv", index_col="Index")
RAW = X_train.join(y_train)

X_test = pd.read_csv(PROJECT_ROOT / "data" / "X_test.csv", index_col="Index")

RAW.shape  # noqa: B018

# %% [markdown]
# ## Table overview — training set (X_train + y_train)
#
# Per-column summary: dtype, fraction missing, and number of unique values.
# The rich interactive report is saved to `data/eda_train.html`.

# %%
report_train = skrub.TableReport(RAW, title="Parkinson's UPDRS — train", verbose=0)
report_train.write_html(EDA_DIR / "eda_train.html")

summary_train = json.loads(report_train.json())
n_rows = summary_train.get("n_rows")
overview_train = [
    {
        "column": col.get("name"),
        "dtype": col.get("dtype"),
        "null_pct": col.get("null_proportion"),
        "n_unique": col.get("n_unique"),
    }
    for col in summary_train.get("columns", [])
]
{"n_rows": n_rows, "n_columns": len(overview_train), "columns": overview_train}  # noqa: B018

# %% [markdown]
# ## Table overview — test set (X_test)
#
# Same structure as X_train but without the target. Useful to spot any
# distribution shift between train and test.

# %%
report_test = skrub.TableReport(X_test, title="Parkinson's UPDRS — test", verbose=0)
report_test.write_html(EDA_DIR / "eda_test.html")

summary_test = json.loads(report_test.json())
overview_test = [
    {
        "column": col.get("name"),
        "dtype": col.get("dtype"),
        "null_pct": col.get("null_proportion"),
        "n_unique": col.get("n_unique"),
    }
    for col in summary_test.get("columns", [])
]
{"n_rows": summary_test.get("n_rows"), "n_columns": len(overview_test), "columns": overview_test}  # noqa: B018

# %% [markdown]
# ## Target
#
# Distribution of the `target` column (UPDRS score). This shapes the
# metric choice and whether cross-validation should stratify.

# %%
TARGET = "target"
target_col = next(
    (col for col in summary_train.get("columns", []) if col.get("name") == TARGET),
    None,
)
target_col  # noqa: B018

# %% [markdown]
# ## Structure signals
#
# Datetime columns (which point to time-based validation) and
# high-cardinality id / group-like columns (patient_id, rater_id)
# which would leak entities across folds if not handled.

# %%
datetime_cols = [
    col.get("name")
    for col in summary_train.get("columns", [])
    if "date" in str(col.get("dtype", "")).lower()
]
unique_ratio = sorted(
    (
        {
            "column": col.get("name"),
            "unique_ratio": (col.get("n_unique") or 0) / n_rows if n_rows else None,
        }
        for col in summary_train.get("columns", [])
    ),
    key=lambda r: (r["unique_ratio"] is not None, r["unique_ratio"]),
    reverse=True,
)
{"datetime_cols": datetime_cols, "top_unique_ratio": unique_ratio[:10]}  # noqa: B018

# %% [markdown]
# ## Associations
#
# Strongest pairwise column associations in the training set.
# Strong feature↔target links are candidate predictors;
# implausibly perfect ones are possible leakage flags.

# %%
assoc = skrub.column_associations(RAW)
rows = assoc.to_dicts() if hasattr(assoc, "to_dicts") else assoc.to_dict(orient="records")
target_links = [
    row
    for row in rows
    if row["left_column_name"] == TARGET or row["right_column_name"] == TARGET
]
{"with_target": target_links[:15], "strongest": rows[:10]}  # noqa: B018

# %% [markdown]
# ## Summary
#
# The findings and their modelling implications are written up in
# `data/eda.md`.
