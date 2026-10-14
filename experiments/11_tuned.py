# %% [markdown]
# # Experiment: 11_tuned
#
# **Date:** 2026-10-08
# **Goal:** Tune HistGradientBoostingRegressor hyperparameters with grouped CV
# to squeeze out gains on top of the best feature set (patient features +
# categoricals from 09_rater_cohort). Search space: learning_rate ~0.05,
# max_iter up to 1000 with early stopping, max_leaf_nodes 15-31,
# min_samples_leaf 20-50, l2_regularization 0-10.
# Random state 0 throughout for reproducibility.
# Hub key: 11_tuned.

# %%
import numpy as np
import pandas as pd
import skore
from skore import login
from sklearn.model_selection import GroupKFold, RandomizedSearchCV
from scipy.stats import loguniform, randint, uniform

from parkinsons import PROJECT_ROOT
from parkinsons.data import load_dataset, load_groups, load_test_dataset
from parkinsons.hub import load_skore_credentials
from parkinsons.pipeline import build_hgbr_tuned, build_hgbr_with_categoricals

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
# ## Data and groups

# %%
DATA_DIR = PROJECT_ROOT / "data"
X, y = load_dataset(data_dir=DATA_DIR)
visits = X[["patient_id", "age"]].copy()
groups = load_groups(data_dir=DATA_DIR)

# %% [markdown]
# ## Precompute 5-fold GroupKFold splits

# %%
cv_splits = list(GroupKFold(n_splits=5).split(X, y, groups=visits["patient_id"]))

# %% [markdown]
# ## Hyperparameter search
#
# RandomizedSearchCV with the same GroupKFold CV splits. The feature
# engineering step is inside the pipeline so it runs inside every CV fold.

# %%
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from parkinsons.pipeline import _apply_patient_features
from sklearn.ensemble import HistGradientBoostingRegressor

# Build the base pipeline for search
base_pipe = Pipeline(
    steps=[
        (
            "engineer",
            FunctionTransformer(_apply_patient_features, validate=False),
        ),
        (
            "hgbr",
            HistGradientBoostingRegressor(
                early_stopping=True,
                n_iter_no_change=20,
                random_state=0,
                categorical_features="from_dtype",
            ),
        ),
    ]
)

param_dist = {
    "hgbr__learning_rate": [0.02, 0.05, 0.07, 0.1],
    "hgbr__max_iter": [500, 1000],
    "hgbr__max_leaf_nodes": [15, 23, 31],
    "hgbr__min_samples_leaf": [20, 30, 50],
    "hgbr__l2_regularization": [0.0, 0.1, 1.0, 5.0, 10.0],
}

search = RandomizedSearchCV(
    base_pipe,
    param_distributions=param_dist,
    n_iter=30,
    scoring="neg_root_mean_squared_error",
    cv=cv_splits,
    random_state=0,
    n_jobs=-1,
    refit=True,
    verbose=1,
)

search.fit(X, y)

print(f"\nBest params: {search.best_params_}")
print(f"Best CV RMSE: {-search.best_score_:.4f}")

# %% [markdown]
# ## Evaluate best configuration with skore

# %%
best_params = search.best_params_

tuned_pipe = build_hgbr_tuned(
    learning_rate=best_params["hgbr__learning_rate"],
    max_iter=best_params["hgbr__max_iter"],
    max_leaf_nodes=best_params["hgbr__max_leaf_nodes"],
    min_samples_leaf=best_params["hgbr__min_samples_leaf"],
    l2_regularization=best_params["hgbr__l2_regularization"],
    random_state=0,
)

report_11 = skore.evaluate(tuned_pipe, X, y, splitter=cv_splits)

# %% [markdown]
# ## Report: RMSE summary

# %%
rmse_agg = report_11.metrics.rmse()
rmse_mean = rmse_agg.iloc[0, 0]
rmse_std  = rmse_agg.iloc[0, 1]
rmse_per_fold = report_11.metrics.rmse(aggregate=None).iloc[0].values
print(f"11_tuned — RMSE per fold: {rmse_per_fold.round(4).tolist()}")
print(f"  Mean: {rmse_mean:.4f}  Std: {rmse_std:.4f}")

# %% [markdown]
# ## Push to hub

# %%
project.put("11_tuned", report_11)

# %% [markdown]
# ## Kaggle submission

# %%
submissions_dir = PROJECT_ROOT / "submissions"
submissions_dir.mkdir(exist_ok=True)

X_test = load_test_dataset(data_dir=DATA_DIR)

final_11 = build_hgbr_tuned(
    learning_rate=best_params["hgbr__learning_rate"],
    max_iter=best_params["hgbr__max_iter"],
    max_leaf_nodes=best_params["hgbr__max_leaf_nodes"],
    min_samples_leaf=best_params["hgbr__min_samples_leaf"],
    l2_regularization=best_params["hgbr__l2_regularization"],
    random_state=0,
)
final_11.fit(X, y)
preds_11 = final_11.predict(X_test)

submission_11 = pd.DataFrame({"target": preds_11}, index=X_test.index)
submission_11.index.name = "Index"
submission_11.to_csv(submissions_dir / "11_tuned.csv")
print(f"Submission written: {submissions_dir / '11_tuned.csv'}")
