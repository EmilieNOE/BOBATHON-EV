# %% [markdown]
# # Experiment: 10_smoothing
#
# **Date:** 2026-10-08
# **Goal:** Post-process out-of-fold predictions by fitting a per-patient linear
# age trend and blending it with the raw prediction. TRUE OFF progresses slowly
# over years, so a linear trend within a patient should reduce random noise.
# The blend weight w is chosen by grouped CV (trying 0, 0.25, 0.5, 0.75, 1).
# Patients with < 3 visits keep the raw prediction (too few points for a trend).
# Smoothing is computed only from predictions — target is never read.
# Hub key: 10_smoothing.

# %%
import numpy as np
import pandas as pd
import skore
from skore import login
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GroupKFold

from parkinsons import PROJECT_ROOT
from parkinsons.data import load_dataset, load_groups, load_test_dataset
from parkinsons.hub import load_skore_credentials
from parkinsons.pipeline import build_hgbr_with_categoricals

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
# ## Build 5-fold GroupKFold splits

# %%
cv_splits = list(GroupKFold(n_splits=5).split(X, y, groups=visits["patient_id"]))


# %% [markdown]
# ## Smoothing helper: blend raw OOF predictions with a per-patient linear trend

# %%
def smooth_predictions(
    raw_preds: np.ndarray,
    ages: np.ndarray,
    patient_ids: np.ndarray,
    w: float,
    min_visits: int = 3,
) -> np.ndarray:
    """Blend raw predictions with a per-patient linear age trend.

    For each patient that has at least ``min_visits`` rows, we fit a OLS line
    ``prediction ~ age`` using only the raw predictions (never target), then
    return ``(1 - w) * raw + w * trend``. Patients with fewer rows keep the
    raw prediction unchanged.

    Parameters
    ----------
    raw_preds : array of shape (n,)
        Out-of-fold predictions from the HGBR.
    ages : array of shape (n,)
        Age of the patient at each row (same order as raw_preds).
    patient_ids : array of shape (n,)
        Patient identifier for each row (same order).
    w : float
        Blend weight in [0, 1]. ``w=0`` → pure raw; ``w=1`` → pure trend.
    min_visits : int
        Minimum visits before smoothing is applied.

    Returns
    -------
    np.ndarray of shape (n,)
        Blended predictions.
    """
    if w == 0.0:
        return raw_preds.copy()

    smoothed = raw_preds.copy()
    df = pd.DataFrame({"pred": raw_preds, "age": ages, "pid": patient_ids})

    for pid, grp in df.groupby("pid"):
        if len(grp) < min_visits:
            continue
        lr = LinearRegression()
        lr.fit(grp[["age"]], grp["pred"])
        trend = lr.predict(grp[["age"]])
        smoothed[grp.index] = (1 - w) * grp["pred"].values + w * trend

    return smoothed


# %% [markdown]
# ## Grouped CV to pick the best blend weight w
#
# We collect out-of-fold HGBR predictions on each fold, apply the smoother at
# five candidate weights, and report the fold-wise RMSE for each.

# %%
learner_09 = build_hgbr_with_categoricals()
W_CANDIDATES = [0.0, 0.25, 0.5, 0.75, 1.0]
fold_rmses = {w: [] for w in W_CANDIDATES}

for fold_idx, (train_idx, val_idx) in enumerate(cv_splits):
    X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_tr = y.iloc[train_idx]

    model = build_hgbr_with_categoricals()
    model.fit(X_tr, y_tr)
    raw = model.predict(X_val)

    ages_val = visits["age"].iloc[val_idx].values
    pids_val = visits["patient_id"].iloc[val_idx].values
    y_val = y.iloc[val_idx].values

    for w in W_CANDIDATES:
        blended = smooth_predictions(raw, ages_val, pids_val, w)
        rmse = np.sqrt(np.mean((blended - y_val) ** 2))
        fold_rmses[w].append(rmse)

print("\nSmoothing CV results:")
print(f"{'w':>5}  {'mean RMSE':>10}  {'std RMSE':>10}")
for w in W_CANDIDATES:
    arr = np.array(fold_rmses[w])
    print(f"{w:>5.2f}  {arr.mean():>10.4f}  {arr.std():>10.4f}")

best_w = min(W_CANDIDATES, key=lambda w: np.mean(fold_rmses[w]))
best_mean = np.mean(fold_rmses[best_w])
best_std = np.std(fold_rmses[best_w])
print(f"\nBest w = {best_w}  ->  RMSE {best_mean:.4f} +/- {best_std:.4f}")

# %% [markdown]
# ## Build a skore report for the best smoothing weight
#
# We use skore.evaluate on the base model (w=0 case already equals 09_rater_cohort)
# and then report the smoothing improvement inline.

# %%
# For hub reporting use the base model report (smoothing is a post-processor).
report_10_base = skore.evaluate(build_hgbr_with_categoricals(), X, y, splitter=cv_splits)

# %% [markdown]
# ## Push to hub

# %%
project.put("10_smoothing", report_10_base)
print(
    f"10_smoothing hub report pushed (base HGBR; best smoothing w={best_w}, "
    f"CV RMSE {best_mean:.4f} ± {best_std:.4f})"
)

# %% [markdown]
# ## Kaggle submission — refit on full training set, apply best smoothing

# %%
submissions_dir = PROJECT_ROOT / "submissions"
submissions_dir.mkdir(exist_ok=True)

X_test = load_test_dataset(data_dir=DATA_DIR)

final_10 = build_hgbr_with_categoricals()
final_10.fit(X, y)
raw_test = final_10.predict(X_test)

# Smooth test predictions using the full-train linear trends per patient.
# We compute per-patient trends from the training raw predictions first.
final_train_raw = final_10.predict(X)
ages_train = visits["age"].values
pids_train = visits["patient_id"].values

ages_test = X_test["age"].values
pids_test = X_test["patient_id"].values

if best_w > 0.0:
    # Build trend lines from training predictions per patient, then apply to test.
    df_train = pd.DataFrame(
        {"pred": final_train_raw, "age": ages_train, "pid": pids_train}
    )
    trends_test = raw_test.copy()
    for pid, grp in df_train.groupby("pid"):
        mask_test = pids_test == pid
        if mask_test.sum() == 0:
            continue
        if len(grp) < 3:
            continue
        lr = LinearRegression()
        lr.fit(grp[["age"]], grp["pred"])
        trends_test[mask_test] = (
            (1 - best_w) * raw_test[mask_test]
            + best_w * lr.predict(pd.DataFrame({"age": ages_test[mask_test]}))
        )
    preds_10 = trends_test
else:
    preds_10 = raw_test

submission_10 = pd.DataFrame({"target": preds_10}, index=X_test.index)
submission_10.index.name = "Index"
submission_10.to_csv(submissions_dir / "10_smoothing.csv")
print(f"Submission written: {submissions_dir / '10_smoothing.csv'}")
