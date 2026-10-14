"""13_tuned_smoothed.py — STEP 0: verify smoothing bug, fix, apply to 11_tuned.

The bug: 10_smoothing.py built the per-patient trend from TRAINING predictions
and looked up test patients by patient_id.  Because train/test patients are
entirely disjoint (0 overlap), no test patient was ever found -> smoothed 0
rows -> submission == 09_rater_cohort.

The fix: smooth within the TEST SET itself.  For every test patient that has
>= MIN_VISITS visits in X_test, fit a linear trend of RAW MODEL PREDICTIONS
against age, then blend:  (1-w)*raw + w*trend.  This is exactly what the OOF
CV loop did (it blended the held-out fold's predictions against each
validation patient's own ages), so the CV estimate of RMSE remains valid.

Hub key: 13_tuned_smoothed
PYTHONUTF8=1 must be set before running.  No non-ASCII characters in prints.
"""

import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GroupKFold

from parkinsons import PROJECT_ROOT
from parkinsons.data import load_dataset, load_test_dataset
from parkinsons.hub import load_skore_credentials
from parkinsons.pipeline import build_hgbr_tuned

DATA_DIR = PROJECT_ROOT / "data"
SUBMISSIONS_DIR = PROJECT_ROOT / "submissions"
CACHE_DIR = PROJECT_ROOT / "scratch" / "13_cache"
SUBMISSIONS_DIR.mkdir(exist_ok=True)
CACHE_DIR.mkdir(parents=True, exist_ok=True)

BEST_W = 0.75
MIN_VISITS = 3
BEST_PARAMS = {
    "learning_rate": 0.02,
    "max_iter": 1000,
    "max_leaf_nodes": 31,
    "min_samples_leaf": 20,
    "l2_regularization": 0.0,
}

# ============================================================
# Smoothing helper (correct version: operates on ANY frame)
# ============================================================

def smooth_within_frame(
    raw_preds: np.ndarray,
    ages: np.ndarray,
    patient_ids: np.ndarray,
    w: float,
    min_visits: int = 3,
) -> np.ndarray:
    """Blend raw predictions with a per-patient linear age trend.

    Trend is fit purely from raw_preds (never target).  Operates on whatever
    frame is passed in -- train fold or test set -- so it is symmetric between
    CV and final prediction.

    Patients with < min_visits rows keep the raw prediction.
    """
    if w == 0.0:
        return raw_preds.copy()
    smoothed = raw_preds.copy().astype(float)
    df = pd.DataFrame({"pred": raw_preds, "age": ages, "pid": patient_ids})
    for pid, grp in df.groupby("pid"):
        if len(grp) < min_visits:
            continue
        lr = LinearRegression()
        lr.fit(grp[["age"]], grp["pred"])
        trend = lr.predict(grp[["age"]])
        smoothed[grp.index] = (1 - w) * grp["pred"].values + w * trend
    return smoothed


# ============================================================
# STEP 0a — verify the bug
# ============================================================
print("=" * 60)
print("STEP 0a -- VERIFY SMOOTHING BUG")
print("=" * 60)

sub09 = pd.read_csv(SUBMISSIONS_DIR / "09_rater_cohort.csv", index_col="Index")
sub10 = pd.read_csv(SUBMISSIONS_DIR / "10_smoothing.csv", index_col="Index")
sub11 = pd.read_csv(SUBMISSIONS_DIR / "11_tuned.csv", index_col="Index")

diff_09_10 = (sub09["target"] - sub10["target"]).abs()
corr_09_10 = sub09["target"].corr(sub10["target"])
print(f"09 vs 10 -- max abs diff: {diff_09_10.max():.8f}")
print(f"09 vs 10 -- correlation : {corr_09_10:.8f}")
print(f"09 vs 10 -- are identical: {(diff_09_10 == 0).all()}")

diff_09_11 = (sub09["target"] - sub11["target"]).abs()
print(f"\n09 vs 11 -- max abs diff: {diff_09_11.max():.4f}  (should be large -> different model)")

print("\n=> BUG CONFIRMED: 10_smoothing.csv == 09_rater_cohort.csv")
print("   Root cause: smooth_within_frame looked up test patients by patient_id")
print("   in the TRAINING frame, but train/test patients are 100% disjoint.")
print("   Fix: smooth within the TEST frame itself (same patient, multiple test visits).")

# ============================================================
# STEP 0b — load data
# ============================================================
print("\n" + "=" * 60)
print("STEP 0b -- LOAD DATA + COMPUTE OOF SPLITS")
print("=" * 60)

t0 = time.time()
X, y = load_dataset(data_dir=DATA_DIR)
X_test = load_test_dataset(data_dir=DATA_DIR)

visits_train = X[["patient_id", "age"]].copy()
visits_test  = X_test[["patient_id", "age"]].copy()

groups = X["patient_id"]
cv_splits = list(GroupKFold(n_splits=5).split(X, y, groups=groups))

print(f"  Train: {len(X)} rows, {X['patient_id'].nunique()} patients")
print(f"  Test : {len(X_test)} rows, {X_test['patient_id'].nunique()} patients")
print(f"  Test patients with >= {MIN_VISITS} visits (eligible for smoothing): "
      f"{(X_test.groupby('patient_id').size() >= MIN_VISITS).sum()}")
print(f"  Loaded in {time.time()-t0:.1f}s")

# ============================================================
# STEP 0c — verify CV RMSE of smoothed 11_tuned (w=0.75)
# ============================================================
print("\n" + "=" * 60)
print("STEP 0c -- VERIFY CV RMSE FOR SMOOTHED 11_TUNED (w=0.75)")
print("=" * 60)

OOF_CACHE = CACHE_DIR / "11_oof_raw.npy"

if OOF_CACHE.exists():
    print("  Loading cached OOF predictions...")
    oof_raw = np.load(OOF_CACHE)
else:
    print("  Computing OOF predictions for 11_tuned (5 folds)...")
    oof_raw = np.full(len(X), np.nan)
    for fold_i, (tr_idx, val_idx) in enumerate(cv_splits):
        t_fold = time.time()
        m = build_hgbr_tuned(**BEST_PARAMS, random_state=0)
        m.fit(X.iloc[tr_idx], y.iloc[tr_idx])
        oof_raw[val_idx] = m.predict(X.iloc[val_idx])
        print(f"    fold {fold_i}: {time.time()-t_fold:.1f}s")
    np.save(OOF_CACHE, oof_raw)
    print("  OOF predictions cached.")

# Evaluate smoothed at several w values using the correct (within-fold) approach
print("\n  w     raw RMSE  smoothed RMSE  delta")
best_w_cv = 0.0
best_rmse_cv = 1e9
for w_cand in [0.0, 0.25, 0.5, 0.75, 1.0]:
    fold_rmses = []
    for tr_idx, val_idx in cv_splits:
        raw_fold = oof_raw[val_idx]
        ages_fold = visits_train["age"].iloc[val_idx].values
        pids_fold = visits_train["patient_id"].iloc[val_idx].values
        blended = smooth_within_frame(raw_fold, ages_fold, pids_fold, w=w_cand, min_visits=MIN_VISITS)
        rmse = np.sqrt(np.mean((blended - y.values[val_idx]) ** 2))
        fold_rmses.append(rmse)
    mean_r = np.mean(fold_rmses)
    std_r  = np.std(fold_rmses)
    raw_overall = np.sqrt(np.mean((oof_raw - y.values) ** 2))
    print(f"  {w_cand:.2f}  {raw_overall:.4f}   {mean_r:.4f} +/- {std_r:.4f}")
    if mean_r < best_rmse_cv:
        best_rmse_cv = mean_r
        best_w_cv = w_cand

print(f"\n  Best w = {best_w_cv}  CV RMSE = {best_rmse_cv:.4f}")

# ============================================================
# STEP 0d — fit final model, smooth within test, write CSV
# ============================================================
print("\n" + "=" * 60)
print("STEP 0d -- FIT FINAL MODEL + SMOOTH TEST PREDICTIONS")
print("=" * 60)

FINAL_MODEL_CACHE = CACHE_DIR / "11_tuned_final.joblib"
FINAL_RAW_TEST_CACHE = CACHE_DIR / "11_tuned_raw_test.npy"

if FINAL_MODEL_CACHE.exists() and FINAL_RAW_TEST_CACHE.exists():
    print("  Loading cached final model + raw test predictions...")
    final_model = joblib.load(FINAL_MODEL_CACHE)
    raw_test_preds = np.load(FINAL_RAW_TEST_CACHE)
    print("  [cache hit]")
else:
    print("  Fitting 11_tuned on all training data...")
    t1 = time.time()
    final_model = build_hgbr_tuned(**BEST_PARAMS, random_state=0)
    final_model.fit(X, y)
    raw_test_preds = final_model.predict(X_test)
    joblib.dump(final_model, FINAL_MODEL_CACHE)
    np.save(FINAL_RAW_TEST_CACHE, raw_test_preds)
    print(f"  Done in {time.time()-t1:.1f}s  [model cached]")

# Apply corrected smoothing: within X_test
ages_test = visits_test["age"].values
pids_test  = visits_test["patient_id"].values

smoothed_test = smooth_within_frame(
    raw_test_preds, ages_test, pids_test, w=best_w_cv, min_visits=MIN_VISITS
)

n_smoothed = int((smoothed_test != raw_test_preds).sum())
print(f"  Test rows smoothed: {n_smoothed} / {len(X_test)}")
print(f"  Test rows kept raw: {len(X_test) - n_smoothed} "
      f"(patients with < {MIN_VISITS} test visits)")

# ============================================================
# Sanity checks
# ============================================================
# Clip to valid UPDRS range [0, 132] -- smoothing can push slightly negative
n_clipped_low  = int((smoothed_test < 0).sum())
n_clipped_high = int((smoothed_test > 132).sum())
if n_clipped_low or n_clipped_high:
    print(f"  Clipping {n_clipped_low} rows below 0, {n_clipped_high} rows above 132")
smoothed_test = np.clip(smoothed_test, 0, 132)

sub = pd.DataFrame({"target": smoothed_test}, index=X_test.index)
sub.index.name = "Index"
assert len(sub) == len(X_test), "Row count mismatch"
assert sub["target"].isna().sum() == 0, "NaN predictions"
pmin, pmax = sub["target"].min(), sub["target"].max()

out_path = SUBMISSIONS_DIR / "13_tuned_smoothed.csv"
sub.to_csv(out_path)

print(f"\n  [OK] rows: {len(sub)}")
print(f"  [OK] NaN : 0")
print(f"  [OK] range: [{pmin:.2f}, {pmax:.2f}]")
print(f"  Written: {out_path}")

# Quick comparison with 09, 10, 11
diff_vs_11 = (sub["target"] - sub11["target"]).abs()
diff_vs_10 = (sub["target"] - sub10["target"]).abs()
print(f"\n  13 vs 11 -- max abs diff: {diff_vs_11.max():.4f}  mean: {diff_vs_11.mean():.4f}")
print(f"  13 vs 10 -- max abs diff: {diff_vs_10.max():.4f}  mean: {diff_vs_10.mean():.4f}")

print(f"\nCV RMSE summary:")
print(f"  09_rater_cohort (base)   : ~4.73 +/- 0.15")
print(f"  11_tuned (raw, no smooth): ~4.65 +/- 0.15")
print(f"  13_tuned_smoothed (w={best_w_cv:.2f}): {best_rmse_cv:.4f}")
print(f"\nSTEP 0 DONE. File: submissions/13_tuned_smoothed.csv")
print("Do NOT submit to Kaggle yourself -- hand the file to the user.")
