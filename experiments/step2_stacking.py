"""step2_stacking.py -- STEP 2: patient-level stacking.
Hub key: 15_stacked.

Stage 1: 11_tuned OOF predictions (cached).
Stage 2: small HGBR using original input features PLUS per-patient aggregates
         of stage-1 predictions computed within each frame:
           - s1_mean, s1_median, s1_min, s1_max  (patient-level)
           - s1_quad_val  (quadratic trend value at current age)
           - s1_quad_resid (raw pred - quad trend)
           - s1_prev, s1_next  (previous/next visit prediction, sorted by age)
           - s1_n_visits
           - off_present_frac  (fraction of patient visits with off present)
           - years_since_first_visit
           - patient_mean_on, patient_mean_off  (from input columns)

Stage-2 CV is strictly nested: stage-2 only ever sees stage-1 OOF predictions
for the validation fold (never the final-model predictions).

Also reports RMSE on the "off missing" subgroup (systematic under-prediction
from Step 1 analysis) before and after stacking.

PYTHONUTF8=1. ASCII only. Cache intermediate results.
"""

import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import skore
from skore import login
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GroupKFold

from parkinsons import PROJECT_ROOT
from parkinsons.data import load_dataset, load_test_dataset
from parkinsons.hub import load_skore_credentials
from parkinsons.pipeline import build_hgbr_tuned

DATA_DIR        = PROJECT_ROOT / "data"
SUBMISSIONS_DIR = PROJECT_ROOT / "submissions"
CACHE_DIR_11    = PROJECT_ROOT / "scratch" / "13_cache"
CACHE_DIR       = PROJECT_ROOT / "scratch" / "15_cache"
SUBMISSIONS_DIR.mkdir(exist_ok=True)
CACHE_DIR.mkdir(parents=True, exist_ok=True)

BEST_PARAMS = dict(learning_rate=0.02, max_iter=1000, max_leaf_nodes=31,
                   min_samples_leaf=20, l2_regularization=0.0)

BEST_W    = 0.75   # from step 0 (13_tuned_smoothed)
PREV_BEST = 4.3975

# ============================================================
# Load data + cached OOF
# ============================================================
t0 = time.time()
X, y   = load_dataset(data_dir=DATA_DIR)
X_test = load_test_dataset(data_dir=DATA_DIR)
groups = X["patient_id"]
cv_splits = list(GroupKFold(n_splits=5).split(X, y, groups=groups))

oof_11 = np.load(CACHE_DIR_11 / "11_oof_raw.npy")
print(f"Data loaded in {time.time()-t0:.1f}s")
print(f"11_tuned raw OOF RMSE: {np.sqrt(np.mean((oof_11 - y.values)**2)):.4f}")

# ============================================================
# Feature builder: add stage-1 aggregates
# ============================================================

def _quad_trend(ages_c, preds, ridge_lam=1.0):
    """Fit ridge quadratic preds ~ centered_age, return trend values."""
    n = len(ages_c)
    if n < 3:
        return np.full(n, np.mean(preds))
    Phi = np.column_stack([ages_c, ages_c**2])
    intercept = np.mean(preds)
    y_c = preds - intercept
    A = Phi.T @ Phi + ridge_lam * np.eye(2)
    try:
        coef = np.linalg.solve(A, Phi.T @ y_c)
    except np.linalg.LinAlgError:
        coef = np.zeros(2)
    return Phi @ coef + intercept


def add_stage1_features(X_frame: pd.DataFrame, s1_preds: np.ndarray) -> pd.DataFrame:
    """Add per-patient stage-1 aggregates to X_frame.

    s1_preds must be aligned to X_frame.index (same order).
    """
    X_out = X_frame.copy()
    X_out["_s1"] = s1_preds

    # Categorical encoding
    for col in ("cohort", "gene", "rater_id"):
        if col in X_out.columns:
            X_out[col] = X_out[col].astype("category")

    # Sort per patient by age for lag/lead
    original_idx = X_out.index
    X_out = X_out.sort_values(["patient_id", "age"])

    s1_col = X_out["_s1"].values.astype(float)
    ages   = X_out["age"].values
    pids   = X_out["patient_id"].values

    # Allocate arrays
    n = len(X_out)
    s1_mean   = np.full(n, np.nan)
    s1_median = np.full(n, np.nan)
    s1_min    = np.full(n, np.nan)
    s1_max    = np.full(n, np.nan)
    s1_quad   = np.full(n, np.nan)
    s1_qresid = np.full(n, np.nan)
    s1_prev   = np.full(n, np.nan)
    s1_next   = np.full(n, np.nan)
    s1_nvis   = np.full(n, np.nan)
    off_frac  = np.full(n, np.nan)
    yrs_first = np.full(n, np.nan)
    pat_mean_on  = np.full(n, np.nan)
    pat_mean_off = np.full(n, np.nan)

    for pid, grp in X_out.groupby("patient_id"):
        idx_pos = [X_out.index.get_loc(i) for i in grp.index]
        preds_g = s1_col[idx_pos]
        ages_g  = ages[idx_pos]
        ages_c  = ages_g - ages_g.mean()
        nv = len(grp)

        s1_mean[idx_pos]   = np.mean(preds_g)
        s1_median[idx_pos] = np.median(preds_g)
        s1_min[idx_pos]    = np.min(preds_g)
        s1_max[idx_pos]    = np.max(preds_g)
        s1_nvis[idx_pos]   = nv

        trend = _quad_trend(ages_c, preds_g)
        s1_quad[idx_pos]   = trend
        s1_qresid[idx_pos] = preds_g - trend

        # prev/next within patient (already sorted by age)
        for k, pos in enumerate(idx_pos):
            s1_prev[pos] = preds_g[k-1] if k > 0     else np.nan
            s1_next[pos] = preds_g[k+1] if k < nv-1  else np.nan

        off_frac[idx_pos]  = grp["off"].notna().mean()
        yrs_first[idx_pos] = ages_g - ages_g.min()
        pat_mean_on[idx_pos]  = grp["on"].mean()
        pat_mean_off[idx_pos] = grp["off"].mean()

    X_out["s1_mean"]      = s1_mean
    X_out["s1_median"]    = s1_median
    X_out["s1_min"]       = s1_min
    X_out["s1_max"]       = s1_max
    X_out["s1_quad"]      = s1_quad
    X_out["s1_qresid"]    = s1_qresid
    X_out["s1_prev"]      = s1_prev
    X_out["s1_next"]      = s1_next
    X_out["s1_nvis"]      = s1_nvis
    X_out["off_frac"]     = off_frac
    X_out["yrs_first"]    = yrs_first
    X_out["pat_mean_on"]  = pat_mean_on
    X_out["pat_mean_off"] = pat_mean_off

    # Also add original patient features (prev/next of on/off/ledd)
    for col in ("on", "off", "ledd"):
        if col not in X_out.columns:
            continue
        grp_col = X_out.groupby("patient_id")[col]
        X_out[f"{col}_prev_s"] = grp_col.shift(1)
        X_out[f"{col}_next_s"] = grp_col.shift(-1)
        X_out[f"{col}_pmean"]  = grp_col.transform("mean")

    # Drop _s1 helper and patient_id
    X_out = X_out.drop(columns=["_s1", "patient_id"])

    # Restore original row order
    X_out = X_out.loc[original_idx]
    return X_out


def hgbr_stage2():
    return HistGradientBoostingRegressor(
        categorical_features="from_dtype",
        early_stopping=True,
        n_iter_no_change=20,
        learning_rate=0.05,
        max_iter=500,
        max_leaf_nodes=31,
        min_samples_leaf=20,
        random_state=0,
    )


# ============================================================
# Nested CV: stage-2 only sees stage-1 OOF predictions
# ============================================================
OOF_S2_CACHE = CACHE_DIR / "15_oof_s2.npy"

if OOF_S2_CACHE.exists():
    print("Loading cached stage-2 OOF...")
    oof_s2 = np.load(OOF_S2_CACHE)
else:
    print("Computing stage-2 OOF (5 folds)...")
    oof_s2 = np.full(len(X), np.nan)
    for fi, (tr_idx, val_idx) in enumerate(cv_splits):
        t_f = time.time()
        # Stage-2 train: original features + stage-1 OOF (from the 4 other folds)
        X_tr_s2 = add_stage1_features(X.iloc[tr_idx].copy(), oof_11[tr_idx])
        X_val_s2 = add_stage1_features(X.iloc[val_idx].copy(), oof_11[val_idx])
        m2 = hgbr_stage2()
        m2.fit(X_tr_s2, y.iloc[tr_idx])
        oof_s2[val_idx] = m2.predict(X_val_s2)
        rmse_f = np.sqrt(np.mean((oof_s2[val_idx] - y.values[val_idx])**2))
        print(f"  fold {fi}: s2 raw RMSE={rmse_f:.4f}  ({time.time()-t_f:.1f}s)")
    np.save(OOF_S2_CACHE, oof_s2)
    print("  Cached.")

rmse_s2_raw = np.sqrt(np.mean((oof_s2 - y.values)**2))
print(f"\nStage-2 raw OOF RMSE: {rmse_s2_raw:.4f}")

# Smoothing sweep on stage-2
def smooth_within_frame(raw_preds, ages, pids, w, min_visits=3):
    if w == 0.0:
        return raw_preds.copy()
    sm = raw_preds.copy().astype(float)
    df_s = pd.DataFrame({"pred": raw_preds, "age": ages, "pid": pids})
    for pid, grp in df_s.groupby("pid"):
        if len(grp) < min_visits:
            continue
        lr = LinearRegression()
        lr.fit(grp[["age"]], grp["pred"])
        sm[grp.index] = (1-w)*grp["pred"].values + w*lr.predict(grp[["age"]])
    return sm

print("\nSmoothing sweep on stage-2:")
print(f"  {'w':>5}  {'RMSE':>8}  {'std':>8}")
best_w_s2   = 0.0
best_rmse_s2 = rmse_s2_raw
fold_rmses_best = []
ages_a = X["age"].values
pids_a = X["patient_id"].values
for w_cand in [0.0, 0.5, 0.75, 0.9, 1.0]:
    fold_rmses = []
    for tr_idx, val_idx in cv_splits:
        bl = smooth_within_frame(oof_s2[val_idx], ages_a[val_idx], pids_a[val_idx], w_cand)
        fold_rmses.append(np.sqrt(np.mean((bl - y.values[val_idx])**2)))
    m, s = np.mean(fold_rmses), np.std(fold_rmses)
    print(f"  {w_cand:>5.2f}  {m:>8.4f}  {s:>8.4f}")
    if m < best_rmse_s2:
        best_rmse_s2 = m
        best_w_s2    = w_cand
        fold_rmses_best = fold_rmses

# ============================================================
# Subgroup analysis: off missing before vs after
# ============================================================
off_miss_mask = X["off"].isna().values
rmse_off_miss_before = np.sqrt(np.mean((
    smooth_within_frame(oof_11, ages_a, pids_a, BEST_W)[off_miss_mask]
    - y.values[off_miss_mask])**2))
oof_s2_smoothed = smooth_within_frame(oof_s2, ages_a, pids_a, best_w_s2)
rmse_off_miss_after = np.sqrt(np.mean((oof_s2_smoothed[off_miss_mask] - y.values[off_miss_mask])**2))
print(f"\nRMSE on 'off missing' subgroup:")
print(f"  Before (13_tuned_smoothed): {rmse_off_miss_before:.4f}")
print(f"  After  (15_stacked):        {rmse_off_miss_after:.4f}")

# ============================================================
# Comparison
# ============================================================
print(f"\nComparison (smoothed OOF RMSE):")
print(f"  13_tuned_smoothed (prev best): {PREV_BEST:.4f}  fold_std=0.139")
print(f"  15_stacked (best w={best_w_s2}):          {best_rmse_s2:.4f}")
delta = PREV_BEST - best_rmse_s2
print(f"  Delta: {delta:.4f}")
if delta > 0.139:
    print("  => KEEP: gain exceeds fold std")
elif delta > 0:
    print("  => MARGINAL: gain < fold std")
else:
    print("  => NO IMPROVEMENT -- keep 13_tuned_smoothed as best")

# ============================================================
# Final model + submission
# ============================================================
STAGE1_FINAL_CACHE   = CACHE_DIR_11 / "11_tuned_final.joblib"
STAGE1_RAW_TEST_CACHE = CACHE_DIR_11 / "11_tuned_raw_test.npy"
STAGE2_MODEL_CACHE   = CACHE_DIR / "15_stage2_final.joblib"
STAGE2_TEST_CACHE    = CACHE_DIR / "15_raw_test.npy"

if STAGE2_MODEL_CACHE.exists() and STAGE2_TEST_CACHE.exists():
    print("\nLoading cached stage-2 final model...")
    m2_final    = joblib.load(STAGE2_MODEL_CACHE)
    raw_test_s2 = np.load(STAGE2_TEST_CACHE)
else:
    print("\nFitting stage-1 final model...")
    t1 = time.time()
    if STAGE1_FINAL_CACHE.exists() and STAGE1_RAW_TEST_CACHE.exists():
        s1_final   = joblib.load(STAGE1_FINAL_CACHE)
        raw_test_s1 = np.load(STAGE1_RAW_TEST_CACHE)
        print(f"  Stage-1 loaded from cache  ({time.time()-t1:.1f}s)")
    else:
        s1_final = build_hgbr_tuned(**BEST_PARAMS, random_state=0)
        s1_final.fit(X, y)
        raw_test_s1 = s1_final.predict(X_test)
        joblib.dump(s1_final, STAGE1_FINAL_CACHE)
        np.save(STAGE1_RAW_TEST_CACHE, raw_test_s1)
        print(f"  Stage-1 fit in {time.time()-t1:.1f}s")

    # Stage-1 OOF on all train (== oof_11)
    print("  Fitting stage-2 final model on all train...")
    t2 = time.time()
    X_train_s2 = add_stage1_features(X.copy(), oof_11)
    m2_final   = hgbr_stage2()
    m2_final.fit(X_train_s2, y)
    X_test_s2   = add_stage1_features(X_test.copy(), raw_test_s1)
    raw_test_s2 = m2_final.predict(X_test_s2)
    joblib.dump(m2_final, STAGE2_MODEL_CACHE)
    np.save(STAGE2_TEST_CACHE, raw_test_s2)
    print(f"  Stage-2 fit in {time.time()-t2:.1f}s  [cached]")

smoothed_test = smooth_within_frame(
    raw_test_s2,
    X_test["age"].values,
    X_test["patient_id"].values,
    w=best_w_s2,
)
n_clip_lo = int((smoothed_test < 0).sum())
n_clip_hi = int((smoothed_test > 132).sum())
if n_clip_lo or n_clip_hi:
    print(f"Clipping {n_clip_lo} below 0, {n_clip_hi} above 132")
smoothed_test = np.clip(smoothed_test, 0, 132)

sub = pd.DataFrame({"target": smoothed_test}, index=X_test.index)
sub.index.name = "Index"
assert len(sub) == len(X_test)
assert sub["target"].isna().sum() == 0
pmin, pmax = sub["target"].min(), sub["target"].max()
out_path = SUBMISSIONS_DIR / "15_stacked.csv"
sub.to_csv(out_path)
print(f"\n[OK] rows={len(sub)}  NaN=0  range=[{pmin:.2f},{pmax:.2f}]")
print(f"Written: {out_path}")

# ============================================================
# Hub report
# ============================================================
print("\nPushing hub report (stage-2 base model)...")
try:
    cfg = load_skore_credentials()
    login(mode="hub")
    project = skore.Project(name="bobathon-esilv", mode="hub", workspace=cfg["workspace"])

    # Use stage-2 evaluated directly via skore (on the full s2 feature set)
    # We pass cv_splits as the splitter; stage-2 features are computed inline
    # Wrap in a simple pipeline that adds features at fit/transform time
    from sklearn.base import BaseEstimator, RegressorMixin

    class Stage2Wrapper(BaseEstimator, RegressorMixin):
        """Wraps the two-stage pipeline for skore.evaluate."""
        def __init__(self):
            self.s1_   = None
            self.s2_   = None
        def fit(self, X, y):
            s1 = build_hgbr_tuned(**BEST_PARAMS, random_state=0)
            s1.fit(X, y)
            self.s1_ = s1
            X_s2 = add_stage1_features(X.copy(), s1.predict(X))
            s2 = hgbr_stage2()
            s2.fit(X_s2, y)
            self.s2_ = s2
            return self
        def predict(self, X):
            s1_preds = self.s1_.predict(X)
            X_s2 = add_stage1_features(X.copy(), s1_preds)
            return self.s2_.predict(X_s2)

    report_15 = skore.evaluate(Stage2Wrapper(), X, y, splitter=cv_splits)
    rmse_agg  = report_15.metrics.rmse()
    print(f"  skore RMSE (non-nested, informative): {rmse_agg.iloc[0,0]:.4f} +/- {rmse_agg.iloc[0,1]:.4f}")
    project.put("15_stacked", report_15)
    print("  Hub report 15_stacked pushed.")
    workspace = cfg["workspace"]
except Exception as e:
    print(f"  Hub put failed (non-fatal): {e}")
    workspace = "evdgreenteam"

hub_url = f"https://skore.probabl.ai/{workspace}/bobathon-esilv/cross-validations"
print(f"\nHub key : 15_stacked")
print(f"Hub URL : {hub_url}")
print(f"Best stacking CV RMSE (nested, smoothed): {best_rmse_s2:.4f}")
print(f"\nSTEP 2 DONE.")
