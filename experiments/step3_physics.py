"""step3_physics.py -- STEP 3: physics-based features + key 13_physics.

Structural insights from STEP 2:
  1. target ~ a*age^2 + b*age + c within each patient (median R2=0.9994).
     => fit per-patient quadratic from OTHER visits in the same CV fold,
        extrapolate to the held-out visit.  For test: fit from test visits.
  2. Rater bias is additive ~9% of variance; estimate per-rater offset from
     the TRAINING fold only (never leaking the held-out fold).
  3. off_missing is correlated (r=0.36) with target -- add as a flag.
  4. PK correction negligible (R2=0.003) -- skip.

New features (all fold-safe, computed inside CV):
  quad_pred       -- per-patient quadratic prediction at this age
                     (fit from other visits in the same patient, same fold)
  quad_residual   -- off_or_on_adj - quad_pred (deviation from patient trend)
  rater_offset    -- per-rater mean of (off - on_adj) computed on train fold only
  corrected_off   -- off - rater_offset  (when off present)
  off_missing_flag-- 1 if off is NaN
  on_adj          -- on + on_avg_offset (global on->target correction)
  target_proxy    -- best available estimate of target:
                     if off present: off - rater_offset
                     else:           on + on_offset

These are computed by a FunctionTransformer that receives the full X frame
(including patient_id, rater_id, age, on, off) and returns the enriched frame.
The transformer is STATEFUL (fit/transform split) so it is wrapped in a
custom sklearn BaseEstimator to be CV-safe.

Hub key: 13_physics
PYTHONUTF8=1 required.  No non-ASCII characters.
"""

import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import skore
from skore import login
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline

from parkinsons import PROJECT_ROOT
from parkinsons.data import load_dataset, load_test_dataset
from parkinsons.hub import load_skore_credentials

DATA_DIR      = PROJECT_ROOT / "data"
SUBMISSIONS_DIR = PROJECT_ROOT / "submissions"
CACHE_DIR     = PROJECT_ROOT / "scratch" / "13_physics_cache"
SUBMISSIONS_DIR.mkdir(exist_ok=True)
CACHE_DIR.mkdir(parents=True, exist_ok=True)

BEST_PARAMS = {
    "learning_rate": 0.02,
    "max_iter": 1000,
    "max_leaf_nodes": 31,
    "min_samples_leaf": 20,
    "l2_regularization": 0.0,
}
BEST_W    = 0.75
MIN_VISITS = 3

# ============================================================
# Physics feature transformer (CV-safe, stateful)
# ============================================================

CAT_COLS = ("cohort", "gene", "rater_id")
DROP_COLS = ("patient_id",)


class PhysicsFeatures(BaseEstimator, TransformerMixin):
    """Fit per-patient quadratic trends and per-rater offsets from training data,
    then apply to any frame (train or test).

    Per-patient quadratic (age -> target_proxy):
        For each patient seen in fit(), stores coefficients of
        target_proxy ~ age + age^2 fitted on all OTHER visits of that patient
        (leave-one-out within the training set).  At transform time the stored
        polynomial is evaluated at the query age.
        For patients NOT seen in fit() (cold-start test patients), a new
        quadratic is fitted from their own visits in the transform frame
        (safe: no target used, only predictions-from-other-features would be
        needed, but we use target_proxy = off if available else on+offset).

    Per-rater offset:
        Mean of (off - target_proxy) over the training frame, per rater.
        Stored as a dict {rater_id: offset}.  At transform, subtract from off.
        Raters not seen in fit get the global mean offset.
    """

    def __init__(self, min_visits_quad: int = 3):
        self.min_visits_quad = min_visits_quad

    # ------------------------------------------------------------------
    def _make_proxy(self, X: pd.DataFrame) -> pd.Series:
        """Target proxy = off when present, else on + global_on_offset."""
        proxy = X["off"].copy()
        on_offset = self.on_offset_ if hasattr(self, "on_offset_") else 22.0
        missing = proxy.isna()
        proxy.loc[missing] = X.loc[missing, "on"] + on_offset
        # If both missing, use a fixed fallback based on ledd (crude)
        still_missing = proxy.isna()
        proxy.loc[still_missing] = 40.0
        return proxy

    # ------------------------------------------------------------------
    def fit(self, X: pd.DataFrame, y=None):
        X = X.copy()
        # 1. Global on->target offset (mean of off - on where both present)
        both = X["off"].notna() & X["on"].notna()
        self.on_offset_ = float((X.loc[both, "off"] - X.loc[both, "on"]).mean())

        # 2. Per-rater offset from off - on_adj (proxy for off - target)
        proxy = self._make_proxy(X)
        rater_group = X.groupby("rater_id")
        self.rater_offset_ = {}
        for rid, grp in rater_group:
            off_vals   = grp["off"].dropna()
            proxy_vals = proxy.loc[off_vals.index]
            if len(off_vals) >= 5:
                self.rater_offset_[rid] = float((off_vals - proxy_vals).mean())
            else:
                self.rater_offset_[rid] = 0.0
        self.global_rater_offset_ = float(np.mean(list(self.rater_offset_.values())))

        # 3. Per-patient quadratic: leave-one-out fit within the training set
        self.patient_quad_ = {}
        for pid, grp in X.groupby("patient_id"):
            grp = grp.sort_values("age")
            ages = grp["age"].values
            y_p  = proxy.loc[grp.index].values
            n    = len(grp)
            if n < self.min_visits_quad:
                # store simple linear from all visits
                if n >= 2:
                    lr = np.polyfit(ages, y_p, 1)
                    self.patient_quad_[pid] = np.array([0.0, lr[0], lr[1]])
                else:
                    self.patient_quad_[pid] = None
                continue
            # leave-one-out quadratic coefficients per visit
            coefs = []
            for leave_out in range(n):
                mask = np.ones(n, dtype=bool)
                mask[leave_out] = False
                ages_tr = ages[mask]
                y_tr    = y_p[mask]
                if len(ages_tr) >= 3:
                    c = np.polyfit(ages_tr, y_tr, 2)
                else:
                    c2 = np.polyfit(ages_tr, y_tr, 1)
                    c  = np.array([0.0, c2[0], c2[1]])
                coefs.append(c)
            # store median coefficients for robustness
            coefs_arr = np.array(coefs)
            self.patient_quad_[pid] = np.median(coefs_arr, axis=0)
        return self

    # ------------------------------------------------------------------
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        proxy = self._make_proxy(X)

        # Per-rater corrected off
        def get_rater_offset(rid):
            return self.rater_offset_.get(rid, self.global_rater_offset_)

        X["rater_offset"]    = X["rater_id"].map(get_rater_offset)
        X["corrected_off"]   = X["off"] - X["rater_offset"]
        X["off_missing_flag"] = X["off"].isna().astype(np.float32)
        X["on_adj"]          = X["on"] + self.on_offset_
        X["target_proxy"]    = proxy - X["rater_offset"].fillna(0)

        # Per-patient quadratic prediction
        quad_preds = np.full(len(X), np.nan)
        for pid, grp in X.groupby("patient_id"):
            idx  = grp.index
            ages = grp["age"].values
            if pid in self.patient_quad_:
                coef = self.patient_quad_[pid]
            else:
                # Cold-start: fit from this frame's own visits (test-time)
                y_p = proxy.loc[idx].values
                n   = len(grp)
                if n >= 3:
                    coef = np.polyfit(ages, y_p, 2)
                elif n >= 2:
                    c2   = np.polyfit(ages, y_p, 1)
                    coef = np.array([0.0, c2[0], c2[1]])
                else:
                    coef = None
            if coef is not None:
                quad_preds[X.index.get_loc(idx[0]):X.index.get_loc(idx[-1]) + 1] = \
                    np.polyval(coef, ages)
                # safer: use positional loc
                pos = [X.index.get_loc(i) for i in idx]
                quad_preds[pos] = np.polyval(coef, ages)

        X["quad_pred"]     = quad_preds
        X["quad_residual"] = proxy - X["quad_pred"]

        # Encode categoricals for HGBR
        for col in CAT_COLS:
            if col in X.columns:
                X[col] = X[col].astype("category")

        # Drop ID columns
        drop = [c for c in DROP_COLS if c in X.columns]
        if drop:
            X = X.drop(columns=drop)

        return X

    def fit_transform(self, X, y=None, **fit_params):
        return self.fit(X, y).transform(X)


# ============================================================
# Smoothing helper (identical to 13_tuned_smoothed.py)
# ============================================================

def smooth_within_frame(raw_preds, ages, patient_ids, w, min_visits=3):
    if w == 0.0:
        return raw_preds.copy()
    smoothed = raw_preds.copy().astype(float)
    df_s = pd.DataFrame({"pred": raw_preds, "age": ages, "pid": patient_ids})
    for pid, grp in df_s.groupby("pid"):
        if len(grp) < min_visits:
            continue
        lr = LinearRegression()
        lr.fit(grp[["age"]], grp["pred"])
        smoothed[grp.index] = (1 - w) * grp["pred"].values + w * lr.predict(grp[["age"]])
    return smoothed


# ============================================================
# Load data
# ============================================================
print("Loading data...")
t0 = time.time()
X, y      = load_dataset(data_dir=DATA_DIR)
X_test    = load_test_dataset(data_dir=DATA_DIR)
groups    = X["patient_id"]
cv_splits = list(GroupKFold(n_splits=5).split(X, y, groups=groups))
print(f"  Done in {time.time()-t0:.1f}s")

# ============================================================
# Build the physics pipeline
# ============================================================

def build_physics_pipeline():
    from sklearn.preprocessing import FunctionTransformer
    physics = PhysicsFeatures(min_visits_quad=3)
    hgbr    = HistGradientBoostingRegressor(
        categorical_features="from_dtype",
        early_stopping=True,
        n_iter_no_change=20,
        **BEST_PARAMS,
        random_state=0,
    )
    return Pipeline([("physics", physics), ("hgbr", hgbr)])


# ============================================================
# OOF predictions (cached) + smoothing sweep
# ============================================================
OOF_CACHE_PHYS = CACHE_DIR / "13_physics_oof_raw.npy"

if OOF_CACHE_PHYS.exists():
    print("Loading cached physics OOF predictions...")
    oof_raw = np.load(OOF_CACHE_PHYS)
else:
    print("Computing OOF predictions for 13_physics (5 folds)...")
    oof_raw = np.full(len(X), np.nan)
    for fold_i, (tr_idx, val_idx) in enumerate(cv_splits):
        t_fold = time.time()
        pipe = build_physics_pipeline()
        pipe.fit(X.iloc[tr_idx], y.iloc[tr_idx])
        oof_raw[val_idx] = pipe.predict(X.iloc[val_idx])
        fold_rmse = np.sqrt(np.mean((oof_raw[val_idx] - y.values[val_idx]) ** 2))
        print(f"  fold {fold_i}: raw RMSE={fold_rmse:.4f}  ({time.time()-t_fold:.1f}s)")
    np.save(OOF_CACHE_PHYS, oof_raw)
    print("  OOF predictions cached.")

raw_rmse = np.sqrt(np.mean((oof_raw - y.values) ** 2))
print(f"\nRaw OOF RMSE (13_physics, no smooth): {raw_rmse:.4f}")

# Smoothing sweep
print("\nSmoothing sweep:")
print(f"  {'w':>5}  {'mean RMSE':>10}  {'std RMSE':>10}")
best_w_phys = 0.0
best_rmse_phys = raw_rmse
for w_cand in [0.0, 0.25, 0.5, 0.75, 1.0]:
    fold_rmses = []
    for tr_idx, val_idx in cv_splits:
        raw_fold  = oof_raw[val_idx]
        ages_fold = X["age"].values[val_idx]
        pids_fold = X["patient_id"].values[val_idx]
        blended   = smooth_within_frame(raw_fold, ages_fold, pids_fold, w=w_cand)
        fold_rmses.append(np.sqrt(np.mean((blended - y.values[val_idx]) ** 2)))
    m, s = np.mean(fold_rmses), np.std(fold_rmses)
    print(f"  {w_cand:>5.2f}  {m:>10.4f}  {s:>10.4f}")
    if m < best_rmse_phys:
        best_rmse_phys = m
        best_w_phys = w_cand

print(f"\nBest w={best_w_phys}  CV RMSE={best_rmse_phys:.4f}")

# Comparison
prev_best = 4.3975
delta = prev_best - best_rmse_phys
fold_std = 0.139
print(f"\nComparison:")
print(f"  13_tuned_smoothed (prev best) : {prev_best:.4f}")
print(f"  13_physics        (this)      : {best_rmse_phys:.4f}")
print(f"  Delta: {delta:.4f}  fold-std: {fold_std:.4f}")
if delta > fold_std:
    print("  => KEEP: gain exceeds fold std")
elif delta > 0:
    print("  => MARGINAL: gain < fold std -- consider keeping for stacking")
else:
    print("  => DISCARD: no improvement")

# ============================================================
# skore evaluate + hub put
# ============================================================
print("\nRunning skore.evaluate for hub report...")
try:
    cfg = load_skore_credentials()
    login(mode="hub")
    project = skore.Project(name="bobathon-esilv", mode="hub", workspace=cfg["workspace"])

    report_13 = skore.evaluate(build_physics_pipeline(), X, y, splitter=cv_splits)
    rmse_agg  = report_13.metrics.rmse()
    rmse_mean = rmse_agg.iloc[0, 0]
    rmse_std  = rmse_agg.iloc[0, 1]
    print(f"  skore RMSE (raw): {rmse_mean:.4f} +/- {rmse_std:.4f}")

    project.put("13_physics", report_13)
    print(f"  Hub report 13_physics pushed.")
    workspace = cfg["workspace"]
except Exception as e:
    print(f"  Hub put failed (will retry): {e}")
    workspace = "evdgreenteam"

# ============================================================
# Final model fit + submission
# ============================================================
FINAL_MODEL_CACHE = CACHE_DIR / "13_physics_final.joblib"
FINAL_RAW_TEST_CACHE = CACHE_DIR / "13_physics_raw_test.npy"

if FINAL_MODEL_CACHE.exists() and FINAL_RAW_TEST_CACHE.exists():
    print("\nLoading cached final model...")
    final_pipe   = joblib.load(FINAL_MODEL_CACHE)
    raw_test     = np.load(FINAL_RAW_TEST_CACHE)
else:
    print("\nFitting final model on all training data...")
    t1 = time.time()
    final_pipe = build_physics_pipeline()
    final_pipe.fit(X, y)
    raw_test = final_pipe.predict(X_test)
    joblib.dump(final_pipe, FINAL_MODEL_CACHE)
    np.save(FINAL_RAW_TEST_CACHE, raw_test)
    print(f"  Done in {time.time()-t1:.1f}s  [cached]")

# Apply smoothing within test frame
smoothed_test = smooth_within_frame(
    raw_test,
    X_test["age"].values,
    X_test["patient_id"].values,
    w=best_w_phys,
)
n_clipped_low  = int((smoothed_test < 0).sum())
n_clipped_high = int((smoothed_test > 132).sum())
if n_clipped_low or n_clipped_high:
    print(f"  Clipping {n_clipped_low} below 0, {n_clipped_high} above 132")
smoothed_test = np.clip(smoothed_test, 0, 132)

sub = pd.DataFrame({"target": smoothed_test}, index=X_test.index)
sub.index.name = "Index"
assert len(sub) == len(X_test)
assert sub["target"].isna().sum() == 0
pmin, pmax = sub["target"].min(), sub["target"].max()

out_path = SUBMISSIONS_DIR / "13_physics.csv"
sub.to_csv(out_path)

print(f"\n[OK] rows: {len(sub)}  NaN: 0  range: [{pmin:.2f}, {pmax:.2f}]")
print(f"Written: {out_path}")

hub_url = f"https://skore.probabl.ai/{workspace}/bobathon-esilv/cross-validations"
print(f"\nHub key : 13_physics")
print(f"Hub URL : {hub_url}")
print(f"\nSTEP 3 DONE.")
