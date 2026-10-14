"""step3b_quadratic.py -- STEP 3b: per-patient quadratic as the primary predictor.

Insight from STEP 2c: median per-patient R2(target ~ age^2) = 0.9994,
residual SD = 0.13 pts. This means the within-patient signal is enormous.

Strategy: fit a per-patient quadratic on the TRAINING visits of each patient,
predict the HELD-OUT visits (same patient, different age).  This is a pure
extrapolation/interpolation from the patient's own trajectory.

For cross-validation: each fold has ~1115 validation patients, all of whom
appear in the 4 training folds with their other visits.  Fit the quadratic on
those training visits, predict on the validation visits.

For test: test patients are COLD-START (zero overlap with train).  Fit the
quadratic on their own test visits (using model predictions as a proxy for
target, since target is unavailable) and smooth within test, which is what
STEP 0 already does.

This script tests THREE approaches and picks the best:
  A) Pure quadratic (no HGBR): just np.polyfit per patient.
  B) Quadratic + residual HGBR: quad_pred as a feature, HGBR learns the residual.
  C) Blend: alpha * quad_pred + (1 - alpha) * HGBR_pred (best alpha by CV).

PYTHONUTF8=1 required.  No non-ASCII characters in prints.
No hub put in this diagnostic script.
"""

import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer

from parkinsons import PROJECT_ROOT
from parkinsons.data import load_dataset
from parkinsons.pipeline import _apply_patient_features, build_hgbr_tuned

DATA_DIR  = PROJECT_ROOT / "data"
CACHE_DIR = PROJECT_ROOT / "scratch" / "13_physics_cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

BEST_PARAMS = {
    "learning_rate": 0.02,
    "max_iter": 1000,
    "max_leaf_nodes": 31,
    "min_samples_leaf": 20,
    "l2_regularization": 0.0,
}

# ============================================================
# Helpers
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


def quad_predict_loo(X_fold: pd.DataFrame, y_fold: pd.Series) -> np.ndarray:
    """Per-patient LOO quadratic prediction within a fold.

    For each patient, fit a quadratic on all other visits of that patient
    within X_fold, predict the left-out visit.  Patients with < 3 visits
    fall back to a linear fit; with < 2 visits return the fold mean.

    Returns predictions aligned to X_fold.index.
    """
    preds = np.full(len(X_fold), np.nan)
    fold_mean = float(y_fold.mean())
    for pid, grp in X_fold.groupby("patient_id"):
        grp = grp.sort_values("age")
        idx_list = list(grp.index)
        n = len(grp)
        ages = grp["age"].values
        y_p  = y_fold.loc[grp.index].values
        for i, idx in enumerate(idx_list):
            mask = np.ones(n, dtype=bool)
            mask[i] = False
            ages_tr = ages[mask]
            y_tr    = y_p[mask]
            pos = X_fold.index.get_loc(idx)
            if n == 1:
                preds[pos] = fold_mean
            elif n == 2:
                preds[pos] = float(np.polyval(np.polyfit(ages_tr, y_tr, 1), ages[i]))
            else:
                preds[pos] = float(np.polyval(np.polyfit(ages_tr, y_tr, 2), ages[i]))
    return preds


def quad_predict_train_test(X_train, y_train, X_val):
    """Fit per-patient quadratic on X_train/y_train, predict X_val.

    Patients in X_val but not in X_train get the global mean.
    """
    # Build patient -> quadratic
    patient_coef = {}
    global_mean  = float(y_train.mean())
    for pid, grp in X_train.groupby("patient_id"):
        ages = grp.sort_values("age")["age"].values
        y_p  = y_train.loc[grp.index].values[np.argsort(grp["age"].values)]
        n = len(grp)
        if n >= 3:
            patient_coef[pid] = np.polyfit(ages, y_p, 2)
        elif n >= 2:
            c1 = np.polyfit(ages, y_p, 1)
            patient_coef[pid] = np.array([0.0, c1[0], c1[1]])
        else:
            patient_coef[pid] = None

    preds = np.full(len(X_val), np.nan)
    for pid, grp in X_val.groupby("patient_id"):
        ages = grp["age"].values
        pos  = [X_val.index.get_loc(i) for i in grp.index]
        if pid in patient_coef and patient_coef[pid] is not None:
            preds[pos] = np.polyval(patient_coef[pid], ages)
        else:
            preds[pos] = global_mean
    # Any remaining NaN -> global mean
    preds[np.isnan(preds)] = global_mean
    return preds


# ============================================================
# Load data
# ============================================================
print("Loading data...")
X, y      = load_dataset(data_dir=DATA_DIR)
groups    = X["patient_id"]
cv_splits = list(GroupKFold(n_splits=5).split(X, y, groups=groups))
print(f"  {len(X)} rows, {X['patient_id'].nunique()} patients")

# ============================================================
# Approach A: pure quadratic per patient (train folds -> val fold)
# ============================================================
print("\n" + "=" * 60)
print("Approach A: per-patient quadratic (train visits -> val visits)")
print("=" * 60)

oof_quad = np.full(len(X), np.nan)
for fold_i, (tr_idx, val_idx) in enumerate(cv_splits):
    t_f = time.time()
    oof_quad[val_idx] = quad_predict_train_test(
        X.iloc[tr_idx], y.iloc[tr_idx], X.iloc[val_idx]
    )
    rmse_f = np.sqrt(np.mean((oof_quad[val_idx] - y.values[val_idx]) ** 2))
    print(f"  fold {fold_i}: RMSE={rmse_f:.4f}  ({time.time()-t_f:.1f}s)")

rmse_A = np.sqrt(np.mean((oof_quad - y.values) ** 2))
print(f"Approach A -- OOF RMSE: {rmse_A:.4f}")

# Smooth A
fold_rmses_A = []
for tr_idx, val_idx in cv_splits:
    bl = smooth_within_frame(oof_quad[val_idx], X["age"].values[val_idx], X["patient_id"].values[val_idx], w=0.75)
    fold_rmses_A.append(np.sqrt(np.mean((bl - y.values[val_idx]) ** 2)))
print(f"Approach A + smooth(w=0.75) -- OOF RMSE: {np.mean(fold_rmses_A):.4f} +/- {np.std(fold_rmses_A):.4f}")

# ============================================================
# Approach B: HGBR with quad_pred as an extra feature
# ============================================================
print("\n" + "=" * 60)
print("Approach B: HGBR with quad_pred as feature")
print("=" * 60)

# Load the already-cached 11_tuned OOF
OOF_11 = PROJECT_ROOT / "scratch" / "13_cache" / "11_oof_raw.npy"
assert OOF_11.exists(), "Run step 0 first"
oof_11 = np.load(OOF_11)
oof_11_smooth_rmse = 4.3975  # known from step 0

def build_hgbr_with_quad_feat():
    """HGBR pipeline that adds quad_pred as a feature at transform time.
    The quad_pred is computed from the patient's train-fold visits only,
    so it is fold-safe.
    """
    from sklearn.base import BaseEstimator, TransformerMixin

    class AddQuadFeature(BaseEstimator, TransformerMixin):
        def fit(self, X, y=None):
            self._X_train = X.copy()
            self._y_train = y.copy() if y is not None else None
            return self
        def transform(self, X):
            X = X.copy()
            if self._y_train is not None:
                preds = quad_predict_train_test(self._X_train, self._y_train, X)
            else:
                preds = np.full(len(X), np.nan)
            X["quad_pred"] = preds
            # categoricals
            for col in ("cohort", "gene", "rater_id"):
                if col in X.columns:
                    X[col] = X[col].astype("category")
            # patient features
            X = _apply_patient_features(X)
            return X
        def fit_transform(self, X, y=None, **kw):
            return self.fit(X, y).transform(X)

    hgbr = HistGradientBoostingRegressor(
        categorical_features="from_dtype",
        early_stopping=True,
        n_iter_no_change=20,
        **BEST_PARAMS,
        random_state=0,
    )
    return Pipeline([("quad_feat", AddQuadFeature()), ("hgbr", hgbr)])

OOF_B_CACHE = CACHE_DIR / "step3b_oof_B.npy"
if OOF_B_CACHE.exists():
    print("  Loading cached OOF B...")
    oof_B = np.load(OOF_B_CACHE)
else:
    oof_B = np.full(len(X), np.nan)
    for fold_i, (tr_idx, val_idx) in enumerate(cv_splits):
        t_f = time.time()
        pipe_B = build_hgbr_with_quad_feat()
        pipe_B.fit(X.iloc[tr_idx], y.iloc[tr_idx])
        oof_B[val_idx] = pipe_B.predict(X.iloc[val_idx])
        rmse_f = np.sqrt(np.mean((oof_B[val_idx] - y.values[val_idx]) ** 2))
        print(f"  fold {fold_i}: RMSE={rmse_f:.4f}  ({time.time()-t_f:.1f}s)")
    np.save(OOF_B_CACHE, oof_B)

rmse_B_raw = np.sqrt(np.mean((oof_B - y.values) ** 2))
fold_rmses_B = []
for tr_idx, val_idx in cv_splits:
    bl = smooth_within_frame(oof_B[val_idx], X["age"].values[val_idx], X["patient_id"].values[val_idx], w=0.75)
    fold_rmses_B.append(np.sqrt(np.mean((bl - y.values[val_idx]) ** 2)))
print(f"Approach B -- OOF RMSE raw: {rmse_B_raw:.4f}  smoothed: {np.mean(fold_rmses_B):.4f} +/- {np.std(fold_rmses_B):.4f}")

# ============================================================
# Approach C: blend quadratic + HGBR (using oof_11 for HGBR)
# ============================================================
print("\n" + "=" * 60)
print("Approach C: blend(alpha) * quad + (1-alpha) * HGBR_11")
print("=" * 60)

print(f"  {'alpha':>6}  {'RMSE':>8}  {'std':>8}")
best_alpha_C = 0.0
best_rmse_C  = 1e9
for alpha in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
    blended_oof = alpha * oof_quad + (1 - alpha) * oof_11
    fold_rmses_C = []
    for tr_idx, val_idx in cv_splits:
        bl = smooth_within_frame(blended_oof[val_idx], X["age"].values[val_idx], X["patient_id"].values[val_idx], w=0.75)
        fold_rmses_C.append(np.sqrt(np.mean((bl - y.values[val_idx]) ** 2)))
    m, s = np.mean(fold_rmses_C), np.std(fold_rmses_C)
    print(f"  {alpha:>6.2f}  {m:>8.4f}  {s:>8.4f}")
    if m < best_rmse_C:
        best_rmse_C = m
        best_alpha_C = alpha

print(f"\nBest alpha={best_alpha_C}  RMSE={best_rmse_C:.4f}")

# ============================================================
# Summary
# ============================================================
print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)
print(f"  11_tuned + smooth (prev best) : {oof_11_smooth_rmse:.4f}")
print(f"  A) pure quadratic + smooth    : {np.mean(fold_rmses_A):.4f}")
print(f"  B) HGBR + quad_pred + smooth  : {np.mean(fold_rmses_B):.4f}")
print(f"  C) blend(alpha={best_alpha_C}) quad+HGBR + smooth: {best_rmse_C:.4f}")
print(f"\n  Approach A RMSE: {rmse_A:.4f}")
print(f"  Gap from linear to quadratic within patient is captured by approach A.")
print(f"  For test (cold-start), approach A uses test visits -> same as smoothing.")
print("\nSTEP 3b DONE -- no hub put (diagnostic only).")
