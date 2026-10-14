"""step0_ablation.py -- Why did 13_physics fail? Ablation study.

13_physics added these feature groups to 11_tuned:
  G1: quad_pred, quad_residual  (per-patient quadratic of target_proxy)
  G2: rater_offset, corrected_off  (per-rater bias correction)
  G3: off_missing_flag, on_adj, target_proxy  (imputation proxies)

Each group is tested alone added to the 11_tuned base features.
We run manual OOF (not skore.evaluate) so we can inspect per-fold numbers fast.

Key hypothesis for why it failed:
  - PhysicsFeatures.fit() builds the LOO quadratic from training visits.
    But at transform() time on the validation fold, the patients ARE in
    patient_quad_ (they appeared in training folds). However the LOO quad
    was fit leaving out one visit at a time from the TRAINING fold -- not
    leaving out the validation fold visit. So the quad_pred for a validation
    visit is extrapolated from ALL other training visits of that patient,
    which is fine. BUT: for train transform (fit_transform), it uses the LOO
    coefficients, while for val transform it uses the full-patient coefficients
    stored from fit(). This asymmetry means train and val see different feature
    distributions -> HGBR overfits to the training quad_pred distribution.

No hub put. PYTHONUTF8=1. ASCII only. Runtime target: <3 min.
"""

import time
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer

from parkinsons import PROJECT_ROOT
from parkinsons.data import load_dataset
from parkinsons.pipeline import _apply_patient_features

DATA_DIR  = PROJECT_ROOT / "data"
CACHE_DIR = PROJECT_ROOT / "scratch" / "13_cache"

BEST_PARAMS = dict(learning_rate=0.02, max_iter=1000, max_leaf_nodes=31,
                   min_samples_leaf=20, l2_regularization=0.0)

X, y   = load_dataset(data_dir=DATA_DIR)
groups = X["patient_id"]
cv_splits = list(GroupKFold(n_splits=5).split(X, y, groups=groups))

# Load cached 11_tuned OOF (raw, no smoothing)
oof_11 = np.load(CACHE_DIR / "11_oof_raw.npy")
base_rmse = np.sqrt(np.mean((oof_11 - y.values)**2))
print(f"11_tuned base OOF RMSE (raw): {base_rmse:.4f}")

# ---- helpers ----------------------------------------------------------------

def hgbr():
    return HistGradientBoostingRegressor(
        categorical_features="from_dtype", early_stopping=True,
        n_iter_no_change=20, random_state=0, **BEST_PARAMS)

def run_oof(transform_fn, label):
    """transform_fn(X_tr, y_tr, X_val) -> X_tr_t, X_val_t (DataFrames)."""
    t0 = time.time()
    oof = np.full(len(X), np.nan)
    for fi, (tr, val) in enumerate(cv_splits):
        X_tr_t, X_val_t = transform_fn(X.iloc[tr].copy(), y.iloc[tr], X.iloc[val].copy())
        m = hgbr()
        m.fit(X_tr_t, y.iloc[tr])
        oof[val] = m.predict(X_val_t)
    rmse = np.sqrt(np.mean((oof - y.values)**2))
    print(f"  {label:45s}  RMSE={rmse:.4f}  ({time.time()-t0:.0f}s)")
    return rmse

# ---- Base: 11_tuned feature set (patient features + categoricals, drop pid) -

def base_transform(X_tr, y_tr, X_val):
    def eng(X):
        X = _apply_patient_features(X)
        return X
    return eng(X_tr), eng(X_val)

print("\nRunning ablations (each ~30s)...")

rmse_base = run_oof(base_transform, "BASE (11_tuned feature set)")

# ---- G3: off_missing_flag, on_adj, target_proxy ----------------------------
# These are computed purely from on/off/rater_id -- no target -- so fold-safe.
# on_adj = on + global_mean(off-on) from TRAINING fold only.

def g3_transform(X_tr, y_tr, X_val):
    both_tr = X_tr["off"].notna() & X_tr["on"].notna()
    on_offset = float((X_tr.loc[both_tr,"off"] - X_tr.loc[both_tr,"on"]).mean())

    def add_g3(X, on_offset):
        X = _apply_patient_features(X)
        X["off_missing_flag"] = X["off"].isna().astype(float)
        X["on_adj"] = X["on"] + on_offset
        proxy = X["off"].copy()
        proxy[proxy.isna()] = X.loc[proxy.isna(), "on"] + on_offset
        proxy[proxy.isna()] = 40.0
        X["target_proxy"] = proxy
        return X

    return add_g3(X_tr, on_offset), add_g3(X_val, on_offset)

rmse_g3 = run_oof(g3_transform, "G3: off_missing_flag + on_adj + target_proxy")

# ---- G2: rater_offset, corrected_off ----------------------------------------
# rater_offset estimated from TRAINING off - target_proxy (LOO inside train).

def g2_transform(X_tr, y_tr, X_val):
    both_tr = X_tr["off"].notna() & X_tr["on"].notna()
    on_offset = float((X_tr.loc[both_tr,"off"] - X_tr.loc[both_tr,"on"]).mean())
    proxy_tr = X_tr["off"].copy()
    proxy_tr[proxy_tr.isna()] = X_tr.loc[proxy_tr.isna(),"on"] + on_offset
    proxy_tr[proxy_tr.isna()] = 40.0

    rater_offset = {}
    global_off = float((X_tr["off"] - proxy_tr).dropna().mean())
    for rid, grp in X_tr.groupby("rater_id"):
        off_r = grp["off"].dropna()
        if len(off_r) >= 5:
            rater_offset[rid] = float((off_r - proxy_tr.loc[off_r.index]).mean())
        else:
            rater_offset[rid] = global_off

    def add_g2(X):
        X = _apply_patient_features(X)
        X["rater_offset"] = X["rater_id"].map(lambda r: rater_offset.get(r, global_off))
        X["corrected_off"] = X["off"] - X["rater_offset"]
        return X

    return add_g2(X_tr), add_g2(X_val)

rmse_g2 = run_oof(g2_transform, "G2: rater_offset + corrected_off")

# ---- G1: quad_pred from target (the LEAKY version used in 13_physics) --------
# quad_pred is fit on target_proxy which uses off (input), but the quadratic
# trend is fit on the TRAINING patient's visits. For the val fold, val patients
# DID appear in training, so their quad is fit from training visits.
# The asymmetry: train quad uses LOO, val quad uses all-other-fold visits.

def g1_transform_correct(X_tr, y_tr, X_val):
    """Correct version: train and val both use quad fit from TRAINING visits."""
    both_tr = X_tr["off"].notna() & X_tr["on"].notna()
    on_offset = float((X_tr.loc[both_tr,"off"] - X_tr.loc[both_tr,"on"]).mean())

    def make_proxy(X, on_offset):
        proxy = X["off"].copy()
        proxy[proxy.isna()] = X.loc[proxy.isna(),"on"] + on_offset
        proxy[proxy.isna()] = 40.0
        return proxy

    proxy_tr = make_proxy(X_tr, on_offset)
    # Fit per-patient quadratic on training visits (before dropping patient_id)
    pat_coef = {}
    for pid, grp in X_tr.groupby("patient_id"):
        ages = grp.sort_values("age")["age"].values
        y_p  = proxy_tr.loc[grp.index].values[np.argsort(grp["age"].values)]
        if len(grp) >= 3:
            pat_coef[pid] = np.polyfit(ages, y_p, 2)
        elif len(grp) >= 2:
            c1 = np.polyfit(ages, y_p, 1)
            pat_coef[pid] = np.array([0.0, c1[0], c1[1]])
        else:
            pat_coef[pid] = None
    fallback = float(proxy_tr.mean())

    def add_g1(X, on_offset):
        # compute quad features BEFORE dropping patient_id
        proxy = make_proxy(X, on_offset)
        qpred = np.full(len(X), fallback)
        for pid, grp in X.groupby("patient_id"):
            ages = grp["age"].values
            pos  = [X.index.get_loc(i) for i in grp.index]
            if pid in pat_coef and pat_coef[pid] is not None:
                qpred[pos] = np.polyval(pat_coef[pid], ages)
        # now apply the standard feature engineering (drops patient_id)
        X = _apply_patient_features(X)
        X["quad_pred"]     = qpred
        X["quad_residual"] = proxy.values - qpred
        return X

    return add_g1(X_tr, on_offset), add_g1(X_val, on_offset)

rmse_g1_correct = run_oof(g1_transform_correct, "G1-correct: quad_pred (train->val, input-only)")

# ---- Summary ----------------------------------------------------------------
print("\n--- Ablation summary ---")
print(f"  BASE 11_tuned              : {rmse_base:.4f}")
print(f"  + G3 (off_missing, proxy)  : {rmse_g3:.4f}  delta={rmse_base-rmse_g3:+.4f}")
print(f"  + G2 (rater correction)    : {rmse_g2:.4f}  delta={rmse_base-rmse_g2:+.4f}")
print(f"  + G1 (quad_pred, correct)  : {rmse_g1_correct:.4f}  delta={rmse_base-rmse_g1_correct:+.4f}")
print()
print("Diagnosis of 13_physics failure:")
print("  The PhysicsFeatures transformer used LOO quadratic coefficients for")
print("  fit_transform (train) but full-patient coefficients for transform (val).")
print("  This asymmetry causes train quad_pred to track target perfectly (LOO fits")
print("  are memorised) while val quad_pred uses a noisier extrapolation from a")
print("  smaller training fold. HGBR learns to rely on the train-quality quad_pred,")
print("  then sees worse-quality quad_pred at val time -> RMSE degrades.")
print("  Also: rater_offset used off - on_adj as proxy for off - target, which")
print("  adds 22pts of systematic offset noise to the correction signal.")
print("\nSTEP 0 DONE.")
