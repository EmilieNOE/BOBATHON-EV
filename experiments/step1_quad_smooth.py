"""step1_quad_smooth.py -- STEP 1: weighted quadratic smoothing of predictions.
Hub key: 14_quad_smooth.

Start from cached 11_tuned OOF. For every patient:
  - Fit a ridge-shrunk polynomial of PREDICTIONS against centered age.
  - Blend: (1-w)*raw + w*trend.
  - Grid over: degree in {1,2}, ridge lambda in {0, 0.1, 1, 10},
    w in {0.5, 0.75, 0.9, 1.0}, weight_scheme in {uniform, subgroup}.
  - Fallback to linear for patients with < 4 visits (degree=2) or < 3 (degree=1).
  - Weights: uniform OR 1/RMSE^2 by subgroup (off_present/missing * on_present/missing).

After finding best config by CV, fit final model on all train, apply same
smoothing to test predictions, write submissions/14_quad_smooth.csv.

PYTHONUTF8=1. ASCII only. Cache OOF. Hub timeout safe.
"""

import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import skore
from skore import login
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import PolynomialFeatures

from parkinsons import PROJECT_ROOT
from parkinsons.data import load_dataset, load_test_dataset
from parkinsons.hub import load_skore_credentials
from parkinsons.pipeline import build_hgbr_tuned

DATA_DIR        = PROJECT_ROOT / "data"
SUBMISSIONS_DIR = PROJECT_ROOT / "submissions"
CACHE_DIR_11    = PROJECT_ROOT / "scratch" / "13_cache"
CACHE_DIR       = PROJECT_ROOT / "scratch" / "14_cache"
SUBMISSIONS_DIR.mkdir(exist_ok=True)
CACHE_DIR.mkdir(parents=True, exist_ok=True)

BEST_PARAMS = dict(learning_rate=0.02, max_iter=1000, max_leaf_nodes=31,
                   min_samples_leaf=20, l2_regularization=0.0)

# ============================================================
# Load data + cached OOF
# ============================================================
t0 = time.time()
X, y   = load_dataset(data_dir=DATA_DIR)
X_test = load_test_dataset(data_dir=DATA_DIR)
groups = X["patient_id"]
cv_splits = list(GroupKFold(n_splits=5).split(X, y, groups=groups))

oof_11 = np.load(CACHE_DIR_11 / "11_oof_raw.npy")
print(f"Data + OOF loaded in {time.time()-t0:.1f}s")
print(f"11_tuned raw OOF RMSE: {np.sqrt(np.mean((oof_11 - y.values)**2)):.4f}")

# Subgroup RMSEs from Step 1 (used as visit weights)
# off_present/missing x on_present/missing
oof_resid = oof_11 - y.values
_off_pres = X["off"].notna()
_on_pres  = X["on"].notna()
subgroup_rmse = {
    (True,  True):  float(np.sqrt(np.mean(oof_resid[ _off_pres &  _on_pres]**2))),
    (True,  False): float(np.sqrt(np.mean(oof_resid[ _off_pres & ~_on_pres]**2))),
    (False, True):  float(np.sqrt(np.mean(oof_resid[~_off_pres &  _on_pres]**2))),
    (False, False): float(np.sqrt(np.mean(oof_resid[~_off_pres & ~_on_pres]**2))),
}
print("\nSubgroup RMSEs (for visit weights):")
for (op, onp), r in subgroup_rmse.items():
    print(f"  off_present={op}, on_present={onp}: {r:.4f}")

def visit_weights(X_frame):
    """1 / rmse^2 by subgroup, normalised within patient."""
    op  = X_frame["off"].notna().values
    onp = X_frame["on"].notna().values
    w = np.array([
        1.0 / subgroup_rmse[(bool(op[i]), bool(onp[i]))]**2
        for i in range(len(X_frame))
    ])
    return w

# ============================================================
# Ridge-smoothed polynomial blend
# ============================================================

def ridge_poly_smooth(raw_preds, ages, patient_ids, w_blend,
                      degree, ridge_lam, visit_w, min_visits_deg2=4):
    """
    For each patient fit a ridge-penalised polynomial of raw_preds ~ centered_age.
    blend = (1-w_blend)*raw + w_blend*trend.
    visit_w: per-row weights (None = uniform).
    """
    smoothed = raw_preds.copy().astype(float)
    df = pd.DataFrame({"pred": raw_preds, "age": ages, "pid": patient_ids,
                       "w": visit_w if visit_w is not None else np.ones(len(raw_preds))})
    for pid, grp in df.groupby("pid"):
        n = len(grp)
        ages_c = grp["age"].values - grp["age"].mean()  # center
        preds_g = grp["pred"].values
        wts     = grp["w"].values
        wts     = wts / wts.sum() * len(wts)            # normalise within patient

        # Decide effective degree
        eff_deg = degree if n >= min_visits_deg2 else min(degree, 1 if n >= 3 else 0)
        if eff_deg == 0 or n < 2:
            # fallback: patient mean
            smoothed[grp.index] = (1 - w_blend) * preds_g + w_blend * np.mean(preds_g)
            continue

        # Build polynomial features (no intercept -> Ridge adds it separately)
        if eff_deg == 1:
            Phi = ages_c.reshape(-1, 1)
        else:
            Phi = np.column_stack([ages_c, ages_c**2])

        W   = np.diag(wts)
        # Weighted ridge: (Phi'W Phi + lam*I) b = Phi'W y
        # Then trend = Phi @ b + intercept (= weighted mean of preds)
        intercept = np.average(preds_g, weights=wts)
        y_c       = preds_g - intercept
        A = Phi.T @ W @ Phi + ridge_lam * np.eye(Phi.shape[1])
        b = Phi.T @ W @ y_c
        try:
            coef = np.linalg.solve(A, b)
        except np.linalg.LinAlgError:
            coef = np.zeros(Phi.shape[1])
        trend = Phi @ coef + intercept

        smoothed[grp.index] = (1 - w_blend) * preds_g + w_blend * trend
    return smoothed

# ============================================================
# Grid search over (degree, ridge_lam, w_blend, weight_scheme)
# ============================================================
degrees    = [1, 2]
ridge_lams = [0.0, 0.1, 1.0, 10.0]
w_blends   = [0.5, 0.75, 0.9, 1.0]
schemes    = ["uniform", "subgroup"]

print("\nGrid search (degree x lam x w x scheme):")
print(f"  {'deg':>3}  {'lam':>6}  {'w':>5}  {'scheme':>10}  {'RMSE':>8}  {'std':>8}")

best_rmse = 4.3975   # prev best (13_tuned_smoothed)
best_cfg  = None
results   = []

ages_all = X["age"].values
pids_all = X["patient_id"].values
vw_all   = visit_weights(X)

for deg in degrees:
    for lam in ridge_lams:
        for w in w_blends:
            for scheme in schemes:
                fold_rmses = []
                for tr_idx, val_idx in cv_splits:
                    raw_f   = oof_11[val_idx]
                    ages_f  = ages_all[val_idx]
                    pids_f  = pids_all[val_idx]
                    vw_f    = vw_all[val_idx] if scheme == "subgroup" else None
                    bl = ridge_poly_smooth(raw_f, ages_f, pids_f, w, deg, lam, vw_f)
                    fold_rmses.append(np.sqrt(np.mean((bl - y.values[val_idx])**2)))
                m, s = np.mean(fold_rmses), np.std(fold_rmses)
                results.append((m, s, deg, lam, w, scheme))
                print(f"  {deg:>3}  {lam:>6.1f}  {w:>5.2f}  {scheme:>10}  {m:>8.4f}  {s:>8.4f}")
                if m < best_rmse:
                    best_rmse = m
                    best_cfg  = (deg, lam, w, scheme)

print(f"\nPrev best (13_tuned_smoothed): 4.3975")
if best_cfg:
    bdeg, blam, bw, bscheme = best_cfg
    print(f"New best: deg={bdeg} lam={blam} w={bw} scheme={bscheme}  RMSE={best_rmse:.4f}")
    delta = 4.3975 - best_rmse
    fold_std = 0.139
    print(f"Delta vs prev: {delta:.4f}  fold_std: {fold_std:.4f}")
    if delta > fold_std:
        print("=> KEEP: gain exceeds fold std")
    elif delta > 0:
        print("=> MARGINAL: gain < fold std")
    else:
        print("=> NO IMPROVEMENT over prev best")
else:
    print("=> NO IMPROVEMENT over prev best -- keeping 13_tuned_smoothed config")
    bdeg, blam, bw, bscheme = 1, 0.0, 0.75, "uniform"

# ============================================================
# Final model + submission
# ============================================================
FINAL_MODEL_CACHE    = CACHE_DIR_11 / "11_tuned_final.joblib"
FINAL_RAW_TEST_CACHE = CACHE_DIR_11 / "11_tuned_raw_test.npy"

if FINAL_MODEL_CACHE.exists() and FINAL_RAW_TEST_CACHE.exists():
    print("\nLoading cached final model + raw test predictions...")
    final_model  = joblib.load(FINAL_MODEL_CACHE)
    raw_test     = np.load(FINAL_RAW_TEST_CACHE)
else:
    print("\nFitting final model on all training data...")
    t1 = time.time()
    final_model = build_hgbr_tuned(**BEST_PARAMS, random_state=0)
    final_model.fit(X, y)
    raw_test = final_model.predict(X_test)
    joblib.dump(final_model, FINAL_MODEL_CACHE)
    np.save(FINAL_RAW_TEST_CACHE, raw_test)
    print(f"  Done in {time.time()-t1:.1f}s")

# Apply best smoothing to test
vw_test = visit_weights(X_test) if bscheme == "subgroup" else None
smoothed_test = ridge_poly_smooth(
    raw_test,
    X_test["age"].values,
    X_test["patient_id"].values,
    bw, bdeg, blam, vw_test,
)
n_clip_lo  = int((smoothed_test < 0).sum())
n_clip_hi  = int((smoothed_test > 132).sum())
if n_clip_lo or n_clip_hi:
    print(f"Clipping {n_clip_lo} below 0, {n_clip_hi} above 132")
smoothed_test = np.clip(smoothed_test, 0, 132)

sub = pd.DataFrame({"target": smoothed_test}, index=X_test.index)
sub.index.name = "Index"
assert len(sub) == len(X_test)
assert sub["target"].isna().sum() == 0
pmin, pmax = sub["target"].min(), sub["target"].max()
out_path = SUBMISSIONS_DIR / "14_quad_smooth.csv"
sub.to_csv(out_path)
print(f"\n[OK] rows={len(sub)}  NaN=0  range=[{pmin:.2f},{pmax:.2f}]")
print(f"Written: {out_path}")

# ============================================================
# Hub report
# ============================================================
print("\nPushing hub report...")
try:
    cfg = load_skore_credentials()
    login(mode="hub")
    project = skore.Project(name="bobathon-esilv", mode="hub", workspace=cfg["workspace"])
    report_14 = skore.evaluate(build_hgbr_tuned(**BEST_PARAMS, random_state=0),
                               X, y, splitter=cv_splits)
    rmse_agg  = report_14.metrics.rmse()
    print(f"  skore RMSE (base, no smooth): {rmse_agg.iloc[0,0]:.4f} +/- {rmse_agg.iloc[0,1]:.4f}")
    project.put("14_quad_smooth", report_14)
    print(f"  Hub report 14_quad_smooth pushed.")
    workspace = cfg["workspace"]
except Exception as e:
    print(f"  Hub put failed (non-fatal): {e}")
    workspace = "evdgreenteam"

hub_url = f"https://skore.probabl.ai/{workspace}/bobathon-esilv/cross-validations"
print(f"\nHub key : 14_quad_smooth")
print(f"Hub URL : {hub_url}")
print(f"Best config: degree={bdeg}, ridge_lam={blam}, w={bw}, weights={bscheme}")
print(f"CV RMSE: {best_rmse:.4f}")
print(f"\nSTEP 1 DONE.")
