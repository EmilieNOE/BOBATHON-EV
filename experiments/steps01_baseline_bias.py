"""step0_baseline.py + step1_bias.py  (combined for speed)

STEP 0: Reproduce 14_quad_smooth CV RMSE from cached OOF (no modelling).
STEP 1: Residual analysis + second-stage HGBR on residuals (key 15_bias).

PYTHONUTF8=1. ASCII only. Cache everything.
"""

import sys, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))  # find shared.py

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import GroupKFold

from parkinsons import PROJECT_ROOT
from parkinsons.data import load_dataset, load_test_dataset
from parkinsons.hub import load_skore_credentials
from parkinsons.pipeline import build_hgbr_tuned
from shared import add_s1_aggregates, ridge_poly_smooth, sanity_check

DATA_DIR  = PROJECT_ROOT / "data"
SUBS_DIR  = PROJECT_ROOT / "submissions"
C11       = PROJECT_ROOT / "scratch" / "13_cache"
C15       = PROJECT_ROOT / "scratch" / "15_cache"
SUBS_DIR.mkdir(exist_ok=True)
C15.mkdir(parents=True, exist_ok=True)

# 14_quad_smooth best config (from grid search output)
CFG14 = dict(degree=2, ridge_lam=0.1, w_blend=0.9, scheme="uniform")
BEST_PARAMS_11 = dict(learning_rate=0.02, max_iter=1000, max_leaf_nodes=31,
                      min_samples_leaf=20, l2_regularization=0.0)
FOLD_STD_14 = 0.1489   # std from grid search row: deg=2 lam=0.1 w=0.90 uniform
PREV_BEST_CV = 4.3604  # best CV RMSE (14_quad_smooth)

t_start = time.time()

# ── load data ─────────────────────────────────────────────────────────────────
X, y   = load_dataset(data_dir=DATA_DIR)
X_test = load_test_dataset(data_dir=DATA_DIR)
groups = X["patient_id"]
cv_splits = list(GroupKFold(n_splits=5).split(X, y, groups=groups))

oof_11 = np.load(C11 / "11_oof_raw.npy")
print(f"Data loaded ({time.time()-t_start:.1f}s)")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 0: reproduce 14_quad_smooth CV RMSE
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("STEP 0 -- 14_quad_smooth baseline reproduction")
print("="*60)
print(f"  Config: degree={CFG14['degree']}, ridge_lam={CFG14['ridge_lam']}, "
      f"w={CFG14['w_blend']}, weights={CFG14['scheme']}")

ages_a = X["age"].values
pids_a = X["patient_id"].values

fold_rmses_14 = []
oof_14_smooth = np.full(len(X), np.nan)
for tr_idx, val_idx in cv_splits:
    raw_f = oof_11[val_idx]
    sm    = ridge_poly_smooth(raw_f, ages_a[val_idx], pids_a[val_idx],
                              CFG14["w_blend"], CFG14["degree"],
                              CFG14["ridge_lam"], visit_w=None)
    oof_14_smooth[val_idx] = sm
    fold_rmses_14.append(np.sqrt(np.mean((sm - y.values[val_idx])**2)))

mean_14, std_14 = np.mean(fold_rmses_14), np.std(fold_rmses_14)
print(f"  Per-fold RMSE: {[round(r,4) for r in fold_rmses_14]}")
print(f"  CV RMSE: {mean_14:.4f} +/- {std_14:.4f}  (target to beat: {PREV_BEST_CV})")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 1a: residual analysis after 14_quad_smooth
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("STEP 1a -- Residual analysis after 14_quad_smooth")
print("="*60)

resid_14 = oof_14_smooth - y.values   # positive = over-predicted
off_miss  = X["off"].isna()
tsd_vals  = (X["age"] - X["age_at_diagnosis"]).values

def rmse(mask):
    r = resid_14[mask]
    return np.sqrt(np.mean(r**2)) if mask.sum() > 0 else np.nan

def bias(mask):
    return resid_14[mask].mean() if mask.sum() > 0 else np.nan

# time_since_diagnosis x off_missing
print("\n  time_since_diag x off_missing -- RMSE / bias (mean residual):")
print(f"  {'tsd_bin':12s}  {'off_pres':8s}  {'n':>6}  {'RMSE':>7}  {'bias':>7}")
for lo, hi in [(0,5),(5,10),(10,15),(15,99)]:
    tsd_mask = (tsd_vals >= lo) & (tsd_vals < hi)
    for om_flag, om_label in [(False,"pres"),(True,"miss")]:
        mask = tsd_mask & (off_miss == om_flag)
        label = f"{lo}-{hi}y" if hi<99 else f">={lo}y"
        print(f"  {label:12s}  off_{om_label} {int(mask.sum()):>6}  "
              f"{rmse(mask):>7.3f}  {bias(mask):>7.3f}")

# predicted-value bins
print("\n  Predicted value bins:")
pred_bins = [(0,20),(20,30),(30,40),(40,50),(50,70),(70,132)]
print(f"  {'pred_bin':12s}  {'n':>6}  {'RMSE':>7}  {'bias':>7}")
for lo, hi in pred_bins:
    mask = (oof_14_smooth >= lo) & (oof_14_smooth < hi)
    print(f"  {lo}-{hi:>3}       {int(mask.sum()):>6}  {rmse(mask):>7.3f}  {bias(mask):>7.3f}")

# focused subgroup
hard_mask = off_miss & (tsd_vals >= 10)
print(f"\n  HARD subgroup (off missing + tsd>=10y): n={int(hard_mask.sum())}, "
      f"RMSE={rmse(hard_mask):.4f}, bias={bias(hard_mask):.4f}")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 1b: stage-2 HGBR on residuals (nested CV)
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("STEP 1b -- Stage-2 stacking (15_bias), nested CV")
print("="*60)

def hgbr_s2():
    return HistGradientBoostingRegressor(
        categorical_features="from_dtype",
        early_stopping=True, n_iter_no_change=20,
        learning_rate=0.05, max_iter=500,
        max_leaf_nodes=23, min_samples_leaf=30,
        random_state=0,
    )

OOF_S2_CACHE = C15 / "15_oof_s2_raw.npy"
if OOF_S2_CACHE.exists():
    print("  Loading cached stage-2 OOF...")
    oof_s2 = np.load(OOF_S2_CACHE)
else:
    print("  Computing nested CV stage-2 OOF...")
    oof_s2 = np.full(len(X), np.nan)
    for fi, (tr_idx, val_idx) in enumerate(cv_splits):
        t_f = time.time()
        # Stage-1 smoothed predictions (14_quad_smooth) on the training fold
        sm_tr = ridge_poly_smooth(
            oof_11[tr_idx], ages_a[tr_idx], pids_a[tr_idx],
            CFG14["w_blend"], CFG14["degree"], CFG14["ridge_lam"])
        sm_val = ridge_poly_smooth(
            oof_11[val_idx], ages_a[val_idx], pids_a[val_idx],
            CFG14["w_blend"], CFG14["degree"], CFG14["ridge_lam"])
        # Build s2 features
        X_tr_s2  = add_s1_aggregates(X.iloc[tr_idx].copy(),  sm_tr)
        X_val_s2 = add_s1_aggregates(X.iloc[val_idx].copy(), sm_val)
        m2 = hgbr_s2()
        m2.fit(X_tr_s2, y.iloc[tr_idx])
        oof_s2[val_idx] = m2.predict(X_val_s2)
        rmse_f = np.sqrt(np.mean((oof_s2[val_idx] - y.values[val_idx])**2))
        print(f"    fold {fi}: RMSE={rmse_f:.4f}  ({time.time()-t_f:.1f}s)")
    np.save(OOF_S2_CACHE, oof_s2)
    print("  Cached.")

rmse_s2_raw = np.sqrt(np.mean((oof_s2 - y.values)**2))
print(f"  Stage-2 raw OOF RMSE: {rmse_s2_raw:.4f}")

# Smoothing sweep on stage-2
print("\n  Smoothing sweep:")
print(f"  {'w':>5}  {'mean':>8}  {'std':>8}  {'per-fold'}")
best_w_s2 = 0.0
best_rmse_s2 = rmse_s2_raw
best_folds_s2 = []
for w_cand in [0.0, 0.5, 0.75, 0.9, 1.0]:
    folds = []
    for tr_idx, val_idx in cv_splits:
        sm = ridge_poly_smooth(oof_s2[val_idx], ages_a[val_idx], pids_a[val_idx],
                               w_cand, degree=2, ridge_lam=0.1)
        folds.append(np.sqrt(np.mean((sm - y.values[val_idx])**2)))
    m, s = np.mean(folds), np.std(folds)
    print(f"  {w_cand:>5.2f}  {m:>8.4f}  {s:>8.4f}  {[round(r,4) for r in folds]}")
    if m < best_rmse_s2:
        best_rmse_s2  = m
        best_w_s2     = w_cand
        best_folds_s2 = folds

# Paired comparison with 14_quad_smooth
print(f"\n  Paired fold comparison (15_bias vs 14_quad_smooth):")
print(f"  {'fold':>5}  {'14':>8}  {'15':>8}  {'delta':>8}")
n_improved = 0
for fi, (r14, r15) in enumerate(zip(fold_rmses_14, best_folds_s2)):
    delta = r14 - r15
    improved = delta > 0
    n_improved += int(improved)
    print(f"  {fi:>5}  {r14:>8.4f}  {r15:>8.4f}  {delta:>+8.4f}  {'BETTER' if improved else ''}")

mean_delta = np.mean([r14 - r15 for r14, r15 in zip(fold_rmses_14, best_folds_s2)])
print(f"\n  Mean gain: {mean_delta:.4f}  Folds improved: {n_improved}/5")
is_improvement = mean_delta > 0.03 and n_improved >= 4
print(f"  Improvement criterion (gain>0.03 AND >=4 folds): {'MET' if is_improvement else 'NOT MET'}")

# Subgroup analysis
oof_s2_best_smooth = np.full(len(X), np.nan)
for tr_idx, val_idx in cv_splits:
    sm = ridge_poly_smooth(oof_s2[val_idx], ages_a[val_idx], pids_a[val_idx],
                           best_w_s2, degree=2, ridge_lam=0.1)
    oof_s2_best_smooth[val_idx] = sm

resid_s2 = oof_s2_best_smooth - y.values
rmse_hard_before = rmse(hard_mask)
rmse_hard_after  = np.sqrt(np.mean(resid_s2[hard_mask]**2))
print(f"\n  HARD subgroup (off_missing + tsd>=10y):")
print(f"    Before (14_quad_smooth): RMSE={rmse_hard_before:.4f}, "
      f"bias={bias(hard_mask):.4f}")
print(f"    After  (15_bias+smooth): RMSE={rmse_hard_after:.4f}, "
      f"bias={resid_s2[hard_mask].mean():.4f}")

# ─────────────────────────────────────────────────────────────────────────────
# OUTPUT RULE: if improvement criterion met, write CSV
# ─────────────────────────────────────────────────────────────────────────────
if is_improvement:
    print("\n" + "="*60)
    print("OUTPUT: Fitting final models + writing submission...")
    print("="*60)

    # Load / fit stage-1 final
    S1_MODEL = C11 / "11_tuned_final.joblib"
    S1_RAW   = C11 / "11_tuned_raw_test.npy"
    if S1_MODEL.exists() and S1_RAW.exists():
        s1_final    = joblib.load(S1_MODEL)
        raw_test_s1 = np.load(S1_RAW)
        print("  Stage-1 loaded from cache.")
    else:
        s1_final = build_hgbr_tuned(**BEST_PARAMS_11, random_state=0)
        s1_final.fit(X, y)
        raw_test_s1 = s1_final.predict(X_test)
        joblib.dump(s1_final, S1_MODEL)
        np.save(S1_RAW, raw_test_s1)
        print("  Stage-1 fit + cached.")

    # Smooth stage-1 test predictions (14_quad_smooth config)
    sm_test_s1 = ridge_poly_smooth(raw_test_s1, X_test["age"].values,
                                   X_test["patient_id"].values,
                                   CFG14["w_blend"], CFG14["degree"], CFG14["ridge_lam"])

    # Smooth stage-1 TRAIN predictions for feeding stage-2
    sm_train_s1 = ridge_poly_smooth(oof_11, ages_a, pids_a,
                                    CFG14["w_blend"], CFG14["degree"], CFG14["ridge_lam"])

    # Fit stage-2 on all train
    S2_MODEL = C15 / "15_s2_final.joblib"
    S2_RAW   = C15 / "15_raw_test.npy"
    if S2_MODEL.exists() and S2_RAW.exists():
        m2_final    = joblib.load(S2_MODEL)
        raw_test_s2 = np.load(S2_RAW)
        print("  Stage-2 loaded from cache.")
    else:
        X_train_s2 = add_s1_aggregates(X.copy(), sm_train_s1)
        m2_final   = hgbr_s2()
        m2_final.fit(X_train_s2, y)
        X_test_s2   = add_s1_aggregates(X_test.copy(), sm_test_s1)
        raw_test_s2 = m2_final.predict(X_test_s2)
        joblib.dump(m2_final, S2_MODEL)
        np.save(S2_RAW, raw_test_s2)
        print("  Stage-2 fit + cached.")

    # Final smoothing on test
    final_test = ridge_poly_smooth(raw_test_s2, X_test["age"].values,
                                   X_test["patient_id"].values,
                                   best_w_s2, degree=2, ridge_lam=0.1)
    final_test = np.clip(final_test, 0, 132)

    sub = pd.DataFrame({"target": final_test}, index=X_test.index)
    sub.index.name = "Index"
    sanity_check(sub, X_test, "15_bias")
    out_path = SUBS_DIR / "15_bias.csv"
    sub.to_csv(out_path)
    print(f"  Written: {out_path}")

    # Hub put (2-minute timeout)
    hub_url = "PENDING"
    try:
        import skore
        from skore import login
        from sklearn.base import BaseEstimator, RegressorMixin

        class S2Wrapper(BaseEstimator, RegressorMixin):
            def fit(self, X, y):
                s1 = build_hgbr_tuned(**BEST_PARAMS_11, random_state=0)
                s1.fit(X, y)
                sm = ridge_poly_smooth(s1.predict(X), X["age"].values, X["patient_id"].values,
                                       CFG14["w_blend"], CFG14["degree"], CFG14["ridge_lam"])
                X_s2 = add_s1_aggregates(X.copy(), sm)
                self._s2 = hgbr_s2()
                self._s2.fit(X_s2, y)
                self._s1 = s1
                return self
            def predict(self, X):
                sm = ridge_poly_smooth(self._s1.predict(X), X["age"].values,
                                       X["patient_id"].values, CFG14["w_blend"],
                                       CFG14["degree"], CFG14["ridge_lam"])
                X_s2 = add_s1_aggregates(X.copy(), sm)
                raw  = self._s2.predict(X_s2)
                return np.clip(ridge_poly_smooth(raw, X["age"].values,
                               X["patient_id"].values, best_w_s2, 2, 0.1), 0, 132)

        cfg = load_skore_credentials()
        login(mode="hub")
        import signal
        project = skore.Project(name="bobathon-esilv", mode="hub", workspace=cfg["workspace"])
        report  = skore.evaluate(S2Wrapper(), X, y, splitter=cv_splits)
        project.put("15_bias", report)
        workspace = cfg["workspace"]
        hub_url   = f"https://skore.probabl.ai/{workspace}/bobathon-esilv/cross-validations"
        print(f"  Hub report 15_bias pushed.")
    except Exception as e:
        print(f"  HUB PUT PENDING: 15_bias  ({e})")

    print(f"\nREADY TO SUBMIT: submissions/15_bias.csv | "
          f"CV RMSE {best_rmse_s2:.4f} +/- {np.std(best_folds_s2):.4f} | "
          f"hub URL: {hub_url}")
else:
    print(f"\nNo CSV written for 15_bias -- criterion not met.")
    print(f"Keeping 14_quad_smooth as best  (CV {PREV_BEST_CV:.4f})")

print(f"\nTotal elapsed: {time.time()-t_start:.0f}s")
print("STEPS 0+1 DONE.")
