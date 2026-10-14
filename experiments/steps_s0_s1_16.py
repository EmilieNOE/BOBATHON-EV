"""steps_s0_s1_16.py -- STEP 0 (describe 15_bias) + STEP 1 (key 16_stage2_tuned).

STEP 0: Print the exact 15_bias pipeline, hyperparameters, features, CV numbers,
        and residual breakdown by subgroup after 15_bias.

STEP 1: Improve stage-2 with:
  a) RandomizedSearch tuning of the stage-2 HGBR (small, reproducible).
  b) Average 5 stage-2 seeds.
  c) Re-tune smoothing post-processing.
  Reuses the cached stage-1 OOF (no stage-1 retraining).

Output rule: if CV beats 15_bias (4.1453), write submissions/16_stage2_tuned.csv,
sanity-check, hub put, then stop.

PYTHONUTF8=1. ASCII-only prints. Cache all OOF arrays.
"""

import sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import joblib
import numpy as np
import pandas as pd
from scipy.stats import randint, uniform
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import GroupKFold, RandomizedSearchCV

from parkinsons import PROJECT_ROOT
from parkinsons.data import load_dataset, load_test_dataset
from parkinsons.hub import load_skore_credentials
from parkinsons.pipeline import build_hgbr_tuned
from shared import add_s1_aggregates, ridge_poly_smooth, sanity_check

DATA_DIR  = PROJECT_ROOT / "data"
SUBS_DIR  = PROJECT_ROOT / "submissions"
C11       = PROJECT_ROOT / "scratch" / "13_cache"
C15       = PROJECT_ROOT / "scratch" / "15_cache"
C16       = PROJECT_ROOT / "scratch" / "16_cache"
SUBS_DIR.mkdir(exist_ok=True)
C16.mkdir(parents=True, exist_ok=True)

CFG14 = dict(degree=2, ridge_lam=0.1, w_blend=0.9)   # 14_quad_smooth smoother
BEST_PARAMS_11 = dict(learning_rate=0.02, max_iter=1000, max_leaf_nodes=31,
                      min_samples_leaf=20, l2_regularization=0.0)

# 15_bias final numbers
PREV_BEST_CV    = 4.1453
PREV_BEST_STD   = 0.1289
PREV_BEST_FOLDS = [4.3990, 4.1213, 4.0873, 4.0548, 4.0642]
PREV_BEST_W_S2  = 0.5     # w used for final smoothing in 15_bias

t0 = time.time()

# ── load data + cached OOF ────────────────────────────────────────────────────
X, y   = load_dataset(data_dir=DATA_DIR)
X_test = load_test_dataset(data_dir=DATA_DIR)
groups = X["patient_id"]
cv_splits = list(GroupKFold(n_splits=5).split(X, y, groups=groups))
ages_a = X["age"].values
pids_a = X["patient_id"].values

oof_11 = np.load(C11 / "11_oof_raw.npy")
oof_s2 = np.load(C15 / "15_oof_s2_raw.npy")   # cached stage-2 OOF from 15_bias
print(f"Data + OOF loaded ({time.time()-t0:.1f}s)")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 0: describe 15_bias and its residual breakdown
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("STEP 0 -- 15_bias description")
print("="*60)

print("""
  Architecture:
    Stage-1 : 11_tuned (HGBR, lr=0.02, 1000 iter, 31 leaves, min_leaf=20, l2=0)
    Smoother: ridge-quadratic within-frame (deg=2, lam=0.1, w=0.90)
              -> 14_quad_smooth config; applied to stage-1 OOF as stage-2 INPUT
    Stage-2 : HGBR (lr=0.05, 500 iter, 23 leaves, min_leaf=30, early_stop, seed=0)
              trained on stage-1 smoothed OOF + per-patient aggregates (s1_mean,
              s1_med, s1_min, s1_max, s1_quad, s1_qres, s1_prev, s1_next,
              s1_nvis, s1_slope, off_frac, yrs_fst, pm_on, pm_off, pm_ledd, tsd,
              on/off/ledd prev2/next2/pm2) + all original input features
    Post-proc: ridge-quadratic smoothing on stage-2 OOF (deg=2, lam=0.1, w=0.50)
    CV:       GroupKFold(5), groups=patient_id, strictly nested
""")
print(f"  15_bias CV RMSE: {PREV_BEST_CV:.4f} +/- {PREV_BEST_STD:.4f}")
print(f"  Per-fold:        {PREV_BEST_FOLDS}")

# Reproduce the 15_bias smoothed OOF for residual analysis
oof_15_smooth = np.full(len(X), np.nan)
for tr_idx, val_idx in cv_splits:
    sm = ridge_poly_smooth(oof_s2[val_idx], ages_a[val_idx], pids_a[val_idx],
                           PREV_BEST_W_S2, degree=2, ridge_lam=0.1)
    oof_15_smooth[val_idx] = sm

resid_15 = oof_15_smooth - y.values
off_miss = X["off"].isna().values
tsd_vals = (X["age"] - X["age_at_diagnosis"]).values

def rmse_m(r): return float(np.sqrt(np.mean(r**2))) if len(r) > 0 else np.nan
def bias_m(r): return float(r.mean()) if len(r) > 0 else np.nan

print("\n  Residual by time_since_diag x off_missing (after 15_bias):")
print(f"  {'tsd_bin':12s}  {'off':8s}  {'n':>6}  {'RMSE':>7}  {'bias':>7}")
for lo, hi in [(0,5),(5,10),(10,15),(15,99)]:
    tsd_m = (tsd_vals >= lo) & (tsd_vals < hi)
    label = f"{lo}-{hi}y" if hi < 99 else f">={lo}y"
    for om_flag, om_lbl in [(False,"pres"),(True,"miss")]:
        mask = tsd_m & (off_miss == om_flag)
        r = resid_15[mask]
        print(f"  {label:12s}  off_{om_lbl} {int(mask.sum()):>6}  "
              f"{rmse_m(r):>7.3f}  {bias_m(r):>7.3f}")

hard_mask = off_miss & (tsd_vals >= 10)
r_hard = resid_15[hard_mask]
print(f"\n  Hard subgroup (off_miss + tsd>=10y): n={int(hard_mask.sum())}, "
      f"RMSE={rmse_m(r_hard):.4f}, bias={bias_m(r_hard):.4f}")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 1a: tune stage-2 hyperparameters with RandomizedSearchCV
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("STEP 1a -- Tune stage-2 hyperparameters")
print("="*60)

# Build the stage-2 training frame on all train (uses OOF as s1, fold-safe)
# For tuning: use OOF_s2_smoothed (14_quad_smooth config) as stage-1 input
print("  Building stage-2 features (all train, using s1=14_quad_smooth OOF)...")
t1 = time.time()
# Smooth oof_11 on full train (the "14_quad_smooth" smoothed OOF)
sm_full = ridge_poly_smooth(oof_11, ages_a, pids_a, CFG14["w_blend"],
                             CFG14["degree"], CFG14["ridge_lam"])
X_s2_full = add_s1_aggregates(X.copy(), sm_full)
print(f"  Feature build: {time.time()-t1:.1f}s  shape={X_s2_full.shape}")

TUNE_CACHE = C16 / "16_best_params.joblib"
if TUNE_CACHE.exists():
    best_params_s2 = joblib.load(TUNE_CACHE)
    print(f"  Loaded cached best params: {best_params_s2}")
else:
    print("  Running RandomizedSearchCV (n_iter=30)...")
    t2 = time.time()
    param_dist = {
        "learning_rate":    [0.02, 0.03, 0.05, 0.08],
        "max_leaf_nodes":   [8, 15, 23, 31],
        "min_samples_leaf": [20, 30, 50, 100],
        "l2_regularization":[0.0, 0.1, 1.0, 5.0],
        "max_iter":         [300, 500, 1000],
    }
    base_s2 = HistGradientBoostingRegressor(
        categorical_features="from_dtype",
        early_stopping=True, n_iter_no_change=20, random_state=0,
    )
    search = RandomizedSearchCV(
        base_s2, param_dist, n_iter=30,
        scoring="neg_root_mean_squared_error",
        cv=cv_splits,   # same GroupKFold splits (indices into X_s2_full)
        random_state=0, n_jobs=-1, refit=False, verbose=0,
    )
    search.fit(X_s2_full, y)
    best_params_s2 = search.best_params_
    joblib.dump(best_params_s2, TUNE_CACHE)
    print(f"  Search done in {time.time()-t2:.0f}s  best: {best_params_s2}")
    print(f"  Best CV score: {-search.best_score_:.4f}")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 1b: compute OOF with tuned stage-2, 5 seeds averaged
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("STEP 1b -- OOF for tuned stage-2 (5 seeds)")
print("="*60)

SEEDS = [0, 1, 2, 3, 4]
OOF_SEED_CACHE = [C16 / f"16_oof_seed{s}.npy" for s in SEEDS]

oof_seeds = []
for s_idx, seed in enumerate(SEEDS):
    cache_path = OOF_SEED_CACHE[s_idx]
    if cache_path.exists():
        oof_seed = np.load(cache_path)
        print(f"  seed {seed}: loaded from cache  "
              f"(raw RMSE={np.sqrt(np.mean((oof_seed-y.values)**2)):.4f})")
    else:
        print(f"  seed {seed}: computing OOF...")
        oof_seed = np.full(len(X), np.nan)
        for fi, (tr_idx, val_idx) in enumerate(cv_splits):
            t_f = time.time()
            sm_tr  = ridge_poly_smooth(oof_11[tr_idx], ages_a[tr_idx], pids_a[tr_idx],
                                       CFG14["w_blend"], CFG14["degree"], CFG14["ridge_lam"])
            sm_val = ridge_poly_smooth(oof_11[val_idx], ages_a[val_idx], pids_a[val_idx],
                                       CFG14["w_blend"], CFG14["degree"], CFG14["ridge_lam"])
            X_tr_s2  = add_s1_aggregates(X.iloc[tr_idx].copy(), sm_tr)
            X_val_s2 = add_s1_aggregates(X.iloc[val_idx].copy(), sm_val)
            m2 = HistGradientBoostingRegressor(
                categorical_features="from_dtype",
                early_stopping=True, n_iter_no_change=20,
                random_state=seed, **best_params_s2,
            )
            m2.fit(X_tr_s2, y.iloc[tr_idx])
            oof_seed[val_idx] = m2.predict(X_val_s2)
            rf = np.sqrt(np.mean((oof_seed[val_idx] - y.values[val_idx])**2))
            print(f"    fold {fi}: {rf:.4f}  ({time.time()-t_f:.1f}s)")
        np.save(cache_path, oof_seed)
    oof_seeds.append(oof_seed)

oof_s2_avg = np.mean(oof_seeds, axis=0)
rmse_avg_raw = np.sqrt(np.mean((oof_s2_avg - y.values)**2))
print(f"\n  Seed-averaged raw OOF RMSE: {rmse_avg_raw:.4f}")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 1c: re-tune smoothing on seed-averaged stage-2
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("STEP 1c -- Smoothing sweep on averaged stage-2")
print("="*60)

print(f"  {'deg':>3}  {'lam':>5}  {'w':>5}  {'mean':>8}  {'std':>8}  per-fold")
best_rmse_16 = rmse_avg_raw
best_cfg_16  = dict(degree=2, ridge_lam=0.1, w_blend=0.0)
best_folds_16 = [np.sqrt(np.mean((oof_s2_avg[val] - y.values[val])**2))
                 for _, val in cv_splits]

for deg in [1, 2]:
    for lam in [0.0, 0.1, 1.0]:
        for w in [0.25, 0.5, 0.75, 0.9, 1.0]:
            folds = []
            for tr_idx, val_idx in cv_splits:
                sm = ridge_poly_smooth(oof_s2_avg[val_idx], ages_a[val_idx],
                                       pids_a[val_idx], w, deg, lam)
                folds.append(np.sqrt(np.mean((sm - y.values[val_idx])**2)))
            m, s = np.mean(folds), np.std(folds)
            print(f"  {deg:>3}  {lam:>5.1f}  {w:>5.2f}  {m:>8.4f}  {s:>8.4f}  "
                  f"{[round(r,4) for r in folds]}")
            if m < best_rmse_16:
                best_rmse_16  = m
                best_cfg_16   = dict(degree=deg, ridge_lam=lam, w_blend=w)
                best_folds_16 = folds

print(f"\n  Best config: {best_cfg_16}  RMSE={best_rmse_16:.4f}")

# ─────────────────────────────────────────────────────────────────────────────
# Paired comparison
# ─────────────────────────────────────────────────────────────────────────────
print("\n  Paired fold comparison (16_stage2_tuned vs 15_bias):")
print(f"  {'fold':>5}  {'15_bias':>8}  {'16':>8}  {'delta':>8}")
n_better = 0
for fi, (r15, r16) in enumerate(zip(PREV_BEST_FOLDS, best_folds_16)):
    delta = r15 - r16
    n_better += int(delta > 0)
    print(f"  {fi:>5}  {r15:>8.4f}  {r16:>8.4f}  {delta:>+8.4f}  "
          f"{'BETTER' if delta > 0 else ''}")
mean_gain = np.mean([r15-r16 for r15,r16 in zip(PREV_BEST_FOLDS, best_folds_16)])
print(f"\n  Mean gain: {mean_gain:.4f}  Folds improved: {n_better}/5")
is_improvement = mean_gain > 0.03 and n_better >= 4
print(f"  Criterion (gain>0.03 AND >=4 folds): {'MET' if is_improvement else 'NOT MET'}")

# ─────────────────────────────────────────────────────────────────────────────
# OUTPUT RULE
# ─────────────────────────────────────────────────────────────────────────────
if is_improvement:
    print("\n" + "="*60)
    print("OUTPUT: writing submissions/16_stage2_tuned.csv")
    print("="*60)

    # Load cached stage-1 test predictions
    raw_test_s1 = np.load(C11 / "11_tuned_raw_test.npy")
    sm_test_s1  = ridge_poly_smooth(raw_test_s1, X_test["age"].values,
                                    X_test["patient_id"].values,
                                    CFG14["w_blend"], CFG14["degree"], CFG14["ridge_lam"])

    # Fit 5 stage-2 models on all train, average test predictions
    # Stage-1 OOF on all train (14_quad_smooth smoothed)
    sm_train_s1 = ridge_poly_smooth(oof_11, ages_a, pids_a,
                                    CFG14["w_blend"], CFG14["degree"], CFG14["ridge_lam"])
    X_train_s2  = add_s1_aggregates(X.copy(), sm_train_s1)
    X_test_s2   = add_s1_aggregates(X_test.copy(), sm_test_s1)

    test_preds_seeds = []
    for seed in SEEDS:
        model_cache = C16 / f"16_s2_final_seed{seed}.joblib"
        test_cache  = C16 / f"16_raw_test_seed{seed}.npy"
        if model_cache.exists() and test_cache.exists():
            test_preds_seeds.append(np.load(test_cache))
            print(f"  seed {seed}: loaded from cache")
        else:
            t_s = time.time()
            m2 = HistGradientBoostingRegressor(
                categorical_features="from_dtype",
                early_stopping=True, n_iter_no_change=20,
                random_state=seed, **best_params_s2,
            )
            m2.fit(X_train_s2, y)
            raw_t = m2.predict(X_test_s2)
            joblib.dump(m2, model_cache)
            np.save(test_cache, raw_t)
            test_preds_seeds.append(raw_t)
            print(f"  seed {seed}: fit in {time.time()-t_s:.1f}s  [cached]")

    raw_test_avg = np.mean(test_preds_seeds, axis=0)
    final_test   = ridge_poly_smooth(raw_test_avg, X_test["age"].values,
                                     X_test["patient_id"].values,
                                     best_cfg_16["w_blend"], best_cfg_16["degree"],
                                     best_cfg_16["ridge_lam"])
    final_test = np.clip(final_test, 0, 132)

    sub = pd.DataFrame({"target": final_test}, index=X_test.index)
    sub.index.name = "Index"
    sanity_check(sub, X_test, "16_stage2_tuned")
    out_path = SUBS_DIR / "16_stage2_tuned.csv"
    sub.to_csv(out_path)
    print(f"  Written: {out_path}")

    # Hub put
    hub_url = "PENDING"
    try:
        import skore
        from skore import login
        from sklearn.base import BaseEstimator, RegressorMixin

        class S2TunedWrapper(BaseEstimator, RegressorMixin):
            def fit(self, X, y):
                s1 = build_hgbr_tuned(**BEST_PARAMS_11, random_state=0)
                s1.fit(X, y)
                sm = ridge_poly_smooth(s1.predict(X), X["age"].values,
                                       X["patient_id"].values, CFG14["w_blend"],
                                       CFG14["degree"], CFG14["ridge_lam"])
                Xs2 = add_s1_aggregates(X.copy(), sm)
                self._models = []
                for seed in SEEDS:
                    m2 = HistGradientBoostingRegressor(
                        categorical_features="from_dtype", early_stopping=True,
                        n_iter_no_change=20, random_state=seed, **best_params_s2)
                    m2.fit(Xs2, y)
                    self._models.append(m2)
                self._s1 = s1
                return self
            def predict(self, X):
                sm = ridge_poly_smooth(self._s1.predict(X), X["age"].values,
                                       X["patient_id"].values, CFG14["w_blend"],
                                       CFG14["degree"], CFG14["ridge_lam"])
                Xs2 = add_s1_aggregates(X.copy(), sm)
                avg = np.mean([m.predict(Xs2) for m in self._models], axis=0)
                return np.clip(ridge_poly_smooth(avg, X["age"].values,
                               X["patient_id"].values, best_cfg_16["w_blend"],
                               best_cfg_16["degree"], best_cfg_16["ridge_lam"]), 0, 132)

        cfg = load_skore_credentials()
        login(mode="hub")
        project = skore.Project(name="bobathon-esilv", mode="hub", workspace=cfg["workspace"])
        report  = skore.evaluate(S2TunedWrapper(), X, y, splitter=cv_splits)
        project.put("16_stage2_tuned", report)
        workspace = cfg["workspace"]
        hub_url   = f"https://skore.probabl.ai/{workspace}/bobathon-esilv/cross-validations"
        print(f"  Hub report 16_stage2_tuned pushed.")
    except Exception as e:
        print(f"  HUB PUT PENDING: 16_stage2_tuned  ({e})")

    print(f"\nREADY TO SUBMIT: submissions/16_stage2_tuned.csv | "
          f"CV RMSE {best_rmse_16:.4f} +/- {np.std(best_folds_16):.4f} | "
          f"hub URL: {hub_url}")
else:
    print(f"\nNo improvement over 15_bias (CV {PREV_BEST_CV:.4f}).")
    print(f"Keeping 15_bias as best.")

print(f"\nTotal elapsed: {time.time()-t0:.0f}s")
print("STEPS 0+1 DONE.")
