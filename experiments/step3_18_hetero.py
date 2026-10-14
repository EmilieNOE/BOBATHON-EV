"""step3_18_hetero.py -- STEP 3: heteroscedastic smoothing (key 18_hetero).

Instead of uniform weights in ridge_poly_smooth, weight each visit by
1 / sigma_hat^2 where sigma_hat is predicted by a small HGBR trained on
OOF squared residuals of the TRAINING fold only.

This is a "meta-error" model: it learns which visits the stage-2 model
tends to get wrong (high LEDD, missing off, advanced disease) and down-
weights them in the within-patient polynomial trend.

Architecture:
  Stage-1  : 11_tuned HGBR (cached OOF)
  Stage-2  : 17_stage2_feats (cached OOF, 5 seeds averaged)
  Hetero   : HGBR(max_depth=4, max_iter=200) fitted on |residual_train|^2
              features: s1_preds, all input cols, on/off/ledd missingness flags
  Smoothing: ridge_poly_smooth with per-visit weights = 1/max(sigma2, eps)

Criterion (keep only if): mean gain > 0.03 AND >= 4/5 folds improve.

Hub key: 18_hetero
PYTHONUTF8=1. ASCII-only prints. Cache all OOF arrays.
"""

import sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

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
# Import the exact transformer from 17 (same file imported by name)
import step2_17_stage2_feats as _s17
InputCurveTransformer = _s17.InputCurveTransformer
build_s2_features     = _s17.build_s2_features

DATA_DIR = PROJECT_ROOT / "data"
SUBS_DIR = PROJECT_ROOT / "submissions"
C11  = PROJECT_ROOT / "scratch" / "13_cache"
C17  = PROJECT_ROOT / "scratch" / "17_cache"
C18  = PROJECT_ROOT / "scratch" / "18_cache"
SUBS_DIR.mkdir(exist_ok=True)
C18.mkdir(parents=True, exist_ok=True)

CFG14       = dict(degree=2, ridge_lam=0.1, w_blend=0.9)
S2_PARAMS   = dict(learning_rate=0.02, max_iter=300, max_leaf_nodes=31,
                   min_samples_leaf=50, l2_regularization=0.0)
SEEDS       = [0, 1, 2, 3, 4]
# Reference: 17_stage2_feats best smoothing was deg=1, lam=1.0, w=0.25
PREV_BEST_CV    = 3.9949
PREV_BEST_FOLDS = [4.2602, 4.0026, 3.9085, 3.8914, 3.9119]

t0 = time.time()

# ── load data + cached artefacts ──────────────────────────────────────────────
X, y   = load_dataset(data_dir=DATA_DIR)
X_test = load_test_dataset(data_dir=DATA_DIR)
groups   = X["patient_id"]
cv_splits = list(GroupKFold(n_splits=5).split(X, y, groups=groups))
ages_a    = X["age"].values
pids_a    = X["patient_id"].values
y_vals    = y.values.astype(float)

oof_11 = np.load(C11 / "11_oof_raw.npy")

# Load 17_stage2_feats OOF (seed-averaged)
oof17_seeds = [np.load(C17 / f"17_oof_seed{s}.npy") for s in SEEDS]
oof17_avg   = np.mean(oof17_seeds, axis=0)
print(f"Data + caches loaded ({time.time()-t0:.1f}s)")
print(f"17 seed-avg raw OOF RMSE: {np.sqrt(np.mean((oof17_avg - y_vals)**2)):.4f}")

# ── build heteroscedastic weights via nested CV ────────────────────────────────
# For each training fold: fit HGBR on squared OOF residuals of TRAINING rows,
# predict expected squared error for VALIDATION rows.
# The error model's features are: the stage-2 features (all of them).

print("\nComputing heteroscedastic error weights (nested CV)...")
SIGMA2_CACHE = C18 / "18_sigma2_oof.npy"
if SIGMA2_CACHE.exists():
    sigma2_oof = np.load(SIGMA2_CACHE)
    print(f"  Loaded from cache.  median sigma2={np.median(sigma2_oof):.4f}")
else:
    sigma2_oof = np.full(len(X), np.nan)
    for fi, (tr_idx, val_idx) in enumerate(cv_splits):
        t_f = time.time()
        # Stage-1 smoothed for this fold
        sm_tr  = ridge_poly_smooth(oof_11[tr_idx], ages_a[tr_idx], pids_a[tr_idx],
                                   CFG14["w_blend"], CFG14["degree"], CFG14["ridge_lam"])
        sm_val = ridge_poly_smooth(oof_11[val_idx], ages_a[val_idx], pids_a[val_idx],
                                   CFG14["w_blend"], CFG14["degree"], CFG14["ridge_lam"])

        # Build S2 features (fit curve transformer on train fold)
        ct = InputCurveTransformer(lam_curve=1.0, lam_off_hat=1.0)
        ct.fit(X.iloc[tr_idx])
        X_tr_s2  = build_s2_features(X.iloc[tr_idx].copy(),  sm_tr,  ct)
        X_val_s2 = build_s2_features(X.iloc[val_idx].copy(), sm_val, ct)

        # Average the 5 seed OOF preds on the training fold to get residuals
        train_preds_17 = np.mean([oof17_seeds[s][tr_idx] for s in range(len(SEEDS))], axis=0)
        resid2_tr = (train_preds_17 - y_vals[tr_idx]) ** 2

        # Fit error model
        err_model = HistGradientBoostingRegressor(
            max_iter=200, max_leaf_nodes=15, min_samples_leaf=30,
            learning_rate=0.05, random_state=0,
        )
        err_model.fit(X_tr_s2, resid2_tr)
        sigma2_oof[val_idx] = err_model.predict(X_val_s2)
        print(f"  fold {fi}: median sigma2_val={np.median(sigma2_oof[val_idx]):.3f}  ({time.time()-t_f:.1f}s)")

    np.save(SIGMA2_CACHE, sigma2_oof)
    print(f"  Saved to {SIGMA2_CACHE}")

# Clip negative/zero predictions (error model can predict negatives)
EPS = 0.5
sigma2_oof = np.clip(sigma2_oof, EPS, None)
visit_weights = 1.0 / sigma2_oof
print(f"  weight range: [{visit_weights.min():.3f}, {visit_weights.max():.3f}]")

# ── Smoothing sweep with heteroscedastic weights ───────────────────────────────
print("\nHetero-weighted smoothing sweep:")
print(f"  {'deg':>3}  {'lam':>5}  {'w':>5}  {'mean':>8}  {'std':>8}")

best_rmse_18  = PREV_BEST_CV
best_cfg_18   = None
best_folds_18 = PREV_BEST_FOLDS[:]

for deg in [1, 2]:
    for lam in [0.0, 0.1, 1.0]:
        for w in [0.0, 0.25, 0.5, 0.75, 0.9]:
            folds = []
            for _, val_idx in cv_splits:
                vw = visit_weights[val_idx]
                sm = ridge_poly_smooth(oof17_avg[val_idx], ages_a[val_idx],
                                       pids_a[val_idx], w, deg, lam,
                                       visit_w=vw)
                folds.append(np.sqrt(np.mean((sm - y_vals[val_idx])**2)))
            m, s = np.mean(folds), np.std(folds)
            marker = " *" if m < best_rmse_18 else ""
            print(f"  {deg:>3}  {lam:>5.1f}  {w:>5.2f}  {m:>8.4f}  {s:>8.4f}{marker}")
            if m < best_rmse_18:
                best_rmse_18  = m
                best_cfg_18   = dict(degree=deg, ridge_lam=lam, w_blend=w)
                best_folds_18 = folds[:]

if best_cfg_18 is None:
    best_cfg_18 = dict(degree=1, ridge_lam=1.0, w_blend=0.25)
    print(f"\nNo improvement. Best config unchanged: {best_cfg_18}")
else:
    print(f"\nBest hetero config: {best_cfg_18}  RMSE={best_rmse_18:.4f}")

# Also try uniform weights for comparison (sanity check baseline for this code)
print("\nUniform weights sweep (deg=1, lam=1.0, w sweep):")
for w in [0.0, 0.25, 0.5, 0.75]:
    folds = [np.sqrt(np.mean((
        ridge_poly_smooth(oof17_avg[val], ages_a[val], pids_a[val], w, 1, 1.0)
        - y_vals[val])**2)) for _, val in cv_splits]
    print(f"  w={w:.2f}  {np.mean(folds):.4f} +/- {np.std(folds):.4f}")

# ── Paired comparison vs 17 ────────────────────────────────────────────────────
print("\nPaired fold comparison (18_hetero vs 17_stage2_feats):")
print(f"  {'fold':>5}  {'17':>8}  {'18':>8}  {'delta':>8}")
n_better = 0
for fi, (r17, r18) in enumerate(zip(PREV_BEST_FOLDS, best_folds_18)):
    delta = r17 - r18
    n_better += int(delta > 0)
    print(f"  {fi:>5}  {r17:>8.4f}  {r18:>8.4f}  {delta:>+8.4f}  "
          f"{'BETTER' if delta > 0 else ''}")
mean_gain = np.mean([r17 - r18 for r17, r18 in zip(PREV_BEST_FOLDS, best_folds_18)])
print(f"\nMean gain: {mean_gain:.4f}  Folds improved: {n_better}/5")
is_improvement = mean_gain > 0.03 and n_better >= 4
print(f"Criterion (gain>0.03 AND >=4 folds): {'MET' if is_improvement else 'NOT MET'}")

# ── OUTPUT RULE ───────────────────────────────────────────────────────────────
if is_improvement:
    print("\n" + "="*60)
    print("OUTPUT: writing submissions/18_hetero.csv")
    print("="*60)

    # Need final sigma2 for test set: fit error model on ALL train
    ct_final = joblib.load(C17 / "17_ct_final.joblib")
    oof_11_raw = np.load(C11 / "11_oof_raw.npy")
    raw_test_s1 = np.load(C11 / "11_tuned_raw_test.npy")

    sm_train_s1 = ridge_poly_smooth(oof_11, ages_a, pids_a,
                                    CFG14["w_blend"], CFG14["degree"], CFG14["ridge_lam"])
    sm_test_s1  = ridge_poly_smooth(raw_test_s1, X_test["age"].values,
                                    X_test["patient_id"].values,
                                    CFG14["w_blend"], CFG14["degree"], CFG14["ridge_lam"])

    X_train_s2 = build_s2_features(X.copy(), sm_train_s1, ct_final)
    X_test_s2  = build_s2_features(X_test.copy(), sm_test_s1, ct_final)

    # Fit error model on all training data
    resid2_all = (oof17_avg - y_vals) ** 2
    err_final = HistGradientBoostingRegressor(
        max_iter=200, max_leaf_nodes=15, min_samples_leaf=30,
        learning_rate=0.05, random_state=0,
    )
    err_final.fit(X_train_s2, resid2_all)
    sigma2_test = np.clip(err_final.predict(X_test_s2), EPS, None)
    vw_test = 1.0 / sigma2_test

    # Load cached final test predictions from seed 0-4
    test_raw = [np.load(C17 / f"17_raw_test_seed{s}.npy") for s in SEEDS]
    raw_test_avg = np.mean(test_raw, axis=0)

    final_test = ridge_poly_smooth(
        raw_test_avg, X_test["age"].values, X_test["patient_id"].values,
        best_cfg_18["w_blend"], best_cfg_18["degree"], best_cfg_18["ridge_lam"],
        visit_w=vw_test,
    )
    final_test = np.clip(final_test, 0, 132)

    sub = pd.DataFrame({"target": final_test}, index=X_test.index)
    sub.index.name = "Index"
    sanity_check(sub, X_test, "18_hetero")
    out_path = SUBS_DIR / "18_hetero.csv"
    sub.to_csv(out_path)
    print(f"\nWritten: {out_path}")

    hub_url = "PENDING"
    try:
        import skore
        from skore import login

        cfg = load_skore_credentials()
        login(mode="hub")
        project = skore.Project(name="bobathon-esilv", mode="hub", workspace=cfg["workspace"])

        class S18Wrapper:
            """Minimal sklearn-compatible wrapper for skore.evaluate."""
            def fit(self, X, y): return self
            def predict(self, X): return np.zeros(len(X))

        # We just store the per-fold RMSE in the project as a note
        import json
        fold_report = {
            "key": "18_hetero",
            "cv_rmse_mean": float(best_rmse_18),
            "cv_rmse_std":  float(np.std(best_folds_18)),
            "folds": [float(f) for f in best_folds_18],
            "best_cfg": best_cfg_18,
        }
        project.put("18_hetero_meta", fold_report)
        workspace = cfg["workspace"]
        hub_url   = f"https://skore.probabl.ai/{workspace}/bobathon-esilv/cross-validations"
        print(f"Hub note pushed.")
    except Exception as e:
        print(f"HUB PUT PENDING: 18_hetero  ({e})")

    std_18 = float(np.std(best_folds_18))
    print(f"\nREADY TO SUBMIT: submissions/18_hetero.csv | "
          f"CV RMSE {best_rmse_18:.4f} +/- {std_18:.4f} | URL: {hub_url}")
else:
    print(f"\nNo improvement over 17 (CV {PREV_BEST_CV:.4f}). No CSV written.")
    print(f"18_hetero CV RMSE: {best_rmse_18:.4f} +/- {np.std(best_folds_18):.4f}")
    print(f"Best hetero cfg: {best_cfg_18}")

print(f"\nTotal elapsed: {time.time()-t0:.0f}s")
print("STEP 3 (18_hetero) DONE.")
