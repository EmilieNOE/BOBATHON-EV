"""step5_20_diverse.py -- STEP 5: stage-1 diversity (key 20_diverse).

Instead of a single stage-1 HGBR, use THREE diverse stage-1 learners
whose OOF predictions feed stage-2 as additional features:
  L1: 11_tuned HGBR (lr=0.02, 1000 iter) -- cached OOF
  L2: ExtraTreesRegressor (500 trees, max_features=0.5, min_samples_leaf=5)
  L3: HGBR variant (lr=0.05, 500 iter, 63 leaves) -- complementary settings

All three are fit within the CV training fold (nested).
Stage-2 sees: s1_L1_smooth, s1_L2_smooth, s1_L3_smooth, plus the 17_stage2_feats
input curve features and per-patient aggregates.

Also test plain average: (L1+L2+L3)/3 smoothed, to check if diversity alone helps
without the stage-2 overhead.

Criterion: mean gain > 0.03 AND >= 4/5 folds improve vs 17_stage2_feats.

Hub key: 20_diverse
PYTHONUTF8=1. ASCII-only prints.
"""

import sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor, ExtraTreesRegressor
from sklearn.model_selection import GroupKFold

from parkinsons import PROJECT_ROOT
from parkinsons.data import load_dataset, load_test_dataset
from parkinsons.hub import load_skore_credentials
from shared import add_s1_aggregates, ridge_poly_smooth, sanity_check
import step2_17_stage2_feats as _s17
InputCurveTransformer = _s17.InputCurveTransformer
build_s2_features     = _s17.build_s2_features

DATA_DIR = PROJECT_ROOT / "data"
SUBS_DIR = PROJECT_ROOT / "submissions"
C11  = PROJECT_ROOT / "scratch" / "13_cache"
C17  = PROJECT_ROOT / "scratch" / "17_cache"
C20  = PROJECT_ROOT / "scratch" / "20_cache"
SUBS_DIR.mkdir(exist_ok=True)
C20.mkdir(parents=True, exist_ok=True)

CFG14     = dict(degree=2, ridge_lam=0.1, w_blend=0.9)
S2_PARAMS = dict(learning_rate=0.02, max_iter=300, max_leaf_nodes=31,
                 min_samples_leaf=50, l2_regularization=0.0)
SEEDS     = [0, 1, 2, 3, 4]
PREV_BEST_CV    = 3.9949
PREV_BEST_FOLDS = [4.2602, 4.0026, 3.9085, 3.8914, 3.9119]

t0 = time.time()

X, y   = load_dataset(data_dir=DATA_DIR)
X_test = load_test_dataset(data_dir=DATA_DIR)
groups    = X["patient_id"]
cv_splits = list(GroupKFold(n_splits=5).split(X, y, groups=groups))
ages_a    = X["age"].values
pids_a    = X["patient_id"].values
y_vals    = y.values.astype(float)

oof_11 = np.load(C11 / "11_oof_raw.npy")
oof17_seeds = [np.load(C17 / f"17_oof_seed{s}.npy") for s in SEEDS]
oof17_avg   = np.mean(oof17_seeds, axis=0)
print(f"Data + caches loaded ({time.time()-t0:.1f}s)")

# ── L2: ExtraTrees OOF ────────────────────────────────────────────────────────
ET_CACHE = C20 / "20_et_oof.npy"
if ET_CACHE.exists():
    oof_et = np.load(ET_CACHE)
    print(f"ExtraTrees OOF loaded. RMSE={np.sqrt(np.mean((oof_et-y_vals)**2)):.4f}")
else:
    print("Computing ExtraTrees OOF...")
    oof_et = np.full(len(X), np.nan)
    # ET needs numeric features; drop patient_id, cast categoricals
    def prep_X_et(X_fr):
        Xn = X_fr.copy()
        for col in ("cohort", "gene", "rater_id"):
            if col in Xn.columns:
                Xn[col] = Xn[col].astype("category").cat.codes.astype(float)
        Xn = Xn.drop(columns=["patient_id"], errors="ignore")
        return Xn.values

    for fi, (tr_idx, val_idx) in enumerate(cv_splits):
        t_f = time.time()
        et = ExtraTreesRegressor(
            n_estimators=500, max_features=0.5,
            min_samples_leaf=5, random_state=0, n_jobs=-1,
        )
        Xtr_et = prep_X_et(X.iloc[tr_idx])
        Xval_et = prep_X_et(X.iloc[val_idx])
        # Fill NaN with median
        med = np.nanmedian(Xtr_et, axis=0)
        for j in range(Xtr_et.shape[1]):
            Xtr_et[np.isnan(Xtr_et[:, j]), j] = med[j]
            Xval_et[np.isnan(Xval_et[:, j]), j] = med[j]
        et.fit(Xtr_et, y_vals[tr_idx])
        oof_et[val_idx] = et.predict(Xval_et)
        rf = np.sqrt(np.mean((oof_et[val_idx] - y_vals[val_idx])**2))
        print(f"  fold {fi}: RMSE={rf:.4f}  ({time.time()-t_f:.1f}s)")
    np.save(ET_CACHE, oof_et)
    print(f"  ExtraTrees OOF cached. Overall RMSE={np.sqrt(np.mean((oof_et-y_vals)**2)):.4f}")

# ── L3: HGBR variant OOF ──────────────────────────────────────────────────────
L3_CACHE = C20 / "20_hgbr_l3_oof.npy"
if L3_CACHE.exists():
    oof_l3 = np.load(L3_CACHE)
    print(f"L3 HGBR OOF loaded. RMSE={np.sqrt(np.mean((oof_l3-y_vals)**2)):.4f}")
else:
    print("Computing L3 HGBR OOF (lr=0.05, 500 iter, 63 leaves)...")
    oof_l3 = np.full(len(X), np.nan)
    for fi, (tr_idx, val_idx) in enumerate(cv_splits):
        t_f = time.time()
        l3 = HistGradientBoostingRegressor(
            learning_rate=0.05, max_iter=500, max_leaf_nodes=63,
            min_samples_leaf=20, l2_regularization=0.0,
            categorical_features="from_dtype",
            early_stopping=True, n_iter_no_change=20,
            random_state=42,
        )
        X_tr_c = X.iloc[tr_idx].copy()
        X_val_c = X.iloc[val_idx].copy()
        for col in ("cohort", "gene", "rater_id"):
            if col in X_tr_c.columns:
                X_tr_c[col]  = X_tr_c[col].astype("category")
                X_val_c[col] = X_val_c[col].astype("category")
        l3.fit(X_tr_c.drop(columns=["patient_id"]), y_vals[tr_idx])
        oof_l3[val_idx] = l3.predict(X_val_c.drop(columns=["patient_id"]))
        rf = np.sqrt(np.mean((oof_l3[val_idx] - y_vals[val_idx])**2))
        print(f"  fold {fi}: RMSE={rf:.4f}  ({time.time()-t_f:.1f}s)")
    np.save(L3_CACHE, oof_l3)
    print(f"  L3 OOF cached. Overall RMSE={np.sqrt(np.mean((oof_l3-y_vals)**2)):.4f}")

# ── Plain average of L1+L2+L3 (no stage-2) ───────────────────────────────────
avg3_oof = (oof_11 + oof_et + oof_l3) / 3.0
print("\nPlain average (L1+L2+L3) smoothing sweep:")
best_avg3_rmse = 1e9
for deg in [1, 2]:
    for lam in [0.1, 1.0]:
        for w in [0.0, 0.25, 0.5, 0.75]:
            folds = [np.sqrt(np.mean((
                ridge_poly_smooth(avg3_oof[val], ages_a[val], pids_a[val], w, deg, lam)
                - y_vals[val])**2)) for _, val in cv_splits]
            m = np.mean(folds)
            if m < best_avg3_rmse:
                best_avg3_rmse = m
print(f"Best plain-avg RMSE: {best_avg3_rmse:.4f}  (vs 17: {PREV_BEST_CV:.4f})")

# ── Multi-learner stage-2: add L2/L3 smoothed preds as extra s2 features ─────
print("\nComputing multi-learner stage-2 OOF (seed 0 only for speed)...")

def build_s2_features_multi(X_frame, s1_L1, s1_L2, s1_L3, curve_transformer):
    """S2 features = 17_feats + smoothed L2 pred + smoothed L3 pred."""
    X_s2 = build_s2_features(X_frame, s1_L1, curve_transformer)
    X_s2["s1_l2_smooth"] = s1_L2
    X_s2["s1_l3_smooth"] = s1_L3
    return X_s2

MULTI_CACHE = C20 / "20_multi_oof_seed0.npy"
if MULTI_CACHE.exists():
    oof_multi = np.load(MULTI_CACHE)
    print(f"  Multi-learner OOF loaded. RMSE={np.sqrt(np.mean((oof_multi-y_vals)**2)):.4f}")
else:
    oof_multi = np.full(len(X), np.nan)
    seed = 0
    for fi, (tr_idx, val_idx) in enumerate(cv_splits):
        t_f = time.time()
        sm_l1_tr  = ridge_poly_smooth(oof_11[tr_idx], ages_a[tr_idx], pids_a[tr_idx],
                                      CFG14["w_blend"], CFG14["degree"], CFG14["ridge_lam"])
        sm_l1_val = ridge_poly_smooth(oof_11[val_idx], ages_a[val_idx], pids_a[val_idx],
                                      CFG14["w_blend"], CFG14["degree"], CFG14["ridge_lam"])
        sm_et_tr  = ridge_poly_smooth(oof_et[tr_idx], ages_a[tr_idx], pids_a[tr_idx], 0.5, 1, 1.0)
        sm_et_val = ridge_poly_smooth(oof_et[val_idx], ages_a[val_idx], pids_a[val_idx], 0.5, 1, 1.0)
        sm_l3_tr  = ridge_poly_smooth(oof_l3[tr_idx], ages_a[tr_idx], pids_a[tr_idx],
                                      CFG14["w_blend"], CFG14["degree"], CFG14["ridge_lam"])
        sm_l3_val = ridge_poly_smooth(oof_l3[val_idx], ages_a[val_idx], pids_a[val_idx],
                                      CFG14["w_blend"], CFG14["degree"], CFG14["ridge_lam"])

        ct = InputCurveTransformer(lam_curve=1.0, lam_off_hat=1.0)
        ct.fit(X.iloc[tr_idx])

        X_tr_s2  = build_s2_features_multi(X.iloc[tr_idx].copy(),
                                            sm_l1_tr, sm_et_tr, sm_l3_tr, ct)
        X_val_s2 = build_s2_features_multi(X.iloc[val_idx].copy(),
                                            sm_l1_val, sm_et_val, sm_l3_val, ct)

        m2 = HistGradientBoostingRegressor(
            categorical_features="from_dtype",
            early_stopping=True, n_iter_no_change=20,
            random_state=seed, **S2_PARAMS,
        )
        m2.fit(X_tr_s2, y.iloc[tr_idx])
        oof_multi[val_idx] = m2.predict(X_val_s2)
        rf = np.sqrt(np.mean((oof_multi[val_idx] - y_vals[val_idx])**2))
        print(f"  fold {fi}: RMSE={rf:.4f}  ({time.time()-t_f:.1f}s)")
    np.save(MULTI_CACHE, oof_multi)

# ── Smoothing sweep for multi-learner ─────────────────────────────────────────
print("\nMulti-learner s2 smoothing sweep:")
best_rmse_20  = PREV_BEST_CV
best_cfg_20   = None
best_folds_20 = PREV_BEST_FOLDS[:]

for deg in [1, 2]:
    for lam in [0.1, 1.0]:
        for w in [0.0, 0.25, 0.5, 0.75]:
            folds = [np.sqrt(np.mean((
                ridge_poly_smooth(oof_multi[val], ages_a[val], pids_a[val], w, deg, lam)
                - y_vals[val])**2)) for _, val in cv_splits]
            m, s = np.mean(folds), np.std(folds)
            marker = " *" if m < best_rmse_20 else ""
            print(f"  deg={deg} lam={lam:.1f} w={w:.2f}  {m:.4f} +/- {s:.4f}{marker}")
            if m < best_rmse_20:
                best_rmse_20  = m
                best_cfg_20   = dict(degree=deg, ridge_lam=lam, w_blend=w)
                best_folds_20 = folds[:]

if best_cfg_20 is None:
    best_cfg_20 = dict(degree=1, ridge_lam=1.0, w_blend=0.25)
print(f"\nBest multi-learner config: {best_cfg_20}  RMSE={best_rmse_20:.4f}")

# ── Paired comparison ─────────────────────────────────────────────────────────
print("\nPaired fold comparison (20_diverse vs 17_stage2_feats):")
print(f"  {'fold':>5}  {'17':>8}  {'20':>8}  {'delta':>8}")
n_better = 0
for fi, (r17, r20) in enumerate(zip(PREV_BEST_FOLDS, best_folds_20)):
    delta = r17 - r20
    n_better += int(delta > 0)
    print(f"  {fi:>5}  {r17:>8.4f}  {r20:>8.4f}  {delta:>+8.4f}  "
          f"{'BETTER' if delta > 0 else ''}")
mean_gain = np.mean([r17 - r20 for r17, r20 in zip(PREV_BEST_FOLDS, best_folds_20)])
print(f"\nMean gain: {mean_gain:.4f}  Folds improved: {n_better}/5")
is_improvement = mean_gain > 0.03 and n_better >= 4
print(f"Criterion (gain>0.03 AND >=4 folds): {'MET' if is_improvement else 'NOT MET'}")

# ── OUTPUT RULE ───────────────────────────────────────────────────────────────
if is_improvement:
    print("\n" + "="*60)
    print("OUTPUT: writing submissions/20_diverse.csv")
    print("="*60)

    # Fit multi-learner stage-2 on all training data (seed 0 only for now)
    FINAL_MULTI_CACHE = C20 / "20_multi_final_test.npy"
    if FINAL_MULTI_CACHE.exists():
        test_preds_multi = np.load(FINAL_MULTI_CACHE)
        print("  Multi-learner test preds loaded from cache.")
    else:
        ct_final = joblib.load(C17 / "17_ct_final.joblib")
        raw_test_s1 = np.load(C11 / "11_tuned_raw_test.npy")

        sm_train_l1 = ridge_poly_smooth(oof_11, ages_a, pids_a,
                                        CFG14["w_blend"], CFG14["degree"], CFG14["ridge_lam"])
        sm_test_l1  = ridge_poly_smooth(raw_test_s1, X_test["age"].values,
                                        X_test["patient_id"].values,
                                        CFG14["w_blend"], CFG14["degree"], CFG14["ridge_lam"])

        # L2 (ET) on all train + predict test
        def prep_X_et(X_fr):
            Xn = X_fr.copy()
            for col in ("cohort", "gene", "rater_id"):
                if col in Xn.columns:
                    Xn[col] = Xn[col].astype("category").cat.codes.astype(float)
            return Xn.drop(columns=["patient_id"], errors="ignore").values

        Xtr_et = prep_X_et(X)
        Xtest_et = prep_X_et(X_test)
        med_et = np.nanmedian(Xtr_et, axis=0)
        for j in range(Xtr_et.shape[1]):
            Xtr_et[np.isnan(Xtr_et[:, j]), j] = med_et[j]
            Xtest_et[np.isnan(Xtest_et[:, j]), j] = med_et[j]
        et_final = ExtraTreesRegressor(
            n_estimators=500, max_features=0.5, min_samples_leaf=5,
            random_state=0, n_jobs=-1).fit(Xtr_et, y_vals)
        et_test_raw = et_final.predict(Xtest_et)
        sm_test_l2 = ridge_poly_smooth(et_test_raw, X_test["age"].values,
                                       X_test["patient_id"].values, 0.5, 1, 1.0)
        sm_train_l2 = ridge_poly_smooth(oof_et, ages_a, pids_a, 0.5, 1, 1.0)

        # L3 on all train + predict test
        X_l3_tr = X.copy()
        X_l3_te = X_test.copy()
        for col in ("cohort", "gene", "rater_id"):
            if col in X_l3_tr.columns:
                X_l3_tr[col] = X_l3_tr[col].astype("category")
                X_l3_te[col] = X_l3_te[col].astype("category")
        l3_final = HistGradientBoostingRegressor(
            learning_rate=0.05, max_iter=500, max_leaf_nodes=63,
            min_samples_leaf=20, l2_regularization=0.0,
            categorical_features="from_dtype",
            early_stopping=False, random_state=42,
        ).fit(X_l3_tr.drop(columns=["patient_id"]), y_vals)
        l3_test_raw = l3_final.predict(X_l3_te.drop(columns=["patient_id"]))
        sm_test_l3  = ridge_poly_smooth(l3_test_raw, X_test["age"].values,
                                        X_test["patient_id"].values,
                                        CFG14["w_blend"], CFG14["degree"], CFG14["ridge_lam"])
        sm_train_l3 = ridge_poly_smooth(oof_l3, ages_a, pids_a,
                                        CFG14["w_blend"], CFG14["degree"], CFG14["ridge_lam"])

        X_train_s2m = build_s2_features_multi(X.copy(),      sm_train_l1, sm_train_l2, sm_train_l3, ct_final)
        X_test_s2m  = build_s2_features_multi(X_test.copy(), sm_test_l1,  sm_test_l2,  sm_test_l3,  ct_final)

        m2_final = HistGradientBoostingRegressor(
            categorical_features="from_dtype", early_stopping=False,
            random_state=0, **S2_PARAMS,
        ).fit(X_train_s2m, y_vals)
        test_preds_multi = m2_final.predict(X_test_s2m)
        np.save(FINAL_MULTI_CACHE, test_preds_multi)
        print("  Multi-learner final model fit and cached.")

    final_test = ridge_poly_smooth(test_preds_multi, X_test["age"].values,
                                   X_test["patient_id"].values,
                                   best_cfg_20["w_blend"], best_cfg_20["degree"],
                                   best_cfg_20["ridge_lam"])
    final_test = np.clip(final_test, 0, 132)

    sub = pd.DataFrame({"target": final_test}, index=X_test.index)
    sub.index.name = "Index"
    sanity_check(sub, X_test, "20_diverse")
    out_path = SUBS_DIR / "20_diverse.csv"
    sub.to_csv(out_path)
    print(f"\nWritten: {out_path}")

    hub_url = "PENDING"
    try:
        import skore
        from skore import login

        cfg = load_skore_credentials()
        login(mode="hub")
        project = skore.Project(name="bobathon-esilv", mode="hub", workspace=cfg["workspace"])
        fold_report = {
            "key": "20_diverse",
            "cv_rmse_mean": float(best_rmse_20),
            "cv_rmse_std":  float(np.std(best_folds_20)),
            "folds": [float(f) for f in best_folds_20],
            "best_cfg": best_cfg_20,
        }
        project.put("20_diverse_meta", fold_report)
        workspace = cfg["workspace"]
        hub_url   = f"https://skore.probabl.ai/{workspace}/bobathon-esilv/cross-validations"
        print(f"Hub note pushed.")
    except Exception as e:
        print(f"HUB PUT PENDING: 20_diverse  ({e})")

    std_20 = float(np.std(best_folds_20))
    print(f"\nREADY TO SUBMIT: submissions/20_diverse.csv | "
          f"CV RMSE {best_rmse_20:.4f} +/- {std_20:.4f} | URL: {hub_url}")
else:
    print(f"\nNo improvement over 17 (CV {PREV_BEST_CV:.4f}). No CSV written.")
    print(f"20_diverse best CV RMSE: {best_rmse_20:.4f}")

print(f"\nTotal elapsed: {time.time()-t0:.0f}s")
print("STEP 5 (20_diverse) DONE.")
