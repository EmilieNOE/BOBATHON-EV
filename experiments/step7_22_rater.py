"""step7_22_rater.py -- STEP 7: rater-aware stage-2 features (key 22_rater).

Rater bias is additive: per-rater mean spread is +-2.6 pts (9% of variance).
This experiment adds rater-level and visit-level features to stage-2 that are
computed safely inside each CV fold (no test-target leakage):

New stage-2 features (all fold-safe):
  rater_bias_hat  -- per-rater mean(target - s1_smooth) from TRAINING fold only;
                     fallback 0 for unseen raters
  rater_n_visits  -- number of training visits this rater administered
  visit_rank      -- 0-based ordinal rank of this visit within the patient
                     (sorted by age); rank 0 = earliest known visit
  off_change_rate -- off_slope minus s1_slope: deviation of off input curve
                     from stage-1 prediction slope; proxy for disease velocity
  patient_age_std -- std of age across patient's visits (visit spread in years)

These are combined with the existing 17_stage2_feats features.

Architecture unchanged from 17_stage2_feats otherwise:
  Stage-1: 11_tuned HGBR (cached OOF)
  Stage-2: HGBR (lr=0.02, 300 iter, 31 leaves, min_leaf=50) x 5 seeds
  Smoothing: best config from 17 (deg=1, lam=1.0, w=0.25)

Criterion: mean gain > 0.03 AND >= 4/5 folds improve vs 17_stage2_feats.

Hub key: 22_rater
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
from shared import add_s1_aggregates, ridge_poly_smooth, sanity_check
import step2_17_stage2_feats as _s17
InputCurveTransformer = _s17.InputCurveTransformer
build_s2_features     = _s17.build_s2_features

DATA_DIR = PROJECT_ROOT / "data"
SUBS_DIR = PROJECT_ROOT / "submissions"
C11  = PROJECT_ROOT / "scratch" / "13_cache"
C17  = PROJECT_ROOT / "scratch" / "17_cache"
C22  = PROJECT_ROOT / "scratch" / "22_cache"
SUBS_DIR.mkdir(exist_ok=True)
C22.mkdir(parents=True, exist_ok=True)

CFG14     = dict(degree=2, ridge_lam=0.1, w_blend=0.9)
S2_PARAMS = dict(learning_rate=0.02, max_iter=300, max_leaf_nodes=31,
                 min_samples_leaf=50, l2_regularization=0.0)
CFG_SMOOTH = dict(degree=1, ridge_lam=1.0, w_blend=0.25)   # best from 17
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
print(f"Data + caches loaded ({time.time()-t0:.1f}s)")


# ── helpers ───────────────────────────────────────────────────────────────────

def add_rater_features(X_frame: pd.DataFrame,
                       s1_smooth: np.ndarray,
                       y_tr_vals: np.ndarray | None,
                       rater_bias_map: dict | None,
                       rater_n_map: dict | None) -> pd.DataFrame:
    """Add rater_bias_hat, rater_n_visits, visit_rank, patient_age_std.

    Parameters
    ----------
    X_frame     : frame to augment (may be train or val/test)
    s1_smooth   : stage-1 smoothed predictions aligned to X_frame positions
    y_tr_vals   : training targets (None for val/test -- not used then)
    rater_bias_map : pre-fitted {rater_id: mean_residual}; None = fit from frame
    rater_n_map    : pre-fitted {rater_id: n_visits};      None = fit from frame
    """
    X_out = X_frame.copy()
    n = len(X_out)

    # ── per-rater bias ────────────────────────────────────────────────────────
    if rater_bias_map is None:
        # Build from this frame (training path): residual = target - s1_smooth
        assert y_tr_vals is not None, "need y for fitting rater bias"
        resid = y_tr_vals - s1_smooth
        rater_ids = X_out["rater_id"].values
        rater_bias_map = {}
        rater_n_map    = {}
        for rid in np.unique(rater_ids):
            mask = rater_ids == rid
            rater_bias_map[rid] = float(np.mean(resid[mask]))
            rater_n_map[rid]    = int(np.sum(mask))

    rater_bias = np.array([rater_bias_map.get(r, 0.0)
                           for r in X_out["rater_id"].values], dtype=float)
    rater_n    = np.array([float(rater_n_map.get(r, 0))
                           for r in X_out["rater_id"].values], dtype=float)

    # ── visit_rank (0-based ordinal within patient, sorted by age) ────────────
    visit_rank  = np.full(n, 0.0)
    pat_age_std = np.full(n, np.nan)

    orig_idx = X_out.index
    X_sorted = X_out.sort_values(["patient_id", "age"])
    for pid, grp in X_sorted.groupby("patient_id"):
        pos = np.array([X_out.index.get_loc(i) for i in grp.index])
        visit_rank[pos] = np.arange(len(pos), dtype=float)
        pat_age_std[pos] = float(np.std(grp["age"].values)) if len(pos) > 1 else 0.0

    X_out["rater_bias_hat"] = rater_bias
    X_out["rater_n_visits"] = rater_n
    X_out["visit_rank"]     = visit_rank
    X_out["patient_age_std"] = pat_age_std

    return X_out, rater_bias_map, rater_n_map


def build_s2_features_rater(X_frame: pd.DataFrame, s1_preds: np.ndarray,
                             curve_transformer: InputCurveTransformer,
                             y_tr_vals,
                             rater_bias_map, rater_n_map):
    """Build full stage-2 feature frame including rater features."""
    # 1. Standard 17_stage2_feats features
    X_s2 = build_s2_features(X_frame, s1_preds, curve_transformer)
    # 2. Rater features (on top of X_frame, not X_s2, to get rater_id still)
    X_rater, rb_map, rn_map = add_rater_features(
        X_frame.copy(), s1_preds, y_tr_vals, rater_bias_map, rater_n_map)
    # 3. Merge: add rater columns to X_s2
    for col in ["rater_bias_hat", "rater_n_visits", "visit_rank", "patient_age_std"]:
        X_s2[col] = X_rater[col].values
    return X_s2, rb_map, rn_map


# ── nested CV: compute OOF for each of 5 seeds ────────────────────────────────
print("Computing 22_rater stage-2 OOF (5 seeds)...")

OOF_SEED_CACHE = [C22 / f"22_oof_seed{s}.npy" for s in SEEDS]
oof_seeds = []

for s_idx, seed in enumerate(SEEDS):
    cache_path = OOF_SEED_CACHE[s_idx]
    if cache_path.exists():
        oof_seed = np.load(cache_path)
        print(f"  seed {seed}: loaded (RMSE={np.sqrt(np.mean((oof_seed-y_vals)**2)):.4f})")
        oof_seeds.append(oof_seed)
        continue

    print(f"  seed {seed}: computing OOF...")
    oof_seed = np.full(len(X), np.nan)

    for fi, (tr_idx, val_idx) in enumerate(cv_splits):
        t_f = time.time()

        # Stage-1 smoothed preds for this fold
        sm_tr  = ridge_poly_smooth(oof_11[tr_idx], ages_a[tr_idx], pids_a[tr_idx],
                                   CFG14["w_blend"], CFG14["degree"], CFG14["ridge_lam"])
        sm_val = ridge_poly_smooth(oof_11[val_idx], ages_a[val_idx], pids_a[val_idx],
                                   CFG14["w_blend"], CFG14["degree"], CFG14["ridge_lam"])

        # Curve transformer on training fold
        ct = InputCurveTransformer(lam_curve=1.0, lam_off_hat=1.0)
        ct.fit(X.iloc[tr_idx])

        # Build features: fit rater map on training fold, apply to both
        X_tr_s2, rb_map, rn_map = build_s2_features_rater(
            X.iloc[tr_idx].copy(), sm_tr, ct,
            y_tr_vals=y_vals[tr_idx],
            rater_bias_map=None, rater_n_map=None)
        X_val_s2, _, _ = build_s2_features_rater(
            X.iloc[val_idx].copy(), sm_val, ct,
            y_tr_vals=None,
            rater_bias_map=rb_map, rater_n_map=rn_map)

        m2 = HistGradientBoostingRegressor(
            categorical_features="from_dtype",
            early_stopping=True, n_iter_no_change=20,
            random_state=seed, **S2_PARAMS,
        )
        m2.fit(X_tr_s2, y.iloc[tr_idx])
        oof_seed[val_idx] = m2.predict(X_val_s2)
        rf = np.sqrt(np.mean((oof_seed[val_idx] - y_vals[val_idx])**2))
        print(f"    fold {fi}: {rf:.4f}  ({time.time()-t_f:.1f}s)")

    np.save(cache_path, oof_seed)
    oof_seeds.append(oof_seed)

oof_avg = np.mean(oof_seeds, axis=0)
rmse_raw = np.sqrt(np.mean((oof_avg - y_vals)**2))
print(f"\n22 seed-avg raw OOF RMSE: {rmse_raw:.4f}")

# ── Smoothing sweep ────────────────────────────────────────────────────────────
print("\nSmoothing sweep:")
print(f"  {'deg':>3}  {'lam':>5}  {'w':>5}  {'mean':>8}  {'std':>8}")
best_rmse_22  = rmse_raw
best_cfg_22   = dict(degree=1, ridge_lam=1.0, w_blend=0.0)
best_folds_22 = [np.sqrt(np.mean((oof_avg[val] - y_vals[val])**2))
                 for _, val in cv_splits]

for deg in [1, 2]:
    for lam in [0.0, 0.1, 1.0]:
        for w in [0.0, 0.25, 0.5, 0.75]:
            folds = []
            for _, val_idx in cv_splits:
                sm = ridge_poly_smooth(oof_avg[val_idx], ages_a[val_idx],
                                       pids_a[val_idx], w, deg, lam)
                folds.append(np.sqrt(np.mean((sm - y_vals[val_idx])**2)))
            m, s = np.mean(folds), np.std(folds)
            marker = " *" if m < best_rmse_22 else ""
            print(f"  {deg:>3}  {lam:>5.1f}  {w:>5.2f}  {m:>8.4f}  {s:>8.4f}{marker}")
            if m < best_rmse_22:
                best_rmse_22  = m
                best_cfg_22   = dict(degree=deg, ridge_lam=lam, w_blend=w)
                best_folds_22 = folds[:]

print(f"\nBest 22 config: {best_cfg_22}  RMSE={best_rmse_22:.4f}")

# ── Paired comparison vs 17 ────────────────────────────────────────────────────
print("\nPaired fold comparison (22_rater vs 17_stage2_feats):")
print(f"  {'fold':>5}  {'17':>8}  {'22':>8}  {'delta':>8}")
n_better = 0
for fi, (r17, r22) in enumerate(zip(PREV_BEST_FOLDS, best_folds_22)):
    delta = r17 - r22
    n_better += int(delta > 0)
    print(f"  {fi:>5}  {r17:>8.4f}  {r22:>8.4f}  {delta:>+8.4f}  "
          f"{'BETTER' if delta > 0 else ''}")
mean_gain = np.mean([r17 - r22 for r17, r22 in zip(PREV_BEST_FOLDS, best_folds_22)])
print(f"\nMean gain: {mean_gain:.4f}  Folds improved: {n_better}/5")
is_improvement = mean_gain > 0.03 and n_better >= 4
print(f"Criterion (gain>0.03 AND >=4 folds): {'MET' if is_improvement else 'NOT MET'}")

# ── OUTPUT RULE ────────────────────────────────────────────────────────────────
if is_improvement:
    print("\n" + "="*60)
    print("OUTPUT: writing submissions/22_rater.csv")
    print("="*60)

    ct_final = joblib.load(C17 / "17_ct_final.joblib")
    raw_test_s1 = np.load(C11 / "11_tuned_raw_test.npy")

    sm_train_s1 = ridge_poly_smooth(oof_11, ages_a, pids_a,
                                    CFG14["w_blend"], CFG14["degree"], CFG14["ridge_lam"])
    sm_test_s1  = ridge_poly_smooth(raw_test_s1, X_test["age"].values,
                                    X_test["patient_id"].values,
                                    CFG14["w_blend"], CFG14["degree"], CFG14["ridge_lam"])

    # Build final rater map on ALL training data
    X_train_s2, rb_map_final, rn_map_final = build_s2_features_rater(
        X.copy(), sm_train_s1, ct_final,
        y_tr_vals=y_vals,
        rater_bias_map=None, rater_n_map=None)
    X_test_s2, _, _ = build_s2_features_rater(
        X_test.copy(), sm_test_s1, ct_final,
        y_tr_vals=None,
        rater_bias_map=rb_map_final, rater_n_map=rn_map_final)

    test_preds = []
    for seed in SEEDS:
        mc = C22 / f"22_s2_final_seed{seed}.joblib"
        tc = C22 / f"22_raw_test_seed{seed}.npy"
        if mc.exists() and tc.exists():
            test_preds.append(np.load(tc))
            print(f"  seed {seed}: loaded from cache")
        else:
            t_s = time.time()
            m2 = HistGradientBoostingRegressor(
                categorical_features="from_dtype",
                early_stopping=True, n_iter_no_change=20,
                random_state=seed, **S2_PARAMS,
            )
            m2.fit(X_train_s2, y)
            raw_t = m2.predict(X_test_s2)
            joblib.dump(m2, mc)
            np.save(tc, raw_t)
            test_preds.append(raw_t)
            print(f"  seed {seed}: fit in {time.time()-t_s:.1f}s  [cached]")

    raw_test_avg = np.mean(test_preds, axis=0)
    final_test = ridge_poly_smooth(
        raw_test_avg, X_test["age"].values, X_test["patient_id"].values,
        best_cfg_22["w_blend"], best_cfg_22["degree"], best_cfg_22["ridge_lam"])
    final_test = np.clip(final_test, 0, 132)

    sub = pd.DataFrame({"target": final_test}, index=X_test.index)
    sub.index.name = "Index"
    sanity_check(sub, X_test, "22_rater")
    out_path = SUBS_DIR / "22_rater.csv"
    sub.to_csv(out_path)
    print(f"\nWritten: {out_path}")

    hub_url = "PENDING"
    try:
        import skore
        from skore import login
        from sklearn.base import BaseEstimator, RegressorMixin

        class S22Wrapper(BaseEstimator, RegressorMixin):
            def fit(self, X, y):
                from parkinsons.pipeline import build_hgbr_tuned
                s1 = build_hgbr_tuned(learning_rate=0.02, max_iter=1000,
                                      max_leaf_nodes=31, min_samples_leaf=20,
                                      l2_regularization=0.0, random_state=0)
                s1.fit(X, y)
                sm = ridge_poly_smooth(s1.predict(X), X["age"].values,
                                       X["patient_id"].values,
                                       CFG14["w_blend"], CFG14["degree"], CFG14["ridge_lam"])
                ct = InputCurveTransformer(); ct.fit(X)
                Xs2, self._rb, self._rn = build_s2_features_rater(
                    X.copy(), sm, ct, y_tr_vals=y.values, rater_bias_map=None, rater_n_map=None)
                self._models = [
                    HistGradientBoostingRegressor(
                        categorical_features="from_dtype", early_stopping=True,
                        n_iter_no_change=20, random_state=s, **S2_PARAMS
                    ).fit(Xs2, y) for s in SEEDS]
                self._s1 = s1; self._ct = ct
                return self

            def predict(self, X):
                sm = ridge_poly_smooth(self._s1.predict(X), X["age"].values,
                                       X["patient_id"].values,
                                       CFG14["w_blend"], CFG14["degree"], CFG14["ridge_lam"])
                Xs2, _, _ = build_s2_features_rater(
                    X.copy(), sm, self._ct, y_tr_vals=None,
                    rater_bias_map=self._rb, rater_n_map=self._rn)
                avg = np.mean([m.predict(Xs2) for m in self._models], axis=0)
                return np.clip(ridge_poly_smooth(avg, X["age"].values,
                               X["patient_id"].values,
                               best_cfg_22["w_blend"], best_cfg_22["degree"],
                               best_cfg_22["ridge_lam"]), 0, 132)

        cfg = load_skore_credentials()
        login(mode="hub")
        project = skore.Project(name="bobathon-esilv", mode="hub", workspace=cfg["workspace"])
        report  = skore.evaluate(S22Wrapper(), X, y, splitter=cv_splits)
        project.put("22_rater", report)
        workspace = cfg["workspace"]
        hub_url   = f"https://skore.probabl.ai/{workspace}/bobathon-esilv/cross-validations"
        print(f"Hub report 22_rater pushed.")
    except Exception as e:
        print(f"HUB PUT PENDING: 22_rater  ({e})")

    std_22 = float(np.std(best_folds_22))
    print(f"\nREADY TO SUBMIT: submissions/22_rater.csv | "
          f"CV RMSE {best_rmse_22:.4f} +/- {std_22:.4f} | URL: {hub_url}")
else:
    print(f"\nNo improvement over 17 (CV {PREV_BEST_CV:.4f}). No CSV written.")
    print(f"22_rater CV RMSE: {best_rmse_22:.4f} +/- {np.std(best_folds_22):.4f}")
    print(f"Best config: {best_cfg_22}")

print(f"\nTotal elapsed: {time.time()-t0:.0f}s")
print("STEP 7 (22_rater) DONE.")
