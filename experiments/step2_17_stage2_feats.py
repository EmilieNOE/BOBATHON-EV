"""step2_17_stage2_feats.py -- STEP 2: richer stage-2 features (key 17_stage2_feats).

New features added to stage-2 (all from INPUT columns, no target):
  For each patient, fit ridge-shrunk quadratics of observed `on` and `off`
  against centered age (fit on training-fold rows; fallback to linear/<mean
  with <4 observations).  At transform time add:
    on_curve   -- fitted on-curve value at this visit's age
    on_slope   -- slope of the on-curve at this visit's age
    on_curv    -- curvature (quadratic coeff) of the on-curve
    on_resid   -- on - on_curve  (deviation from patient's on trend)
    off_curve, off_slope, off_curv, off_resid  (same for off where observed)
    off_hat    -- for visits with off MISSING: estimate from on using a
                  ridge regression trained on rows where BOTH are present
                  (fit on the TRAINING fold only, applied to held-out fold)
    off_or_hat -- off if present, else off_hat
    off_hat_curve -- patient's off_or_hat curve value at this age
    on_miss_frac, off_miss_frac  -- fraction of patient's visits with missing on/off

These are fit inside the CV training fold via a stateful transformer that:
  1. Fits per-patient quadratic curves on input columns in training data.
  2. Fits off_hat model (Ridge) on training rows where both on and off are present.
  At transform() time applies the stored coefficients to any frame.

The base model remains 11_tuned (stage-1). Stage-2 uses the tuned hyperparameters
from STEP 1 (lr=0.02, max_leaf_nodes=31, min_samples_leaf=50, max_iter=300,
l2_regularization=0.0) averaged over 5 seeds. Post-processing: sweep to find
the best smoothing config.

Hub key: 17_stage2_feats
PYTHONUTF8=1. ASCII-only prints. Cache all OOF arrays.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from shared import add_s1_aggregates

# ─────────────────────────────────────────────────────────────────────────────
# Input curve transformer  (module-level so joblib can unpickle from imports)
# ─────────────────────────────────────────────────────────────────────────────

def _fit_ridge_quad(ages_obs, vals_obs, lam=1.0):
    """Fit ridge-penalised quadratic: vals ~ a*age_c^2 + b*age_c + intercept.
    Returns (coef_a, coef_b, intercept, age_mean) or None if fewer than 2 obs.
    Falls back to linear with < 4 obs.
    """
    n = len(ages_obs)
    if n < 2:
        return None
    age_mean = ages_obs.mean()
    ac = ages_obs - age_mean
    if n < 4:
        # linear only
        Phi = ac.reshape(-1, 1)
        A = Phi.T @ Phi + lam * np.eye(1)
        b_vec = np.linalg.solve(A, Phi.T @ vals_obs)
        return (0.0, float(b_vec[0]), float(np.mean(vals_obs)), float(age_mean))
    Phi = np.column_stack([ac, ac**2])
    intercept = np.mean(vals_obs)
    yc = vals_obs - intercept
    A = Phi.T @ Phi + lam * np.eye(2)
    try:
        coef = np.linalg.solve(A, Phi.T @ yc)
    except np.linalg.LinAlgError:
        coef = np.zeros(2)
    return (float(coef[1]), float(coef[0]), float(intercept), float(age_mean))


def _eval_quad(coef_tuple, age):
    """Evaluate fitted curve at age (scalar or array)."""
    if coef_tuple is None:
        return np.full_like(np.atleast_1d(age), np.nan, dtype=float)
    a, b, intercept, age_mean = coef_tuple
    ac = np.atleast_1d(age) - age_mean
    return a * ac**2 + b * ac + intercept


def _slope_quad(coef_tuple, age):
    """Slope of fitted curve at age."""
    if coef_tuple is None:
        return np.full_like(np.atleast_1d(age), np.nan, dtype=float)
    a, b, _, age_mean = coef_tuple
    ac = np.atleast_1d(age) - age_mean
    return 2 * a * ac + b


class InputCurveTransformer:
    """Fit per-patient on/off curves and off_hat model on training data.
    Apply to any frame at transform time (fold-safe).
    """
    def __init__(self, lam_curve=1.0, lam_off_hat=1.0):
        self.lam_curve   = lam_curve
        self.lam_off_hat = lam_off_hat

    def fit(self, X_tr: pd.DataFrame):
        self.on_coef_  = {}
        self.off_coef_ = {}
        fallback_on  = float(X_tr["on"].mean())
        fallback_off = float(X_tr["off"].mean())
        self.fallback_on_  = fallback_on
        self.fallback_off_ = fallback_off

        for pid, grp in X_tr.groupby("patient_id"):
            grp = grp.sort_values("age")
            # on curve: use all rows with on present
            on_rows = grp[grp["on"].notna()]
            if len(on_rows) >= 2:
                self.on_coef_[pid] = _fit_ridge_quad(
                    on_rows["age"].values, on_rows["on"].values, self.lam_curve)
            else:
                self.on_coef_[pid] = None
            # off curve: use all rows with off present
            off_rows = grp[grp["off"].notna()]
            if len(off_rows) >= 2:
                self.off_coef_[pid] = _fit_ridge_quad(
                    off_rows["age"].values, off_rows["off"].values, self.lam_curve)
            else:
                self.off_coef_[pid] = None

        # off_hat model: Ridge(on, age, ledd, time_since_intake_on) -> off
        # fit on rows where both on and off are present
        both = X_tr["on"].notna() & X_tr["off"].notna()
        Xboth = X_tr[both][["on", "age", "ledd", "time_since_intake_on"]].copy()
        Xboth = Xboth.fillna(Xboth.median())
        yboth = X_tr[both]["off"].values
        self.off_hat_scaler_ = StandardScaler()
        Xs = self.off_hat_scaler_.fit_transform(Xboth.values)
        self.off_hat_ridge_ = Ridge(alpha=self.lam_off_hat)
        self.off_hat_ridge_.fit(Xs, yboth)
        return self

    def transform(self, X_fr: pd.DataFrame) -> pd.DataFrame:
        X_out = X_fr.copy()
        n = len(X_out)
        on_curve  = np.full(n, np.nan)
        on_slope  = np.full(n, np.nan)
        on_curv   = np.full(n, np.nan)
        on_resid  = np.full(n, np.nan)
        off_curve = np.full(n, np.nan)
        off_slope = np.full(n, np.nan)
        off_curv  = np.full(n, np.nan)
        off_resid = np.full(n, np.nan)
        on_mf     = np.full(n, np.nan)
        off_mf    = np.full(n, np.nan)

        # per-patient curves -- use positional indexing (frame already has int index)
        age_vals = X_out["age"].values
        on_vals  = X_out["on"].values.astype(float)
        off_vals = X_out["off"].values.astype(float)

        # group by patient (frame need not be sorted here)
        for pid, grp in X_out.groupby("patient_id"):
            pos = [X_out.index.get_loc(i) for i in grp.index]
            pos = np.array(pos)
            ages_g = age_vals[pos]
            on_g   = on_vals[pos]
            off_g  = off_vals[pos]

            # on curve
            coef_on = self.on_coef_.get(pid, None)
            if coef_on is None:
                # cold-start: fit from this frame's own on observations
                on_rows = ~np.isnan(on_g)
                if on_rows.sum() >= 2:
                    coef_on = _fit_ridge_quad(ages_g[on_rows], on_g[on_rows], self.lam_curve)
            if coef_on is not None:
                on_curve[pos] = _eval_quad(coef_on, ages_g)
                on_slope[pos] = _slope_quad(coef_on, ages_g)
                on_curv[pos]  = coef_on[0]   # quadratic coefficient
                on_resid[pos] = on_g - on_curve[pos]
            else:
                on_curve[pos] = self.fallback_on_
                on_slope[pos] = 0.0
                on_curv[pos]  = 0.0

            # off curve
            coef_off = self.off_coef_.get(pid, None)
            if coef_off is None:
                off_rows = ~np.isnan(off_g)
                if off_rows.sum() >= 2:
                    coef_off = _fit_ridge_quad(ages_g[off_rows], off_g[off_rows], self.lam_curve)
            if coef_off is not None:
                off_curve[pos] = _eval_quad(coef_off, ages_g)
                off_slope[pos] = _slope_quad(coef_off, ages_g)
                off_curv[pos]  = coef_off[0]
                off_resid[pos] = off_g - off_curve[pos]
            else:
                off_curve[pos] = self.fallback_off_
                off_slope[pos] = 0.0
                off_curv[pos]  = 0.0

            on_mf[pos]  = np.mean(np.isnan(on_g))
            off_mf[pos] = np.mean(np.isnan(off_g))

        # off_hat for rows with off missing
        feat_cols = ["on", "age", "ledd", "time_since_intake_on"]
        X_feat = X_out[feat_cols].copy().fillna(
            pd.Series({c: float(X_out[c].median()) for c in feat_cols}))
        Xs = self.off_hat_scaler_.transform(X_feat.values)
        off_hat_all = self.off_hat_ridge_.predict(Xs)

        off_arr     = X_out["off"].values.astype(float)
        off_or_hat  = np.where(np.isnan(off_arr), off_hat_all, off_arr)

        # off_or_hat patient curve (fit from combined series)
        off_hat_curve = np.full(n, np.nan)
        for pid, grp in X_out.groupby("patient_id"):
            pos   = np.array([X_out.index.get_loc(i) for i in grp.index])
            ages_g = age_vals[pos]
            ooh_g  = off_or_hat[pos]
            coef = _fit_ridge_quad(ages_g, ooh_g, self.lam_curve)
            if coef is not None:
                off_hat_curve[pos] = _eval_quad(coef, ages_g)

        X_out["on_curve"]      = on_curve
        X_out["on_slope"]      = on_slope
        X_out["on_curv"]       = on_curv
        X_out["on_resid"]      = on_resid
        X_out["off_curve"]     = off_curve
        X_out["off_slope"]     = off_slope
        X_out["off_curv"]      = off_curv
        X_out["off_resid"]     = off_resid
        X_out["off_hat"]       = off_hat_all
        X_out["off_or_hat"]    = off_or_hat
        X_out["off_hat_curve"] = off_hat_curve
        X_out["on_miss_frac"]  = on_mf
        X_out["off_miss_frac"] = off_mf
        return X_out


def build_s2_features(X_frame: pd.DataFrame, s1_preds: np.ndarray,
                      curve_transformer: "InputCurveTransformer") -> pd.DataFrame:
    """Apply input-curve features then stage-1 aggregates."""
    X_with_curves = curve_transformer.transform(X_frame)
    return add_s1_aggregates(X_with_curves, s1_preds)


# ─────────────────────────────────────────────────────────────────────────────
# Execution (only when run as __main__, not when imported)
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import time
    import joblib
    from sklearn.ensemble import HistGradientBoostingRegressor
    from sklearn.model_selection import GroupKFold

    from parkinsons import PROJECT_ROOT
    from parkinsons.data import load_dataset, load_test_dataset
    from parkinsons.hub import load_skore_credentials
    from parkinsons.pipeline import build_hgbr_tuned
    from shared import ridge_poly_smooth, sanity_check

    DATA_DIR  = PROJECT_ROOT / "data"
    SUBS_DIR  = PROJECT_ROOT / "submissions"
    C11       = PROJECT_ROOT / "scratch" / "13_cache"
    C17       = PROJECT_ROOT / "scratch" / "17_cache"
    SUBS_DIR.mkdir(exist_ok=True)
    C17.mkdir(parents=True, exist_ok=True)

    CFG14 = dict(degree=2, ridge_lam=0.1, w_blend=0.9)
    BEST_PARAMS_11 = dict(learning_rate=0.02, max_iter=1000, max_leaf_nodes=31,
                          min_samples_leaf=20, l2_regularization=0.0)
    S2_PARAMS = dict(learning_rate=0.02, max_iter=300, max_leaf_nodes=31,
                     min_samples_leaf=50, l2_regularization=0.0)
    SEEDS = [0, 1, 2, 3, 4]
    PREV_BEST_CV    = 4.1453
    PREV_BEST_FOLDS = [4.3990, 4.1213, 4.0873, 4.0548, 4.0642]

    t0 = time.time()

    # ── load data + cached OOF ────────────────────────────────────────────────
    X, y   = load_dataset(data_dir=DATA_DIR)
    X_test = load_test_dataset(data_dir=DATA_DIR)
    groups = X["patient_id"]
    cv_splits = list(GroupKFold(n_splits=5).split(X, y, groups=groups))
    ages_a = X["age"].values
    pids_a = X["patient_id"].values
    oof_11 = np.load(C11 / "11_oof_raw.npy")
    print(f"Data + OOF loaded ({time.time()-t0:.1f}s)")

    # ── Nested CV: compute OOF for each of 5 seeds ───────────────────────────
    print("Computing stage-2 OOF with input curves (5 seeds)...")

    OOF_SEED_CACHE = [C17 / f"17_oof_seed{s}.npy" for s in SEEDS]
    oof_seeds = []

    for s_idx, seed in enumerate(SEEDS):
        cache_path = OOF_SEED_CACHE[s_idx]
        if cache_path.exists():
            oof_seed = np.load(cache_path)
            print(f"  seed {seed}: loaded  (raw RMSE={np.sqrt(np.mean((oof_seed-y.values)**2)):.4f})")
            oof_seeds.append(oof_seed)
            continue

        print(f"  seed {seed}: computing OOF...")
        oof_seed = np.full(len(X), np.nan)
        for fi, (tr_idx, val_idx) in enumerate(cv_splits):
            t_f = time.time()
            sm_tr  = ridge_poly_smooth(oof_11[tr_idx], ages_a[tr_idx], pids_a[tr_idx],
                                       CFG14["w_blend"], CFG14["degree"], CFG14["ridge_lam"])
            sm_val = ridge_poly_smooth(oof_11[val_idx], ages_a[val_idx], pids_a[val_idx],
                                       CFG14["w_blend"], CFG14["degree"], CFG14["ridge_lam"])

            ct = InputCurveTransformer(lam_curve=1.0, lam_off_hat=1.0)
            ct.fit(X.iloc[tr_idx])

            X_tr_s2  = build_s2_features(X.iloc[tr_idx].copy(),  sm_tr,  ct)
            X_val_s2 = build_s2_features(X.iloc[val_idx].copy(), sm_val, ct)

            m2 = HistGradientBoostingRegressor(
                categorical_features="from_dtype",
                early_stopping=True, n_iter_no_change=20,
                random_state=seed, **S2_PARAMS,
            )
            m2.fit(X_tr_s2, y.iloc[tr_idx])
            oof_seed[val_idx] = m2.predict(X_val_s2)
            rf = np.sqrt(np.mean((oof_seed[val_idx] - y.values[val_idx])**2))
            print(f"    fold {fi}: {rf:.4f}  ({time.time()-t_f:.1f}s)")

        np.save(cache_path, oof_seed)
        oof_seeds.append(oof_seed)

    oof_avg = np.mean(oof_seeds, axis=0)
    rmse_raw = np.sqrt(np.mean((oof_avg - y.values)**2))
    print(f"\nSeed-averaged raw OOF RMSE: {rmse_raw:.4f}")

    # ── Smoothing sweep ───────────────────────────────────────────────────────
    print("\nSmoothing sweep:")
    print(f"  {'deg':>3}  {'lam':>5}  {'w':>5}  {'mean':>8}  {'std':>8}")
    best_rmse_17  = rmse_raw
    best_cfg_17   = dict(degree=2, ridge_lam=0.1, w_blend=0.0)
    best_folds_17 = [np.sqrt(np.mean((oof_avg[val] - y.values[val])**2))
                     for _, val in cv_splits]

    for deg in [1, 2]:
        for lam in [0.0, 0.1, 1.0]:
            for w in [0.25, 0.5, 0.75, 0.9]:
                folds = []
                for _, val_idx in cv_splits:
                    sm = ridge_poly_smooth(oof_avg[val_idx], ages_a[val_idx],
                                           pids_a[val_idx], w, deg, lam)
                    folds.append(np.sqrt(np.mean((sm - y.values[val_idx])**2)))
                m, s = np.mean(folds), np.std(folds)
                print(f"  {deg:>3}  {lam:>5.1f}  {w:>5.2f}  {m:>8.4f}  {s:>8.4f}")
                if m < best_rmse_17:
                    best_rmse_17  = m
                    best_cfg_17   = dict(degree=deg, ridge_lam=lam, w_blend=w)
                    best_folds_17 = folds

    print(f"\nBest smoothing: {best_cfg_17}  RMSE={best_rmse_17:.4f}")

    # ── Paired comparison vs 15_bias ─────────────────────────────────────────
    print("\nPaired fold comparison (17_stage2_feats vs 15_bias):")
    print(f"  {'fold':>5}  {'15_bias':>8}  {'17':>8}  {'delta':>8}")
    n_better = 0
    for fi, (r15, r17) in enumerate(zip(PREV_BEST_FOLDS, best_folds_17)):
        delta = r15 - r17
        n_better += int(delta > 0)
        print(f"  {fi:>5}  {r15:>8.4f}  {r17:>8.4f}  {delta:>+8.4f}  "
              f"{'BETTER' if delta > 0 else ''}")
    mean_gain = np.mean([r15-r17 for r15,r17 in zip(PREV_BEST_FOLDS, best_folds_17)])
    print(f"\nMean gain: {mean_gain:.4f}  Folds improved: {n_better}/5")
    is_improvement = mean_gain > 0.03 and n_better >= 4
    print(f"Criterion (gain>0.03 AND >=4 folds): {'MET' if is_improvement else 'NOT MET'}")

    # ── OUTPUT RULE ───────────────────────────────────────────────────────────
    if is_improvement:
        print("\n" + "="*60)
        print("OUTPUT: writing submissions/17_stage2_feats.csv")
        print("="*60)

        CT_CACHE = C17 / "17_ct_final.joblib"
        if CT_CACHE.exists():
            ct_final = joblib.load(CT_CACHE)
            print("  Curve transformer loaded from cache.")
        else:
            ct_final = InputCurveTransformer(lam_curve=1.0, lam_off_hat=1.0)
            ct_final.fit(X)
            joblib.dump(ct_final, CT_CACHE)
            print("  Curve transformer fit on all train + cached.")

        raw_test_s1 = np.load(C11 / "11_tuned_raw_test.npy")
        sm_train_s1 = ridge_poly_smooth(oof_11, ages_a, pids_a,
                                        CFG14["w_blend"], CFG14["degree"], CFG14["ridge_lam"])
        sm_test_s1  = ridge_poly_smooth(raw_test_s1, X_test["age"].values,
                                        X_test["patient_id"].values,
                                        CFG14["w_blend"], CFG14["degree"], CFG14["ridge_lam"])

        X_train_s2 = build_s2_features(X.copy(),      sm_train_s1, ct_final)
        X_test_s2  = build_s2_features(X_test.copy(), sm_test_s1,  ct_final)

        test_preds = []
        for seed in SEEDS:
            mc = C17 / f"17_s2_final_seed{seed}.joblib"
            tc = C17 / f"17_raw_test_seed{seed}.npy"
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
        final_test   = ridge_poly_smooth(raw_test_avg, X_test["age"].values,
                                         X_test["patient_id"].values,
                                         best_cfg_17["w_blend"], best_cfg_17["degree"],
                                         best_cfg_17["ridge_lam"])
        final_test = np.clip(final_test, 0, 132)

        sub = pd.DataFrame({"target": final_test}, index=X_test.index)
        sub.index.name = "Index"
        sanity_check(sub, X_test, "17_stage2_feats")
        out_path = SUBS_DIR / "17_stage2_feats.csv"
        sub.to_csv(out_path)
        print(f"\nWritten: {out_path}")

        hub_url = "PENDING"
        try:
            import skore
            from skore import login
            from sklearn.base import BaseEstimator, RegressorMixin

            class S17Wrapper(BaseEstimator, RegressorMixin):
                def fit(self, X, y):
                    s1 = build_hgbr_tuned(**BEST_PARAMS_11, random_state=0)
                    s1.fit(X, y)
                    sm = ridge_poly_smooth(s1.predict(X), X["age"].values,
                                           X["patient_id"].values, CFG14["w_blend"],
                                           CFG14["degree"], CFG14["ridge_lam"])
                    ct = InputCurveTransformer(); ct.fit(X)
                    Xs2 = build_s2_features(X.copy(), sm, ct)
                    self._models = [
                        HistGradientBoostingRegressor(
                            categorical_features="from_dtype", early_stopping=True,
                            n_iter_no_change=20, random_state=s, **S2_PARAMS
                        ).fit(Xs2, y) for s in SEEDS]
                    self._s1 = s1; self._ct = ct
                    return self
                def predict(self, X):
                    sm = ridge_poly_smooth(self._s1.predict(X), X["age"].values,
                                           X["patient_id"].values, CFG14["w_blend"],
                                           CFG14["degree"], CFG14["ridge_lam"])
                    Xs2 = build_s2_features(X.copy(), sm, self._ct)
                    avg = np.mean([m.predict(Xs2) for m in self._models], axis=0)
                    return np.clip(ridge_poly_smooth(avg, X["age"].values,
                                   X["patient_id"].values, best_cfg_17["w_blend"],
                                   best_cfg_17["degree"], best_cfg_17["ridge_lam"]), 0, 132)

            cfg = load_skore_credentials()
            login(mode="hub")
            project = skore.Project(name="bobathon-esilv", mode="hub", workspace=cfg["workspace"])
            report  = skore.evaluate(S17Wrapper(), X, y, splitter=cv_splits)
            project.put("17_stage2_feats", report)
            workspace = cfg["workspace"]
            hub_url   = f"https://skore.probabl.ai/{workspace}/bobathon-esilv/cross-validations"
            print(f"Hub report 17_stage2_feats pushed.")
        except Exception as e:
            print(f"HUB PUT PENDING: 17_stage2_feats  ({e})")

        print(f"\nREADY TO SUBMIT: submissions/17_stage2_feats.csv | "
              f"CV RMSE {best_rmse_17:.4f} +/- {np.std(best_folds_17):.4f} | "
              f"hub URL: {hub_url}")
    else:
        print(f"\nNo improvement over 15_bias (CV {PREV_BEST_CV:.4f}). No CSV written.")

    print(f"\nTotal elapsed: {time.time()-t0:.0f}s")
    print("STEP 2 DONE.")
