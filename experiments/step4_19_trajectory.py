"""step4_19_trajectory.py -- STEP 4: patient trajectory model (key 19_trajectory).

Key insight: per-patient target is nearly perfectly quadratic in age
(median R^2=0.9994, residual SD~0.13). So we can:

  1. For each TRAINING-fold patient, fit a quadratic: target ~ a*age_c^2 + b*age_c + c
     using ridge regularisation. This gives trajectory parameters (a, b, c).

  2. Build a model to predict (a, b, c) from input features ONLY (no target):
       - per-patient summaries of on, off, ledd, age_at_diagnosis, cohort, gene
       - on/off input curve features (from InputCurveTransformer)
       - stage-1 prediction curve parameters (mean, slope, quad coef)
       This model is called "TrajModel".

  3. At prediction time: predict (a_hat, b_hat, c_hat) from inputs,
     then evaluate a_hat*(age-age_mean)^2 + b_hat*(age-age_mean) + c_hat.

  4. Blend: final = (1-alpha)*stage2_pred + alpha*traj_pred.
     Sweep alpha in [0, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0] by grouped CV.

LEAKAGE CHECK: trajectory parameters (a, b, c) are fitted on TRAINING FOLD
targets only. The parameter model is trained on those training patients.
At val/test time we predict (a, b, c) purely from inputs.

Hub key: 19_trajectory
PYTHONUTF8=1. ASCII-only prints.
"""

import sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
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
C19  = PROJECT_ROOT / "scratch" / "19_cache"
SUBS_DIR.mkdir(exist_ok=True)
C19.mkdir(parents=True, exist_ok=True)

CFG14     = dict(degree=2, ridge_lam=0.1, w_blend=0.9)
S2_PARAMS = dict(learning_rate=0.02, max_iter=300, max_leaf_nodes=31,
                 min_samples_leaf=50, l2_regularization=0.0)
SEEDS     = [0, 1, 2, 3, 4]
PREV_BEST_CV    = 3.9949
PREV_BEST_FOLDS = [4.2602, 4.0026, 3.9085, 3.8914, 3.9119]
TRAJ_LAM  = 0.1   # ridge penalty for trajectory fit (small: data is clean)

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


# ── helpers ───────────────────────────────────────────────────────────────────

def fit_quad_traj(ages, targets, lam=TRAJ_LAM):
    """Fit ridge-penalised quadratic target ~ a*age_c^2 + b*age_c + c.
    Returns (a, b, c, age_mean) or None if n<2.
    """
    n = len(ages)
    if n < 2:
        return None
    age_mean = float(ages.mean())
    ac = ages - age_mean
    c  = float(np.mean(targets))
    yc = targets - c
    if n < 4:
        # linear only (a=0)
        Phi = ac.reshape(-1, 1)
        A = Phi.T @ Phi + lam * np.eye(1)
        b_v = float(np.linalg.solve(A, Phi.T @ yc)[0])
        return (0.0, b_v, c, age_mean)
    Phi = np.column_stack([ac, ac**2])
    A = Phi.T @ Phi + lam * np.eye(2)
    try:
        coef = np.linalg.solve(A, Phi.T @ yc)
    except np.linalg.LinAlgError:
        coef = np.zeros(2)
    return (float(coef[1]), float(coef[0]), c, age_mean)


def eval_quad_traj(params, ages):
    """Evaluate quadratic at given ages (array or scalar)."""
    if params is None:
        return np.full(len(np.atleast_1d(ages)), np.nan)
    a, b, c, age_mean = params
    ac = np.atleast_1d(ages) - age_mean
    return a * ac**2 + b * ac + c


def build_traj_features(X_frame, s1_preds, curve_transformer):
    """Build per-patient trajectory-prediction feature rows.

    One row per PATIENT (not per visit). Features are input-only
    summaries: no target information.
    """
    X_with_curves = curve_transformer.transform(X_frame.copy())
    # Stage-1 aggregates give us per-patient prediction stats
    X_s2 = add_s1_aggregates(X_with_curves.copy(), s1_preds)

    # Per-patient summary: take the mean over visits (one row per patient)
    numeric_cols = X_s2.select_dtypes(include=[np.number]).columns.tolist()
    # Remove per-visit columns that are noisy; keep patient-level summaries
    per_pat = (X_s2[numeric_cols]
               .assign(patient_id=X_frame["patient_id"].values)
               .groupby("patient_id")
               .mean())
    return per_pat


# ── nested CV: compute OOF trajectory predictions ─────────────────────────────
print("\nComputing trajectory OOF predictions (nested CV)...")
TRAJ_CACHE = C19 / "19_traj_oof.npy"
if TRAJ_CACHE.exists():
    traj_oof = np.load(TRAJ_CACHE)
    print(f"  Loaded from cache. raw RMSE={np.sqrt(np.mean((traj_oof-y_vals)**2)):.4f}")
else:
    traj_oof = np.full(len(X), np.nan)

    for fi, (tr_idx, val_idx) in enumerate(cv_splits):
        t_f = time.time()

        # --- Stage-1 smoothed preds for this fold
        sm_tr  = ridge_poly_smooth(oof_11[tr_idx], ages_a[tr_idx], pids_a[tr_idx],
                                   CFG14["w_blend"], CFG14["degree"], CFG14["ridge_lam"])
        sm_val = ridge_poly_smooth(oof_11[val_idx], ages_a[val_idx], pids_a[val_idx],
                                   CFG14["w_blend"], CFG14["degree"], CFG14["ridge_lam"])

        # --- Curve transformer: fit on training fold
        ct = InputCurveTransformer(lam_curve=1.0, lam_off_hat=1.0)
        ct.fit(X.iloc[tr_idx])

        # --- Fit trajectory parameters on training fold patients (uses TARGET)
        X_tr    = X.iloc[tr_idx]
        y_tr    = y_vals[tr_idx]
        tr_pids = pids_a[tr_idx]
        tr_ages = ages_a[tr_idx]

        # Per-patient trajectory coefficients for training patients
        traj_params = {}   # pid -> (a, b, c, age_mean)
        for pid, grp_idx in pd.Series(tr_pids).groupby(pd.Series(tr_pids)).groups.items():
            grp_idx = np.array(grp_idx)
            ages_p  = tr_ages[grp_idx]
            targs_p = y_tr[grp_idx]
            sort_o  = np.argsort(ages_p)
            params  = fit_quad_traj(ages_p[sort_o], targs_p[sort_o], lam=TRAJ_LAM)
            traj_params[pid] = params

        # Build feature rows (one per TRAINING patient)
        # These features are input-only (no target)
        pat_feats_tr = build_traj_features(X_tr, sm_tr, ct)

        # Target for the trajectory model: (a, b, c) parameters
        # We exclude the age_mean since that is purely from inputs
        traj_train_pids = pat_feats_tr.index.values
        a_tr = np.array([traj_params[p][0] if traj_params.get(p) else 0.0
                         for p in traj_train_pids])
        b_tr = np.array([traj_params[p][1] if traj_params.get(p) else 0.0
                         for p in traj_train_pids])
        c_tr = np.array([traj_params[p][2] if traj_params.get(p) else float(np.mean(y_tr))
                         for p in traj_train_pids])
        # Also store age_mean per patient (from the INPUTS, not target)
        age_mean_tr = np.array([traj_params[p][3] if traj_params.get(p) else
                                 X_tr[pids_a[tr_idx] == p]["age"].mean()
                                 for p in traj_train_pids])

        Xf = pat_feats_tr.values
        # Replace NaN with median
        col_medians = np.nanmedian(Xf, axis=0)
        for j in range(Xf.shape[1]):
            mask = np.isnan(Xf[:, j])
            Xf[mask, j] = col_medians[j]

        # Train HGBR for each coefficient (a, b, c)
        def _train_coef_model(Xf, ytgt, seed=0):
            m = HistGradientBoostingRegressor(
                max_iter=200, max_leaf_nodes=15, min_samples_leaf=10,
                learning_rate=0.05, random_state=seed,
            )
            m.fit(Xf, ytgt)
            return m

        m_a = _train_coef_model(Xf, a_tr)
        m_b = _train_coef_model(Xf, b_tr)
        m_c = _train_coef_model(Xf, c_tr)

        # --- Predict trajectory on validation patients (no target used)
        X_val = X.iloc[val_idx]
        pat_feats_val = build_traj_features(X_val, sm_val, ct)
        val_pids = pat_feats_val.index.values

        Xv = pat_feats_val.values
        for j in range(Xv.shape[1]):
            mask = np.isnan(Xv[:, j])
            Xv[mask, j] = col_medians[j]

        a_hat = m_a.predict(Xv)
        b_hat = m_b.predict(Xv)
        c_hat = m_c.predict(Xv)

        # Evaluate trajectory at each validation visit's age
        # age_mean for val patients: mean of their own input ages (not target)
        val_ages_by_pat = {}
        for pid_v, grp_df in X_val.groupby("patient_id"):
            val_ages_by_pat[pid_v] = float(grp_df["age"].mean())

        pid_to_coef = {
            pid_v: (a_hat[i], b_hat[i], c_hat[i], val_ages_by_pat.get(pid_v, 0.0))
            for i, pid_v in enumerate(val_pids)
        }

        val_pids_arr = pids_a[val_idx]
        val_ages_arr = ages_a[val_idx]
        traj_pred_val = np.array([
            float(eval_quad_traj(pid_to_coef.get(p), np.array([a]))[0])
            for p, a in zip(val_pids_arr, val_ages_arr)
        ])
        traj_oof[val_idx] = traj_pred_val

        raw_rmse = np.sqrt(np.mean((traj_pred_val - y_vals[val_idx])**2))
        print(f"  fold {fi}: traj RMSE={raw_rmse:.4f}  ({time.time()-t_f:.1f}s)")

    np.save(TRAJ_CACHE, traj_oof)
    print(f"  Saved to {TRAJ_CACHE}")

# Per-fold raw trajectory RMSE
traj_fold_rmse = [
    np.sqrt(np.mean((traj_oof[val] - y_vals[val])**2))
    for _, val in cv_splits
]
print(f"\nTrajectory-only OOF RMSE: {np.mean(traj_fold_rmse):.4f} +/- {np.std(traj_fold_rmse):.4f}")

# ── Blend sweep: alpha * traj + (1-alpha) * stage2 ────────────────────────────
print("\nBlend sweep (alpha*traj + (1-alpha)*s2):")
print(f"  {'alpha':>6}  {'mean':>8}  {'std':>8}")

best_rmse_19  = PREV_BEST_CV
best_alpha_19 = 0.0
best_folds_19 = PREV_BEST_FOLDS[:]

# Also sweep smoothing on top of blended
for alpha in [0.0, 0.05, 0.10, 0.15, 0.20, 0.30, 0.50, 0.70, 1.0]:
    blended_oof = (1 - alpha) * oof17_avg + alpha * traj_oof
    folds = []
    for _, val_idx in cv_splits:
        # Apply best-17 smoothing on top of blend
        sm = ridge_poly_smooth(blended_oof[val_idx], ages_a[val_idx],
                               pids_a[val_idx], 0.25, 1, 1.0)
        folds.append(np.sqrt(np.mean((sm - y_vals[val_idx])**2)))
    m, s = np.mean(folds), np.std(folds)
    marker = " *" if m < best_rmse_19 else ""
    print(f"  {alpha:>6.2f}  {m:>8.4f}  {s:>8.4f}{marker}")
    if m < best_rmse_19:
        best_rmse_19  = m
        best_alpha_19 = alpha
        best_folds_19 = folds[:]

print(f"\nBest blend alpha: {best_alpha_19}  RMSE={best_rmse_19:.4f}")

# ── Paired comparison ─────────────────────────────────────────────────────────
print("\nPaired fold comparison (19_trajectory vs 17_stage2_feats):")
print(f"  {'fold':>5}  {'17':>8}  {'19':>8}  {'delta':>8}")
n_better = 0
for fi, (r17, r19) in enumerate(zip(PREV_BEST_FOLDS, best_folds_19)):
    delta = r17 - r19
    n_better += int(delta > 0)
    print(f"  {fi:>5}  {r17:>8.4f}  {r19:>8.4f}  {delta:>+8.4f}  "
          f"{'BETTER' if delta > 0 else ''}")
mean_gain = np.mean([r17 - r19 for r17, r19 in zip(PREV_BEST_FOLDS, best_folds_19)])
print(f"\nMean gain: {mean_gain:.4f}  Folds improved: {n_better}/5")
is_improvement = mean_gain > 0.03 and n_better >= 4
print(f"Criterion (gain>0.03 AND >=4 folds): {'MET' if is_improvement else 'NOT MET'}")

# ── OUTPUT RULE ───────────────────────────────────────────────────────────────
if is_improvement:
    print("\n" + "="*60)
    print("OUTPUT: writing submissions/19_trajectory.csv")
    print("="*60)

    # Refit full trajectory model on ALL training data
    TRAJ_FINAL_CACHE = C19 / "19_traj_final_preds.npy"
    if TRAJ_FINAL_CACHE.exists():
        traj_test_preds = np.load(TRAJ_FINAL_CACHE)
        print("  Loaded test trajectory preds from cache.")
    else:
        ct_final = joblib.load(C17 / "17_ct_final.joblib")
        raw_test_s1 = np.load(C11 / "11_tuned_raw_test.npy")
        sm_train_s1 = ridge_poly_smooth(oof_11, ages_a, pids_a,
                                        CFG14["w_blend"], CFG14["degree"], CFG14["ridge_lam"])
        sm_test_s1  = ridge_poly_smooth(raw_test_s1, X_test["age"].values,
                                        X_test["patient_id"].values,
                                        CFG14["w_blend"], CFG14["degree"], CFG14["ridge_lam"])

        # Fit trajectory params on ALL training patients
        traj_params_all = {}
        for pid_all, grp in X.groupby("patient_id"):
            ages_p  = grp["age"].values
            targs_p = y_vals[X.index.get_indexer(grp.index)]
            sort_o  = np.argsort(ages_p)
            traj_params_all[pid_all] = fit_quad_traj(
                ages_p[sort_o], targs_p[sort_o], lam=TRAJ_LAM)

        pat_feats_all = build_traj_features(X.copy(), sm_train_s1, ct_final)
        train_pids_all = pat_feats_all.index.values

        a_all = np.array([traj_params_all[p][0] if traj_params_all.get(p) else 0.0
                          for p in train_pids_all])
        b_all = np.array([traj_params_all[p][1] if traj_params_all.get(p) else 0.0
                          for p in train_pids_all])
        c_all = np.array([traj_params_all[p][2] if traj_params_all.get(p) else float(np.mean(y_vals))
                          for p in train_pids_all])

        Xf_all = pat_feats_all.values.copy()
        medians_all = np.nanmedian(Xf_all, axis=0)
        for j in range(Xf_all.shape[1]):
            mask = np.isnan(Xf_all[:, j])
            Xf_all[mask, j] = medians_all[j]

        m_a_f = HistGradientBoostingRegressor(
            max_iter=200, max_leaf_nodes=15, min_samples_leaf=10,
            learning_rate=0.05, random_state=0).fit(Xf_all, a_all)
        m_b_f = HistGradientBoostingRegressor(
            max_iter=200, max_leaf_nodes=15, min_samples_leaf=10,
            learning_rate=0.05, random_state=0).fit(Xf_all, b_all)
        m_c_f = HistGradientBoostingRegressor(
            max_iter=200, max_leaf_nodes=15, min_samples_leaf=10,
            learning_rate=0.05, random_state=0).fit(Xf_all, c_all)

        # Predict (a, b, c) for test patients
        pat_feats_test = build_traj_features(X_test.copy(), sm_test_s1, ct_final)
        test_pids_arr  = pat_feats_test.index.values
        Xv_test = pat_feats_test.values.copy()
        for j in range(Xv_test.shape[1]):
            mask = np.isnan(Xv_test[:, j])
            Xv_test[mask, j] = medians_all[j]

        a_hat_t = m_a_f.predict(Xv_test)
        b_hat_t = m_b_f.predict(Xv_test)
        c_hat_t = m_c_f.predict(Xv_test)

        test_ages_by_pat = {}
        for pid_t, grp_df in X_test.groupby("patient_id"):
            test_ages_by_pat[pid_t] = float(grp_df["age"].mean())

        pid_to_coef_t = {
            pid_t: (a_hat_t[i], b_hat_t[i], c_hat_t[i], test_ages_by_pat.get(pid_t, 0.0))
            for i, pid_t in enumerate(test_pids_arr)
        }

        test_pids_arr2 = X_test["patient_id"].values
        test_ages_arr2 = X_test["age"].values
        traj_test_preds = np.array([
            float(eval_quad_traj(pid_to_coef_t.get(p), np.array([a]))[0])
            for p, a in zip(test_pids_arr2, test_ages_arr2)
        ])
        np.save(TRAJ_FINAL_CACHE, traj_test_preds)
        print("  Trajectory test preds computed and cached.")

    # Blend with stage-2 test preds
    test_raw = [np.load(C17 / f"17_raw_test_seed{s}.npy") for s in SEEDS]
    raw_test_avg = np.mean(test_raw, axis=0)

    blended_test = (1 - best_alpha_19) * raw_test_avg + best_alpha_19 * traj_test_preds
    final_test = ridge_poly_smooth(blended_test, X_test["age"].values,
                                   X_test["patient_id"].values,
                                   0.25, 1, 1.0)
    final_test = np.clip(final_test, 0, 132)

    sub = pd.DataFrame({"target": final_test}, index=X_test.index)
    sub.index.name = "Index"
    sanity_check(sub, X_test, "19_trajectory")
    out_path = SUBS_DIR / "19_trajectory.csv"
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
            "key": "19_trajectory",
            "cv_rmse_mean": float(best_rmse_19),
            "cv_rmse_std":  float(np.std(best_folds_19)),
            "folds": [float(f) for f in best_folds_19],
            "best_alpha": float(best_alpha_19),
        }
        project.put("19_trajectory_meta", fold_report)
        workspace = cfg["workspace"]
        hub_url   = f"https://skore.probabl.ai/{workspace}/bobathon-esilv/cross-validations"
        print(f"Hub note pushed.")
    except Exception as e:
        print(f"HUB PUT PENDING: 19_trajectory  ({e})")

    std_19 = float(np.std(best_folds_19))
    print(f"\nREADY TO SUBMIT: submissions/19_trajectory.csv | "
          f"CV RMSE {best_rmse_19:.4f} +/- {std_19:.4f} | URL: {hub_url}")
else:
    print(f"\nNo improvement over 17 (CV {PREV_BEST_CV:.4f}). No CSV written.")
    traj_fold_rmse_smooth = []
    for _, val_idx in cv_splits:
        blended = (1 - best_alpha_19) * oof17_avg[val_idx] + best_alpha_19 * traj_oof[val_idx]
        sm = ridge_poly_smooth(blended, ages_a[val_idx], pids_a[val_idx], 0.25, 1, 1.0)
        traj_fold_rmse_smooth.append(np.sqrt(np.mean((sm - y_vals[val_idx])**2)))
    print(f"19_trajectory best CV RMSE: {best_rmse_19:.4f} +/- {np.std(traj_fold_rmse_smooth):.4f}")
    print(f"Best alpha: {best_alpha_19}")

print(f"\nTotal elapsed: {time.time()-t0:.0f}s")
print("STEP 4 (19_trajectory) DONE.")
