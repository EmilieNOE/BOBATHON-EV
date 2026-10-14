"""step8_23_rater_lpo.py -- B1-rigor: leave-patient-out rater bias (key 23_rater_lpo).

The standard per-rater bias (22_rater) computes mean(target - s1_smooth) over
ALL training-fold rows with that rater_id.  A single patient may have many
visits with the same rater, so the patient's own target enters its own feature.
This is not direct leakage (the fold already excludes val patients) but it is
a form of within-training over-smoothing.

Fix: for each training-fold row, compute per-rater mean(target - s1_smooth)
using only rows whose patient_id DIFFERS from that row's patient_id.
This is leave-patient-out (LPO) shrinkage of the rater bias.

Also adds a count-based reliability weight: for raters with few patients
(<= 5 unique patients), shrink the bias estimate toward 0 with
  bias_lpo_shrunk = bias_lpo * n_patients / (n_patients + shrink_k)
where shrink_k is tuned on CV (try 5, 10, 20).

No other architecture changes from 22_rater.

Hub key: 23_rater_lpo
PYTHONUTF8=1. ASCII-only prints.
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
C23  = PROJECT_ROOT / "scratch" / "23_cache"
SUBS_DIR.mkdir(exist_ok=True)
C23.mkdir(parents=True, exist_ok=True)

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
print(f"Data + caches loaded ({time.time()-t0:.1f}s)")


# ── LPO rater bias helper ──────────────────────────────────────────────────────

def compute_lpo_rater_bias(X_tr: pd.DataFrame, resid: np.ndarray,
                           shrink_k: float = 10.0):
    """Leave-patient-out per-rater mean residual with count shrinkage.

    For each row i with rater_id r and patient_id p:
      bias_lpo[i] = mean(resid[j] for j where rater_id[j]==r AND patient_id[j]!=p)
    Then shrink toward 0:
      bias_shrunk[i] = bias_lpo[i] * n_unique_patients(r,excluding p) /
                                     (n_unique_patients(r,excluding p) + shrink_k)

    Returns
    -------
    bias_arr : (n,) float array, bias estimate for each row
    rater_global_map : dict {rater_id: (sum_resid, n_patients, patient_set)}
                       for applying to val/test rows at transform time
    """
    n = len(X_tr)
    rater_ids = X_tr["rater_id"].values
    pid_vals  = X_tr["patient_id"].values

    # Build per-rater table: for each rater, list of (pid, mean_resid_for_pid)
    # First group by (rater, patient) -> mean residual per patient-rater pair
    df = pd.DataFrame({
        "rater": rater_ids,
        "pid":   pid_vals,
        "resid": resid,
    })
    rater_pid_mean = (df.groupby(["rater", "pid"])["resid"]
                       .mean().rename("rp_mean").reset_index())

    # Per-rater summary: sum of pid-means and count of unique pids
    rater_summary = {}
    for r, grp in rater_pid_mean.groupby("rater"):
        pids_r   = grp["pid"].values
        means_r  = grp["rp_mean"].values
        total    = float(means_r.sum())
        n_pat    = len(pids_r)
        rater_summary[r] = {"total": total, "n_pat": n_pat,
                             "pid_means": dict(zip(pids_r, means_r))}

    # LPO: for each row, exclude its own patient
    bias_arr = np.zeros(n, dtype=float)
    for i in range(n):
        r   = rater_ids[i]
        p   = pid_vals[i]
        rs  = rater_summary.get(r)
        if rs is None or rs["n_pat"] == 0:
            bias_arr[i] = 0.0
            continue
        # Subtract this patient's contribution
        p_mean   = rs["pid_means"].get(p, 0.0)
        excl_sum = rs["total"] - p_mean
        excl_n   = rs["n_pat"] - (1 if p in rs["pid_means"] else 0)
        if excl_n <= 0:
            raw_bias = 0.0
            n_eff    = 0
        else:
            raw_bias = excl_sum / excl_n
            n_eff    = excl_n
        bias_arr[i] = raw_bias * n_eff / (n_eff + shrink_k)

    # Global map for val/test (uses all training rows): bias using all patients
    rater_global_map = {}
    for r, rs in rater_summary.items():
        n_eff = rs["n_pat"]
        raw   = rs["total"] / n_eff if n_eff > 0 else 0.0
        rater_global_map[r] = raw * n_eff / (n_eff + shrink_k)

    return bias_arr, rater_global_map


def add_rater_lpo_features(X_frame: pd.DataFrame,
                           s1_smooth: np.ndarray,
                           y_tr_vals,
                           rater_global_map,
                           shrink_k: float = 10.0):
    """Add LPO rater bias + visit_rank + patient_age_std to X_frame.

    For training frames: y_tr_vals != None → compute LPO bias.
    For val/test frames: y_tr_vals is None → use rater_global_map.
    """
    X_out = X_frame.copy()
    n = len(X_out)

    if y_tr_vals is not None:
        resid = y_tr_vals - s1_smooth
        bias_arr, rater_global_map_out = compute_lpo_rater_bias(
            X_out, resid, shrink_k=shrink_k)
    else:
        # Val/test: look up from global map
        bias_arr = np.array([rater_global_map.get(r, 0.0)
                              for r in X_out["rater_id"].values], dtype=float)
        rater_global_map_out = rater_global_map

    # visit_rank and patient_age_std
    visit_rank  = np.zeros(n, dtype=float)
    pat_age_std = np.zeros(n, dtype=float)
    X_sorted = X_out.sort_values(["patient_id", "age"])
    for pid, grp in X_sorted.groupby("patient_id"):
        pos = np.array([X_out.index.get_loc(i) for i in grp.index])
        visit_rank[pos]  = np.arange(len(pos), dtype=float)
        pat_age_std[pos] = float(np.std(grp["age"].values)) if len(pos) > 1 else 0.0

    X_out["rater_bias_lpo"]  = bias_arr
    X_out["visit_rank"]      = visit_rank
    X_out["patient_age_std"] = pat_age_std
    return X_out, rater_global_map_out


def build_s2_features_lpo(X_frame, s1_preds, curve_transformer,
                           y_tr_vals, rater_global_map, shrink_k=10.0):
    """Full stage-2 features: standard 17 + LPO rater."""
    X_s2 = build_s2_features(X_frame, s1_preds, curve_transformer)
    X_rater, rgm = add_rater_lpo_features(
        X_frame.copy(), s1_preds, y_tr_vals, rater_global_map, shrink_k)
    for col in ["rater_bias_lpo", "visit_rank", "patient_age_std"]:
        X_s2[col] = X_rater[col].values
    return X_s2, rgm


# ── nested CV: sweep shrink_k and compute OOF ────────────────────────────────
# First sweep shrink_k with seed=0 only to pick the best k, then run 5 seeds.

def run_oof_seed(seed, shrink_k, suffix=""):
    """Run one seed; return oof array. Loads cache if available."""
    cache_path = C23 / f"23_oof_k{int(shrink_k)}_seed{seed}{suffix}.npy"
    if cache_path.exists():
        return np.load(cache_path)

    oof = np.full(len(X), np.nan)
    for fi, (tr_idx, val_idx) in enumerate(cv_splits):
        sm_tr  = ridge_poly_smooth(oof_11[tr_idx], ages_a[tr_idx], pids_a[tr_idx],
                                   CFG14["w_blend"], CFG14["degree"], CFG14["ridge_lam"])
        sm_val = ridge_poly_smooth(oof_11[val_idx], ages_a[val_idx], pids_a[val_idx],
                                   CFG14["w_blend"], CFG14["degree"], CFG14["ridge_lam"])
        ct = InputCurveTransformer(lam_curve=1.0, lam_off_hat=1.0)
        ct.fit(X.iloc[tr_idx])
        X_tr_s2, rgm = build_s2_features_lpo(
            X.iloc[tr_idx].copy(), sm_tr, ct,
            y_tr_vals=y_vals[tr_idx], rater_global_map=None, shrink_k=shrink_k)
        X_val_s2, _ = build_s2_features_lpo(
            X.iloc[val_idx].copy(), sm_val, ct,
            y_tr_vals=None, rater_global_map=rgm, shrink_k=shrink_k)
        m2 = HistGradientBoostingRegressor(
            categorical_features="from_dtype", early_stopping=True,
            n_iter_no_change=20, random_state=seed, **S2_PARAMS)
        m2.fit(X_tr_s2, y.iloc[tr_idx])
        oof[val_idx] = m2.predict(X_val_s2)
    np.save(cache_path, oof)
    return oof


print("Sweeping shrink_k (seed=0)...")
best_k, best_k_rmse = 10.0, 1e9
for k in [5.0, 10.0, 20.0, 50.0]:
    oof_k = run_oof_seed(0, k)
    folds_k = [np.sqrt(np.mean((oof_k[val] - y_vals[val])**2)) for _, val in cv_splits]
    m_k = np.mean(folds_k)
    print(f"  k={k:5.0f}  RMSE={m_k:.4f}")
    if m_k < best_k_rmse:
        best_k_rmse = m_k
        best_k = k
print(f"Best k: {best_k}  RMSE={best_k_rmse:.4f}")

print(f"\nRunning 5 seeds with k={best_k}...")
oof_seeds = []
for seed in SEEDS:
    t_s = time.time()
    oof_s = run_oof_seed(seed, best_k)
    rmse_s = np.sqrt(np.mean((oof_s - y_vals)**2))
    print(f"  seed {seed}: RMSE={rmse_s:.4f}  ({time.time()-t_s:.1f}s)")
    oof_seeds.append(oof_s)

oof_avg = np.mean(oof_seeds, axis=0)
rmse_raw = np.sqrt(np.mean((oof_avg - y_vals)**2))
print(f"\n23 seed-avg raw OOF RMSE: {rmse_raw:.4f}")

# ── smoothing sweep ───────────────────────────────────────────────────────────
print("\nSmoothing sweep:")
best_rmse_23  = rmse_raw
best_cfg_23   = dict(degree=1, ridge_lam=1.0, w_blend=0.0)
best_folds_23 = [np.sqrt(np.mean((oof_avg[val] - y_vals[val])**2))
                 for _, val in cv_splits]

for deg in [1, 2]:
    for lam in [0.0, 0.1, 1.0]:
        for w in [0.0, 0.25, 0.5]:
            folds = []
            for _, val_idx in cv_splits:
                sm = ridge_poly_smooth(oof_avg[val_idx], ages_a[val_idx],
                                       pids_a[val_idx], w, deg, lam)
                folds.append(np.sqrt(np.mean((sm - y_vals[val_idx])**2)))
            m, s = np.mean(folds), np.std(folds)
            marker = " *" if m < best_rmse_23 else ""
            print(f"  deg={deg} lam={lam:.1f} w={w:.2f}  {m:.4f} +/- {s:.4f}{marker}")
            if m < best_rmse_23:
                best_rmse_23  = m
                best_cfg_23   = dict(degree=deg, ridge_lam=lam, w_blend=w)
                best_folds_23 = folds[:]

print(f"\nBest 23 config: {best_cfg_23}  RMSE={best_rmse_23:.4f}")

# ── paired comparison ─────────────────────────────────────────────────────────
print("\nPaired fold comparison (23_rater_lpo vs 17):")
print(f"  {'fold':>5}  {'17':>8}  {'23':>8}  {'delta':>8}")
n_better = 0
for fi, (r17, r23) in enumerate(zip(PREV_BEST_FOLDS, best_folds_23)):
    delta = r17 - r23
    n_better += int(delta > 0)
    print(f"  {fi:>5}  {r17:>8.4f}  {r23:>8.4f}  {delta:>+8.4f}  "
          f"{'BETTER' if delta > 0 else ''}")
mean_gain = np.mean([r17 - r23 for r17, r23 in zip(PREV_BEST_FOLDS, best_folds_23)])
print(f"\nMean gain: {mean_gain:.4f}  Folds improved: {n_better}/5")
is_improvement = mean_gain > 0.03 and n_better >= 4
print(f"Criterion (gain>0.03 AND >=4 folds): {'MET' if is_improvement else 'NOT MET'}")

# ── output rule ────────────────────────────────────────────────────────────────
if is_improvement:
    print("\n" + "="*60)
    print("OUTPUT: writing submissions/23_rater_lpo.csv")
    print("="*60)

    ct_final = joblib.load(C17 / "17_ct_final.joblib")
    raw_test_s1 = np.load(C11 / "11_tuned_raw_test.npy")
    sm_train_s1 = ridge_poly_smooth(oof_11, ages_a, pids_a,
                                    CFG14["w_blend"], CFG14["degree"], CFG14["ridge_lam"])
    sm_test_s1  = ridge_poly_smooth(raw_test_s1, X_test["age"].values,
                                    X_test["patient_id"].values,
                                    CFG14["w_blend"], CFG14["degree"], CFG14["ridge_lam"])

    X_train_s2, rgm_final = build_s2_features_lpo(
        X.copy(), sm_train_s1, ct_final,
        y_tr_vals=y_vals, rater_global_map=None, shrink_k=best_k)
    X_test_s2, _ = build_s2_features_lpo(
        X_test.copy(), sm_test_s1, ct_final,
        y_tr_vals=None, rater_global_map=rgm_final, shrink_k=best_k)

    test_preds = []
    for seed in SEEDS:
        mc = C23 / f"23_s2_final_k{int(best_k)}_seed{seed}.joblib"
        tc = C23 / f"23_raw_test_k{int(best_k)}_seed{seed}.npy"
        if mc.exists() and tc.exists():
            test_preds.append(np.load(tc))
        else:
            t_s = time.time()
            m2 = HistGradientBoostingRegressor(
                categorical_features="from_dtype", early_stopping=True,
                n_iter_no_change=20, random_state=seed, **S2_PARAMS)
            m2.fit(X_train_s2, y)
            raw_t = m2.predict(X_test_s2)
            joblib.dump(m2, mc)
            np.save(tc, raw_t)
            test_preds.append(raw_t)
            print(f"  seed {seed}: fit {time.time()-t_s:.1f}s")

    raw_test_avg = np.mean(test_preds, axis=0)
    final_test = ridge_poly_smooth(
        raw_test_avg, X_test["age"].values, X_test["patient_id"].values,
        best_cfg_23["w_blend"], best_cfg_23["degree"], best_cfg_23["ridge_lam"])
    final_test = np.clip(final_test, 0, 132)

    sub = pd.DataFrame({"target": final_test}, index=X_test.index)
    sub.index.name = "Index"
    sanity_check(sub, X_test, "23_rater_lpo")
    out_path = SUBS_DIR / "23_rater_lpo.csv"
    sub.to_csv(out_path)
    print(f"Written: {out_path}")

    hub_url = "PENDING"
    try:
        import skore; from skore import login
        cfg = load_skore_credentials()
        login(mode="hub")
        project = skore.Project(name="bobathon-esilv", mode="hub", workspace=cfg["workspace"])
        fold_report = {"key": "23_rater_lpo", "cv_rmse_mean": float(best_rmse_23),
                       "cv_rmse_std": float(np.std(best_folds_23)),
                       "folds": [float(f) for f in best_folds_23], "best_k": float(best_k)}
        project.put("23_rater_lpo_meta", fold_report)
        hub_url = f"https://skore.probabl.ai/{cfg['workspace']}/bobathon-esilv/cross-validations"
    except Exception as e:
        print(f"HUB PUT PENDING: ({e})")

    std_23 = float(np.std(best_folds_23))
    print(f"\nREADY TO SUBMIT: submissions/23_rater_lpo.csv | "
          f"CV RMSE {best_rmse_23:.4f} +/- {std_23:.4f} | URL: {hub_url}")
else:
    print(f"\nNo improvement. CV RMSE: {best_rmse_23:.4f} +/- {np.std(best_folds_23):.4f}")

print(f"\nTotal elapsed: {time.time()-t0:.0f}s")
print("STEP 8 (23_rater_lpo) DONE.")
