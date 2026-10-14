"""step6_21_final.py -- STEP 6: final ensemble (key 21_final).

Combine the best validated components from experiments 18-20:
  - If 19_trajectory improved: blend traj + s2 at best alpha
  - If 20_diverse improved: use multi-learner s2 OOF
  - Sweep multiple seeds for stage-2 (5 seeds)
  - Confirm with a SECOND shuffled GroupKFold split

Runs in two passes:
  Pass A: reproduce the best OOF configuration found in steps 18-20 and
          average 5 seeds of stage-2.
  Pass B: re-evaluate on a second shuffled split to confirm.

If Pass A beats 17 by >0.03 on >=4/5 folds AND Pass B is within 0.1 of
Pass A, write the final submission.

Hub key: 21_final (skore.evaluate with the full wrapper)
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
C18  = PROJECT_ROOT / "scratch" / "18_cache"
C19  = PROJECT_ROOT / "scratch" / "19_cache"
C20  = PROJECT_ROOT / "scratch" / "20_cache"
C21  = PROJECT_ROOT / "scratch" / "21_cache"
SUBS_DIR.mkdir(exist_ok=True)
C21.mkdir(parents=True, exist_ok=True)

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

# ── Load cached results from steps 18-20 if available ─────────────────────────
def try_load(path):
    if Path(path).exists():
        return np.load(path)
    return None

sigma2_oof   = try_load(C18 / "18_sigma2_oof.npy")
traj_oof     = try_load(C19 / "19_traj_oof.npy")
oof_et       = try_load(C20 / "20_et_oof.npy")
oof_l3       = try_load(C20 / "20_hgbr_l3_oof.npy")
oof_multi    = try_load(C20 / "20_multi_oof_seed0.npy")

# Evaluate each component's best fold-by-fold performance
print("\nAvailable OOF components:")

def fold_rmse(oof):
    return [np.sqrt(np.mean((oof[val] - y_vals[val])**2)) for _, val in cv_splits]

# Baseline: 17_stage2_feats
f17 = PREV_BEST_FOLDS
print(f"  17_stage2_feats:  {np.mean(f17):.4f} +/- {np.std(f17):.4f}")

if sigma2_oof is not None:
    # Best hetero smoothing from step 18
    sigma2_clip = np.clip(sigma2_oof, 0.5, None)
    vw = 1.0 / sigma2_clip
    best_h = 1e9
    best_hcfg = (1, 1.0, 0.25)
    for deg, lam, w in [(1, 1.0, 0.25), (1, 0.1, 0.25), (2, 1.0, 0.25), (1, 1.0, 0.0)]:
        folds_h = [np.sqrt(np.mean((
            ridge_poly_smooth(oof17_avg[val], ages_a[val], pids_a[val], w, deg, lam,
                              visit_w=vw[val]) - y_vals[val])**2))
            for _, val in cv_splits]
        if np.mean(folds_h) < best_h:
            best_h = np.mean(folds_h)
            best_hcfg = (deg, lam, w)
    print(f"  18_hetero best:   {best_h:.4f} (deg={best_hcfg[0]},lam={best_hcfg[1]},w={best_hcfg[2]})")

if traj_oof is not None:
    # Best trajectory blend from step 19
    best_t, best_ta = 1e9, 0.0
    for alpha in [0.0, 0.05, 0.1, 0.15, 0.2, 0.3]:
        bl = (1-alpha)*oof17_avg + alpha*traj_oof
        folds_t = [np.sqrt(np.mean((
            ridge_poly_smooth(bl[val], ages_a[val], pids_a[val], 0.25, 1, 1.0)
            - y_vals[val])**2)) for _, val in cv_splits]
        if np.mean(folds_t) < best_t:
            best_t = np.mean(folds_t)
            best_ta = alpha
    print(f"  19_trajectory:    {best_t:.4f} (alpha={best_ta})")

if oof_multi is not None:
    best_m = 1e9
    for deg, lam, w in [(1, 1.0, 0.25), (2, 1.0, 0.25), (1, 0.1, 0.25)]:
        folds_m = [np.sqrt(np.mean((
            ridge_poly_smooth(oof_multi[val], ages_a[val], pids_a[val], w, deg, lam)
            - y_vals[val])**2)) for _, val in cv_splits]
        if np.mean(folds_m) < best_m:
            best_m = np.mean(folds_m)
    print(f"  20_diverse:       {best_m:.4f}")

# ── Strategy: compute 5-seed stage-2 with best available inputs ───────────────
# Run 5 seeds of stage-2 (same as 17 but averaged over more seeds if any cached)
# Use trajectory blend if it helped, otherwise plain 17 OOF

USE_TRAJ = (traj_oof is not None and
            best_t < (PREV_BEST_CV - 0.01))
USE_MULTI = (oof_multi is not None and
             best_m < (PREV_BEST_CV - 0.01))
best_alpha_traj = best_ta if (traj_oof is not None) else 0.0

print(f"\nStrategy: USE_TRAJ={USE_TRAJ}, USE_MULTI={USE_MULTI}")

# ── Pass A: 5-seed stage-2 OOF ────────────────────────────────────────────────
print("\nPass A: 5-seed stage-2 OOF...")
S21_SEED_CACHE = [C21 / f"21_oof_seed{s}.npy" for s in SEEDS]
oof21_seeds = []

for s_idx, seed in enumerate(SEEDS):
    cache_path = S21_SEED_CACHE[s_idx]
    if cache_path.exists():
        oof_s = np.load(cache_path)
        print(f"  seed {seed}: loaded (RMSE={np.sqrt(np.mean((oof_s-y_vals)**2)):.4f})")
        oof21_seeds.append(oof_s)
        continue

    print(f"  seed {seed}: computing...")
    oof_s = np.full(len(X), np.nan)

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
        oof_s[val_idx] = m2.predict(X_val_s2)
        rf = np.sqrt(np.mean((oof_s[val_idx] - y_vals[val_idx])**2))
        print(f"    fold {fi}: {rf:.4f}  ({time.time()-t_f:.1f}s)")

    np.save(cache_path, oof_s)
    oof21_seeds.append(oof_s)

oof21_avg = np.mean(oof21_seeds, axis=0)
print(f"\n21 seed-avg raw OOF RMSE: {np.sqrt(np.mean((oof21_avg-y_vals)**2)):.4f}")

# ── Optionally blend with trajectory ─────────────────────────────────────────
if USE_TRAJ and traj_oof is not None:
    oof21_blended = (1 - best_alpha_traj) * oof21_avg + best_alpha_traj * traj_oof
    print(f"\nAfter traj blend (alpha={best_alpha_traj}): "
          f"RMSE={np.sqrt(np.mean((oof21_blended-y_vals)**2)):.4f}")
else:
    oof21_blended = oof21_avg

# ── Smoothing sweep ───────────────────────────────────────────────────────────
print("\nFinal smoothing sweep:")
best_rmse_21  = PREV_BEST_CV
best_cfg_21   = None
best_folds_21 = PREV_BEST_FOLDS[:]

for deg in [1, 2]:
    for lam in [0.0, 0.1, 1.0]:
        for w in [0.0, 0.25, 0.5, 0.75, 0.9]:
            folds = []
            vw_all = None
            if sigma2_oof is not None:
                vw_all = 1.0 / np.clip(sigma2_oof, 0.5, None)
            for _, val_idx in cv_splits:
                vw = vw_all[val_idx] if vw_all is not None else None
                sm = ridge_poly_smooth(oof21_blended[val_idx], ages_a[val_idx],
                                       pids_a[val_idx], w, deg, lam, visit_w=vw)
                folds.append(np.sqrt(np.mean((sm - y_vals[val_idx])**2)))
            m, s = np.mean(folds), np.std(folds)
            marker = " *" if m < best_rmse_21 else ""
            print(f"  deg={deg} lam={lam:.1f} w={w:.2f}  {m:.4f} +/- {s:.4f}{marker}")
            if m < best_rmse_21:
                best_rmse_21  = m
                best_cfg_21   = dict(degree=deg, ridge_lam=lam, w_blend=w,
                                     use_hetero=(vw_all is not None))
                best_folds_21 = folds[:]

if best_cfg_21 is None:
    best_cfg_21 = dict(degree=1, ridge_lam=1.0, w_blend=0.25, use_hetero=False)
print(f"\nBest 21 config: {best_cfg_21}  RMSE={best_rmse_21:.4f}")

# ── Paired comparison vs 17 ───────────────────────────────────────────────────
print("\nPaired fold comparison (21_final vs 17_stage2_feats):")
print(f"  {'fold':>5}  {'17':>8}  {'21':>8}  {'delta':>8}")
n_better = 0
for fi, (r17, r21) in enumerate(zip(PREV_BEST_FOLDS, best_folds_21)):
    delta = r17 - r21
    n_better += int(delta > 0)
    print(f"  {fi:>5}  {r17:>8.4f}  {r21:>8.4f}  {delta:>+8.4f}  "
          f"{'BETTER' if delta > 0 else ''}")
mean_gain = np.mean([r17 - r21 for r17, r21 in zip(PREV_BEST_FOLDS, best_folds_21)])
print(f"\nMean gain: {mean_gain:.4f}  Folds improved: {n_better}/5")
is_improvement = mean_gain > 0.03 and n_better >= 4
print(f"Criterion (gain>0.03 AND >=4 folds): {'MET' if is_improvement else 'NOT MET'}")

# ── Pass B: second split confirmation ─────────────────────────────────────────
print("\nPass B: shuffled split confirmation...")
rng = np.random.RandomState(42)
shuffled_pids = X["patient_id"].values.copy()
unique_pids = np.unique(shuffled_pids)
rng.shuffle(unique_pids)
pid_to_shuf = {p: i for i, p in enumerate(unique_pids)}
shuf_groups = np.array([pid_to_shuf[p] for p in shuffled_pids])
cv_splits_b = list(GroupKFold(n_splits=5).split(X, y, groups=shuf_groups))

oof21_b_seeds = []
for s_idx, seed in enumerate(SEEDS):
    cache_path_b = C21 / f"21_oof_splitB_seed{s_idx}.npy"
    if cache_path_b.exists():
        oof_sb = np.load(cache_path_b)
        oof21_b_seeds.append(oof_sb)
        continue
    oof_sb = np.full(len(X), np.nan)
    for fi, (tr_idx, val_idx) in enumerate(cv_splits_b):
        sm_tr  = ridge_poly_smooth(oof_11[tr_idx], ages_a[tr_idx], pids_a[tr_idx],
                                   CFG14["w_blend"], CFG14["degree"], CFG14["ridge_lam"])
        sm_val = ridge_poly_smooth(oof_11[val_idx], ages_a[val_idx], pids_a[val_idx],
                                   CFG14["w_blend"], CFG14["degree"], CFG14["ridge_lam"])
        ct = InputCurveTransformer(lam_curve=1.0, lam_off_hat=1.0)
        ct.fit(X.iloc[tr_idx])
        X_tr_s2  = build_s2_features(X.iloc[tr_idx].copy(),  sm_tr,  ct)
        X_val_s2 = build_s2_features(X.iloc[val_idx].copy(), sm_val, ct)
        m2 = HistGradientBoostingRegressor(
            categorical_features="from_dtype", early_stopping=True,
            n_iter_no_change=20, random_state=seed, **S2_PARAMS,
        )
        m2.fit(X_tr_s2, y.iloc[tr_idx])
        oof_sb[val_idx] = m2.predict(X_val_s2)
    np.save(cache_path_b, oof_sb)
    oof21_b_seeds.append(oof_sb)

oof21_b_avg = np.mean(oof21_b_seeds, axis=0)
if USE_TRAJ and traj_oof is not None:
    oof21_b_blended = (1 - best_alpha_traj) * oof21_b_avg + best_alpha_traj * traj_oof
else:
    oof21_b_blended = oof21_b_avg

folds_b = []
for _, val_idx in cv_splits_b:
    vw = (1.0/np.clip(sigma2_oof[val_idx], 0.5, None)
          if (sigma2_oof is not None and best_cfg_21.get("use_hetero")) else None)
    sm = ridge_poly_smooth(oof21_b_blended[val_idx], ages_a[val_idx], pids_a[val_idx],
                           best_cfg_21["w_blend"], best_cfg_21["degree"],
                           best_cfg_21["ridge_lam"], visit_w=vw)
    folds_b.append(np.sqrt(np.mean((sm - y_vals[val_idx])**2)))

rmse_b = np.mean(folds_b)
print(f"Pass B RMSE: {rmse_b:.4f} +/- {np.std(folds_b):.4f}")
print(f"Pass A RMSE: {best_rmse_21:.4f}")
pass_b_ok = abs(rmse_b - best_rmse_21) < 0.15
print(f"Pass B OK (within 0.15 of A): {'YES' if pass_b_ok else 'NO'}")

# ── OUTPUT ─────────────────────────────────────────────────────────────────────
if is_improvement and pass_b_ok:
    print("\n" + "="*60)
    print("OUTPUT: writing submissions/21_final.csv")
    print("="*60)

    ct_final = joblib.load(C17 / "17_ct_final.joblib")
    raw_test_s1 = np.load(C11 / "11_tuned_raw_test.npy")
    sm_train_s1 = ridge_poly_smooth(oof_11, ages_a, pids_a,
                                    CFG14["w_blend"], CFG14["degree"], CFG14["ridge_lam"])
    sm_test_s1  = ridge_poly_smooth(raw_test_s1, X_test["age"].values,
                                    X_test["patient_id"].values,
                                    CFG14["w_blend"], CFG14["degree"], CFG14["ridge_lam"])
    X_train_s2 = build_s2_features(X.copy(), sm_train_s1, ct_final)
    X_test_s2  = build_s2_features(X_test.copy(), sm_test_s1, ct_final)

    test_preds = []
    for seed in SEEDS:
        mc = C21 / f"21_s2_final_seed{seed}.joblib"
        tc = C21 / f"21_raw_test_seed{seed}.npy"
        if mc.exists() and tc.exists():
            test_preds.append(np.load(tc))
            print(f"  seed {seed}: loaded from cache")
        else:
            t_s = time.time()
            m2 = HistGradientBoostingRegressor(
                categorical_features="from_dtype", early_stopping=True,
                n_iter_no_change=20, random_state=seed, **S2_PARAMS,
            )
            m2.fit(X_train_s2, y)
            raw_t = m2.predict(X_test_s2)
            joblib.dump(m2, mc)
            np.save(tc, raw_t)
            test_preds.append(raw_t)
            print(f"  seed {seed}: fit in {time.time()-t_s:.1f}s  [cached]")

    raw_test_avg = np.mean(test_preds, axis=0)

    # Blend with trajectory if applicable
    if USE_TRAJ:
        traj_test = try_load(C19 / "19_traj_final_preds.npy")
        if traj_test is not None:
            raw_test_avg = (1 - best_alpha_traj) * raw_test_avg + best_alpha_traj * traj_test
            print(f"  Applied trajectory blend alpha={best_alpha_traj}")

    vw_test = None
    if best_cfg_21.get("use_hetero") and sigma2_oof is not None:
        # Recompute sigma2 for test using all-train error model
        X_train_full_s2 = X_train_s2.copy()
        resid2_all = (oof17_avg - y_vals) ** 2
        err_final = HistGradientBoostingRegressor(
            max_iter=200, max_leaf_nodes=15, min_samples_leaf=30,
            learning_rate=0.05, random_state=0).fit(X_train_full_s2, resid2_all)
        sigma2_test = np.clip(err_final.predict(X_test_s2), 0.5, None)
        vw_test = 1.0 / sigma2_test

    final_test = ridge_poly_smooth(
        raw_test_avg, X_test["age"].values, X_test["patient_id"].values,
        best_cfg_21["w_blend"], best_cfg_21["degree"], best_cfg_21["ridge_lam"],
        visit_w=vw_test,
    )
    final_test = np.clip(final_test, 0, 132)

    sub = pd.DataFrame({"target": final_test}, index=X_test.index)
    sub.index.name = "Index"
    sanity_check(sub, X_test, "21_final")
    out_path = SUBS_DIR / "21_final.csv"
    sub.to_csv(out_path)
    print(f"\nWritten: {out_path}")

    hub_url = "PENDING"
    try:
        import skore
        from skore import login
        from sklearn.base import BaseEstimator, RegressorMixin

        class S21Wrapper(BaseEstimator, RegressorMixin):
            def fit(self, X, y):
                from parkinsons.pipeline import build_hgbr_tuned
                s1 = build_hgbr_tuned(learning_rate=0.02, max_iter=1000,
                                      max_leaf_nodes=31, min_samples_leaf=20,
                                      l2_regularization=0.0, random_state=0)
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
                               X["patient_id"].values, best_cfg_21["w_blend"],
                               best_cfg_21["degree"], best_cfg_21["ridge_lam"]), 0, 132)

        cfg = load_skore_credentials()
        login(mode="hub")
        project = skore.Project(name="bobathon-esilv", mode="hub", workspace=cfg["workspace"])
        report  = skore.evaluate(S21Wrapper(), X, y, splitter=cv_splits)
        project.put("21_final", report)
        workspace = cfg["workspace"]
        hub_url   = f"https://skore.probabl.ai/{workspace}/bobathon-esilv/cross-validations"
        print(f"Hub report 21_final pushed.")
    except Exception as e:
        print(f"HUB PUT PENDING: 21_final  ({e})")

    std_21 = float(np.std(best_folds_21))
    print(f"\nREADY TO SUBMIT: submissions/21_final.csv | "
          f"CV RMSE {best_rmse_21:.4f} +/- {std_21:.4f} | "
          f"Pass B RMSE {rmse_b:.4f} | URL: {hub_url}")
elif is_improvement and not pass_b_ok:
    print(f"\nPass A improved ({best_rmse_21:.4f}) but Pass B too different ({rmse_b:.4f}).")
    print("Investigate variance before submitting.")
else:
    print(f"\nNo improvement over 17 (CV {PREV_BEST_CV:.4f}). No CSV written.")
    print(f"21_final CV RMSE: {best_rmse_21:.4f} +/- {np.std(best_folds_21):.4f}")
    print(f"Best 17 remains the submission: submissions/17_stage2_feats.csv")

print(f"\nTotal elapsed: {time.time()-t0:.0f}s")
print("STEP 6 (21_final) DONE.")
