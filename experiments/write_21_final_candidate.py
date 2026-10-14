"""write_21_final_candidate.py -- Task 2: write submissions/21_final.csv as bundle candidate.

Uses the cached 21_final OOF and final-model artefacts:
  - 5-seed stage-2 (scratch/21_cache/21_oof_seed{0..4}.npy)
  - trajectory blend at alpha=0.2 (scratch/19_cache/19_traj_final_preds.npy)
  - best smoothing: deg=1, lam=0.0, w=0.0 (no smoothing was best in 21_final)

Sanity-checks the output and prints CANDIDATE notice.
Does NOT push to hub as the primary 21_final skore report (that requires
the full skore.evaluate wrapper and is slow); instead puts a metadata dict.

PYTHONUTF8=1. ASCII-only prints.
"""

import sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import joblib
import numpy as np
import pandas as pd

from parkinsons import PROJECT_ROOT
from parkinsons.data import load_dataset, load_test_dataset
from parkinsons.hub import load_skore_credentials
from shared import ridge_poly_smooth, sanity_check
import step2_17_stage2_feats as _s17
InputCurveTransformer = _s17.InputCurveTransformer
build_s2_features     = _s17.build_s2_features

DATA_DIR = PROJECT_ROOT / "data"
SUBS_DIR = PROJECT_ROOT / "submissions"
C11  = PROJECT_ROOT / "scratch" / "13_cache"
C17  = PROJECT_ROOT / "scratch" / "17_cache"
C19  = PROJECT_ROOT / "scratch" / "19_cache"
C21  = PROJECT_ROOT / "scratch" / "21_cache"
SUBS_DIR.mkdir(exist_ok=True)

CFG14    = dict(degree=2, ridge_lam=0.1, w_blend=0.9)
S2_PARAMS = dict(learning_rate=0.02, max_iter=300, max_leaf_nodes=31,
                 min_samples_leaf=50, l2_regularization=0.0)
SEEDS    = [0, 1, 2, 3, 4]
# Results from 21_final run
BEST_FOLDS_21 = None   # will be recomputed below
BEST_CFG_21   = dict(degree=1, ridge_lam=0.0, w_blend=0.0, use_hetero=False)
BEST_ALPHA    = 0.2    # trajectory blend alpha
PASS_A_RMSE   = 3.9832
PASS_B_RMSE   = 3.9743

t0 = time.time()

X, y   = load_dataset(data_dir=DATA_DIR)
X_test = load_test_dataset(data_dir=DATA_DIR)
ages_a = X["age"].values
pids_a = X["patient_id"].values
y_vals = y.values.astype(float)

from sklearn.model_selection import GroupKFold
cv_splits = list(GroupKFold(n_splits=5).split(X, y, groups=X["patient_id"]))

# ── Load cached 21_final OOF ──────────────────────────────────────────────────
print("Loading 21_final cached OOF seeds...")
oof21_seeds = []
for seed in SEEDS:
    p = C21 / f"21_oof_seed{seed}.npy"
    if p.exists():
        oof21_seeds.append(np.load(p))
        print(f"  seed {seed}: loaded")
    else:
        print(f"  seed {seed}: MISSING - cannot write 21_final.csv without full OOF")
        sys.exit(1)

oof21_avg = np.mean(oof21_seeds, axis=0)

# Apply trajectory blend
traj_oof_path = C19 / "19_traj_oof.npy"
if traj_oof_path.exists():
    traj_oof = np.load(traj_oof_path)
    oof21_blended = (1 - BEST_ALPHA) * oof21_avg + BEST_ALPHA * traj_oof
    print(f"Traj blend applied (alpha={BEST_ALPHA}): "
          f"RMSE={np.sqrt(np.mean((oof21_blended-y_vals)**2)):.4f}")
else:
    oof21_blended = oof21_avg
    print("No traj OOF found; using 21_avg only")

# Recompute per-fold metrics with best cfg (w=0.0 means no smoothing)
folds_21 = []
for _, val_idx in cv_splits:
    sm = ridge_poly_smooth(oof21_blended[val_idx], ages_a[val_idx],
                           pids_a[val_idx],
                           BEST_CFG_21["w_blend"], BEST_CFG_21["degree"],
                           BEST_CFG_21["ridge_lam"])
    folds_21.append(np.sqrt(np.mean((sm - y_vals[val_idx])**2)))

print(f"\n21_final OOF (Pass A):  {np.mean(folds_21):.4f} +/- {np.std(folds_21):.4f}")
print(f"Per-fold: {[round(f,4) for f in folds_21]}")
print(f"Pass B (shuffled):      {PASS_B_RMSE:.4f}")
print(f"Mean gain vs 17:        {3.9949 - np.mean(folds_21):.4f}")

# ── Build final test predictions ──────────────────────────────────────────────
print("\nBuilding test predictions...")

ct_final    = joblib.load(C17 / "17_ct_final.joblib")
oof_11      = np.load(C11 / "11_oof_raw.npy")
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
        print(f"  seed {seed}: fitting final model...")
        m2 = HistGradientBoostingRegressor(
            categorical_features="from_dtype", early_stopping=True,
            n_iter_no_change=20, random_state=seed, **S2_PARAMS)
        m2.fit(X_train_s2, y)
        raw_t = m2.predict(X_test_s2)
        joblib.dump(m2, mc)
        np.save(tc, raw_t)
        test_preds.append(raw_t)

from sklearn.ensemble import HistGradientBoostingRegressor

raw_test_avg = np.mean(test_preds, axis=0)

# Apply trajectory blend to test
traj_test_path = C19 / "19_traj_final_preds.npy"
if traj_test_path.exists():
    traj_test = np.load(traj_test_path)
    raw_test_blended = (1 - BEST_ALPHA) * raw_test_avg + BEST_ALPHA * traj_test
    print(f"Traj blend applied to test (alpha={BEST_ALPHA})")
else:
    raw_test_blended = raw_test_avg
    print("No traj test preds found; using 21 only")

# No post-smoothing (w=0.0 was best)
final_test = np.clip(raw_test_blended, 0, 132)

sub = pd.DataFrame({"target": final_test}, index=X_test.index)
sub.index.name = "Index"
sanity_check(sub, X_test, "21_final")

out_path = SUBS_DIR / "21_final.csv"
sub.to_csv(out_path)
print(f"\nWritten: {out_path}")

# ── Hub metadata put ──────────────────────────────────────────────────────────
hub_url = "PENDING"
try:
    import skore
    from skore import login
    cfg = load_skore_credentials()
    login(mode="hub")
    project = skore.Project(name="bobathon-esilv", mode="hub", workspace=cfg["workspace"])
    meta = {
        "key": "21_final",
        "note": "CANDIDATE - 5-seed s2 + traj blend alpha=0.2, no post-smooth",
        "cv_rmse_pass_a": float(np.mean(folds_21)),
        "cv_rmse_pass_a_std": float(np.std(folds_21)),
        "cv_rmse_pass_b": float(PASS_B_RMSE),
        "folds_pass_a": [float(f) for f in folds_21],
        "mean_gain_vs_17": float(3.9949 - np.mean(folds_21)),
        "folds_improved_vs_17": int(sum(
            r17 > r21 for r17, r21 in
            zip([4.2602, 4.0026, 3.9085, 3.8914, 3.9119], folds_21))),
    }
    project.put("21_final_meta", meta)
    hub_url = (f"https://skore.probabl.ai/{cfg['workspace']}/"
               f"bobathon-esilv/cross-validations")
    print(f"Hub metadata pushed.")
except Exception as e:
    print(f"HUB PUT PENDING: ({e})")

print("\n" + "="*60)
print("CANDIDATE (DO NOT SUBMIT YET): submissions/21_final.csv")
print(f"  CV RMSE Pass A: {np.mean(folds_21):.4f} +/- {np.std(folds_21):.4f}")
print(f"  CV RMSE Pass B: {PASS_B_RMSE:.4f}")
print(f"  Mean gain vs 17: {3.9949 - np.mean(folds_21):.4f}  (threshold: 0.03)")
print(f"  Folds improved: {sum(r17>r21 for r17,r21 in zip([4.2602,4.0026,3.9085,3.8914,3.9119],folds_21))}/5")
print(f"  Hub URL: {hub_url}")
print("="*60)
print(f"\nTotal elapsed: {time.time()-t0:.0f}s")
