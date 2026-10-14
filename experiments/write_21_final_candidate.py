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


"""

FINAL PIPELINE - "TRUE OFF" SCORE PREDICTION (PARKINSON'S)


1. ARCHITECTURE AND MODEL PROGRESSION

The goal is to predict the "true OFF" motor score (without medication effect), 
which is often missing or biased. The final pipeline combines several approaches:
  - Stage 1: A baseline model (HGBR) with temporal smoothing per patient.
  - Stage 2: An ensemble of 5 HGBR models (multiple seeds) trained on 
    enriched features (InputCurveTransformer), capturing the quadratic dynamics 
    of the ON and OFF states for each patient.
  - Trajectory Model: A dedicated model predicting the long-term quadratic 
    evolution of the disease for a given patient.
  - Final Blend: A mix of Stage 2 predictions (80%) and the trajectory 
    model (20%, alpha=0.2).

2. FEATURE ENGINEERING AND HANDLING MISSING VALUES (NaNs)

Given the high rate of missing data (e.g., 42% missing OFF scores):
  - Native handling: The HGBR model uses the absence of data as a 
    predictive piece of information in its own right.
  - Smart Imputation ("off_hat"): A Ridge model infers missing OFF scores 
    based on the ON score, age, and LEDD dose.
  - Missingness Flags: Creation of binary variables and frequencies 
    (e.g., on_miss_frac) indicating the absence of certain measurements.
  - Fallbacks: Use of global means/medians to impute data for new patients 
    in the test set (cold-start).


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

# COMMENT: Import key functions developed in previous steps.
# ridge_poly_smooth performs intra-patient temporal smoothing.
# InputCurveTransformer (from step 17) handles the quadratic curves of inputs (ON/OFF) and imputation (off_hat).
from shared import ridge_poly_smooth, sanity_check
import step2_17_stage2_feats as _s17
InputCurveTransformer = _s17.InputCurveTransformer
build_s2_features     = _s17.build_s2_features

# Define paths to data and caches (to avoid recalculating everything)
DATA_DIR = PROJECT_ROOT / "data"
SUBS_DIR = PROJECT_ROOT / "submissions"
C11  = PROJECT_ROOT / "scratch" / "13_cache"  # Contains Stage 1 OOF (11_tuned)
C17  = PROJECT_ROOT / "scratch" / "17_cache"  # Contains Stage 2 features (Curves)
C19  = PROJECT_ROOT / "scratch" / "19_cache"  # Contains trajectory model predictions
C21  = PROJECT_ROOT / "scratch" / "21_cache"  # Cache for this final script
SUBS_DIR.mkdir(exist_ok=True)

# Selected configuration for Stage 1 smoothing (from exp. 14)
CFG14    = dict(degree=2, ridge_lam=0.1, w_blend=0.9)
# Optimized hyperparameters for Stage 2 HGBR
S2_PARAMS = dict(learning_rate=0.02, max_iter=300, max_leaf_nodes=31,
                 min_samples_leaf=50, l2_regularization=0.0)
SEEDS    = [0, 1, 2, 3, 4] # For ensembling (average of 5 models)

# Results from 21_final run
BEST_FOLDS_21 = None   # Will be recomputed below
# The final smoothing configuration (w_blend=0.0 means no smoothing on the final test prediction)
BEST_CFG_21   = dict(degree=1, ridge_lam=0.0, w_blend=0.0, use_hetero=False)
BEST_ALPHA    = 0.2    # Trajectory model weight (20% Trajectory, 80% Stage 2)
PASS_A_RMSE   = 3.9832
PASS_B_RMSE   = 3.9743

t0 = time.time()

# Load datasets
X, y   = load_dataset(data_dir=DATA_DIR)
X_test = load_test_dataset(data_dir=DATA_DIR)
ages_a = X["age"].values
pids_a = X["patient_id"].values
y_vals = y.values.astype(float)

# Create cross-validation splits.
# GroupKFold guarantees that a validation patient is NEVER seen in training.
from sklearn.model_selection import GroupKFold
cv_splits = list(GroupKFold(n_splits=5).split(X, y, groups=X["patient_id"]))

# ── Load cached 21_final OOF 
print("Loading 21_final cached OOF seeds...")
oof21_seeds = []
# COMMENT: Load the pre-calculated "Out-Of-Fold" (OOF) predictions for each of the 5 models.
for seed in SEEDS:
    p = C21 / f"21_oof_seed{seed}.npy"
    if p.exists():
        oof21_seeds.append(np.load(p))
        print(f"  seed {seed}: loaded")
    else:
        print(f"  seed {seed}: MISSING - cannot write 21_final.csv without full OOF")
        sys.exit(1)

# Average the 5 models to get the final Stage 2 OOF
oof21_avg = np.mean(oof21_seeds, axis=0)

# Apply trajectory blend
# COMMENT: The "Trajectory" model was trained to guess the evolution curve.
# We "blend" (mix) it with the ensemble model's predictions.
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
# COMMENT: Recompute the per-fold error on the training data (Cross Validation RMSE).
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

# COMMENT: Retrieve the fundamental components: 
# 1. The curve transformer (which contains the off_hat imputer).
# 2. Stage 1 predictions (for OOF and Test).
ct_final    = joblib.load(C17 / "17_ct_final.joblib")
oof_11      = np.load(C11 / "11_oof_raw.npy")
raw_test_s1 = np.load(C11 / "11_tuned_raw_test.npy")

# COMMENT: Smooth Stage 1. Stage 2 will use these smoothed predictions as features.
sm_train_s1 = ridge_poly_smooth(oof_11, ages_a, pids_a,
                                CFG14["w_blend"], CFG14["degree"], CFG14["ridge_lam"])
sm_test_s1  = ridge_poly_smooth(raw_test_s1, X_test["age"].values,
                                X_test["patient_id"].values,
                                CFG14["w_blend"], CFG14["degree"], CFG14["ridge_lam"])

# COMMENT: Build Stage 2 features. 
# This is where 'build_s2_features' calls 'InputCurveTransformer' which applies 
# the Ridge to impute missing 'off' values and builds the aggregation features.
X_train_s2 = build_s2_features(X.copy(), sm_train_s1, ct_final)
X_test_s2  = build_s2_features(X_test.copy(), sm_test_s1, ct_final)

# COMMENT: Inference (Prediction) with our 5 HGBR models (one per seed)
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

# Average the 5 models on the Test Set
raw_test_avg = np.mean(test_preds, axis=0)

# Apply trajectory blend to test
# COMMENT: Final blend of Test predictions with the Trajectory model (Alpha = 0.2)
traj_test_path = C19 / "19_traj_final_preds.npy"
if traj_test_path.exists():
    traj_test = np.load(traj_test_path)
    raw_test_blended = (1 - BEST_ALPHA) * raw_test_avg + BEST_ALPHA * traj_test
    print(f"Traj blend applied to test (alpha={BEST_ALPHA})")
else:
    raw_test_blended = raw_test_avg
    print("No traj test preds found; using 21 only")

# No post-smoothing (w=0.0 was best)
# COMMENT: Final bounds imposed by UPDRS (0 = Normal, 132 = Maximum severity)
final_test = np.clip(raw_test_blended, 0, 132)

sub = pd.DataFrame({"target": final_test}, index=X_test.index)
sub.index.name = "Index"

# Consistency checks (no NaNs introduced, correct format) before saving.
sanity_check(sub, X_test, "21_final")

out_path = SUBS_DIR / "21_final.csv"
sub.to_csv(out_path)
print(f"\nWritten: {out_path}")

# ── Hub metadata put ──────────────────────────────────────────────────────────
# COMMENT: Send model metadata to the Skore Hub platform for monitoring
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

# Summary printed to the console
print("\n" + "="*60)
print("CANDIDATE (DO NOT SUBMIT YET): submissions/21_final.csv")
print(f"  CV RMSE Pass A: {np.mean(folds_21):.4f} +/- {np.std(folds_21):.4f}")
print(f"  CV RMSE Pass B: {PASS_B_RMSE:.4f}")
print(f"  Mean gain vs 17: {3.9949 - np.mean(folds_21):.4f}  (threshold: 0.03)")
print(f"  Folds improved: {sum(r17>r21 for r17,r21 in zip([4.2602,4.0026,3.9085,3.8914,3.9119],folds_21))}/5")
print(f"  Hub URL: {hub_url}")
print("="*60)
print(f"\nTotal elapsed: {time.time()-t0:.0f}s")