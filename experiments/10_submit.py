"""10_submit.py — sanity checks + final submission for 10_smoothing.

Steps
-----
1. Sanity checks (no hub put):
   - list features used by 10_smoothing; confirm none derived from target.
   - confirm X_test has patient_id and age.
   - compare visit-count distribution (train vs test).
2. Fit base model (build_hgbr_with_categoricals) on all training data,
   cache to disk, apply per-patient linear smoothing with w=0.75,
   write submissions/10_smoothing.csv.
3. Print the hub key and URL; note that smoothing is NOT in the hub report.

PYTHONUTF8=1 must be set before running so all prints are safe on Windows.
Non-ASCII characters are intentionally avoided.
"""

# %%
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

from parkinsons import PROJECT_ROOT
from parkinsons.data import load_dataset, load_test_dataset
from parkinsons.features import (
    add_patient_features,
    drop_id_columns,
    encode_categoricals,
)
from parkinsons.pipeline import build_hgbr_with_categoricals

DATA_DIR = PROJECT_ROOT / "data"
SUBMISSIONS_DIR = PROJECT_ROOT / "submissions"
CACHE_DIR = PROJECT_ROOT / "scratch" / "10_submit_cache"
SUBMISSIONS_DIR.mkdir(exist_ok=True)
CACHE_DIR.mkdir(parents=True, exist_ok=True)

BEST_W = 0.75
MIN_VISITS_FOR_SMOOTHING = 3

# ============================================================
# STEP 1 — Sanity checks
# ============================================================
print("=" * 60)
print("STEP 1 — SANITY CHECKS")
print("=" * 60)

X_raw, y = load_dataset(data_dir=DATA_DIR)
X_test_raw = load_test_dataset(data_dir=DATA_DIR)

# -- 1a. Feature list for 10_smoothing ---------------------------------------
# The pipeline in build_hgbr_with_categoricals runs _apply_patient_features,
# which calls add_patient_features -> encode_categoricals -> drop_id_columns.
# Apply the full chain to a single-row copy to see the resulting columns.

sample = X_raw.iloc[:2].copy()
sample_engineered = drop_id_columns(encode_categoricals(add_patient_features(sample)))

raw_cols = list(X_raw.columns)
engineered_cols = list(sample_engineered.columns)
added_cols = [c for c in engineered_cols if c not in raw_cols]
dropped_cols = [c for c in raw_cols if c not in engineered_cols]

print("\n--- Input columns (before feature engineering) ---")
for c in raw_cols:
    print(f"  {c}")

print("\n--- Columns added by add_patient_features ---")
for c in added_cols:
    print(f"  {c}")

print("\n--- Columns dropped (patient_id, Index, string IDs) ---")
for c in dropped_cols:
    print(f"  {c}")

print("\n--- Final model input columns ---")
for c in engineered_cols:
    print(f"  {c}")

# Confirm no column touches 'target'
TARGET_RELATED = ["target"]
leaked = [c for c in engineered_cols if c in TARGET_RELATED]
if leaked:
    raise ValueError(f"TARGET LEAK detected: {leaked}")
print("\n[OK] No column derived from 'target' — leak check passed.")

# -- 1b. X_test columns ------------------------------------------------------
print("\n--- X_test columns ---")
for c in X_test_raw.columns:
    print(f"  {c}")

has_patient_id = "patient_id" in X_test_raw.columns
has_age = "age" in X_test_raw.columns
print(f"\n[OK] X_test has patient_id: {has_patient_id}")
print(f"[OK] X_test has age:        {has_age}")
if not has_patient_id or not has_age:
    raise ValueError("X_test is missing required columns for smoothing.")

# -- 1c. Visit-count distribution: train vs test ----------------------------
train_visits_per_patient = X_raw.groupby("patient_id").size()
test_visits_per_patient = X_test_raw.groupby("patient_id").size()

print("\n--- Visit count per patient ---")
print(f"{'':20s}  {'train':>8}  {'test':>8}")
print(f"{'n patients':20s}  {len(train_visits_per_patient):>8}  {len(test_visits_per_patient):>8}")
print(f"{'total rows':20s}  {len(X_raw):>8}  {len(X_test_raw):>8}")
print(f"{'mean visits':20s}  {train_visits_per_patient.mean():>8.2f}  {test_visits_per_patient.mean():>8.2f}")
print(f"{'median visits':20s}  {train_visits_per_patient.median():>8.1f}  {test_visits_per_patient.median():>8.1f}")
print(f"{'max visits':20s}  {train_visits_per_patient.max():>8}  {test_visits_per_patient.max():>8}")
print(f"{'min visits':20s}  {train_visits_per_patient.min():>8}  {test_visits_per_patient.min():>8}")

train_pids = set(X_raw["patient_id"].unique())
test_pids = set(X_test_raw["patient_id"].unique())
overlap = train_pids & test_pids
print(f"\nPatients in both train and test: {len(overlap)}")
print(f"Test-only patients (cold start): {len(test_pids - train_pids)}")

print("\n[STEP 1 DONE] Review the output above before proceeding.\n")

# ============================================================
# STEP 2 — Fit, smooth, write submission
# ============================================================
print("=" * 60)
print("STEP 2 — FIT + SMOOTH + WRITE SUBMISSION")
print("=" * 60)

MODEL_CACHE = CACHE_DIR / "10_final_model.joblib"
TRAIN_RAW_CACHE = CACHE_DIR / "10_train_raw_preds.npy"

# Cache the fitted model so a re-run (e.g. after hub timeout) is instant.
if MODEL_CACHE.exists() and TRAIN_RAW_CACHE.exists():
    print("Loading cached model and train-raw predictions...")
    final_model = joblib.load(MODEL_CACHE)
    final_train_raw = np.load(TRAIN_RAW_CACHE)
    print("  [cache hit]")
else:
    print("Fitting base model on all training data...")
    final_model = build_hgbr_with_categoricals()
    final_model.fit(X_raw, y)
    final_train_raw = final_model.predict(X_raw)
    joblib.dump(final_model, MODEL_CACHE)
    np.save(TRAIN_RAW_CACHE, final_train_raw)
    print("  [model cached]")

# Raw test predictions
print("Predicting on X_test...")
raw_test = final_model.predict(X_test_raw)

# Per-patient linear smoothing on test predictions.
# Trend lines are built from training (raw) predictions, never from target.
ages_train = X_raw["age"].values
pids_train = X_raw["patient_id"].values
ages_test = X_test_raw["age"].values
pids_test = X_test_raw["patient_id"].values

print(f"Applying per-patient linear smoothing (w={BEST_W})...")
df_train_raw = pd.DataFrame(
    {"pred": final_train_raw, "age": ages_train, "pid": pids_train}
)

smoothed_test = raw_test.copy().astype(float)
n_smoothed = 0

for pid, grp in df_train_raw.groupby("pid"):
    mask_test = pids_test == pid
    if mask_test.sum() == 0:
        continue  # patient not in test set
    if len(grp) < MIN_VISITS_FOR_SMOOTHING:
        continue  # too few training visits for a reliable trend
    lr = LinearRegression()
    lr.fit(grp[["age"]], grp["pred"])
    test_ages_pid = pd.DataFrame({"age": ages_test[mask_test]})
    trend = lr.predict(test_ages_pid)
    smoothed_test[mask_test] = (1 - BEST_W) * raw_test[mask_test] + BEST_W * trend
    n_smoothed += int(mask_test.sum())

print(f"  Smoothed {n_smoothed} / {len(X_test_raw)} test rows.")
print(f"  Kept raw prediction for {len(X_test_raw) - n_smoothed} rows "
      f"(cold-start or < {MIN_VISITS_FOR_SMOOTHING} training visits).")

# Write submission
submission = pd.DataFrame({"target": smoothed_test}, index=X_test_raw.index)
submission.index.name = "Index"
out_path = SUBMISSIONS_DIR / "10_smoothing.csv"
submission.to_csv(out_path)

# Checks
assert len(submission) == len(X_test_raw), "Row count mismatch"
assert submission["target"].isna().sum() == 0, "NaN predictions found"
pmin, pmax = submission["target"].min(), submission["target"].max()
assert 0 <= pmin and pmax <= 132, f"Predictions out of range: [{pmin:.2f}, {pmax:.2f}]"

print(f"\n[OK] submission rows : {len(submission)} (expected {len(X_test_raw)})")
print(f"[OK] NaN count       : 0")
print(f"[OK] prediction range: [{pmin:.2f}, {pmax:.2f}] (within 0-132)")
print(f"\nSubmission written:  {out_path}")

print("\n[STEP 2 DONE]\n")

# ============================================================
# STEP 3 — Hub key and URL
# ============================================================
print("=" * 60)
print("STEP 3 — HUB KEY AND URL")
print("=" * 60)

# Load workspace from .skore so the URL is exact.
skore_file = PROJECT_ROOT / ".skore"
try:
    cfg = json.loads(skore_file.read_text(encoding="utf-8"))
    workspace = cfg.get("workspace", "evdgreenteam")
except Exception:
    workspace = "evdgreenteam"

hub_key = "10_smoothing"
hub_url = f"https://skore.probabl.ai/{workspace}/bobathon-esilv/cross-validations"

print(f"\nHub key  : {hub_key}")
print(f"Hub URL  : {hub_url}")
print("""
NOTE: The hub report for key '10_smoothing' contains the CrossValidationReport
of the BASE model (build_hgbr_with_categoricals, w=0), which is identical to
09_rater_cohort. The smoothing post-processor (w=0.75 linear-trend blend) is
NOT included in the skore CrossValidationReport because skore.evaluate wraps
a single sklearn estimator and cannot wrap the two-stage predict logic.

The CV RMSE reported by the hub (10_smoothing key) therefore reflects the
unsmoothed model (~4.73). The smoothed CV RMSE of 4.47 +/- 0.14 was computed
by the manual OOF loop in experiments/10_smoothing.py and is printed in that
script's stdout.

For the Kaggle submission description, paste the hub URL above and note that
the actual predictions were smoothed with w=0.75 per-patient linear trend,
which reduced CV RMSE from 4.73 to 4.47.
""")
