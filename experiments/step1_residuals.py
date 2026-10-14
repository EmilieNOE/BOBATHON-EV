"""step1_residuals.py -- STEP 1: residual analysis of best OOF predictions.

Uses the cached 11_tuned OOF predictions (w=0, raw) and the corrected smoothed
OOF predictions (w=0.75) to slice RMSE by every requested subgroup.  Saves all
plots to scratch/step1_plots/.  No hub put.

PYTHONUTF8=1 must be set.  No non-ASCII characters in prints.
"""

import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GroupKFold

from parkinsons import PROJECT_ROOT
from parkinsons.data import load_dataset
from parkinsons.pipeline import build_hgbr_tuned

warnings.filterwarnings("ignore")

DATA_DIR    = PROJECT_ROOT / "data"
CACHE_DIR   = PROJECT_ROOT / "scratch" / "13_cache"
PLOT_DIR    = PROJECT_ROOT / "scratch" / "step1_plots"
PLOT_DIR.mkdir(parents=True, exist_ok=True)

BEST_PARAMS = {
    "learning_rate": 0.02,
    "max_iter": 1000,
    "max_leaf_nodes": 31,
    "min_samples_leaf": 20,
    "l2_regularization": 0.0,
}
BEST_W    = 0.75
MIN_VISITS = 3

# ------------------------------------------------------------------
# Load data + cached OOF
# ------------------------------------------------------------------
X, y = load_dataset(data_dir=DATA_DIR)
groups = X["patient_id"]
cv_splits = list(GroupKFold(n_splits=5).split(X, y, groups=groups))

OOF_CACHE = CACHE_DIR / "11_oof_raw.npy"
assert OOF_CACHE.exists(), "Run 13_tuned_smoothed.py first to generate OOF cache."
oof_raw = np.load(OOF_CACHE)

# Build smoothed OOF (within-fold, same logic as 13_tuned_smoothed.py)
oof_smoothed = oof_raw.copy().astype(float)
for tr_idx, val_idx in cv_splits:
    raw_fold  = oof_raw[val_idx]
    ages_fold = X["age"].values[val_idx]
    pids_fold = X["patient_id"].values[val_idx]
    df = pd.DataFrame({"pred": raw_fold, "age": ages_fold, "pid": pids_fold})
    sm = raw_fold.copy().astype(float)
    for pid, grp in df.groupby("pid"):
        if len(grp) < MIN_VISITS:
            continue
        lr = LinearRegression()
        lr.fit(grp[["age"]], grp["pred"])
        sm[grp.index] = (1 - BEST_W) * grp["pred"].values + BEST_W * lr.predict(grp[["age"]])
    oof_smoothed[val_idx] = sm

residuals = oof_smoothed - y.values   # positive = over-predicted

overall_rmse = np.sqrt(np.mean(residuals ** 2))
overall_mae  = np.mean(np.abs(residuals))
print(f"Overall smoothed OOF RMSE: {overall_rmse:.4f}  MAE: {overall_mae:.4f}")

# Helper: rmse for a boolean mask
def rmse(mask):
    r = residuals[mask]
    return np.sqrt(np.mean(r ** 2)) if len(r) > 0 else np.nan

def n_count(mask):
    return int(mask.sum())

# ------------------------------------------------------------------
# 1. off missing vs present
# ------------------------------------------------------------------
print("\n--- off missing vs present ---")
off_miss  = X["off"].isna()
off_pres  = ~off_miss
print(f"  off missing  (n={n_count(off_miss):6d}):  RMSE {rmse(off_miss):.4f}")
print(f"  off present  (n={n_count(off_pres):6d}):  RMSE {rmse(off_pres):.4f}")

# ------------------------------------------------------------------
# 2. on missing vs present
# ------------------------------------------------------------------
print("\n--- on missing vs present ---")
on_miss = X["on"].isna()
on_pres = ~on_miss
print(f"  on missing   (n={n_count(on_miss):6d}):  RMSE {rmse(on_miss):.4f}")
print(f"  on present   (n={n_count(on_pres):6d}):  RMSE {rmse(on_pres):.4f}")

# ------------------------------------------------------------------
# 3. visits per patient (bins)
# ------------------------------------------------------------------
print("\n--- visits per patient (bins) ---")
vcount = X.groupby("patient_id").size()
visit_bin = X["patient_id"].map(vcount)
for lo, hi in [(4, 5), (6, 7), (8, 9), (10, 12)]:
    mask = (visit_bin >= lo) & (visit_bin <= hi)
    print(f"  {lo}-{hi} visits  (n={n_count(mask):6d}):  RMSE {rmse(mask):.4f}")

# ------------------------------------------------------------------
# 4. time_since_intake_off bins
# ------------------------------------------------------------------
print("\n--- time_since_intake_off bins (hours) ---")
toff = X["time_since_intake_off"]
miss_mask = toff.isna()
print(f"  missing      (n={n_count(miss_mask):6d}):  RMSE {rmse(miss_mask):.4f}")
for lo, hi in [(0, 5), (5, 12), (12, 24), (24, 9999)]:
    mask = (toff >= lo) & (toff < hi) & (~miss_mask)
    label = f"{lo}-{hi}h" if hi < 9999 else f">={lo}h"
    print(f"  {label:12s} (n={n_count(mask):6d}):  RMSE {rmse(mask):.4f}")

# ------------------------------------------------------------------
# 5. time_since_intake_on bins
# ------------------------------------------------------------------
print("\n--- time_since_intake_on bins (hours) ---")
ton = X["time_since_intake_on"]
miss_on = ton.isna()
print(f"  missing      (n={n_count(miss_on):6d}):  RMSE {rmse(miss_on):.4f}")
for lo, hi in [(0, 1), (1, 2), (2, 4), (4, 9999)]:
    mask = (ton >= lo) & (ton < hi) & (~miss_on)
    label = f"{lo}-{hi}h" if hi < 9999 else f">={lo}h"
    print(f"  {label:12s} (n={n_count(mask):6d}):  RMSE {rmse(mask):.4f}")

# ------------------------------------------------------------------
# 6. cohort
# ------------------------------------------------------------------
print("\n--- cohort ---")
for cohort in sorted(X["cohort"].unique()):
    mask = X["cohort"] == cohort
    print(f"  cohort {cohort}    (n={n_count(mask):6d}):  RMSE {rmse(mask):.4f}")

# ------------------------------------------------------------------
# 7. rater_id (top and bottom 10 by RMSE)
# ------------------------------------------------------------------
print("\n--- rater_id (top/bottom 10 by RMSE) ---")
rater_stats = []
for rid, grp in X.groupby("rater_id"):
    idx = grp.index
    mask = X.index.isin(idx)
    rater_stats.append({
        "rater_id": rid,
        "n": len(grp),
        "rmse": rmse(mask),
        "bias": residuals[mask].mean(),
    })
rater_df = pd.DataFrame(rater_stats).sort_values("rmse")
print("\n  Top 10 (lowest RMSE):")
print(rater_df.head(10).to_string(index=False))
print("\n  Bottom 10 (highest RMSE):")
print(rater_df.tail(10).to_string(index=False))

# ------------------------------------------------------------------
# 8. time_since_diagnosis bins
# ------------------------------------------------------------------
print("\n--- time_since_diagnosis bins (years) ---")
tsd = X["age"] - X["age_at_diagnosis"]
for lo, hi in [(0, 5), (5, 10), (10, 15), (15, 20), (20, 9999)]:
    mask = (tsd >= lo) & (tsd < hi)
    label = f"{lo}-{hi}y" if hi < 9999 else f">={lo}y"
    print(f"  {label:10s} (n={n_count(mask):6d}):  RMSE {rmse(mask):.4f}")

# ------------------------------------------------------------------
# 9. Worst 10 patients
# ------------------------------------------------------------------
print("\n--- 10 worst patients by mean squared error ---")
patient_mse = (
    pd.Series(residuals ** 2, index=X.index)
    .groupby(X["patient_id"].values)
    .mean()
    .sort_values(ascending=False)
)
worst_pids = patient_mse.head(10).index.tolist()
print(f"\n{'patient_id':15s}  {'n':>4}  {'MSE':>8}  {'RMSE':>8}")
for pid in worst_pids:
    mask_p = X["patient_id"] == pid
    mse_p  = patient_mse[pid]
    print(f"{pid:15s}  {n_count(mask_p):>4}  {mse_p:>8.2f}  {np.sqrt(mse_p):>8.4f}")

print("\n--- Visit detail for the 10 worst patients ---")
detail_cols = ["patient_id", "age", "on", "off", "time_since_intake_off", "ledd"]
for pid in worst_pids:
    mask_p = X["patient_id"] == pid
    rows   = X[mask_p][detail_cols].copy()
    rows["target"]     = y[mask_p].values
    rows["prediction"] = oof_smoothed[mask_p]
    rows["residual"]   = residuals[mask_p]
    rows = rows.sort_values("age")
    print(f"\n  Patient {pid}  (RMSE {np.sqrt(patient_mse[pid]):.2f})")
    print(rows.to_string(index=False))

# ------------------------------------------------------------------
# 10. Summary bar chart: RMSE by subgroup
# ------------------------------------------------------------------
subgroup_labels = ["off present", "off missing", "on present", "on missing",
                   "cohort A", "cohort B"]
subgroup_rmses  = [
    rmse(off_pres), rmse(off_miss),
    rmse(on_pres),  rmse(on_miss),
    rmse(X["cohort"] == "A"), rmse(X["cohort"] == "B"),
]

fig, ax = plt.subplots(figsize=(10, 4))
colors = ["#3b82d4" if r <= overall_rmse else "#e05252" for r in subgroup_rmses]
ax.barh(subgroup_labels, subgroup_rmses, color=colors)
ax.axvline(overall_rmse, color="black", linestyle="--", label=f"overall {overall_rmse:.2f}")
ax.set_xlabel("RMSE")
ax.set_title("RMSE by subgroup (smoothed 11_tuned OOF)")
ax.legend()
plt.tight_layout()
fig.savefig(PLOT_DIR / "rmse_by_subgroup.png", dpi=120)
plt.close()
print(f"\nPlot saved: {PLOT_DIR / 'rmse_by_subgroup.png'}")

# Rater bias scatter
fig2, ax2 = plt.subplots(figsize=(8, 4))
ax2.scatter(rater_df["bias"], rater_df["rmse"], s=rater_df["n"] / 5, alpha=0.7, color="#3b82d4")
for _, row in rater_df.iterrows():
    ax2.annotate(row["rater_id"], (row["bias"], row["rmse"]), fontsize=6, alpha=0.7)
ax2.set_xlabel("mean residual (bias)")
ax2.set_ylabel("RMSE")
ax2.set_title("Rater: bias vs RMSE (bubble = n visits)")
plt.tight_layout()
fig2.savefig(PLOT_DIR / "rater_bias_rmse.png", dpi=120)
plt.close()
print(f"Plot saved: {PLOT_DIR / 'rater_bias_rmse.png'}")

print("\nSTEP 1 DONE.")
