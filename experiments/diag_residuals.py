"""diag_residuals.py -- Task 3: decompose OOF residuals of 17_stage2_feats.

Decomposes the total MSE of experiment 17 into three components:
  (a) patient-level mean residual  -- constant offset per patient
  (b) within-patient quadratic trend of residuals in age -- slow drift
  (c) per-visit noise              -- remainder

Reports each share of total MSE, overall and split by off-missing vs
off-present subgroups.

If component (a) dominates: patient-level input statistics are missing.
  -> Report top correlates of the patient-level offset with training-fold
     input aggregates (which inputs predict the per-patient bias?).
If component (c) dominates: per-visit noise is the lever; smoothing/pooling.

No modelling. Read-only on cached OOF. Target: < 10 min.
PYTHONUTF8=1. ASCII-only prints.
"""

import sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

from parkinsons import PROJECT_ROOT
from parkinsons.data import load_dataset

DATA_DIR = PROJECT_ROOT / "data"
C11 = PROJECT_ROOT / "scratch" / "13_cache"
C17 = PROJECT_ROOT / "scratch" / "17_cache"

t0 = time.time()

X, y = load_dataset(data_dir=DATA_DIR)
y_vals = y.values.astype(float)
ages_a = X["age"].values
pids_a = X["patient_id"].values

cv_splits = list(GroupKFold(n_splits=5).split(X, y, groups=X["patient_id"]))

# ── load 17_stage2_feats OOF (5-seed average, best smoothing w=0.25 deg=1) ───
from shared import ridge_poly_smooth
SEEDS = [0, 1, 2, 3, 4]
oof17_seeds = [np.load(C17 / f"17_oof_seed{s}.npy") for s in SEEDS]
oof17_raw   = np.mean(oof17_seeds, axis=0)

# Apply the best smoothing from 17 (deg=1, lam=1.0, w=0.25)
oof17_smooth = ridge_poly_smooth(oof17_raw, ages_a, pids_a, 0.25, 1, 1.0)

resid_raw    = y_vals - oof17_raw
resid_smooth = y_vals - oof17_smooth

print(f"Loaded 17_stage2_feats OOF ({time.time()-t0:.1f}s)")
print(f"  Raw    OOF RMSE: {np.sqrt(np.mean(resid_raw**2)):.4f}")
print(f"  Smooth OOF RMSE: {np.sqrt(np.mean(resid_smooth**2)):.4f}")


# ── decomposition helper ──────────────────────────────────────────────────────

def decompose_residuals(resid, ages, pids):
    """Decompose residuals into patient-mean, quad-trend, and noise.

    Returns
    -------
    comp_a : (n,) patient-level mean residual per row
    comp_b : (n,) within-patient quadratic trend (centred age) of residual
    comp_c : (n,) per-visit noise = resid - comp_a - comp_b
    """
    n = len(resid)
    comp_a = np.zeros(n)
    comp_b = np.zeros(n)

    df = pd.DataFrame({"resid": resid, "age": ages, "pid": pids})
    df_sorted = df.sort_values(["pid", "age"])

    for pid, grp in df_sorted.groupby("pid"):
        pos = [df.index.get_loc(i) for i in grp.index]
        pos = np.array(pos)
        r   = grp["resid"].values
        a   = grp["age"].values
        nv  = len(r)

        # (a) patient mean
        pat_mean = float(np.mean(r))
        comp_a[pos] = pat_mean

        # (b) within-patient quadratic trend of (resid - mean)
        r_demeaned = r - pat_mean
        if nv >= 4:
            ac = a - a.mean()
            Phi = np.column_stack([ac, ac**2])
            try:
                coef, _, _, _ = np.linalg.lstsq(Phi, r_demeaned, rcond=None)
                trend = Phi @ coef
            except Exception:
                trend = np.zeros(nv)
        elif nv >= 2:
            ac = a - a.mean()
            Phi = ac.reshape(-1, 1)
            try:
                coef, _, _, _ = np.linalg.lstsq(Phi, r_demeaned, rcond=None)
                trend = Phi @ coef
            except Exception:
                trend = np.zeros(nv)
        else:
            trend = np.zeros(nv)
        comp_b[pos] = trend

    comp_c = resid - comp_a - comp_b
    return comp_a, comp_b, comp_c


def mse_shares(resid, comp_a, comp_b, comp_c, label=""):
    total_mse = float(np.mean(resid**2))
    mse_a = float(np.mean(comp_a**2))
    mse_b = float(np.mean(comp_b**2))
    mse_c = float(np.mean(comp_c**2))
    # Cross terms: resid = a+b+c, so MSE(resid) != MSE(a)+MSE(b)+MSE(c)
    # We report the "explained" shares via R^2-style:
    #   frac_a = 1 - MSE(resid - a) / MSE(resid)   etc.
    # Better: just report variance of each component / variance of resid
    var_total = float(np.var(resid))
    var_a = float(np.var(comp_a))
    var_b = float(np.var(comp_b))
    var_c = float(np.var(comp_c))
    # Also report mean squared value of each component
    print(f"\n  {label}")
    print(f"    Total MSE:         {total_mse:.4f}  (RMSE={np.sqrt(total_mse):.4f})")
    print(f"    Var of components (sum may exceed total due to covariance):")
    print(f"      (a) patient-mean:    {var_a:.4f}  ({100*var_a/var_total:.1f}% of resid var)")
    print(f"      (b) within-pat quad: {var_b:.4f}  ({100*var_b/var_total:.1f}%)")
    print(f"      (c) per-visit noise: {var_c:.4f}  ({100*var_c/var_total:.1f}%)")
    print(f"    MSE of components:")
    print(f"      (a): {mse_a:.4f}  (b): {mse_b:.4f}  (c): {mse_c:.4f}")
    # Fraction of total MSE reducible if we perfectly corrected each component:
    # If we subtract comp_a from resid, new MSE = MSE(b+c)
    mse_minus_a = float(np.mean((resid - comp_a)**2))
    mse_minus_b = float(np.mean((resid - comp_b)**2))
    mse_minus_c = float(np.mean((resid - comp_c)**2))
    print(f"    Reducible MSE (subtract component perfectly):")
    print(f"      -a: {total_mse - mse_minus_a:.4f}  ({100*(total_mse-mse_minus_a)/total_mse:.1f}%)")
    print(f"      -b: {total_mse - mse_minus_b:.4f}  ({100*(total_mse-mse_minus_b)/total_mse:.1f}%)")
    print(f"      -c: {total_mse - mse_minus_c:.4f}  ({100*(total_mse-mse_minus_c)/total_mse:.1f}%)")
    return dict(total_mse=total_mse, var_a=var_a, var_b=var_b, var_c=var_c,
                mse_a=mse_a, mse_b=mse_b, mse_c=mse_c,
                red_a=total_mse-mse_minus_a, red_b=total_mse-mse_minus_b,
                red_c=total_mse-mse_minus_c)


# ── Overall decomposition ─────────────────────────────────────────────────────
print("\n" + "="*60)
print("RESIDUAL DECOMPOSITION -- 17_stage2_feats (smoothed OOF)")
print("="*60)

comp_a, comp_b, comp_c = decompose_residuals(resid_smooth, ages_a, pids_a)
overall = mse_shares(resid_smooth, comp_a, comp_b, comp_c, "OVERALL")

# ── By off-missing subgroup ────────────────────────────────────────────────────
off_miss = X["off"].isna().values
mask_miss = off_miss
mask_pres = ~off_miss

print("\n  By off-missing subgroup:")
for mask, name in [(mask_miss, "off MISSING (42%)"), (mask_pres, "off PRESENT (58%)")]:
    r_sub = resid_smooth[mask]
    a_sub = comp_a[mask]
    b_sub = comp_b[mask]
    c_sub = comp_c[mask]
    mse_shares(r_sub, a_sub, b_sub, c_sub, name)

# ── Correlates of patient-level offset with input features ────────────────────
print("\n" + "="*60)
print("CORRELATES OF PATIENT-LEVEL OFFSET (comp_a) WITH INPUT FEATURES")
print("="*60)

# Build per-patient summary frame
pat_df = pd.DataFrame({
    "pid":      pids_a,
    "comp_a":   comp_a,
    "age":      ages_a,
    "off_miss": off_miss.astype(float),
    "on":       X["on"].values,
    "off":      X["off"].values,
    "ledd":     X["ledd"].values,
    "age_at_diag": X["age_at_diagnosis"].values,
    "tsd":      ages_a - X["age_at_diagnosis"].values,
    "sexM":     X["sexM"].values.astype(float),
    "cohort":   X["cohort"].astype("category").cat.codes.values.astype(float),
    "gene":     X["gene"].astype("category").cat.codes.values.astype(float),
})

# Per-patient mean of each feature + mean offset
pat_agg = pat_df.groupby("pid").agg(
    comp_a_mean=("comp_a", "mean"),
    off_miss_frac=("off_miss", "mean"),
    age_mean=("age", "mean"),
    on_mean=("on", "nanmean") if False else ("on", lambda x: np.nanmean(x)),
    off_mean=("off", lambda x: np.nanmean(x)),
    ledd_mean=("ledd", lambda x: np.nanmean(x)),
    tsd_mean=("tsd", "mean"),
    sexM=("sexM", "first"),
    cohort=("cohort", "first"),
    gene=("gene", "first"),
    n_visits=("pid", "count"),
).reset_index()

# Point-biserial / Pearson correlations of comp_a_mean with each feature
print(f"  n_patients: {len(pat_agg)}")
print(f"  Patient-level offset: mean={pat_agg['comp_a_mean'].mean():.4f}  "
      f"std={pat_agg['comp_a_mean'].std():.4f}  "
      f"abs_mean={pat_agg['comp_a_mean'].abs().mean():.4f}")
print(f"\n  Pearson r between patient-mean offset and input aggregates:")
feat_cols = ["off_miss_frac", "age_mean", "on_mean", "off_mean",
             "ledd_mean", "tsd_mean", "sexM", "cohort", "gene", "n_visits"]
corrs = []
for col in feat_cols:
    sub = pat_agg[["comp_a_mean", col]].dropna()
    if len(sub) < 10:
        r = float("nan")
    else:
        r = float(np.corrcoef(sub["comp_a_mean"], sub[col])[0, 1])
    corrs.append((abs(r) if not np.isnan(r) else 0, col, r))
corrs.sort(reverse=True)
for _, col, r in corrs:
    print(f"    {col:20s}  r = {r:+.4f}")

# ── Distribution of offset sizes ──────────────────────────────────────────────
print(f"\n  Patient offset distribution (abs):")
pct = np.percentile(pat_agg["comp_a_mean"].abs(), [25, 50, 75, 90, 95])
print(f"    P25={pct[0]:.2f}  P50={pct[1]:.2f}  P75={pct[2]:.2f}  "
      f"P90={pct[3]:.2f}  P95={pct[4]:.2f}")
n_large = int((pat_agg["comp_a_mean"].abs() > 5).sum())
print(f"    Patients with |offset| > 5 pts: {n_large} ({100*n_large/len(pat_agg):.1f}%)")

# ── Conditional: off-missing patients ─────────────────────────────────────────
miss_pats = pat_agg[pat_agg["off_miss_frac"] > 0.5]
pres_pats = pat_agg[pat_agg["off_miss_frac"] <= 0.5]
print(f"\n  off-missing patients (frac>0.5): n={len(miss_pats)}")
print(f"    offset abs-mean={miss_pats['comp_a_mean'].abs().mean():.4f}  "
      f"std={miss_pats['comp_a_mean'].std():.4f}")
print(f"  off-present patients (frac<=0.5): n={len(pres_pats)}")
print(f"    offset abs-mean={pres_pats['comp_a_mean'].abs().mean():.4f}  "
      f"std={pres_pats['comp_a_mean'].std():.4f}")

# ── Summary verdict ────────────────────────────────────────────────────────────
red_a_frac = overall["red_a"] / overall["total_mse"]
red_c_frac = overall["red_c"] / overall["total_mse"]
print("\n" + "="*60)
print("VERDICT")
print("="*60)
if red_a_frac > 0.30:
    print(f"  Patient-level offset (a) accounts for {100*red_a_frac:.0f}% of reducible MSE.")
    print(f"  -> Patient-level information is MISSING from the model.")
    print(f"     Top correlates above indicate which inputs to aggregate.")
elif red_c_frac > 0.50:
    print(f"  Per-visit noise (c) accounts for {100*red_c_frac:.0f}% of reducible MSE.")
    print(f"  -> Smoothing / pooling is the primary lever.")
else:
    print(f"  Mixed: (a) reducible {100*red_a_frac:.0f}%,  (c) reducible {100*red_c_frac:.0f}%.")
print(f"  Elapsed: {time.time()-t0:.1f}s")
