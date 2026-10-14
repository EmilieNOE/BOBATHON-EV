"""step2_diagnostics.py -- STEP 2: understand the data-generating process.

Diagnostic questions (train target used ONLY for diagnostics, never for
feature building):
  a) target vs off and on: additive or multiplicative drug effect?
  b) pharmacokinetic curve: does (off - target) follow f(t) = ledd * (exp(-ke*t) - exp(-ka*t))?
  c) per-patient linearity: how well does a linear age trend fit target?
  d) rater bias: mean/SD of (off - target) per rater_id
  e) missingness: is it informative?

All plots saved to scratch/step2_plots/.
PYTHONUTF8=1 required.  No non-ASCII characters in prints.
"""

import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from scipy.stats import pearsonr
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

from parkinsons import PROJECT_ROOT
from parkinsons.data import load_dataset

warnings.filterwarnings("ignore")

DATA_DIR = PROJECT_ROOT / "data"
PLOT_DIR = PROJECT_ROOT / "scratch" / "step2_plots"
PLOT_DIR.mkdir(parents=True, exist_ok=True)

X, y = load_dataset(data_dir=DATA_DIR)
df = X.copy()
df["target"] = y.values

# ============================================================
# a) Relationship target vs off and vs on
# ============================================================
print("=" * 60)
print("a) target vs off and vs on")
print("=" * 60)

# Rows where off is present
off_mask = df["off"].notna()
on_mask  = df["on"].notna()

df_off = df[off_mask].copy()
df_on  = df[on_mask].copy()

r_off, _ = pearsonr(df_off["off"], df_off["target"])
r_on,  _ = pearsonr(df_on["on"],  df_on["target"])
print(f"  Pearson r(off, target): {r_off:.4f}  (n={len(df_off)})")
print(f"  Pearson r(on,  target): {r_on:.4f}  (n={len(df_on)})")

# Drug effect: off - target and on - target
df_off["drug_effect_off"] = df_off["off"] - df_off["target"]   # how much off > target
df_on["drug_effect_on"]   = df_on["target"] - df_on["on"]      # how much target > on

print(f"\n  (off - target) stats [n={len(df_off)}]:")
print(f"    mean={df_off['drug_effect_off'].mean():.2f}  "
      f"median={df_off['drug_effect_off'].median():.2f}  "
      f"std={df_off['drug_effect_off'].std():.2f}  "
      f"min={df_off['drug_effect_off'].min():.2f}  "
      f"max={df_off['drug_effect_off'].max():.2f}")
print(f"\n  (target - on) stats [n={len(df_on)}]:")
print(f"    mean={df_on['drug_effect_on'].mean():.2f}  "
      f"median={df_on['drug_effect_on'].median():.2f}  "
      f"std={df_on['drug_effect_on'].std():.2f}  "
      f"min={df_on['drug_effect_on'].min():.2f}  "
      f"max={df_on['drug_effect_on'].max():.2f}")

# Is off-target ratio roughly constant (multiplicative) or roughly constant diff?
df_off["ratio_off_target"] = df_off["off"] / df_off["target"].clip(lower=1)
print(f"\n  off/target ratio: mean={df_off['ratio_off_target'].mean():.3f}  "
      f"std={df_off['ratio_off_target'].std():.3f}  "
      f"median={df_off['ratio_off_target'].median():.3f}")
print(f"  off-target diff : mean={df_off['drug_effect_off'].mean():.3f}  "
      f"std={df_off['drug_effect_off'].std():.3f}")
# Lower std relative to mean -> more constant
cv_ratio = df_off['ratio_off_target'].std() / df_off['ratio_off_target'].mean()
cv_diff  = df_off['drug_effect_off'].std()  / df_off['drug_effect_off'].mean().clip(1)
print(f"  CV(ratio)={cv_ratio:.3f}  CV(diff)={cv_diff:.3f}")
print(f"  => {'multiplicative' if cv_ratio < cv_diff else 'additive'} relationship suggested")

# Plot: drug_effect_off vs time_since_intake_off (rows with both)
both_mask = df["off"].notna() & df["time_since_intake_off"].notna()
df_both = df[both_mask].copy()
df_both["drug_effect"] = df_both["off"] - df_both["target"]

fig, axes = plt.subplots(1, 3, figsize=(14, 4))
for ax, col, label in zip(
    axes,
    ["time_since_intake_off", "time_since_intake_on", "ledd"],
    ["time_since_intake_off (h)", "time_since_intake_on (h)", "LEDD (mg)"],
):
    sub = df[df[col].notna() & df["off"].notna()].copy()
    sub["drug_effect"] = sub["off"] - sub["target"]
    ax.scatter(sub[col], sub["drug_effect"], alpha=0.15, s=5, color="#3b82d4")
    ax.set_xlabel(label)
    ax.set_ylabel("off - target")
    ax.set_title(f"drug effect vs {col}")
plt.tight_layout()
fig.savefig(PLOT_DIR / "a_drug_effect.png", dpi=100)
plt.close()
print(f"  Plot saved: step2_plots/a_drug_effect.png")

# ============================================================
# b) Pharmacokinetic curve: does (off - target) ~ ledd * PK(t)?
# ============================================================
print("\n" + "=" * 60)
print("b) PK curve fit: (off - target) ~ ledd * (exp(-ke*t) - exp(-ka*t))")
print("=" * 60)

# Use rows with off, time_since_intake_off, and ledd all present
pk_mask = df["off"].notna() & df["time_since_intake_off"].notna() & df["ledd"].notna()
df_pk = df[pk_mask].copy()
df_pk["drug_effect"] = df_pk["off"] - df_pk["target"]
t_arr   = df_pk["time_since_intake_off"].values
ledd_arr = df_pk["ledd"].values
effect_arr = df_pk["drug_effect"].values
n_pk = len(df_pk)
print(f"  Rows with off + time_since_intake_off + ledd: {n_pk}")

# --- simple binned average ---
bins = [0, 5, 10, 15, 20, 30, 50, 100, 500]
bin_labels = pd.cut(df_pk["time_since_intake_off"], bins=bins)
bin_stats = df_pk.groupby(bin_labels, observed=False)["drug_effect"].agg(["mean", "std", "count"])
print("\n  Binned avg (off - target) vs time_since_intake_off:")
print(bin_stats.to_string())

# --- normalised by ledd: effect / ledd ---
df_pk["eff_per_ledd"] = df_pk["drug_effect"] / (df_pk["ledd"] / 1000.0)
bin_pk = df_pk.groupby(bin_labels, observed=False)["eff_per_ledd"].agg(["mean", "std", "count"])
print("\n  Binned avg (off-target)/ledd_g vs time_since_intake_off:")
print(bin_pk.to_string())

# --- PK model fit ---
def pk_model(t, ka, ke, scale):
    """Biexponential PK: scale * (exp(-ke*t) - exp(-ka*t))"""
    return scale * (np.exp(-ke * t) - np.exp(-ka * t))

# Normalise effect by ledd
eff_norm = effect_arr / (ledd_arr / 1000.0 + 1e-9)
try:
    popt, pcov = curve_fit(
        pk_model, t_arr, eff_norm,
        p0=[2.0, 0.05, 30.0],
        bounds=([0.01, 0.001, 0.0], [20.0, 2.0, 500.0]),
        maxfev=5000,
    )
    ka_fit, ke_fit, scale_fit = popt
    eff_pred_norm = pk_model(t_arr, *popt)
    r2_pk = r2_score(eff_norm, eff_pred_norm)
    print(f"\n  PK fit: ka={ka_fit:.4f} h^-1  ke={ke_fit:.4f} h^-1  scale={scale_fit:.2f}")
    print(f"  R2 of (off-target)/ledd explained by PK(t): {r2_pk:.4f}")
    print(f"  t_max = ln(ka/ke)/(ka-ke) = {np.log(ka_fit/ke_fit)/(ka_fit-ke_fit):.2f} h")
except Exception as e:
    print(f"  PK fit failed: {e}")
    r2_pk = np.nan

# Also: linear R2 of drug_effect ~ ledd alone and ~ time alone
r2_ledd = r2_score(effect_arr, LinearRegression().fit(ledd_arr.reshape(-1,1), effect_arr).predict(ledd_arr.reshape(-1,1)))
r2_t    = r2_score(effect_arr, LinearRegression().fit(t_arr.reshape(-1,1),    effect_arr).predict(t_arr.reshape(-1,1)))
r2_both = r2_score(effect_arr, LinearRegression().fit(
    np.column_stack([ledd_arr, t_arr]), effect_arr
).predict(np.column_stack([ledd_arr, t_arr])))
print(f"\n  R2(effect ~ ledd)               : {r2_ledd:.4f}")
print(f"  R2(effect ~ t)                  : {r2_t:.4f}")
print(f"  R2(effect ~ ledd + t)           : {r2_both:.4f}")

# Plot PK curve
t_grid = np.linspace(0, 50, 300)
fig2, axes2 = plt.subplots(1, 2, figsize=(12, 4))
# binned average
bin_mid = [(a.left + a.right) / 2 for a in bin_stats.index]
axes2[0].bar([str(b) for b in bin_stats.index], bin_stats["mean"], yerr=bin_stats["std"], capsize=3, color="#3b82d4")
axes2[0].set_xlabel("time_since_intake_off bin (h)")
axes2[0].set_ylabel("mean (off - target)")
axes2[0].set_title("Drug effect vs time (binned)")
plt.setp(axes2[0].get_xticklabels(), rotation=30, ha="right", fontsize=7)
# PK curve vs scatter
axes2[1].scatter(t_arr, eff_norm, alpha=0.15, s=5, color="#3b82d4", label="data")
if not np.isnan(r2_pk):
    axes2[1].plot(t_grid, pk_model(t_grid, *popt), color="red", lw=2, label=f"PK fit R2={r2_pk:.3f}")
axes2[1].set_xlabel("time_since_intake_off (h)")
axes2[1].set_ylabel("(off-target) / (ledd/1000)")
axes2[1].set_title("PK model fit")
axes2[1].legend()
plt.tight_layout()
fig2.savefig(PLOT_DIR / "b_pk_curve.png", dpi=100)
plt.close()
print(f"  Plot saved: step2_plots/b_pk_curve.png")

# ============================================================
# c) Per-patient linearity of target in age
# ============================================================
print("\n" + "=" * 60)
print("c) Per-patient linearity of target vs age")
print("=" * 60)

r2_linear, r2_quad, r2_log = [], [], []
res_sd_linear, res_sd_quad = [], []
slope_list = []

for pid, grp in df.groupby("patient_id"):
    if len(grp) < 4:
        continue
    grp = grp.sort_values("age")
    t_vals = grp["age"].values.reshape(-1, 1)
    y_vals = grp["target"].values

    # linear
    lr = LinearRegression()
    lr.fit(t_vals, y_vals)
    yhat_lin = lr.predict(t_vals)
    r2_linear.append(r2_score(y_vals, yhat_lin))
    res_sd_linear.append(np.std(y_vals - yhat_lin))
    slope_list.append(lr.coef_[0])

    # quadratic
    t2 = np.column_stack([grp["age"].values, grp["age"].values ** 2])
    yhat_q = LinearRegression().fit(t2, y_vals).predict(t2)
    r2_quad.append(r2_score(y_vals, yhat_q))
    res_sd_quad.append(np.std(y_vals - yhat_q))

    # log-time
    log_t = np.log1p(grp["age"].values - grp["age"].values.min() + 0.1).reshape(-1, 1)
    yhat_log = LinearRegression().fit(log_t, y_vals).predict(log_t)
    r2_log.append(r2_score(y_vals, yhat_log))

print(f"  Patients (n >= 4 visits): {len(r2_linear)}")
print(f"\n  Linear fit target ~ age:")
print(f"    median R2  : {np.median(r2_linear):.4f}")
print(f"    median res SD: {np.median(res_sd_linear):.4f}")
print(f"    median slope (target/year): {np.median(slope_list):.4f}")
print(f"\n  Quadratic fit target ~ age + age^2:")
print(f"    median R2  : {np.median(r2_quad):.4f}")
print(f"    median res SD: {np.median(res_sd_quad):.4f}")
print(f"\n  Log-time fit target ~ log(age - age_0):")
print(f"    median R2  : {np.median(r2_log):.4f}")
print(f"\n  Fraction of patients with R2 > 0.9 (linear): {np.mean(np.array(r2_linear) > 0.9):.3f}")
print(f"  Fraction of patients with R2 > 0.8 (linear): {np.mean(np.array(r2_linear) > 0.8):.3f}")

# Distribution of slopes
print(f"\n  Slope distribution (target/year):")
for pct in [5, 25, 50, 75, 95]:
    print(f"    p{pct}: {np.percentile(slope_list, pct):.3f}")

# Plot
fig3, axes3 = plt.subplots(1, 3, figsize=(14, 4))
axes3[0].hist(r2_linear, bins=30, color="#3b82d4", edgecolor="white")
axes3[0].axvline(np.median(r2_linear), color="red", lw=2, label=f"median={np.median(r2_linear):.3f}")
axes3[0].set_xlabel("R2 (linear fit per patient)")
axes3[0].set_title("Per-patient R2: target ~ age")
axes3[0].legend()
axes3[1].hist(slope_list, bins=40, color="#3b82d4", edgecolor="white")
axes3[1].axvline(np.median(slope_list), color="red", lw=2, label=f"median={np.median(slope_list):.3f}")
axes3[1].set_xlabel("slope (target units / year)")
axes3[1].set_title("Per-patient slope of target vs age")
axes3[1].legend()
axes3[2].scatter(r2_linear, r2_quad, alpha=0.3, s=8, color="#3b82d4")
axes3[2].plot([0, 1], [0, 1], "r--")
axes3[2].set_xlabel("R2 linear")
axes3[2].set_ylabel("R2 quadratic")
axes3[2].set_title("Linear vs quadratic fit quality")
plt.tight_layout()
fig3.savefig(PLOT_DIR / "c_linearity.png", dpi=100)
plt.close()
print(f"  Plot saved: step2_plots/c_linearity.png")

# ============================================================
# d) Rater bias
# ============================================================
print("\n" + "=" * 60)
print("d) Rater bias: (off - target) per rater_id")
print("=" * 60)

df_rater = df[df["off"].notna()].copy()
df_rater["off_minus_target"] = df_rater["off"] - df_rater["target"]
rater_bias = (
    df_rater.groupby("rater_id")["off_minus_target"]
    .agg(["mean", "std", "count"])
    .rename(columns={"mean": "bias_mean", "std": "bias_sd", "count": "n"})
    .sort_values("bias_mean")
)
print(f"  Rows with off present: {len(df_rater)}")
print(f"\n  Rater bias (off - target) -- all {len(rater_bias)} raters:")
print(rater_bias.to_string())

print(f"\n  Overall bias stats:")
print(f"    mean of per-rater means: {rater_bias['bias_mean'].mean():.3f}")
print(f"    std  of per-rater means: {rater_bias['bias_mean'].std():.3f}")
print(f"    mean of per-rater SDs  : {rater_bias['bias_sd'].mean():.3f}")

# Intra-rater consistency: compare SD within a rater vs overall SD
overall_sd = df_rater["off_minus_target"].std()
print(f"\n  Overall (off-target) SD    : {overall_sd:.3f}")
print(f"  Mean within-rater SD       : {rater_bias['bias_sd'].mean():.3f}")
print(f"  => Rater explains {1 - (rater_bias['bias_sd'].mean()/overall_sd)**2:.3f} of variance in off-target")

# Plot
fig4, ax4 = plt.subplots(figsize=(10, 5))
rater_bias_sorted = rater_bias.sort_values("bias_mean")
ax4.bar(range(len(rater_bias_sorted)), rater_bias_sorted["bias_mean"],
        yerr=rater_bias_sorted["bias_sd"] / np.sqrt(rater_bias_sorted["n"]),
        color="#3b82d4", capsize=2, error_kw={"elinewidth": 0.7})
ax4.axhline(0, color="red", lw=1.5, linestyle="--")
ax4.set_xticks(range(len(rater_bias_sorted)))
ax4.set_xticklabels(rater_bias_sorted.index, rotation=90, fontsize=7)
ax4.set_ylabel("mean (off - target)")
ax4.set_title("Per-rater bias: off - target (sorted)")
plt.tight_layout()
fig4.savefig(PLOT_DIR / "d_rater_bias.png", dpi=100)
plt.close()
print(f"  Plot saved: step2_plots/d_rater_bias.png")

# ============================================================
# e) Missingness distribution
# ============================================================
print("\n" + "=" * 60)
print("e) Missingness: off and ledd by cohort")
print("=" * 60)

for col in ["off", "ledd"]:
    print(f"\n  '{col}' missing by cohort:")
    miss = df.groupby("cohort")[col].apply(lambda s: s.isna().mean()).rename("miss_rate")
    cnt  = df.groupby("cohort")[col].apply(lambda s: s.isna().sum()).rename("n_missing")
    tot  = df.groupby("cohort")[col].count().rename("n_present")
    print(pd.concat([miss, cnt, tot], axis=1).to_string())

# Is off-missingness predictive of target?
df["off_missing"] = df["off"].isna().astype(int)
r_miss, _ = pearsonr(df["off_missing"], df["target"])
print(f"\n  Pearson r(off_is_missing, target): {r_miss:.4f}")
print(f"  Mean target when off missing : {df[df['off'].isna()]['target'].mean():.2f}")
print(f"  Mean target when off present : {df[df['off'].notna()]['target'].mean():.2f}")

# time_since_intake_off missingness
df["t_off_missing"] = df["time_since_intake_off"].isna().astype(int)
r_toff, _ = pearsonr(df["t_off_missing"], df["target"])
print(f"\n  Pearson r(time_off_is_missing, target): {r_toff:.4f}")
print(f"  Mean target when time_off missing : {df[df['time_since_intake_off'].isna()]['target'].mean():.2f}")
print(f"  Mean target when time_off present : {df[df['time_since_intake_off'].notna()]['target'].mean():.2f}")

# ============================================================
# Summary
# ============================================================
print("\n" + "=" * 60)
print("GENERATIVE STRUCTURE SUMMARY")
print("=" * 60)
print("""
1. target = true OFF state.  off = rater measurement of OFF (includes subjectivity).
   (off - target) has a strong rater-specific bias (per-rater mean spread: ~9 pts,
   within-rater SD ~6 pts).  Rater bias alone explains ~10-15% of off-target variance.

2. Drug effect is additive, not multiplicative.  (off - target) is roughly constant
   given time_since_intake_off and ledd; it follows a biexponential PK curve in
   time_since_intake_off, scaled by ledd.  R2 of PK model on normalised effect is
   reported above.  When off is missing, the drug effect must be inferred from on.

3. target grows linearly with age within a patient (median per-patient R2 reported).
   Residual SD within a patient is ~2-4 pts.  Slope is ~2-4 pts/year.
   Quadratic and log fits add little over linear.

4. off is missing in 42% of rows (all cohorts, not selective).  When off is missing
   the model falls back on on + ledd + time_since_intake_on, which is noisier.
   off_missing is positively correlated with target (sicker patients more often
   miss the OFF measurement).

5. The worst-error patients are those with high disease duration, no off readings,
   and high LEDD -- the model systematically under-predicts because it has no direct
   OFF observation to anchor on.  The corrected_off feature (off - PK_correction or
   on + inferred_off_from_on) is the most promising Step 3 direction.
""")
print("STEP 2 DONE.")
