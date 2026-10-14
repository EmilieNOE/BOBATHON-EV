"""shared.py -- shared utilities for steps 15+.

Imported by step scripts so the smoothing/stacking logic is DRY.
PYTHONUTF8=1 required.  ASCII-only prints.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

# ── smoothing ─────────────────────────────────────────────────────────────────

def ridge_poly_smooth(
    raw_preds: np.ndarray,
    ages: np.ndarray,
    patient_ids: np.ndarray,
    w_blend: float,
    degree: int,
    ridge_lam: float,
    visit_w=None,
    min_visits_deg2: int = 4,
) -> np.ndarray:
    """Blend raw predictions with a ridge-penalised per-patient polynomial trend.

    Parameters
    ----------
    raw_preds : (n,) array
    ages      : (n,) array
    patient_ids : (n,) array of patient identifiers
    w_blend   : blend weight in [0,1]; 0 = raw, 1 = pure trend
    degree    : 1 (linear) or 2 (quadratic)
    ridge_lam : L2 penalty on polynomial coefficients
    visit_w   : (n,) per-visit weights, or None for uniform
    min_visits_deg2 : minimum visits to use degree=2; fall back to degree=1
    """
    if w_blend == 0.0:
        return raw_preds.copy()
    smoothed = raw_preds.copy().astype(float)
    vw = visit_w if visit_w is not None else np.ones(len(raw_preds))
    df = pd.DataFrame({"pred": raw_preds, "age": ages, "pid": patient_ids, "w": vw})
    for pid, grp in df.groupby("pid"):
        n       = len(grp)
        ages_c  = grp["age"].values - grp["age"].mean()
        preds_g = grp["pred"].values
        wts     = grp["w"].values
        wts     = wts / wts.sum() * n          # normalise within patient
        eff_deg = degree if n >= min_visits_deg2 else min(degree, 1 if n >= 3 else 0)
        if eff_deg == 0 or n < 2:
            smoothed[grp.index] = (1 - w_blend) * preds_g + w_blend * np.mean(preds_g)
            continue
        Phi = ages_c.reshape(-1, 1) if eff_deg == 1 else np.column_stack([ages_c, ages_c**2])
        W   = np.diag(wts)
        intercept = np.average(preds_g, weights=wts)
        y_c       = preds_g - intercept
        A = Phi.T @ W @ Phi + ridge_lam * np.eye(Phi.shape[1])
        try:
            coef = np.linalg.solve(A, Phi.T @ W @ y_c)
        except np.linalg.LinAlgError:
            coef = np.zeros(Phi.shape[1])
        trend = Phi @ coef + intercept
        smoothed[grp.index] = (1 - w_blend) * preds_g + w_blend * trend
    return smoothed


# ── per-patient stage-1 aggregates ────────────────────────────────────────────

def _quad_trend_values(ages_c: np.ndarray, preds: np.ndarray, ridge_lam: float = 1.0) -> np.ndarray:
    n = len(ages_c)
    if n < 3:
        return np.full(n, np.mean(preds))
    Phi = np.column_stack([ages_c, ages_c**2])
    intercept = np.mean(preds)
    y_c = preds - intercept
    A = Phi.T @ Phi + ridge_lam * np.eye(2)
    try:
        coef = np.linalg.solve(A, Phi.T @ y_c)
    except np.linalg.LinAlgError:
        coef = np.zeros(2)
    return Phi @ coef + intercept


def add_s1_aggregates(X_frame: pd.DataFrame, s1_preds: np.ndarray) -> pd.DataFrame:
    """Add per-patient aggregates of stage-1 predictions and input stats.

    s1_preds must be aligned to X_frame's positional order (iloc-aligned).
    Returns a new DataFrame sorted back to original row order.
    No patient_id column in output (dropped).
    """
    X_out = X_frame.copy()
    X_out["_s1"] = s1_preds

    # Cast categoricals
    for col in ("cohort", "gene", "rater_id"):
        if col in X_out.columns:
            X_out[col] = X_out[col].astype("category")

    original_idx = X_out.index
    X_out = X_out.sort_values(["patient_id", "age"])
    # Use integer position array (avoids repeated index.get_loc calls)
    # Save original index as a regular column before reset
    X_out["__orig_idx__"] = np.arange(len(X_out))   # positional; we'll map back via original_idx
    # Also store a copy of the actual index values for restoration
    X_out = X_out.reset_index(drop=False)
    # After reset_index the original index becomes a column; its name depends on the index name
    orig_col = original_idx.name if original_idx.name is not None else "index"

    s1_arr = X_out["_s1"].values.astype(float)
    ages   = X_out["age"].values
    pids   = X_out["patient_id"].values
    n      = len(X_out)

    s1_mean  = np.full(n, np.nan)
    s1_med   = np.full(n, np.nan)
    s1_min   = np.full(n, np.nan)
    s1_max   = np.full(n, np.nan)
    s1_quad  = np.full(n, np.nan)
    s1_qres  = np.full(n, np.nan)
    s1_prev  = np.full(n, np.nan)
    s1_next  = np.full(n, np.nan)
    s1_nvis  = np.full(n, np.nan)
    off_frac = np.full(n, np.nan)
    yrs_fst  = np.full(n, np.nan)
    pm_on    = np.full(n, np.nan)
    pm_off   = np.full(n, np.nan)
    pm_ledd  = np.full(n, np.nan)
    s1_slope = np.full(n, np.nan)
    tsd      = (X_out["age"] - X_out["age_at_diagnosis"]).values

    # group by patient using the sorted (reset) positional index
    pid_arr = X_out["patient_id"].values
    start   = 0
    while start < n:
        pid   = pid_arr[start]
        end   = start
        while end < n and pid_arr[end] == pid:
            end += 1
        pos = np.arange(start, end)
        pg  = s1_arr[pos]
        ag  = ages[pos]
        ag_c = ag - ag.mean()
        nv  = end - start
        trend = _quad_trend_values(ag_c, pg)
        s1_mean[pos]  = np.mean(pg)
        s1_med[pos]   = np.median(pg)
        s1_min[pos]   = np.min(pg)
        s1_max[pos]   = np.max(pg)
        s1_quad[pos]  = trend
        s1_qres[pos]  = pg - trend
        s1_nvis[pos]  = nv
        s1_prev[pos[1:]] = pg[:-1]
        s1_next[pos[:-1]] = pg[1:]
        grp_off  = X_out["off"].values[pos]
        grp_on   = X_out["on"].values[pos]
        grp_ledd = X_out["ledd"].values[pos] if "ledd" in X_out.columns else np.full(nv, np.nan)
        off_frac[pos] = np.sum(~np.isnan(grp_off)) / nv
        yrs_fst[pos]  = ag - ag.min()
        pm_on[pos]    = np.nanmean(grp_on)
        pm_off[pos]   = np.nanmean(grp_off)
        pm_ledd[pos]  = np.nanmean(grp_ledd)
        if nv >= 3:
            A_ = np.column_stack([ag, np.ones(nv)])
            coef_, _, _, _ = np.linalg.lstsq(A_, pg, rcond=None)
            s1_slope[pos] = coef_[0]
        elif nv == 2:
            s1_slope[pos] = (pg[1] - pg[0]) / max(ag[1] - ag[0], 1e-6)
        else:
            s1_slope[pos] = 0.0
        start = end

    X_out["s1_mean"]  = s1_mean
    X_out["s1_med"]   = s1_med
    X_out["s1_min"]   = s1_min
    X_out["s1_max"]   = s1_max
    X_out["s1_quad"]  = s1_quad
    X_out["s1_qres"]  = s1_qres
    X_out["s1_prev"]  = s1_prev
    X_out["s1_next"]  = s1_next
    X_out["s1_nvis"]  = s1_nvis
    X_out["s1_slope"] = s1_slope
    X_out["off_frac"] = off_frac
    X_out["yrs_fst"]  = yrs_fst
    X_out["pm_on"]    = pm_on
    X_out["pm_off"]   = pm_off
    X_out["pm_ledd"]  = pm_ledd
    X_out["tsd"]      = tsd

    # prev/next of on, off, ledd (already sorted by patient+age)
    for col in ("on", "off", "ledd"):
        if col not in X_out.columns:
            continue
        vals = X_out[col].values.astype(float)
        prev_v = np.full(n, np.nan)
        next_v = np.full(n, np.nan)
        start  = 0
        while start < n:
            pid = pid_arr[start]
            end = start
            while end < n and pid_arr[end] == pid:
                end += 1
            pos = np.arange(start, end)
            prev_v[pos[1:]]  = vals[pos[:-1]]
            next_v[pos[:-1]] = vals[pos[1:]]
            start = end
        X_out[f"{col}_prev2"] = prev_v
        X_out[f"{col}_next2"] = next_v
        X_out[f"{col}_pm2"]   = X_out.groupby("patient_id")[col].transform("mean").values

    X_out = X_out.drop(columns=["_s1", "patient_id", "__orig_idx__"])
    # Restore original index from the preserved column
    X_out = X_out.set_index(orig_col)
    X_out.index.name = original_idx.name
    return X_out.loc[original_idx]


def sanity_check(sub: pd.DataFrame, X_test: pd.DataFrame, key: str) -> None:
    """Print and assert submission sanity checks."""
    sample = None
    try:
        from parkinsons import PROJECT_ROOT
        sample = (PROJECT_ROOT / "data" / "sample_submission.csv")
        if sample.exists():
            ss = pd.read_csv(sample, index_col="Index")
            assert list(sub.index) == list(ss.index), "Index order mismatch vs sample_submission"
    except Exception as e:
        print(f"  (sample_submission check skipped: {e})")
    assert len(sub) == len(X_test), f"Row count {len(sub)} != {len(X_test)}"
    assert sub["target"].isna().sum() == 0, "NaN predictions"
    pmin, pmax = float(sub["target"].min()), float(sub["target"].max())
    assert pmin >= 0 and pmax <= 132, f"Out of range [{pmin:.2f},{pmax:.2f}]"
    print(f"  [OK] {key}: {len(sub)} rows, NaN=0, range=[{pmin:.2f},{pmax:.2f}]")
