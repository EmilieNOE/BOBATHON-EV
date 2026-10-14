# %% [markdown]
# # Experiments 08-12: Beat 07_best_model (CV RMSE 7.530)
#
# **Date:** 2026-10-12
# **Goal:** Improve on grouped-CV RMSE of 7.530 through five incremental steps:
#   08_patient_features - patient-level engineered features
#   09_rater_cohort     - categorical rater_id / cohort via pandas category
#   10_smoothing        - per-patient linear smoothing of OOF predictions
#   11_tuned            - RandomizedSearch tuning of HGBR
#   12_ensemble         - average several HGBR seeds + optional Ridge blend

# %%
import warnings
warnings.filterwarnings("ignore")
import sys
import os
os.environ.setdefault("PYTHONIOENCODING", "utf-8")

# Redirect stdout to a log file so background runs capture output
_LOG = os.path.join(os.path.dirname(__file__), "_run_08_12_utf8.log")
_logf = open(_LOG, "w", encoding="utf-8", buffering=1)
sys.stdout = _logf
sys.stderr = _logf

import numpy as np
import pandas as pd
import skore
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from skrub import TableVectorizer
import sys
sys.path.insert(0, "src")

from parkinsons import PROJECT_ROOT
from parkinsons.hub import load_skore_credentials

# -- Hub login ------------------------------------------------------------------
cfg = load_skore_credentials()
print(f"Workspace : {cfg['workspace']}  hub={cfg['hub_url']}")
project = skore.Project(name="bobathon-esilv", mode="hub", workspace=cfg["workspace"])
print(f"Connected : {project}")

# -- Load raw data --------------------------------------------------------------
DATA_DIR = PROJECT_ROOT / "data"

visits_raw = pd.read_csv(DATA_DIR / "X_train.csv", index_col="Index").join(
    pd.read_csv(DATA_DIR / "y_train.csv", index_col="Index")
)
X_test_raw = pd.read_csv(DATA_DIR / "X_test.csv", index_col="Index")

BASELINE_RMSE = 7.530
print(f"\nBaseline CV RMSE to beat: {BASELINE_RMSE}")
print(f"Training visits : {len(visits_raw)} rows, {visits_raw['patient_id'].nunique()} patients")
print(f"Test rows       : {len(X_test_raw)}")

# -- Grouped CV splits (computed once, reused every step) ----------------------
y_base = visits_raw["target"]
X_base_for_split = visits_raw.drop(columns=["patient_id", "target"])
groups = visits_raw["patient_id"]

cv_splits = list(GroupKFold(n_splits=5).split(X_base_for_split, y_base, groups))
print(f"\n5-fold GroupKFold: {len(cv_splits)} folds")


# ===============================================================================
# Helper: compute RMSE from a skore CrossValidationReport
# ===============================================================================
def rmse_from_report(report) -> tuple[float, float]:
    """Return (mean_rmse, std_rmse) from a skore CrossValidationReport.

    Uses report.metrics.rmse() which returns a DataFrame with index 'RMSE'
    and columns (estimator_name, 'mean') / (estimator_name, 'std').
    """
    df = report.metrics.rmse()
    # df has multi-level column: (estimator_label, aggregate)
    # flatten to find mean/std
    if isinstance(df.columns, pd.MultiIndex):
        # columns like ('Estimator HistGBR', 'mean'), ('Estimator ...', 'std')
        mean_cols = [c for c in df.columns if c[1] == "mean"]
        std_cols  = [c for c in df.columns if c[1] == "std"]
        mean_rmse = float(df[mean_cols[0]].iloc[0])
        std_rmse  = float(df[std_cols[0]].iloc[0])
    else:
        # flat columns
        mean_rmse = float(df.filter(like="mean").iloc[0, 0])
        std_rmse  = float(df.filter(like="std").iloc[0, 0])
    return mean_rmse, std_rmse


def report_result(name, report, prev_mean, prev_std):
    mean, std = rmse_from_report(report)
    delta = prev_mean - mean
    significant = abs(delta) > prev_std
    flag = "OK KEEP" if (delta > 0 and significant) else ("~ small gain" if delta > 0 else "WORSE WORSE")
    print(f"\n{'-'*60}")
    print(f"  {name}")
    print(f"  CV RMSE  : {mean:.4f} +/- {std:.4f}")
    print(f"  vs prev  : {prev_mean:.4f}  delta={delta:+.4f}  fold-std={prev_std:.4f}  -> {flag}")
    print(f"{'-'*60}")
    return mean, std


# ===============================================================================
# STEP 1 - Patient-level features  ->  08_patient_features
# ===============================================================================
print("\n" + "="*60)
print("STEP 1: Patient-level features (08_patient_features)")
print("="*60)


def add_patient_features(df: pd.DataFrame, is_test: bool = False) -> pd.DataFrame:
    """Add patient-level features.

    Rules
    -----
    - Uses ONLY columns from input X (never `target`).
    - Works identically on train and test splits.
    - Preserves original row order.
    """
    df = df.copy()

    # time_since_diagnosis (only if not already present)
    if "time_since_diagnosis" not in df.columns:
        df["time_since_diagnosis"] = df["age"] - df["age_at_diagnosis"]

    # on_minus_off
    df["on_minus_off"] = df["on"] - df["off"]

    # -- per-patient rolling features ------------------------------------------
    original_index = df.index

    # Sort by (patient_id, age) for shift operations
    df = df.sort_values(["patient_id", "age"])

    grp = df.groupby("patient_id", sort=False)

    # n_visits
    df["n_visits"] = grp["age"].transform("count")

    # years_since_first_visit
    df["years_since_first_visit"] = df["age"] - grp["age"].transform("min")

    # Per patient: mean, prev, next for on, off, ledd
    for col in ["on", "off", "ledd"]:
        df[f"{col}_patient_mean"] = grp[col].transform("mean")
        df[f"{col}_prev"]         = grp[col].transform(lambda s: s.shift(1))
        df[f"{col}_next"]         = grp[col].transform(lambda s: s.shift(-1))

    # Restore original row order
    df = df.loc[original_index]
    return df


# Build feature frames
visits_feat = add_patient_features(visits_raw.drop(columns=["target"]))
visits_feat["target"] = visits_raw["target"]

y_s1 = visits_feat["target"]
X_s1 = visits_feat.drop(columns=["patient_id", "target"])
# add_patient_features needs patient_id for groupby; drop it after
X_test_s1 = add_patient_features(X_test_raw).drop(columns=["patient_id"])

print(f"X_s1 shape: {X_s1.shape}  new cols: {[c for c in X_s1.columns if c not in X_base_for_split.columns]}")

model_s1 = Pipeline([
    ("vectorizer", TableVectorizer()),
    ("hgbr", HistGradientBoostingRegressor(random_state=0)),
])

report_s1 = skore.evaluate(model_s1, X_s1, y_s1, splitter=cv_splits)
project.put("08_patient_features", report_s1)
mean_s1, std_s1 = report_result("08_patient_features", report_s1, BASELINE_RMSE, 0.137)
best_mean, best_std, best_key, best_step = BASELINE_RMSE, 0.137, "07_best_model", 0

if mean_s1 < best_mean:
    best_mean, best_std, best_key, best_step = mean_s1, std_s1, "08_patient_features", 1
    print("  -> New best!")


# ===============================================================================
# STEP 2 - rater_id and cohort as pandas category  ->  09_rater_cohort
# ===============================================================================
print("\n" + "="*60)
print("STEP 2: rater_id + cohort as category (09_rater_cohort)")
print("="*60)

# Report cardinality
for col in ["rater_id", "cohort", "gene"]:
    n = visits_raw[col].nunique()
    print(f"  {col}: {n} unique values -> using pandas 'category' (low-card ordinal via TableVectorizer)")


def add_category_cols(df: pd.DataFrame) -> pd.DataFrame:
    """Cast rater_id, cohort, gene to pandas Categorical."""
    df = df.copy()
    for col in ["rater_id", "cohort", "gene"]:
        if col in df.columns:
            df[col] = df[col].astype("category")
    return df


X_s2 = add_category_cols(X_s1)
# X_test: patient features already computed in X_test_s1; just cast categories
X_test_s2 = add_category_cols(X_test_s1)

print(f"X_s2 shape: {X_s2.shape}")

model_s2 = Pipeline([
    ("vectorizer", TableVectorizer()),
    ("hgbr", HistGradientBoostingRegressor(random_state=0)),
])

report_s2 = skore.evaluate(model_s2, X_s2, y_s1, splitter=cv_splits)
project.put("09_rater_cohort", report_s2)
mean_s2, std_s2 = report_result("09_rater_cohort", report_s2, mean_s1, std_s1)

if mean_s2 < best_mean:
    best_mean, best_std, best_key, best_step = mean_s2, std_s2, "09_rater_cohort", 2
    print("  -> New best!")

# Choose which X to carry forward into smoothing
X_carry = X_s2 if mean_s2 < mean_s1 else X_s1
X_test_carry = X_test_s2 if mean_s2 < mean_s1 else X_test_s1
mean_carry, std_carry = (mean_s2, std_s2) if mean_s2 < mean_s1 else (mean_s1, std_s1)


# ===============================================================================
# STEP 3 - Per-patient smoothing of OOF predictions  ->  10_smoothing
# ===============================================================================
print("\n" + "="*60)
print("STEP 3: Per-patient smoothing (10_smoothing)")
print("="*60)
print("  Blending linear per-patient age trend with raw OOF predictions")
print("  Patients with < 3 visits keep raw prediction.")
print("  Blend weight w tried in {0, 0.25, 0.50, 0.75, 1.0}")


def smooth_predictions(preds: np.ndarray, ages: np.ndarray, pids: np.ndarray, w: float) -> np.ndarray:
    """Blend raw preds with a per-patient linear age trend.

    Parameters
    ----------
    preds : raw predictions (shape n,)
    ages  : age values aligned with preds
    pids  : patient_id aligned with preds
    w     : blend weight for trend (0 = all raw, 1 = all trend)

    Returns
    -------
    smoothed predictions
    """
    if w == 0.0:
        return preds.copy()
    result = preds.copy()
    for pid in np.unique(pids):
        mask = pids == pid
        n = mask.sum()
        if n < 3:
            continue
        age_p = ages[mask]
        pred_p = preds[mask]
        # fit linear trend: pred ~ a*age + b
        A = np.column_stack([age_p, np.ones(n)])
        try:
            coef, _, _, _ = np.linalg.lstsq(A, pred_p, rcond=None)
            trend = A @ coef
            result[mask] = (1 - w) * pred_p + w * trend
        except Exception:
            pass
    return result


def smooth_cv_rmse(X_feat, y, groups_s, cv_spl, w):
    """Compute smoothed OOF RMSE for weight w."""
    all_oof = np.full(len(y), np.nan)
    model = Pipeline([
        ("vectorizer", TableVectorizer()),
        ("hgbr", HistGradientBoostingRegressor(random_state=0)),
    ])
    # need ages and pids aligned with X_feat index
    ages = visits_raw.loc[X_feat.index, "age"].values
    pids_arr = visits_raw.loc[X_feat.index, "patient_id"].values
    y_arr = y.values

    fold_rmses = []
    for tr_idx, val_idx in cv_spl:
        m = Pipeline([
            ("vectorizer", TableVectorizer()),
            ("hgbr", HistGradientBoostingRegressor(random_state=0)),
        ])
        m.fit(X_feat.iloc[tr_idx], y_arr[tr_idx])
        raw = m.predict(X_feat.iloc[val_idx])
        smoothed = smooth_predictions(raw, ages[val_idx], pids_arr[val_idx], w)
        rmse = np.sqrt(np.mean((smoothed - y_arr[val_idx]) ** 2))
        fold_rmses.append(rmse)
    return np.mean(fold_rmses), np.std(fold_rmses)


# Search over w values
best_w = 0.0
best_w_rmse = mean_carry
best_w_std  = std_carry
smooth_results = {}
for w_try in [0.0, 0.25, 0.50, 0.75, 1.0]:
    w_mean, w_std = smooth_cv_rmse(X_carry, y_s1, groups, cv_splits, w_try)
    smooth_results[w_try] = (w_mean, w_std)
    print(f"  w={w_try:.2f} -> RMSE={w_mean:.4f} +/- {w_std:.4f}")
    if w_mean < best_w_rmse:
        best_w_rmse = w_mean
        best_w_std  = w_std
        best_w = w_try

print(f"\n  Best w = {best_w}  RMSE = {best_w_rmse:.4f}")

# Now publish the smoothed result using best_w via a wrapper estimator
from sklearn.base import BaseEstimator, RegressorMixin


class SmoothedHGBR(BaseEstimator, RegressorMixin):
    """HGBR + per-patient linear smoothing of predictions."""

    def __init__(self, w=0.5, random_state=0):
        self.w = w
        self.random_state = random_state
        self._inner = None
        self._train_index = None

    def fit(self, X, y):
        self._inner = Pipeline([
            ("vectorizer", TableVectorizer()),
            ("hgbr", HistGradientBoostingRegressor(random_state=self.random_state)),
        ])
        self._inner.fit(X, y)
        self._train_index = X.index if hasattr(X, "index") else None
        return self

    def predict(self, X):
        raw = self._inner.predict(X)
        idx = X.index if hasattr(X, "index") else np.arange(len(X))
        ages = visits_raw.loc[idx, "age"].values if hasattr(X, "index") else np.zeros(len(X))
        pids = visits_raw.loc[idx, "patient_id"].values if hasattr(X, "index") else np.zeros(len(X))
        return smooth_predictions(raw, ages, pids, self.w)


model_s3 = SmoothedHGBR(w=best_w, random_state=0)
report_s3 = skore.evaluate(model_s3, X_carry, y_s1, splitter=cv_splits)
project.put("10_smoothing", report_s3)
mean_s3, std_s3 = report_result("10_smoothing", report_s3, mean_carry, std_carry)

if mean_s3 < best_mean:
    best_mean, best_std, best_key, best_step = mean_s3, std_s3, "10_smoothing", 3
    print("  -> New best!")

# Carry forward: use smoothed if it's better, otherwise raw pipeline
use_smoothing = mean_s3 < mean_carry
mean_s3_carry = mean_s3 if use_smoothing else mean_carry
std_s3_carry  = std_s3  if use_smoothing else std_carry


# ===============================================================================
# STEP 4 - Hyperparameter tuning of HGBR  ->  11_tuned
# ===============================================================================
print("\n" + "="*60)
print("STEP 4: Hyperparameter tuning (11_tuned)")
print("="*60)


class TunedHGBRPipeline(BaseEstimator, RegressorMixin):
    """Fit a RandomizedSearchCV over HGBR inside the CV loop."""

    def __init__(self, n_iter=30, random_state=0):
        self.n_iter = n_iter
        self.random_state = random_state
        self._best = None

    def fit(self, X, y):
        param_dist = {
            "hgbr__learning_rate": [0.03, 0.05, 0.07, 0.1],
            "hgbr__max_iter": [300, 500, 700, 1000],
            "hgbr__max_leaf_nodes": [15, 20, 25, 31],
            "hgbr__min_samples_leaf": [20, 30, 40, 50],
            "hgbr__l2_regularization": [0.0, 0.1, 1.0, 5.0, 10.0],
        }
        base = Pipeline([
            ("vectorizer", TableVectorizer()),
            ("hgbr", HistGradientBoostingRegressor(
                early_stopping=True,
                validation_fraction=0.1,
                random_state=self.random_state,
            )),
        ])
        inner_cv = GroupKFold(n_splits=3)
        # inner CV uses the same group labels
        search = RandomizedSearchCV(
            base,
            param_distributions=param_dist,
            n_iter=self.n_iter,
            cv=3,   # plain 3-fold for speed (no groups inside groups)
            scoring="neg_root_mean_squared_error",
            random_state=self.random_state,
            n_jobs=-1,
            refit=True,
        )
        search.fit(X, y)
        self._best = search.best_estimator_
        print(f"    Best params: {search.best_params_}  inner-RMSE={-search.best_score_:.4f}")
        return self

    def predict(self, X):
        return self._best.predict(X)


model_s4 = TunedHGBRPipeline(n_iter=30, random_state=0)
report_s4 = skore.evaluate(model_s4, X_carry, y_s1, splitter=cv_splits)
project.put("11_tuned", report_s4)
mean_s4, std_s4 = report_result("11_tuned", report_s4, mean_s3_carry, std_s3_carry)

if mean_s4 < best_mean:
    best_mean, best_std, best_key, best_step = mean_s4, std_s4, "11_tuned", 4
    print("  -> New best!")


# ===============================================================================
# STEP 5 - Ensemble  ->  12_ensemble
# ===============================================================================
print("\n" + "="*60)
print("STEP 5: Ensemble (12_ensemble)")
print("="*60)


class EnsembleHGBR(BaseEstimator, RegressorMixin):
    """Average predictions from several HGBR seeds."""

    SEEDS = [0, 1, 2, 3, 7]

    def __init__(self):
        self._models = []

    def fit(self, X, y):
        self._models = []
        for seed in self.SEEDS:
            m = Pipeline([
                ("vectorizer", TableVectorizer()),
                ("hgbr", HistGradientBoostingRegressor(
                    random_state=seed,
                    learning_rate=0.05,
                    max_iter=500,
                    max_leaf_nodes=31,
                    min_samples_leaf=20,
                    l2_regularization=0.1,
                )),
            ])
            m.fit(X, y)
            self._models.append(m)
        return self

    def predict(self, X):
        preds = np.column_stack([m.predict(X) for m in self._models])
        return preds.mean(axis=1)


model_s5 = EnsembleHGBR()
report_s5 = skore.evaluate(model_s5, X_carry, y_s1, splitter=cv_splits)
project.put("12_ensemble", report_s5)
mean_s5, std_s5 = report_result("12_ensemble", report_s5, mean_s4, std_s4)

if mean_s5 < best_mean:
    best_mean, best_std, best_key, best_step = mean_s5, std_s5, "12_ensemble", 5
    print("  -> New best!")


# ===============================================================================
# Summary
# ===============================================================================
print("\n" + "="*60)
print("SUMMARY OF ALL STEPS")
print("="*60)
step_names = {
    "07_best_model"      : f"07 baseline (TV+HGBR)         RMSE={BASELINE_RMSE:.4f} +/- 0.1370",
    "08_patient_features": f"08 +patient features          RMSE={mean_s1:.4f} +/- {std_s1:.4f}",
    "09_rater_cohort"    : f"09 +rater_id/cohort category  RMSE={mean_s2:.4f} +/- {std_s2:.4f}",
    "10_smoothing"       : f"10 +smoothing w={best_w:.2f}         RMSE={mean_s3:.4f} +/- {std_s3:.4f}",
    "11_tuned"           : f"11 +HGBR tuned                RMSE={mean_s4:.4f} +/- {std_s4:.4f}",
    "12_ensemble"        : f"12 ensemble (5 seeds)         RMSE={mean_s5:.4f} +/- {std_s5:.4f}",
}
for k, v in step_names.items():
    marker = " <- BEST" if k == best_key else ""
    print(f"  {v}{marker}")

print(f"\nBest report key: {best_key}")
print(f"Best CV RMSE   : {best_mean:.4f} +/- {best_std:.4f}")
improvement = BASELINE_RMSE - best_mean
print(f"Improvement    : {improvement:+.4f} vs 07_best_model")


# ===============================================================================
# Fit best model on ALL training data and write submission CSV
# ===============================================================================
print("\n" + "="*60)
print(f"Fitting best model ({best_key}) on all training data ?")
print("="*60)

# Select the model and feature set corresponding to best_step
def make_best_model(step):
    if step <= 2:
        return Pipeline([
            ("vectorizer", TableVectorizer()),
            ("hgbr", HistGradientBoostingRegressor(random_state=0)),
        ])
    elif step == 3:
        return SmoothedHGBR(w=best_w, random_state=0)
    elif step == 4:
        return TunedHGBRPipeline(n_iter=30, random_state=0)
    else:
        return EnsembleHGBR()


final_model = make_best_model(best_step)
final_model.fit(X_carry, y_s1)

# Apply same feature engineering to test set
final_test = X_test_carry  # already engineered

preds_test = final_model.predict(final_test)

submissions_dir = PROJECT_ROOT / "submissions"
submissions_dir.mkdir(exist_ok=True)
out_path = submissions_dir / f"{best_key}.csv"

submission = pd.DataFrame({"target": preds_test}, index=final_test.index)
submission.index.name = "Index"
submission.to_csv(out_path)
print(f"\nSubmission written : {out_path}  ({len(submission)} rows)")

# -- Hub URL -------------------------------------------------------------------
hub_url = cfg["hub_url"].rstrip("/")
workspace = cfg["workspace"]
# skore hub URL pattern: https://skore.probabl.ai/<workspace>/projects/bobathon-esilv/runs/<key>
hub_report_url = f"https://skore.probabl.ai/{workspace}/projects/bobathon-esilv"
print(f"\nHub project URL : {hub_report_url}")
print(f"Best report key : {best_key}")
print(f"\nPaste this in Kaggle Submission Description:")
print(f"  skore hub report: {best_key}  CV RMSE {best_mean:.4f}")
