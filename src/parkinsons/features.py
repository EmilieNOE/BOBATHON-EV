"""Feature engineering for patient-level temporal features.

All functions operate on the raw X frame (columns: patient_id, cohort, sexM,
gene, age_at_diagnosis, age, ledd, time_since_intake_on, time_since_intake_off,
rater_id, on, off) and return a new DataFrame with the same index in the same
row order. The ``target`` column is never read — these features are safe to
apply identically to X_train and X_test.

The public entry point is ``add_patient_features``. The category-encoding
helper ``encode_categoricals`` lives here too and is used by Step 2 onwards.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# Columns whose per-patient lag/lead/mean are computed.
_LAG_LEAD_COLS = ["on", "off", "ledd"]


def add_patient_features(X: pd.DataFrame) -> pd.DataFrame:
    """Add patient-level temporal features to ``X``.

    Features added (all derived from non-target columns):

    ``time_since_diagnosis``
        ``age - age_at_diagnosis``. Only added when not already present.
    ``on_minus_off``
        ``on - off``.
    ``n_visits``
        Number of rows for this patient in the frame.
    ``years_since_first_visit``
        ``age - min(age)`` within the patient.
    ``{col}_patient_mean``
        Per-patient mean of ``on``, ``off``, ``ledd`` (computed on the whole
        frame for that patient — not leaking target).
    ``{col}_prev``
        Previous visit value (shift +1 within patient, sorted by age).
    ``{col}_next``
        Next visit value (shift -1 within patient, sorted by age).

    Parameters
    ----------
    X : pd.DataFrame
        Raw feature frame with ``patient_id`` and ``age`` columns.

    Returns
    -------
    pd.DataFrame
        Same index and row order as ``X``, with additional columns.
    """
    X = X.copy()

    # ── 1. Simple derived columns ─────────────────────────────────────────────
    if "time_since_diagnosis" not in X.columns:
        X["time_since_diagnosis"] = X["age"] - X["age_at_diagnosis"]

    X["on_minus_off"] = X["on"] - X["off"]

    # ── 2. Sort by (patient_id, age) to compute temporal features ─────────────
    original_index = X.index
    X = X.sort_values(["patient_id", "age"])

    # ── 3. Per-patient counts and first-visit reference ────────────────────────
    X["n_visits"] = X.groupby("patient_id")["age"].transform("count")
    X["years_since_first_visit"] = X["age"] - X.groupby("patient_id")["age"].transform(
        "min"
    )

    # ── 4. Per-patient mean, prev, next for on / off / ledd ───────────────────
    for col in _LAG_LEAD_COLS:
        if col not in X.columns:
            continue
        grp = X.groupby("patient_id")[col]
        X[f"{col}_patient_mean"] = grp.transform("mean")
        X[f"{col}_prev"] = grp.shift(1)
        X[f"{col}_next"] = grp.shift(-1)

    # ── 5. Restore original row order ─────────────────────────────────────────
    X = X.loc[original_index]

    return X


def encode_categoricals(X: pd.DataFrame) -> pd.DataFrame:
    """Cast ``cohort``, ``gene``, and ``rater_id`` to pandas ``category`` dtype.

    These three columns are already present in the raw data. Casting to
    ``category`` is enough for ``HistGradientBoostingRegressor`` (sklearn ≥ 1.4)
    to use its native categorical split logic — no ordinal encoding or OHE
    needed.

    Parameters
    ----------
    X : pd.DataFrame
        Raw (or partially engineered) feature frame.

    Returns
    -------
    pd.DataFrame
        Same frame with the categorical columns re-cast in-place.
    """
    X = X.copy()
    for col in ("cohort", "gene", "rater_id"):
        if col in X.columns:
            X[col] = X[col].astype("category")
    return X


def drop_id_columns(X: pd.DataFrame) -> pd.DataFrame:
    """Drop ``Index`` (if present as a column) and ``patient_id``.

    ``patient_id`` is used only for feature engineering and grouping — it must
    not be fed as a model input. ``Index`` should already be the DataFrame
    index, not a column, but we guard against it here.

    Parameters
    ----------
    X : pd.DataFrame
        Feature frame, possibly still carrying ``patient_id`` and/or ``Index``.

    Returns
    -------
    pd.DataFrame
        Frame without ``patient_id`` and ``Index``.
    """
    drop = [c for c in ("Index", "patient_id") if c in X.columns]
    return X.drop(columns=drop) if drop else X
