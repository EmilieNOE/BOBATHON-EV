"""Learner declarations for Parkinson's UPDRS prediction experiments.

Each function returns an unfit estimator ready for ``skore.evaluate`` or
``.fit(X, y)``. Adding a new function (rather than modifying existing ones)
preserves the reproducibility of prior experiments.

Experiments 01–04 use numeric-only pipelines (``_STRING_COLS`` dropped).
Experiments 08+ use ``features.add_patient_features`` and, from 09 onward,
``features.encode_categoricals`` for HGBR's native categorical split.
"""

from __future__ import annotations

import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer

# Columns that are non-numeric and must be dropped before a numeric-only model.
# Ridge and SimpleImputer(median) cannot handle strings.
_STRING_COLS = ["patient_id", "cohort", "gene", "rater_id"]


def _drop_strings(X: pd.DataFrame) -> pd.DataFrame:
    """Drop known string columns, ignoring any that are already absent."""
    return X.drop(columns=[c for c in _STRING_COLS if c in X.columns])


def build_learner() -> DummyRegressor:
    """Return an unfit ``DummyRegressor(strategy='mean')``.

    Used by ``experiments/01_dummy.py``. Returns the always-predict-mean
    baseline — the performance floor every real model must beat.

    Returns
    -------
    DummyRegressor
        Unfit estimator.
    """
    return DummyRegressor(strategy="mean")


def build_ridge_learner() -> Pipeline:
    """Return an unfit Ridge pipeline with median imputation.

    Drops known string columns, imputes remaining NaNs with the column
    median, then fits a Ridge regression. Used by
    ``experiments/02_ridge.py``.

    Returns
    -------
    Pipeline
        Unfit ``Pipeline([("drop_strings", ...), ("imputer", SimpleImputer),
        ("ridge", Ridge)])`` ready for ``fit(X, y)`` or ``skore.evaluate``.
    """
    return Pipeline(
        steps=[
            (
                "drop_strings",
                FunctionTransformer(_drop_strings, validate=False),
            ),
            (
                "imputer",
                SimpleImputer(strategy="median"),
            ),
            (
                "ridge",
                Ridge(),
            ),
        ]
    )


def build_hgbr_learner() -> Pipeline:
    """Return an unfit HistGradientBoostingRegressor pipeline.

    Drops known string columns (HGBR cannot encode arbitrary strings in
    sklearn 1.x), then passes the numeric frame directly to
    ``HistGradientBoostingRegressor``. No imputer — HGBR handles NaN
    natively and treats missingness as a learnable split signal.
    Used by ``experiments/04_hgbr.py``.

    Returns
    -------
    Pipeline
        Unfit ``Pipeline([("drop_strings", ...), ("hgbr", HGBR)])``
        ready for ``fit(X, y)`` or ``skore.evaluate``.
    """
    return Pipeline(
        steps=[
            (
                "drop_strings",
                FunctionTransformer(_drop_strings, validate=False),
            ),
            (
                "hgbr",
                HistGradientBoostingRegressor(random_state=0),
            ),
        ]
    )


# ── Experiments 08–12: patient-feature + categorical pipelines ────────────────


def _apply_patient_features(X: pd.DataFrame) -> pd.DataFrame:
    """Run the full feature-engineering chain: patient features, categoricals, drop IDs."""
    from parkinsons.features import (
        add_patient_features,
        drop_id_columns,
        encode_categoricals,
    )

    X = add_patient_features(X)
    X = encode_categoricals(X)
    X = drop_id_columns(X)
    return X


def _apply_patient_features_no_cat(X: pd.DataFrame) -> pd.DataFrame:
    """Patient features only (no categorical encoding) — used by Step 1."""
    from parkinsons.features import add_patient_features, drop_id_columns

    # For Step 1 we only add numeric patient features; string cols are still
    # dropped by a subsequent step (or here directly).
    X = add_patient_features(X)
    X = drop_id_columns(X)
    return X.drop(columns=[c for c in _STRING_COLS if c in X.columns])


def build_hgbr_patient_features() -> Pipeline:
    """HGBR with patient-level temporal features (experiment 08).

    Adds ``time_since_diagnosis``, ``on_minus_off``, ``n_visits``,
    ``years_since_first_visit``, and per-patient mean/prev/next for on, off,
    ledd. String ID columns are dropped; no categorical encoding yet.

    Returns
    -------
    Pipeline
        Unfit pipeline ready for ``fit(X, y)`` or ``skore.evaluate``.
    """
    return Pipeline(
        steps=[
            (
                "patient_features",
                FunctionTransformer(_apply_patient_features_no_cat, validate=False),
            ),
            (
                "hgbr",
                HistGradientBoostingRegressor(random_state=0),
            ),
        ]
    )


def build_hgbr_with_categoricals() -> Pipeline:
    """HGBR with patient features + native categorical encoding (experiment 09).

    Adds all Step 1 features, then casts ``cohort``, ``gene``, and
    ``rater_id`` to ``category`` dtype so HGBR can use its native categorical
    split logic. ``patient_id`` is dropped.

    Returns
    -------
    Pipeline
        Unfit pipeline ready for ``fit(X, y)`` or ``skore.evaluate``.
    """
    return Pipeline(
        steps=[
            (
                "engineer",
                FunctionTransformer(_apply_patient_features, validate=False),
            ),
            (
                "hgbr",
                HistGradientBoostingRegressor(
                    random_state=0, categorical_features="from_dtype"
                ),
            ),
        ]
    )


def build_hgbr_tuned(
    *,
    learning_rate: float = 0.05,
    max_iter: int = 1000,
    max_leaf_nodes: int = 31,
    min_samples_leaf: int = 20,
    l2_regularization: float = 0.1,
    random_state: int = 0,
) -> Pipeline:
    """HGBR with tuned hyperparameters + early stopping (experiment 11).

    All feature engineering (patient features + categoricals) is included.

    Parameters
    ----------
    learning_rate : float
        Shrinkage applied to each tree.
    max_iter : int
        Maximum number of boosting rounds (early stopping may stop earlier).
    max_leaf_nodes : int
        Maximum leaves per tree.
    min_samples_leaf : int
        Minimum samples required in a leaf.
    l2_regularization : float
        L2 regularization on leaf values.
    random_state : int
        Reproducibility seed.

    Returns
    -------
    Pipeline
        Unfit pipeline ready for ``fit(X, y)`` or ``skore.evaluate``.
    """
    return Pipeline(
        steps=[
            (
                "engineer",
                FunctionTransformer(_apply_patient_features, validate=False),
            ),
            (
                "hgbr",
                HistGradientBoostingRegressor(
                    learning_rate=learning_rate,
                    max_iter=max_iter,
                    max_leaf_nodes=max_leaf_nodes,
                    min_samples_leaf=min_samples_leaf,
                    l2_regularization=l2_regularization,
                    early_stopping=True,
                    n_iter_no_change=20,
                    random_state=random_state,
                    categorical_features="from_dtype",
                ),
            ),
        ]
    )


# -- Experiment 05 (teammate): skrub TableVectorizer + HGBR ------------------


def build_skrub_learner() -> Pipeline:
    """Return an unfit skrub TableVectorizer + HGBR pipeline.

    Uses ``skrub.TableVectorizer`` to handle all column types — including
    categorical string columns such as ``gene``, ``cohort``, and ``rater_id``
    — followed by ``HistGradientBoostingRegressor``. No manual string-dropping
    or imputation needed: TableVectorizer encodes categoricals and handles NaN.
    Used by ``experiments/05_skrub_tabular.py``.

    Returns
    -------
    Pipeline
        Unfit ``Pipeline([("vectorizer", TableVectorizer()), ("hgbr", HGBR)])``
        ready for ``fit(X, y)`` or ``skore.evaluate``.
    """
    from skrub import TableVectorizer

    return Pipeline(
        steps=[
            ("vectorizer", TableVectorizer()),
            ("hgbr", HistGradientBoostingRegressor(random_state=0)),
        ]
    )
