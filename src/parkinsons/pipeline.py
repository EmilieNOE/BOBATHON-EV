"""Learner declarations for Parkinson's UPDRS prediction experiments.

Each function returns an unfit estimator ready for ``skore.evaluate`` or
``.fit(X, y)``. Adding a new function here (rather than modifying ``build_learner``)
preserves the reproducibility of prior experiments.
"""

from __future__ import annotations

import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from skrub import TableVectorizer

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
    from sklearn.preprocessing import FunctionTransformer

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
    from sklearn.preprocessing import FunctionTransformer

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
    return Pipeline(
        steps=[
            ("vectorizer", TableVectorizer()),
            ("hgbr", HistGradientBoostingRegressor(random_state=0)),
        ]
    )
