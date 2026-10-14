"""Data loading for the Parkinson's UPDRS prediction workspace.

Loads the raw CSV files and returns ``(X, y)`` ready for fitting,
plus ``X_test`` for the Kaggle submission.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from parkinsons import PROJECT_ROOT

DATA_DIR = PROJECT_ROOT / "data"

TARGET_COL = "target"


def load_dataset(data_dir: Path = DATA_DIR) -> tuple[pd.DataFrame, pd.Series]:
    """Return ``(X, y)`` from the training CSV files.

    Parameters
    ----------
    data_dir : Path, optional
        Directory that contains ``X_train.csv`` and ``y_train.csv``.

    Returns
    -------
    tuple[pd.DataFrame, pd.Series]
        ``X`` is the feature frame (index preserved from the CSV);
        ``y`` is the ``target`` series with the same index.
    """
    X = pd.read_csv(data_dir / "X_train.csv", index_col="Index")
    y = pd.read_csv(data_dir / "y_train.csv", index_col="Index")[TARGET_COL]
    return X, y


def load_groups(data_dir: Path = DATA_DIR) -> pd.Series:
    """Return the ``patient_id`` series from the training set for grouped CV.

    Parameters
    ----------
    data_dir : Path, optional
        Directory that contains ``X_train.csv``.

    Returns
    -------
    pd.Series
        ``patient_id`` values with the same index as ``load_dataset``'s ``X``.
        Pass directly as ``groups`` when computing GroupKFold splits.
    """
    return pd.read_csv(data_dir / "X_train.csv", index_col="Index")["patient_id"]


def load_test_dataset(data_dir: Path = DATA_DIR) -> pd.DataFrame:
    """Return ``X_test`` for the Kaggle submission CSV.

    Parameters
    ----------
    data_dir : Path, optional
        Directory that contains ``X_test.csv``.

    Returns
    -------
    pd.DataFrame
        Test feature frame (index preserved from the CSV).
    """
    return pd.read_csv(data_dir / "X_test.csv", index_col="Index")
