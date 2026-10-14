"""Smoke test for ``experiments/04_hgbr.py``.

Verifies HGBR handles NaN natively across GroupKFold folds and produces
one prediction per validation row with no patient leakage.
"""

import numpy as np
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import GroupKFold

from parkinsons import PROJECT_ROOT
from parkinsons.data import load_dataset, load_groups
from parkinsons.pipeline import build_hgbr_learner

# Conservative upper bound — HGBR should beat Ridge (MAE ~8.5).
# Updated from journal/04_hgbr.md § Status.headline after the first run.
CV_MAE_UPPER_BOUND = 10.0


def test_04_hgbr_no_patient_leakage():
    """Each fold's validation set must contain only patients absent from training."""
    X, y = load_dataset(data_dir=PROJECT_ROOT / "data")
    groups = load_groups(data_dir=PROJECT_ROOT / "data")

    cv = GroupKFold(n_splits=5)
    for fold_idx, (train_idx, val_idx) in enumerate(cv.split(X, y, groups)):
        overlap = set(groups.iloc[train_idx]) & set(groups.iloc[val_idx])
        assert len(overlap) == 0, (
            f"Fold {fold_idx}: {len(overlap)} patient(s) in both train and val."
        )


def test_04_hgbr_row_count_and_nan():
    """HGBR must return one prediction per val row and handle NaN without error."""
    X, y = load_dataset(data_dir=PROJECT_ROOT / "data")
    groups = load_groups(data_dir=PROJECT_ROOT / "data")

    cv = GroupKFold(n_splits=5)
    # Test only the first fold to keep the smoke test fast.
    train_idx, val_idx = next(iter(cv.split(X, y, groups)))

    learner = build_hgbr_learner()
    learner.fit(X.iloc[train_idx], y.iloc[train_idx])
    preds = learner.predict(X.iloc[val_idx])

    # HARD: row count.
    assert len(preds) == len(val_idx), (
        f"Got {len(preds)} predictions for {len(val_idx)} val rows."
    )

    # SOFT: no NaN predictions (HGBR should never produce NaN).
    assert not np.isnan(preds).any(), "HGBR produced NaN predictions."

    # SOFT: MAE within a reasonable bound.
    mae = mean_absolute_error(y.iloc[val_idx], preds)
    assert mae < CV_MAE_UPPER_BOUND, (
        f"Fold 0 MAE {mae:.2f} >= upper bound {CV_MAE_UPPER_BOUND}."
    )
