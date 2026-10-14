"""Smoke test for ``experiments/03_ridge_cv.py``.

Verifies that GroupKFold splits produced from patient_id never leak a patient
across the train/validation boundary, and that Ridge predictions cover every
held-out row in each fold.
"""

import numpy as np
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import GroupKFold

from parkinsons import PROJECT_ROOT
from parkinsons.data import load_dataset, load_groups
from parkinsons.pipeline import build_ridge_learner

# Conservative upper bound — grouped CV is expected to be higher than 02_ridge
# (RMSE 10.49). Updated from journal/03_ridge_cv.md § Status.headline after run.
CV_MAE_UPPER_BOUND = 16.0


def test_03_ridge_cv_no_patient_leakage():
    """Each fold's validation set must contain only patients absent from training."""
    X, y = load_dataset(data_dir=PROJECT_ROOT / "data")
    groups = load_groups(data_dir=PROJECT_ROOT / "data")

    cv = GroupKFold(n_splits=5)
    for fold_idx, (train_idx, val_idx) in enumerate(cv.split(X, y, groups)):
        train_patients = set(groups.iloc[train_idx])
        val_patients = set(groups.iloc[val_idx])
        overlap = train_patients & val_patients
        assert len(overlap) == 0, (
            f"Fold {fold_idx}: {len(overlap)} patient(s) appear in both train and "
            f"validation — GroupKFold is not working as expected."
        )


def test_03_ridge_cv_row_count():
    """Ridge must return exactly one prediction per validation row in every fold."""
    X, y = load_dataset(data_dir=PROJECT_ROOT / "data")
    groups = load_groups(data_dir=PROJECT_ROOT / "data")

    cv = GroupKFold(n_splits=5)
    for fold_idx, (train_idx, val_idx) in enumerate(cv.split(X, y, groups)):
        X_train_fold = X.iloc[train_idx]
        y_train_fold = y.iloc[train_idx]
        X_val_fold = X.iloc[val_idx]

        learner = build_ridge_learner()
        learner.fit(X_train_fold, y_train_fold)
        preds = learner.predict(X_val_fold)

        # HARD: one prediction per validation row.
        assert len(preds) == len(X_val_fold), (
            f"Fold {fold_idx}: got {len(preds)} predictions for "
            f"{len(X_val_fold)} validation rows."
        )

        # SOFT: predictions are not degenerate.
        mae = mean_absolute_error(y.iloc[val_idx], preds)
        assert mae < CV_MAE_UPPER_BOUND, (
            f"Fold {fold_idx}: MAE {mae:.2f} >= upper bound {CV_MAE_UPPER_BOUND}. "
            "Ridge may have a bug in this fold."
        )

    # Confirm total coverage: each row appears in exactly one validation fold.
    all_val_indices = np.concatenate(
        [val for _, val in cv.split(X, y, groups)]
    )
    assert len(all_val_indices) == len(X), (
        f"Expected {len(X)} total validation rows across folds, "
        f"got {len(all_val_indices)}."
    )
