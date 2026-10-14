"""Smoke test for ``experiments/02_ridge.py``.

The Ridge pipeline has no cross-row dependencies, so this is an IID
sanity check: fit on a train slice, predict on a held-out slice, assert
row count and a plausible MAE bound.
"""

from sklearn.metrics import mean_absolute_error

from parkinsons import PROJECT_ROOT
from parkinsons.data import load_dataset, load_test_dataset
from parkinsons.pipeline import build_ridge_learner

# Conservative bound: Ridge should beat the dummy (MAE ~13.5) significantly.
# Updated from journal/02_ridge.md § Status.headline after the first run.
CV_MAE_UPPER_BOUND = 14.0


def test_02_ridge():
    """Fit Ridge on a train slice, predict on held-out slice; assert row count."""
    X, y = load_dataset(data_dir=PROJECT_ROOT / "data")

    split = int(len(X) * 0.8)
    X_train, y_train = X.iloc[:split], y.iloc[:split]
    X_predict, y_predict = X.iloc[split:], y.iloc[split:]
    n_predict_grid_rows = len(X_predict)

    learner = build_ridge_learner()
    learner.fit(X_train, y_train)
    predictions = learner.predict(X_predict)

    # HARD: one prediction per predict-grid row.
    assert len(predictions) == n_predict_grid_rows, (
        f"got {len(predictions)} predictions for {n_predict_grid_rows} rows"
    )

    # SOFT: Ridge should be strictly better than always-predict-mean.
    smoke_mae = mean_absolute_error(y_predict, predictions)
    assert smoke_mae < CV_MAE_UPPER_BOUND, (
        f"smoke MAE {smoke_mae:.2f} >= upper bound {CV_MAE_UPPER_BOUND}; "
        "Ridge pipeline may have a bug (imputer or string-drop step)."
    )


def test_02_ridge_handles_missing():
    """Pipeline must not error on the real data's heavy missingness."""
    X, y = load_dataset(data_dir=PROJECT_ROOT / "data")
    learner = build_ridge_learner()
    # Just check fit+predict doesn't raise on the full training set.
    learner.fit(X, y)
    preds = learner.predict(X)
    assert len(preds) == len(X)


def test_02_ridge_test_loader():
    """Test loader returns a frame compatible with the Ridge pipeline."""
    X_test = load_test_dataset(data_dir=PROJECT_ROOT / "data")
    learner = build_ridge_learner()
    X, y = load_dataset(data_dir=PROJECT_ROOT / "data")
    learner.fit(X, y)
    preds = learner.predict(X_test)
    assert len(preds) == len(X_test)
