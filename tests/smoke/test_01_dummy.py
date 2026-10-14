"""Smoke test for ``experiments/01_dummy.py``.

The DummyRegressor has no cross-row dependencies, so the hard
row-count assertion is a sanity check on the loaders rather than
a pipeline-topology proof. The soft assertion checks that predictions
are not NaN-poisoned and stay in a plausible range for the UPDRS
target (mean ≈ 37.5, std ≈ 16.5).
"""

from sklearn.metrics import mean_absolute_error

from parkinsons import PROJECT_ROOT
from parkinsons.data import load_dataset, load_test_dataset
from parkinsons.pipeline import build_learner

# Hardcoded from the target distribution (std ≈ 16.5).  Updated from
# journal/01_dummy.md § Status.headline after the first run.
CV_MAE_APPROX = 16.5


def test_01_dummy():
    """Fit on a train slice, predict on a held-out slice; assert row count."""
    X, y = load_dataset(data_dir=PROJECT_ROOT / "data")

    # Use the last 20 % of rows as the predict grid (IID holdout).
    split = int(len(X) * 0.8)
    X_train, y_train = X.iloc[:split], y.iloc[:split]
    X_predict, y_predict = X.iloc[split:], y.iloc[split:]
    n_predict_grid_rows = len(X_predict)

    learner = build_learner()
    learner.fit(X_train, y_train)
    predictions = learner.predict(X_predict)

    # HARD: one prediction per predict-grid row.
    assert len(predictions) == n_predict_grid_rows, (
        f"got {len(predictions)} predictions for {n_predict_grid_rows} rows"
    )

    # SOFT: predictions are finite and within a plausible MAE bound.
    smoke_mae = mean_absolute_error(y_predict, predictions)
    assert smoke_mae < 3 * CV_MAE_APPROX, (
        f"smoke MAE {smoke_mae:.1f} > 3 × expected ({CV_MAE_APPROX:.1f}); "
        "predictions may be degenerate."
    )


def test_01_dummy_test_loader():
    """Test loader returns a frame with the right shape."""
    X_test = load_test_dataset(data_dir=PROJECT_ROOT / "data")
    assert len(X_test) > 0, "X_test is empty"
    assert "patient_id" in X_test.columns
