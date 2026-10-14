# JOURNAL

<!--
Durable index of every experiment in this workspace. Four sections,
in order: Status, Data understanding (EDA), History, Backlog - keep
them so the file stays quick to scan. Each journal/NN_short_name.md
design note pairs one-to-one with experiments/NN_short_name.py (same
stem).
-->

## Status

- **Project / dataset:** Parkinson's disease UPDRS motor score regression
- **Goal:** Predict a patient's UPDRS motor score (continuous) from clinical visit features
- **Last experiment:** 04_hgbr - done
- **Last result:** RMSE 7.76 ± 0.15, MAE 6.10 ± 0.10, R² 0.779 ± 0.004 (5-fold GroupKFold) — HGBR no imputer

<!--
Workspace decisions: one-time project-setup choices. Record each when
it is made and treat it as fixed unless you deliberately change one
(e.g. switch pandas → polars), updating the recorded date. Reading
this block on later sessions avoids re-deciding what's already settled.

A row is recorded only when the value is a concrete choice and a date.
Angle brackets left in the value (`<pandas | polars>`, `<YYYY-MM-DD>`)
mean this template is still here. That is not a decision.
-->

- **Workspace decisions** (immutable unless the user pivots):
  - tabular library: pandas - recorded: 2026-10-07
  - env manager: pip+venv - recorded: 2026-10-07
  - agent feature: <installed> - recorded: <YYYY-MM-DD>
  - optional features: none - recorded: 2026-10-07
  - package name (`src/<pkg>/`): parkinsons - recorded: 2026-10-07
  - skore mode: hub - recorded: 2026-10-07
  - skore hub workspace: evdgreenteam - recorded: 2026-10-07
  - skore mlflow tracking uri: n/a - recorded: 2026-10-07
  - student prior: some-sklearn - recorded: 2026-10-07
  - CV splitter family: holdout-0.2 (01_dummy/02_ridge); GroupKFold for grouped-CV experiments - recorded: 2026-10-07

## Data understanding (EDA)

- **Status:** done - 2026-10-07
- **Summary:** 44,590 train rows × 12 features + target; 5,576 unique patients (~8 visits
  each). Target (UPDRS motor score) is continuous, roughly bell-shaped (mean 37.5, std 16.5),
  no missing values. Key findings: `off` (r=0.87) and `on` (r=0.67) are the strongest
  predictors but may be leaky — confirm their relationship to `target` before including.
  Patient-level grouping dominates structure → GroupKFold by `patient_id` is strongly
  indicated. Heavy missingness in `gene` (32 %), `ledd` (37 %), `time_since_intake_on` (46 %),
  `time_since_intake_off` (79 %).
- **Report:** [data/eda.md](../data/eda.md)

## History

<!--
One row per experiment, in chronological order. Newest at the bottom.
Status values: planned | approved | running | done | abandoned.
-->

| Stem | Intent (one line) | Status | Headline result | Design note |
|---|---|---|---|---|
| `01_dummy` | DummyRegressor(mean) — performance floor | done | RMSE 16.48, MAE 13.48, R² ≈ 0.0 (80/20 holdout) | [design note](01_dummy.md) |
| `02_ridge` | Ridge + median imputer vs dummy comparison | done | RMSE 10.49, MAE 8.44, R² 0.595 (80/20 holdout) | [design note](02_ridge.md) |
| `03_ridge_cv` | Ridge with 5-fold GroupKFold (patient_id) — honest holdout | done | RMSE 10.60 ± 0.27, MAE 8.53 ± 0.18, R² 0.587 ± 0.016 | [design note](03_ridge_cv.md) |
| `04_hgbr` | HGBR (random_state=0), no imputer, 5-fold GroupKFold | done | RMSE 7.76 ± 0.15, MAE 6.10 ± 0.10, R² 0.779 ± 0.004 | [design note](04_hgbr.md) |

## Backlog

<!--
Ideas not yet committed to a journal/NN_*.md design note. Each row
has a stable B<N> index so it can be picked by number ("go with B2").
-->

| # | Item | Source |
|---|---|---|
