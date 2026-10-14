# EDA: Parkinson's Disease UPDRS Score Prediction

Exploratory data analysis run on the raw CSV files before any modelling.
Interactive column-level reports: [eda_train.html](eda_train.html) · [eda_test.html](eda_test.html).

---

## Dataset at a glance

| Split | Rows | Columns (incl. target) |
|---|---|---|
| Train (`X_train` + `y_train`) | 44,590 | 13 |
| Test (`X_test`) | 11,013 | 12 |

**Task:** regression — predict a continuous UPDRS (Unified Parkinson's Disease Rating Scale)
motor score (`target`).  
**Patient cohorts:** A and B (2 unique values). **Genes:** LRRK2+, LRRK2-, SNCA+, missing
(4 values + ~32 % NaN).

---

## Per-column findings

| Column | Dtype | % Missing | # Unique | Notes |
|---|---|---|---|---|
| `patient_id` | string | 0 % | 5,576 | High-cardinality ID. Each patient has multiple rows (visits). Unique-ratio = 0.125. |
| `cohort` | string | 0 % | 2 | Study cohort A / B. |
| `sexM` | int | 0 % | 2 | Binary. |
| `gene` | string | **32 %** | 4 | Heavy missingness (~1 in 3 rows). |
| `age_at_diagnosis` | float | **5 %** | 563 | Highly correlated with `age` (r = 0.94). |
| `age` | float | 0 % | 1,150 | — |
| `ledd` | float | **37 %** | 1,320 | Levodopa equivalent daily dose. Moderate missing. |
| `time_since_intake_on` | float | **46 %** | 64 | Time since medication intake in ON state. Heavily sparse. |
| `time_since_intake_off` | float | **79 %** | 178 | Time since intake in OFF state. Very sparse — only ~21 % of rows populated. |
| `rater_id` | string | 0 % | 50 | Clinical rater who performed the assessment. Potential confounder. |
| `on` | float | **30 %** | 83 | UPDRS score in the ON (medicated) state. Strong predictor of `target` (r = 0.67). |
| `off` | float | **42 %** | 101 | UPDRS score in the OFF (unmedicated) state. **Strongest predictor** (r = 0.87). |
| `target` | float | 0 % | 867 | The continuous regression target. |

Missingness patterns largely co-occur: rows missing `time_since_intake_off` also tend
to be missing `off`. This is structurally meaningful — OFF assessments are harder to collect.

---

## Target

**Distribution (44,590 rows, no missing values):**

| Stat | Value |
|---|---|
| Mean | 37.5 |
| Std | 16.5 |
| Median (Q50) | 37.3 |
| IQR (Q25–Q75) | 25.6 – 49.3 |
| Min / Max | 0.0 / 109.5 |
| # Unique values | 867 |

The target is roughly bell-shaped around the mid-30s with a slight right skew
(a long tail extending to ~110). Not class-balanced in the classification sense.
No hard floor at zero causes concern. **Stratification is not needed**;
standard K-Fold is fine, though GroupKFold per patient is preferred (see below).

---

## Structure signals

**No datetime columns detected.** The dataset has no explicit timestamps → `TimeSeriesSplit`
is not indicated.

**Patient-level grouping is the dominant structure.** `patient_id` appears 5,576 times
across 44,590 rows — on average ~8 visits per patient. If random K-Fold is used,
the same patient can appear in both train and validation, leaking patient-specific
baseline severity. This is likely the biggest source of optimistic bias in any naive CV.

`rater_id` has 50 unique values and could also confound — raters differ in scoring
stringency — but splitting by patient already captures most of this.

---

## Associations

Top associations (skrub `column_associations`, Cramér V + Pearson r):

| Pair | Cramér V | Pearson r | Comment |
|---|---|---|---|
| `age_at_diagnosis` ↔ `age` | 0.56 | **0.94** | Nearly redundant — consider dropping one or using age at diagnosis only |
| `off` ↔ `target` | 0.41 | **0.87** | Strongest predictor. Check for leakage (see below). |
| `on` ↔ `target` | 0.37 | **0.67** | Second strongest. |
| `on` ↔ `off` | 0.33 | **0.87** | The ON and OFF scores are co-linear. |
| `ledd` ↔ `on` | 0.29 | 0.20 | Dose relates to ON-state score. |
| `time_since_intake_on` ↔ `on` | 0.25 | −0.22 | Inverse: more time → lower ON score. |
| `ledd` ↔ `target` | 0.24 | 0.30 | Moderate. |

**Leakage flag — `off` and `on`:** These columns represent UPDRS scores in different
medication states for the *same visit*. The target is described as UPDRS but its exact
definition relative to `on` and `off` is ambiguous from the raw data alone. If `target`
is derived from `on` and `off`, including them in the model would be leakage. If they
are independent assessments, they are legitimate features. **This must be confirmed with
the problem statement** before proceeding.

`age_at_diagnosis` ↔ `age` Pearson r = 0.94: these two are near-redundant; keeping both
adds noise. Consider dropping `age_at_diagnosis` or computing `disease_duration = age − age_at_diagnosis`.

---

## Modelling implications

| Finding | Implication |
|---|---|
| Repeated patient visits (avg 8 visits/patient) | Use **`GroupKFold`** (groups = `patient_id`) to prevent patient leakage across folds |
| Continuous, near-normally distributed target | **RMSE or MAE** as primary metric; no need to stratify |
| Heavy missingness in `gene`, `ledd`, `time_since_intake_on/off`, `on`, `off` | Imputation needed in the pipeline (median for numeric, constant/"missing" for string) |
| `off` r=0.87 and `on` r=0.67 with `target` | These are the most predictive features **if not leakage** — confirm before including |
| `age_at_diagnosis` / `age` near-redundant (r=0.94) | Consider engineering `disease_duration`; drop `age_at_diagnosis` |
| 50 raters (`rater_id`) | `rater_id` is a useful categorical feature but may need target-encoding; cardinality is manageable |
| No datetime column | `TimeSeriesSplit` not needed |

---

## Open questions

1. **Is `target` derived from `on` or `off`?** The near-perfect correlation (r=0.87 with `off`)
   warrants explicit confirmation. If yes → drop both from features.
2. **What does `target` measure exactly?** UPDRS Part III (motor)? A composite? The
   `sample_submission.csv` shows the same target value repeated per patient, suggesting
   `target` may be a patient-level aggregated score rather than a per-visit score — if so
   the unit of prediction is the patient, not the visit.
3. **Does `cohort` (A/B) introduce distribution shift** that the model should be aware of?
4. **Should `ledd` be log-transformed?** The distribution is likely right-skewed (drugs dosage).
