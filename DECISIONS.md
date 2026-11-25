# Decisions and deviations from the build spec

## Why these methods
- **WoE + logistic regression.** Each variable becomes a few bins with a single risk number, the
  model stays additive, and every score can be broken down into points and reason codes.
- **No resampling or class weights.** The bad rate is about 8%; reweighting would distort the
  predicted probabilities.
- **Fit on TRAIN only.** Bins, WoE, dropped columns, selection and coefficients never see TEST.

## Deviations

| Spec | What was done | Why |
|---|---|---|
| Dataset | Published results use synthetic data | The Kaggle files require a signed-in user who accepts the competition rules |
| FR-1.1 | pandas infers dtypes; no dtype map in config | Time limit. Child tables do load only the needed columns |
| FR-3.5 | Columns are dropped based on TRAIN only | The spec's leakage rule overrides the whole-table wording |
| FR-5.1 | optbinning's tree pre-binning (up to 20 bins) replaces quantile fine classing | One library step does both |
| FR-5.5 | Manual cut points work for numeric variables only | Categorical overrides were not needed |
| M6 | Raw `DAYS_*` columns are not candidates | They duplicate the readable `*_YEARS` features |
| FR-6.4 | Stepwise keeps adding below 8 variables even if the gain is under 0.002 | Otherwise it can conflict with the 8-variable minimum (FR-6.6) |
| FR-8.4 | Published score is the sum of rounded points; PD comes from the unrounded model | The printed points table must add up to the score |
| FR-11.2 | Cut-off is chosen on VALIDATION, reported on TEST | Choosing it on TEST would be tuning on TEST |
| FR-12.5 | The API takes bureau and history aggregates pre-computed | As the spec allows |
| Smoke run | `--smoke` uses 8,000 synthetic rows, 3 trials, minimum 5 variables | Small samples support fewer significant variables |
| Testing | Not every requirement has its own named test | Time limit; coverage of `src/scorecard` is above 80% |
| Phase 2 | `bureau_balance`, POS and credit card tables are not used | Optional in the spec |

## Known miss: train-test Gini gap

With seed 42 the gap is about 0.046 (limit 0.03). Refitting the final variables on eight different
split seeds gave gaps from -0.014 to 0.047, averaging about 0.019. So much of it is the particular
split. The seed was **not** changed to pick a better-looking result. Possible improvements: fewer
bins, fewer variables, or cross-validation instead of a single split.

## Not verified
- The Docker image was not built on the development machine (Docker is not installed). The CI
  workflow builds and calls it.
- `make` is not installed on the development machine, so the Makefile targets were not run there.

## Leakage review: EXT_SOURCE_MEAN (IV 0.54)

Its IV is above 0.50. It is the row-wise mean of three applicant inputs and uses no outcome data. It is not in the final model (removed by the VIF step).
