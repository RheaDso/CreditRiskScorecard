# Run summary (synthetic data)

| metric | target | actual | met |
|---|---|---|---|
| TEST Gini | >= 0.45 | 0.5142 | yes |
| TEST KS | >= 0.3 | 0.3855 | yes |
| Train-Test Gini gap | <= 0.03 | 0.0458 | NO |
| Score PSI train vs test | < 0.1 | 0.0009 | yes |
| Max VIF | < 5.0 | 1.011 | yes |
| All coefficients negative, p < 0.05 | yes | yes | yes |

## Notes

- Variables in the final model: 12
- Largest possible score difference caused by rounding points: 5.32
- Recommended cut-off: 545 (TEST approval rate 80.6%, bad rate among approved 4.86%)
- Champion: logistic scorecard (challenger Gini uplift -0.0113)

Test count and coverage: run `pytest`. Deviations: see [DECISIONS.md](../DECISIONS.md).
