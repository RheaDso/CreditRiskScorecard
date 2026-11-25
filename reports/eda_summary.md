# Exploratory data analysis

## Target distribution

| split | rows | bads | bad rate |
|---|---|---|---|
| test | 20,000 | 1,567 | 7.83% |
| train | 60,000 | 4,702 | 7.84% |
| validation | 20,000 | 1,568 | 7.84% |

## Observations (TRAIN)

- The data has 100,000 applicants with an overall bad rate of 7.84%.
- EXT_SOURCE_MEAN has the strongest single-variable AUC of 0.701.
- The next strongest are EXT_SOURCE_MIN (0.668) and EXT_SOURCE_MAX (0.659).
- 15 of 46 numeric variables have AUC above 0.55.
- The most often missing variable is EXT_SOURCE_1 (45.0% missing).
- 18 variables have at least one missing value.
