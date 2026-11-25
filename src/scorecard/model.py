"""M7: unregularised logistic regression on WoE values."""

from dataclasses import dataclass

import numpy as np
import pandas as pd
import statsmodels.api as sm


@dataclass
class LogisticModel:
    variables: list[str]
    intercept: float
    coefficients: dict[str, float]
    summary: str = ""

    def predict_pd(self, woe: pd.DataFrame) -> np.ndarray:
        beta = np.array([self.coefficients[v] for v in self.variables])
        z = self.intercept + woe[self.variables].to_numpy() @ beta
        return 1 / (1 + np.exp(-z))


def fit_logistic(woe_train: pd.DataFrame, y: pd.Series, variables: list[str]) -> LogisticModel:
    """No class weights or resampling, so predicted probabilities stay calibrated."""
    result = sm.Logit(y, sm.add_constant(woe_train[variables])).fit(disp=0)
    return LogisticModel(
        variables=list(variables),
        intercept=float(result.params["const"]),
        coefficients={v: float(result.params[v]) for v in variables},
        summary=str(result.summary()),
    )
