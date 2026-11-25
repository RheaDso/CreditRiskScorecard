"""M6: reduce all candidates to a small, stable, explainable set of variables."""

import logging

import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.linear_model import LogisticRegression
from statsmodels.stats.outliers_influence import variance_inflation_factor

from scorecard.config import Config
from scorecard.exceptions import SelectionError
from scorecard.features import source_of
from scorecard.stability import gini

log = logging.getLogger(__name__)


def iv_filter(iv: dict[str, float], min_iv: float) -> list[str]:
    """Keep variables with enough predictive power, strongest first."""
    return sorted((v for v in iv if iv[v] >= min_iv), key=lambda v: -iv[v])


def correlation_filter(woe: pd.DataFrame, ordered: list[str], max_corr: float) -> list[str]:
    """Walk from the highest IV down; skip anything too correlated with one already kept."""
    corr = woe[ordered].corr().abs()
    kept: list[str] = []
    for var in ordered:
        if all(corr.loc[var, other] <= max_corr for other in kept):
            kept.append(var)
    return kept


def vif_table(woe: pd.DataFrame, variables: list[str]) -> pd.Series:
    x = sm.add_constant(woe[variables]).to_numpy()
    return pd.Series(
        [variance_inflation_factor(x, i + 1) for i in range(len(variables))], index=variables
    )


def vif_filter(woe: pd.DataFrame, variables: list[str], max_vif: float) -> list[str]:
    variables = list(variables)
    while len(variables) > 1:
        vif = vif_table(woe, variables)
        if vif.max() < max_vif:
            break
        log.info("Dropping %s (VIF %.2f)", vif.idxmax(), vif.max())
        variables.remove(str(vif.idxmax()))
    return variables


def _validation_gini(x_tr, y_tr, x_val, y_val, variables: list[str]) -> float:
    model = LogisticRegression(penalty=None, max_iter=1000).fit(x_tr[variables], y_tr)
    return gini(y_val, model.predict_proba(x_val[variables])[:, 1])


def forward_stepwise(
    x_tr: pd.DataFrame, y_tr: pd.Series, x_val: pd.DataFrame, y_val: pd.Series,
    candidates: list[str], cfg: Config,
) -> tuple[list[str], pd.DataFrame]:
    """Add the variable that most improves VALIDATION Gini, one at a time.

    Below `min_vars` the best variable is added even if its gain is small, so the
    model can reach the minimum size (see DECISIONS.md).
    """
    s = cfg.selection
    chosen: list[str] = []
    history = []
    best = 0.0
    remaining = list(candidates)
    while remaining and len(chosen) < s.max_vars:
        scores = {v: _validation_gini(x_tr, y_tr, x_val, y_val, [*chosen, v]) for v in remaining}
        winner = max(scores, key=scores.__getitem__)
        gain = scores[winner] - best
        if gain < s.stepwise_min_gini_gain and len(chosen) >= s.min_vars:
            break
        chosen.append(winner)
        remaining.remove(winner)
        best = scores[winner]
        history.append({"step": len(chosen), "added": winner, "validation_gini": best,
                        "gain": gain})

    # Business diversity: try to cover application, bureau and payment history.
    for source in ("application", "bureau", "instalments"):
        options = [v for v in remaining if source_of(v) == source]
        if options and not any(source_of(v) == source for v in chosen):
            scores = {
                v: _validation_gini(x_tr, y_tr, x_val, y_val, [*chosen, v]) for v in options
            }
            winner = max(scores, key=scores.__getitem__)
            chosen.append(winner)
            history.append({"step": len(chosen), "added": winner,
                            "validation_gini": scores[winner], "gain": scores[winner] - best})
            best = scores[winner]
    return chosen, pd.DataFrame(history)


def sign_and_significance(
    x_tr: pd.DataFrame, y_tr: pd.Series, variables: list[str], max_pvalue: float
) -> list[str]:
    """Every coefficient must be negative and significant. Remove one offender, refit."""
    variables = list(variables)
    while variables:
        result = sm.Logit(y_tr, sm.add_constant(x_tr[variables])).fit(disp=0)
        beta, p = result.params[variables], result.pvalues[variables]
        if (beta >= 0).any():
            drop = str(beta.idxmax())
            log.info("Dropping %s: wrong sign (beta %.3f)", drop, beta.max())
        elif (p >= max_pvalue).any():
            drop = str(p.idxmax())
            log.info("Dropping %s: not significant (p %.3f)", drop, p.max())
        else:
            return variables
        variables.remove(drop)
    raise SelectionError("No variable has a negative, significant coefficient")


def select_variables(
    x_tr: pd.DataFrame, y_tr: pd.Series, x_val: pd.DataFrame, y_val: pd.Series,
    iv: dict[str, float], cfg: Config,
) -> tuple[list[str], pd.DataFrame, pd.DataFrame, pd.Series]:
    """Run the five steps. Returns variables, funnel, stepwise log and final VIF."""
    s = cfg.selection
    funnel = [("candidates", len(iv))]
    step1 = iv_filter(iv, s.min_iv)
    funnel.append(("iv_filter", len(step1)))
    step2 = correlation_filter(x_tr, step1, s.max_corr)
    funnel.append(("correlation_filter", len(step2)))
    step3 = vif_filter(x_tr, step2, s.max_vif)
    funnel.append(("vif_filter", len(step3)))
    step4, history = forward_stepwise(x_tr, y_tr, x_val, y_val, step3, cfg)
    funnel.append(("stepwise", len(step4)))
    final = sign_and_significance(x_tr, y_tr, step4, s.max_pvalue)
    funnel.append(("sign_and_significance", len(final)))

    if len(final) < s.min_vars:
        raise SelectionError(
            f"Only {len(final)} variables survived; at least {s.min_vars} are required"
        )
    vif = vif_table(x_tr, final)
    assert np.all(vif < s.max_vif)
    return final, pd.DataFrame(funnel, columns=["step", "variables"]), history, vif
