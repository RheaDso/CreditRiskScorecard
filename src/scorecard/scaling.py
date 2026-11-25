"""M8: turn the logistic model into a points table.

    factor = PDO / ln(2)
    offset = base_score - factor * ln(base_odds)
    score  = offset + factor * ln(odds_good)
    points_ij = -(beta_i * WoE_ij + alpha / n) * factor + offset / n
"""

import math
from pathlib import Path

import numpy as np
import pandas as pd
from openpyxl import Workbook
from openpyxl.styles import Font

from scorecard.binning import VariableBins
from scorecard.config import Scaling
from scorecard.model import LogisticModel


def factor_offset(s: Scaling) -> tuple[float, float]:
    factor = s.pdo / math.log(2)
    return factor, s.base_score - factor * math.log(s.base_odds)


def score_from_odds(odds_good: float, s: Scaling) -> float:
    factor, offset = factor_offset(s)
    return offset + factor * math.log(odds_good)


def attribute_points(
    beta: float, woe: np.ndarray, intercept: float, n_variables: int, s: Scaling
) -> np.ndarray:
    """Unrounded points for each bin of one variable."""
    factor, offset = factor_offset(s)
    return -(beta * np.asarray(woe) + intercept / n_variables) * factor + offset / n_variables


def build_points_table(
    model: LogisticModel, bins: dict[str, VariableBins], s: Scaling
) -> pd.DataFrame:
    rows = []
    n = len(model.variables)
    for var in model.variables:
        vb = bins[var]
        beta = model.coefficients[var]
        raw = attribute_points(beta, np.array(vb.woe), model.intercept, n, s)
        for i, label in enumerate(vb.labels):
            rows.append(
                {"variable": var, "bin": i, "range": label, "woe": vb.woe[i],
                 "points_unrounded": raw[i], "points": int(round(raw[i]))}
            )
    return pd.DataFrame(rows)


def export_scorecard(table: pd.DataFrame, reports_dir: Path) -> None:
    out = table[["variable", "range", "woe", "points"]]
    out.to_csv(reports_dir / "scorecard.csv", index=False)
    book = Workbook()
    sheet = book.active
    sheet.title = "Scorecard"
    sheet.append(["Variable", "Bin range", "WoE", "Points"])
    for cell in sheet[1]:
        cell.font = Font(bold=True)
    for row in out.itertuples(index=False):
        sheet.append([row.variable, row.range, round(float(row.woe), 4), int(row.points)])
    for column, width in zip("ABCD", [32, 36, 10, 10], strict=False):
        sheet.column_dimensions[column].width = width
    sheet.freeze_panes = "A2"
    book.save(reports_dir / "scorecard.xlsx")
