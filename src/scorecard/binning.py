"""M5: coarse classing, Weight of Evidence and Information Value.

optbinning only chooses the cut points (on TRAIN). WoE and IV are computed here
so the formulas are visible and testable.

    WoE_j = ln(%Good_j / %Bad_j)          positive = safer than average
    IV    = sum_j (%Good_j - %Bad_j) * WoE_j
"""

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from optbinning import OptimalBinning

from scorecard.config import Config

log = logging.getLogger(__name__)

MISSING = "Missing"
OTHER = "OTHER"


def woe_iv(good: np.ndarray, bad: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """WoE and IV contribution per bin from good and bad counts."""
    good = np.asarray(good, dtype=float)
    bad = np.asarray(bad, dtype=float)
    zero = (good == 0) | (bad == 0)
    if zero.any():
        log.info("Zero-count adjustment applied to %d bin(s)", zero.sum())
        good = good + 0.5 * zero
        bad = bad + 0.5 * zero
    pct_good = good / good.sum()
    pct_bad = bad / bad.sum()
    woe = np.log(pct_good / pct_bad)
    return woe, (pct_good - pct_bad) * woe


@dataclass
class VariableBins:
    """Fitted bins for one variable; `transform` applies them to any data."""

    name: str
    categorical: bool
    splits: list[float] = field(default_factory=list)
    groups: dict[str, int] = field(default_factory=dict)  # category -> bin index
    labels: list[str] = field(default_factory=list)
    woe: list[float] = field(default_factory=list)
    iv: float = 0.0
    table: pd.DataFrame = field(default_factory=pd.DataFrame)

    def assign(self, values: pd.Series) -> np.ndarray:
        """Bin index per value. -1 means missing (or unseen with no OTHER bin)."""
        if self.categorical:
            text = values.astype(object).where(values.notna(), None)
            fallback = self.groups.get(OTHER, -2)
            return np.array(
                [-1 if v is None else self.groups.get(str(v), fallback) for v in text], dtype=int
            )
        x = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)
        index = np.searchsorted(self.splits, x, side="right")  # bins are [low, high)
        return np.where(np.isnan(x), -1, index)

    def positions(self, values: pd.Series) -> np.ndarray:
        """Row of `labels`/`woe` for each value; -1 if there is no matching bin."""
        index = self.assign(values)
        if MISSING in self.labels:
            index = np.where(index == -1, self.labels.index(MISSING), index)
        return np.where(index < 0, -1, index)

    def transform(self, values: pd.Series) -> np.ndarray:
        """WoE per value. A value with no matching bin gets the neutral WoE of 0."""
        position = self.positions(values)
        woe = np.append(np.array(self.woe), 0.0)  # index -1 -> 0.0
        return woe[position]


def _find_splits(x: pd.Series, y: pd.Series, categorical: bool, cfg: Config) -> list:
    b = cfg.binning
    optimiser = OptimalBinning(
        name=str(x.name),
        dtype="categorical" if categorical else "numerical",
        monotonic_trend=None if categorical else b.monotonic_trend,
        min_prebin_size=b.min_bin_share,
        min_bin_size=b.min_bin_share,
        max_n_bins=b.max_n_bins,
        max_n_prebins=b.fine_bins,
    )
    present = x.notna()
    optimiser.fit(x[present].to_numpy(), y[present].to_numpy())
    return list(optimiser.splits)


def fit_variable(
    x: pd.Series, y: pd.Series, cfg: Config, override: list[float] | None = None
) -> VariableBins:
    """Fit bins for one variable using TRAIN rows only."""
    name = str(x.name)
    categorical = not pd.api.types.is_numeric_dtype(x)
    vb = VariableBins(name=name, categorical=categorical)

    if categorical:
        text = x.astype(object).where(x.notna(), None)
        share = text.value_counts() / len(x)
        rare = set(share[share < cfg.features.rare_category_share].index)
        grouped = text.map(lambda v: None if v is None else (OTHER if v in rare else str(v)))
        grouped.name = name
        splits = _find_splits(grouped, y, True, cfg)
        for i, members in enumerate(splits):
            labels = [str(m) for m in members]
            vb.groups.update({m: i for m in labels})
            vb.labels.append(", ".join(sorted(labels)))
        if OTHER in vb.groups:  # rare raw levels share the OTHER bin
            vb.groups.update({str(v): vb.groups[OTHER] for v in rare})
    else:
        if override is not None:
            log.info("Manual cut points used for %s: %s", name, override)
            vb.splits = sorted(float(v) for v in override)
        elif x.notna().sum() > 0 and x.nunique() > 1:
            vb.splits = [float(s) for s in _find_splits(x, y, False, cfg)]
        edges = [-np.inf, *vb.splits, np.inf]
        vb.labels = [f"[{lo:.4g}, {hi:.4g})" for lo, hi in zip(edges[:-1], edges[1:], strict=False)]

    index = vb.assign(x)
    if (index == -1).any():
        vb.labels.append(MISSING)
        index = np.where(index == -1, len(vb.labels) - 1, index)

    n_bins = len(vb.labels)
    count = np.bincount(index, minlength=n_bins)
    bad = np.bincount(index, weights=y.to_numpy(), minlength=n_bins)
    good = count - bad
    woe, iv = woe_iv(good, bad)
    vb.woe = woe.tolist()
    vb.iv = float(iv.sum())
    vb.table = pd.DataFrame(
        {
            "variable": name,
            "bin": range(n_bins),
            "range": vb.labels,
            "count": count,
            "pct": count / len(x),
            "bad_rate": np.divide(bad, count, out=np.zeros(n_bins), where=count > 0),
            "woe": woe,
            "iv_contribution": iv,
        }
    )
    return vb


def fit_binning(
    train: pd.DataFrame, y: pd.Series, columns: list[str], cfg: Config,
    overrides: dict[str, list[float]] | None = None,
) -> dict[str, VariableBins]:
    overrides = overrides or {}
    return {c: fit_variable(train[c], y, cfg, overrides.get(c)) for c in columns}


def to_woe(df: pd.DataFrame, bins: dict[str, VariableBins], columns: list[str]) -> pd.DataFrame:
    return pd.DataFrame({c: bins[c].transform(df[c]) for c in columns}, index=df.index)


def iv_summary(bins: dict[str, VariableBins]) -> pd.DataFrame:
    return pd.DataFrame(
        {"variable": list(bins), "iv": [b.iv for b in bins.values()]}
    ).sort_values("iv", ascending=False, ignore_index=True)
