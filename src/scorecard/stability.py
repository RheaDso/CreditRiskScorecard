"""Ranking and stability measures: Gini, KS and PSI."""

import numpy as np
from sklearn.metrics import roc_auc_score


def gini(y: np.ndarray, p_bad: np.ndarray) -> float:
    return float(2 * roc_auc_score(y, p_bad) - 1)


def ks_statistic(y: np.ndarray, p_bad: np.ndarray) -> float:
    """Largest gap between the cumulative bad and cumulative good distributions."""
    y = np.asarray(y)
    p_bad = np.asarray(p_bad)
    values = np.unique(p_bad)
    bad = np.sort(p_bad[y == 1])
    good = np.sort(p_bad[y == 0])
    cdf_bad = np.searchsorted(bad, values, side="right") / len(bad)
    cdf_good = np.searchsorted(good, values, side="right") / len(good)
    return float(np.abs(cdf_bad - cdf_good).max())


def psi_from_shares(expected: np.ndarray, actual: np.ndarray) -> float:
    e = np.clip(expected, 1e-4, None)
    a = np.clip(actual, 1e-4, None)
    return float(np.sum((a - e) * np.log(a / e)))


def psi(expected: np.ndarray, actual: np.ndarray, n_bins: int = 10) -> float:
    """Population Stability Index with bins taken from the expected sample's deciles."""
    cuts = np.unique(np.quantile(expected, np.linspace(0, 1, n_bins + 1)[1:-1]))
    size = len(cuts) + 1
    e = np.bincount(np.searchsorted(cuts, expected, side="right"), minlength=size)
    a = np.bincount(np.searchsorted(cuts, actual, side="right"), minlength=size)
    return psi_from_shares(e / e.sum(), a / a.sum())


def csi(expected_bins: np.ndarray, actual_bins: np.ndarray, n_bins: int) -> float:
    """Characteristic Stability Index over a variable's own bins (-1 = unmatched)."""
    e = np.bincount(expected_bins + 1, minlength=n_bins + 1)
    a = np.bincount(actual_bins + 1, minlength=n_bins + 1)
    return psi_from_shares(e / e.sum(), a / a.sum())
