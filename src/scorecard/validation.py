"""M9: how well the model ranks, how well calibrated it is, and how stable it is."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import chi2  # noqa: E402
from sklearn.metrics import brier_score_loss, roc_auc_score, roc_curve  # noqa: E402

from scorecard.stability import gini, ks_statistic  # noqa: E402


def metrics(y: np.ndarray, p_bad: np.ndarray) -> dict[str, float]:
    return {
        "rows": len(y),
        "bad_rate": float(np.mean(y)),
        "auc": float(roc_auc_score(y, p_bad)),
        "gini": gini(y, p_bad),
        "ks": ks_statistic(y, p_bad),
    }


def decile_table(y: np.ndarray, score: np.ndarray) -> pd.DataFrame:
    """Decile 1 holds the lowest scores (riskiest applicants)."""
    df = pd.DataFrame({"y": y, "score": score})
    df["decile"] = pd.qcut(df["score"].rank(method="first"), 10, labels=range(1, 11)).astype(int)
    t = df.groupby("decile").agg(
        min_score=("score", "min"), max_score=("score", "max"),
        count=("y", "size"), bads=("y", "sum"),
    )
    t["bad_rate"] = t["bads"] / t["count"]
    t["cum_pct_bads"] = t["bads"].cumsum() / t["bads"].sum()
    goods = t["count"] - t["bads"]
    t["cum_pct_goods"] = goods.cumsum() / goods.sum()
    t["ks"] = (t["cum_pct_bads"] - t["cum_pct_goods"]).abs()
    t["lift"] = t["bad_rate"] / df["y"].mean()
    return t.reset_index()


def monotonic_reversals(table: pd.DataFrame) -> int:
    """Times the bad rate goes UP as the score decile improves."""
    return int((np.diff(table["bad_rate"].to_numpy()) > 0).sum())


def calibration(y: np.ndarray, p_bad: np.ndarray) -> tuple[pd.DataFrame, dict[str, float]]:
    df = pd.DataFrame({"y": y, "p": p_bad})
    df["group"] = pd.qcut(df["p"].rank(method="first"), 10, labels=False)
    t = df.groupby("group").agg(
        count=("y", "size"), predicted=("p", "mean"), actual=("y", "mean")
    )
    expected = t["count"] * t["predicted"]
    observed = t["count"] * t["actual"]
    hl = float((((observed - expected) ** 2) / (expected * (1 - t["predicted"]))).sum())
    stats = {
        "hosmer_lemeshow": hl,
        "hl_pvalue": float(chi2.sf(hl, len(t) - 2)),
        "brier": float(brier_score_loss(y, p_bad)),
    }
    return t.reset_index(), stats


def plots(
    y: np.ndarray, p_bad: np.ndarray, score: np.ndarray, calib: pd.DataFrame, figures: Path
) -> None:
    figures.mkdir(parents=True, exist_ok=True)

    def save(fig: plt.Figure, name: str) -> None:
        fig.tight_layout()
        fig.savefig(figures / name, dpi=100)
        plt.close(fig)

    fpr, tpr, _ = roc_curve(y, p_bad)
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot(fpr, tpr, label=f"Gini {gini(y, p_bad):.3f}")
    ax.plot([0, 1], [0, 1], "--", color="grey")
    ax.set(title="ROC (TEST)", xlabel="false positive rate", ylabel="true positive rate")
    ax.legend()
    save(fig, "roc.png")

    order = np.argsort(score)
    cum_bad = np.cumsum(y[order]) / y.sum()
    cum_good = np.cumsum(1 - y[order]) / (1 - y).sum()
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(score[order], cum_bad, label="cumulative bads")
    ax.plot(score[order], cum_good, label="cumulative goods")
    ax.set(title=f"KS curve (TEST), KS {ks_statistic(y, p_bad):.3f}", xlabel="score")
    ax.legend()
    save(fig, "ks.png")

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(score[y == 0], bins=40, density=True, alpha=0.6, label="good")
    ax.hist(score[y == 1], bins=40, density=True, alpha=0.6, label="bad")
    ax.set(title="Score distribution (TEST)", xlabel="score")
    ax.legend()
    save(fig, "score_distribution.png")

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot(calib["predicted"], calib["actual"], "o-")
    top = max(calib["predicted"].max(), calib["actual"].max())
    ax.plot([0, top], [0, top], "--", color="grey")
    ax.set(title="Calibration (TEST)", xlabel="predicted PD", ylabel="actual bad rate")
    save(fig, "calibration.png")
