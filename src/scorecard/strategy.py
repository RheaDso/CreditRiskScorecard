"""M11: choose a cut-off score and split applicants into risk bands."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402


def cutoff_table(y: np.ndarray, score: np.ndarray, step: int) -> pd.DataFrame:
    """For each cut-off: approve everyone scoring at or above it."""
    start = int(np.floor(score.min() / step) * step)
    rows = []
    for cutoff in range(start, int(score.max()) + 1, step):
        approved = score >= cutoff
        if not approved.any():
            continue
        rows.append(
            {
                "cutoff": cutoff,
                "approval_rate": approved.mean(),
                "bad_rate_approved": y[approved].mean(),
                "bads_avoided": int(y[~approved].sum()),
                "pct_bads_avoided": y[~approved].sum() / y.sum(),
            }
        )
    return pd.DataFrame(rows)


def recommend_cutoff(table: pd.DataFrame, target_bad_rate: float) -> int | None:
    """Lowest cut-off meeting the bad-rate target, i.e. the most approvals."""
    ok = table[table["bad_rate_approved"] <= target_bad_rate]
    return None if ok.empty else int(ok["cutoff"].min())


def band_edges(train_scores: np.ndarray, n_bands: int) -> list[float]:
    """Score boundaries giving roughly equal TRAIN population per band."""
    return [float(q) for q in np.quantile(train_scores, np.linspace(0, 1, n_bands + 1)[1:-1])]


def band_table(y: np.ndarray, bands: np.ndarray, score: np.ndarray) -> pd.DataFrame:
    df = pd.DataFrame({"band": bands, "y": y, "score": score})
    t = df.groupby("band").agg(
        min_score=("score", "min"), max_score=("score", "max"),
        count=("y", "size"), bad_rate=("y", "mean"),
    )
    return t.reset_index()


def tradeoff_plot(table: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(table["approval_rate"], table["bad_rate_approved"], "o-", markersize=3)
    ax.set(title="Approval rate vs bad rate", xlabel="approval rate",
           ylabel="bad rate among approved")
    fig.tight_layout()
    fig.savefig(path, dpi=100)
    plt.close(fig)
