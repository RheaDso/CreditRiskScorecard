"""M4: exploratory analysis on TRAIN. Every sentence is generated from computed numbers."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402


def univariate_auc(train: pd.DataFrame, y: pd.Series, columns: list[str]) -> pd.Series:
    """AUC of each numeric variable on its own (direction ignored), strongest first."""
    scores = {}
    for col in columns:
        x = train[col]
        if not pd.api.types.is_numeric_dtype(x) or x.nunique() < 2:
            continue
        auc = roc_auc_score(y, x.fillna(x.median()))
        scores[col] = max(auc, 1 - auc)
    return pd.Series(scores).sort_values(ascending=False)


def _decile_plot(x: pd.Series, y: pd.Series, path: Path) -> None:
    groups = pd.qcut(x, 10, duplicates="drop")
    rate = y.groupby(groups, observed=True).mean()
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(range(len(rate)), rate.to_numpy())
    ax.set(title=str(x.name), xlabel="decile (low to high)", ylabel="bad rate")
    fig.tight_layout()
    fig.savefig(path, dpi=90)
    plt.close(fig)


def run_eda(
    features: pd.DataFrame, split: pd.DataFrame, columns: list[str], reports_dir: Path
) -> None:
    figures = reports_dir / "figures" / "eda"
    figures.mkdir(parents=True, exist_ok=True)
    data = features.merge(split, on="SK_ID_CURR")
    train = data[data["split"] == "train"]
    y = train["TARGET"]

    by_split = data.groupby("split")["TARGET"].agg(["size", "sum", "mean"])
    auc = univariate_auc(train, y, columns)
    top = auc.head(25)
    for name in top.index:
        _decile_plot(train[name], y, figures / f"{name}.png")

    corr = train[list(top.index)].corr()
    fig, ax = plt.subplots(figsize=(11, 9))
    image = ax.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_xticks(range(len(corr)), corr.columns, rotation=90, fontsize=7)
    ax.set_yticks(range(len(corr)), corr.columns, fontsize=7)
    fig.colorbar(image)
    fig.tight_layout()
    fig.savefig(figures / "correlation_heatmap.png", dpi=90)
    plt.close(fig)

    missing = train[columns].isna().mean().sort_values(ascending=False)
    observations = [
        f"The data has {len(data):,} applicants with an overall bad rate of "
        f"{data['TARGET'].mean():.2%}.",
        f"{auc.index[0]} has the strongest single-variable AUC of {auc.iloc[0]:.3f}.",
        f"The next strongest are {auc.index[1]} ({auc.iloc[1]:.3f}) and "
        f"{auc.index[2]} ({auc.iloc[2]:.3f}).",
        f"{int((auc > 0.55).sum())} of {len(auc)} numeric variables have AUC above 0.55.",
        f"The most often missing variable is {missing.index[0]} ({missing.iloc[0]:.1%} missing).",
        f"{int((missing > 0).sum())} variables have at least one missing value.",
    ]
    table = "\n".join(
        f"| {name} | {int(r['size']):,} | {int(r['sum']):,} | {r['mean']:.2%} |"
        for name, r in by_split.iterrows()
    )
    text = (
        "# Exploratory data analysis\n\n## Target distribution\n\n"
        "| split | rows | bads | bad rate |\n|---|---|---|---|\n"
        f"{table}\n\n## Observations (TRAIN)\n\n"
        + "\n".join(f"- {line}" for line in observations)
        + "\n"
    )
    (reports_dir / "eda_summary.md").write_text(text, encoding="utf-8")
