"""M10: a LightGBM model on raw features, to see what the simple scorecard gives up."""

from pathlib import Path
from typing import Any

import lightgbm as lgb
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import shap  # noqa: E402

from scorecard.config import Config  # noqa: E402
from scorecard.stability import gini  # noqa: E402


def encode(df: pd.DataFrame, columns: list[str], levels: dict[str, list]) -> pd.DataFrame:
    """Categories become integer codes using TRAIN levels (unseen -> -1)."""
    out = df[columns].copy()
    for col, values in levels.items():
        out[col] = pd.Categorical(out[col], categories=values).codes
    return out.astype(float)


def train_challenger(
    train: pd.DataFrame, y_train: pd.Series, validation: pd.DataFrame, y_val: pd.Series,
    columns: list[str], cfg: Config,
) -> tuple[lgb.LGBMClassifier, dict[str, list], pd.DataFrame]:
    """Random search judged on VALIDATION Gini only; TEST is never used."""
    levels = {
        c: sorted(train[c].dropna().astype(str).unique())
        for c in columns
        if not pd.api.types.is_numeric_dtype(train[c])
    }
    x_tr, x_val = encode(train, columns, levels), encode(validation, columns, levels)
    rng = np.random.default_rng(cfg.seed)
    best, best_gini, log = None, -1.0, []
    for trial in range(cfg.challenger.n_trials):
        params: dict[str, Any] = {
            "num_leaves": int(rng.choice([7, 15, 31])),
            "learning_rate": float(rng.choice([0.01, 0.03, 0.05])),
            "min_child_samples": int(rng.choice([50, 100, 200, 400])),
            "subsample": float(rng.choice([0.7, 0.9, 1.0])),
            "colsample_bytree": float(rng.choice([0.6, 0.8, 1.0])),
            "reg_lambda": float(rng.choice([0.0, 1.0, 10.0])),
        }
        model = lgb.LGBMClassifier(
            n_estimators=2000, subsample_freq=1, random_state=cfg.seed,
            verbosity=-1, deterministic=True, force_row_wise=True, **params,
        )
        model.fit(
            x_tr, y_train, eval_set=[(x_val, y_val)], eval_metric="auc",
            callbacks=[lgb.early_stopping(50, verbose=False)],
        )
        score = gini(y_val, np.asarray(model.predict_proba(x_val))[:, 1])
        log.append({"trial": trial, "validation_gini": score,
                    "trees": model.best_iteration_, **params})
        if score > best_gini:
            best, best_gini = model, score
    assert best is not None
    return best, levels, pd.DataFrame(log)


def shap_plots(
    model: lgb.LGBMClassifier, x_train: pd.DataFrame, figures: Path, cfg: Config
) -> None:
    figures.mkdir(parents=True, exist_ok=True)
    sample = x_train.sample(min(cfg.challenger.shap_background, len(x_train)),
                            random_state=cfg.seed)
    values = shap.TreeExplainer(model).shap_values(sample)
    if isinstance(values, list):
        values = values[1]
    for kind, name in [("bar", "shap_importance.png"), ("dot", "shap_beeswarm.png")]:
        np.random.seed(cfg.seed)
        shap.summary_plot(values, sample, plot_type=kind, max_display=20, show=False)
        plt.tight_layout()
        plt.savefig(figures / name, dpi=100)
        plt.close("all")
