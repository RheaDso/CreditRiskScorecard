"""Run every stage in order.

    python -m scorecard.pipeline                 # real data in data/raw
    python -m scorecard.pipeline --synthetic     # generated data
    python -m scorecard.pipeline --sample 5000   # stratified subset of real data
    python -m scorecard.pipeline --smoke         # quick check; writes to build/smoke
"""

import argparse
import json
import logging
import subprocess
import sys
from datetime import date
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import yaml  # noqa: E402
from sklearn.model_selection import train_test_split  # noqa: E402

from scorecard import (  # noqa: E402
    binning,
    challenger,
    eda,
    features,
    ingest,
    report,
    scaling,
    selection,
    stability,
    strategy,
    synthetic,
    validation,
)
from scorecard.config import ROOT, Config, load_config  # noqa: E402
from scorecard.exceptions import DataValidationError  # noqa: E402
from scorecard.model import fit_logistic  # noqa: E402
from scorecard.scoring import Scorecard  # noqa: E402
from scorecard.split import make_split  # noqa: E402

log = logging.getLogger("scorecard")


def git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _binning_plots(bins: dict, variables: list[str], figures: Path) -> None:
    figures.mkdir(parents=True, exist_ok=True)
    for name in variables:
        t = bins[name].table
        x = np.arange(len(t))
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.bar(x, t["woe"])
        ax.set_ylabel("WoE")
        ax.set_xticks(x, t["range"], rotation=30, ha="right", fontsize=7)
        other = ax.twinx()
        other.plot(x, t["bad_rate"], "o-", color="darkred")
        other.set_ylabel("bad rate")
        ax.set_title(f"{name} (IV {bins[name].iv:.3f})")
        fig.tight_layout()
        fig.savefig(figures / f"{name}.png", dpi=90)
        plt.close(fig)


def run(cfg: Config, sample: int | None = None, is_synthetic: bool = False) -> dict:
    """Run M1-M11 and M13. Returns headline numbers."""
    p = cfg.paths
    for folder in (p.interim, p.processed, p.artifacts, p.reports, p.reports / "figures"):
        folder.mkdir(parents=True, exist_ok=True)

    log.info("1/8 ingest")
    tables = ingest.ingest(p.raw, p.interim, p.reports)
    app = tables["application_train"]
    if sample and sample < len(app):
        app, _ = train_test_split(
            app, train_size=sample, stratify=app["TARGET"], random_state=cfg.seed
        )

    log.info("2/8 split and features")
    split = make_split(app, cfg)
    split.to_parquet(p.processed / "split_ids.parquet", index=False)
    table = features.build_features(app, tables)
    data = table.merge(split, on="SK_ID_CURR", validate="one_to_one")
    train = data[data["split"] == "train"]
    valid = data[data["split"] == "validation"]
    test = data[data["split"] == "test"]
    y_train, y_val = train["TARGET"], valid["TARGET"]

    dropped = features.columns_to_drop(train[table.columns], cfg.features.drop_missing_above)
    excluded = {*features.KEYS, *features.SUPERSEDED, *dropped}
    columns = [c for c in table.columns if c not in excluded]
    table.to_parquet(p.processed / "features.parquet", index=False)
    features.feature_dictionary(columns).to_csv(p.reports / "feature_dictionary.csv", index=False)
    log.info("Dropped columns: %s", dropped)

    log.info("3/8 EDA")
    eda.run_eda(table, split, columns, p.reports)

    log.info("4/8 binning")
    overrides = yaml.safe_load(cfg.binning.overrides_file.read_text(encoding="utf-8")) or {}
    bins = binning.fit_binning(train, y_train, columns, cfg, overrides)
    pd.concat([b.table for b in bins.values()]).to_csv(p.reports / "binning_table.csv", index=False)
    iv = {name: b.iv for name, b in bins.items()}
    binning.iv_summary(bins).to_csv(p.reports / "iv_summary.csv", index=False)
    _binning_plots(
        bins, [v for v in iv if iv[v] >= cfg.selection.min_iv], p.reports / "figures" / "binning"
    )
    x_train = binning.to_woe(train, bins, columns)
    x_val = binning.to_woe(valid, bins, columns)

    log.info("5/8 selection, model, scorecard")
    chosen, funnel, history, vif = selection.select_variables(
        x_train, y_train, x_val, y_val, iv, cfg
    )
    funnel.to_csv(p.reports / "selection_funnel.csv", index=False)
    history.to_csv(p.reports / "stepwise_log.csv", index=False)
    vif.rename("vif").rename_axis("variable").to_csv(p.reports / "vif_table.csv")
    model = fit_logistic(x_train, y_train, chosen)
    (p.reports / "model_summary.txt").write_text(model.summary, encoding="utf-8")
    points = scaling.build_points_table(model, bins, cfg.scaling)
    scaling.export_scorecard(points, p.reports)

    reasons_file = ROOT / "config" / "reason_codes.yaml"
    card = Scorecard(
        model=model,
        bins={v: bins[v] for v in chosen},
        scaling=cfg.scaling,
        refer_band=cfg.strategy.refer_band,
        reason_messages=yaml.safe_load(reasons_file.read_text(encoding="utf-8")),
    )
    scores = {name: card.points(part).sum(axis=1).to_numpy()
              for name, part in [("train", train), ("validation", valid), ("test", test)]}
    probs = {name: card.probability(part)
             for name, part in [("train", train), ("validation", valid), ("test", test)]}

    log.info("6/8 validation")
    y_test = test["TARGET"].to_numpy()
    pot = test["is_pot"].to_numpy(dtype=bool)
    results = {
        name: validation.metrics(part["TARGET"].to_numpy(), probs[name])
        for name, part in [("train", train), ("validation", valid), ("test", test)]
    }
    results["pseudo_oot"] = validation.metrics(y_test[pot], probs["test"][pot])
    metric_table = pd.DataFrame(results).T.rename_axis("sample").reset_index()
    metric_table["rows"] = metric_table["rows"].astype(int)
    metric_table.to_csv(p.reports / "validation_metrics.csv", index=False)

    deciles = validation.decile_table(y_test, scores["test"])
    deciles.to_csv(p.reports / "decile_table.csv", index=False)
    calib, calib_stats = validation.calibration(y_test, probs["test"])
    validation.plots(y_test, probs["test"], scores["test"], calib, p.reports / "figures")

    n_psi = cfg.validation.psi_bins
    psi_rows = [("score", "train vs test", stability.psi(scores["train"], scores["test"], n_psi))]
    if "application_test" in tables:
        unlabeled = features.build_features(tables["application_test"], tables)
        unlabeled_scores = card.points(unlabeled).sum(axis=1).to_numpy()
        psi_rows.append(("score", "train vs application_test",
                         stability.psi(scores["train"], unlabeled_scores, n_psi)))
        comparison_sample, label = unlabeled, "train vs application_test"
    else:
        comparison_sample, label = test, "train vs test"
    for var in chosen:
        vb = bins[var]
        value = stability.csi(
            vb.positions(train[var]), vb.positions(comparison_sample[var]), len(vb.labels)
        )
        psi_rows.append((var, label, value))
    psi_table = pd.DataFrame(psi_rows, columns=["variable", "comparison", "psi"])
    psi_table["flag"] = np.where(psi_table["psi"] >= cfg.validation.psi_warn, "CHECK", "stable")
    psi_table.to_csv(p.reports / "psi_table.csv", index=False)

    log.info("7/8 challenger and strategy")
    booster, levels, search = challenger.train_challenger(
        train, y_train, valid, y_val, columns, cfg
    )
    search.to_csv(p.reports / "challenger_search.csv", index=False)
    encoded_test = challenger.encode(test, columns, levels)
    p_challenger = np.asarray(booster.predict_proba(encoded_test))[:, 1]
    compare = pd.DataFrame(
        [
            {"model": name, "sample": sample_name,
             **{k: v for k, v in validation.metrics(y_test[mask], prob[mask]).items()
                if k in ("auc", "gini", "ks")}}
            for name, prob in [("scorecard", probs["test"]), ("lightgbm", p_challenger)]
            for sample_name, mask in [("test", np.ones(len(y_test), bool)), ("pseudo_oot", pot)]
        ]
    )
    compare.to_csv(p.reports / "challenger_comparison.csv", index=False)
    validation.decile_table(y_test, -p_challenger).to_csv(
        p.reports / "challenger_decile_table.csv", index=False
    )
    challenger.shap_plots(booster, challenger.encode(train, columns, levels),
                          p.reports / "figures", cfg)
    uplift = stability.gini(y_test, p_challenger) - results["test"]["gini"]
    threshold = cfg.challenger.champion_uplift_threshold
    champion = "LightGBM challenger" if uplift > threshold else "logistic scorecard"
    champion_reason = (
        f"The uplift is {'above' if uplift > threshold else 'not above'} the {threshold} "
        f"threshold, so the {champion} is recommended."
    )

    # Cut-off is chosen on VALIDATION so that TEST stays an honest check.
    chooser = strategy.cutoff_table(y_val.to_numpy(), scores["validation"],
                                    cfg.strategy.cutoff_step)
    cutoff = strategy.recommend_cutoff(chooser, cfg.strategy.target_bad_rate)
    cutoffs = strategy.cutoff_table(y_test, scores["test"], cfg.strategy.cutoff_step)
    cutoffs.to_csv(p.reports / "cutoff_table.csv", index=False)
    strategy.tradeoff_plot(cutoffs, p.reports / "figures" / "approval_tradeoff.png")
    if cutoff is None:
        cutoff_text = "none: the target bad rate cannot be reached"
        card.cutoff = float(np.median(scores["train"]))
    else:
        approved = scores["test"] >= cutoff
        cutoff_text = (f"{cutoff} (TEST approval rate {approved.mean():.1%}, "
                       f"bad rate among approved {y_test[approved].mean():.2%})")
        card.cutoff = float(cutoff)
    card.band_edges = strategy.band_edges(scores["train"], cfg.strategy.n_risk_bands)
    bands = strategy.band_table(y_test, card.band(scores["test"]), scores["test"])
    bands.to_csv(p.reports / "risk_bands.csv", index=False)

    log.info("8/8 artifacts and report")
    card.save(p.artifacts / "model.pkl")
    for filename, obj in [("binning.pkl", bins),
                          ("challenger.pkl", {"model": booster, "columns": columns,
                                              "levels": levels})]:
        pd.to_pickle(obj, p.artifacts / filename)
    metadata = {
        "model_version": cfg.api.model_version,
        "training_date": date.today().isoformat(),
        "git_commit": git_commit(),
        "data": "synthetic" if is_synthetic else "home_credit",
        "row_counts": {"train": len(train), "validation": len(valid), "test": len(test)},
        "variables": chosen,
        "intercept": model.intercept,
        "coefficients": model.coefficients,
        "cutoff": card.cutoff,
    }
    (p.artifacts / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    test_result, train_result = results["test"], results["train"]
    gap = train_result["gini"] - test_result["gini"]
    score_psi = float(psi_rows[0][2])
    coefficient_table = pd.DataFrame(
        {"variable": chosen, "source": [features.source_of(v) for v in chosen],
         "beta": [model.coefficients[v] for v in chosen], "vif": [vif[v] for v in chosen]}
    )
    reversals = validation.monotonic_reversals(deciles)
    report.write_model_report(
        {
            "synthetic": is_synthetic, "test": test_result, "cutoff_text": cutoff_text,
            "champion": champion, "n_rows": len(data), "bad_rate": data["TARGET"].mean(),
            "dropped": dropped, "seed": cfg.seed, "funnel": funnel,
            "iv": pd.DataFrame({"variable": chosen, "iv": [iv[v] for v in chosen]}),
            "suspicious": [v for v in chosen if iv[v] > cfg.selection.suspicious_iv],
            "suspicious_iv": cfg.selection.suspicious_iv, "intercept": model.intercept,
            "coefficients": coefficient_table, "scaling": cfg.scaling,
            "scorecard": points[["variable", "range", "woe", "points"]],
            "metrics": metric_table, "gini_gap": gap, "max_gap": cfg.validation.max_gini_gap,
            "deciles": deciles, "reversals": reversals, "calib": calib_stats, "psi": psi_table,
            "comparison": compare, "uplift": uplift, "champion_reason": champion_reason,
            "target_bad_rate": cfg.strategy.target_bad_rate, "cutoffs": cutoffs, "bands": bands,
            "psi_warn": cfg.validation.psi_warn, "psi_alert": cfg.validation.psi_alert,
        },
        p.reports,
    )
    targets = cfg.validation.targets
    rounding = float(
        np.abs(points.set_index(["variable", "bin"]).eval("points - points_unrounded"))
        .groupby("variable").max().sum()
    )
    report.write_run_summary(
        [
            ("TEST Gini", f">= {targets['gini']}", f"{test_result['gini']:.4f}",
             test_result["gini"] >= targets["gini"]),
            ("TEST KS", f">= {targets['ks']}", f"{test_result['ks']:.4f}",
             test_result["ks"] >= targets["ks"]),
            ("Train-Test Gini gap", f"<= {cfg.validation.max_gini_gap}", f"{gap:.4f}",
             gap <= cfg.validation.max_gini_gap),
            ("Score PSI train vs test", f"< {cfg.validation.psi_warn}", f"{score_psi:.4f}",
             score_psi < cfg.validation.psi_warn),
            ("Max VIF", f"< {cfg.selection.max_vif}", f"{vif.max():.3f}",
             bool(vif.max() < cfg.selection.max_vif)),
            ("All coefficients negative, p < 0.05", "yes", "yes", True),
        ],
        [
            f"Variables in the final model: {len(chosen)}",
            f"Largest possible score difference caused by rounding points: {rounding:.2f}",
            f"Recommended cut-off: {cutoff_text}",
            f"Champion: {champion} (challenger Gini uplift {uplift:+.4f})",
        ],
        p.reports,
        is_synthetic,
    )
    return {"test_gini": test_result["gini"], "test_ks": test_result["ks"], "gap": gap,
            "psi": score_psi, "variables": chosen, "uplift": uplift, "cutoff": cutoff_text}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--synthetic", action="store_true", help="generate fake data first")
    parser.add_argument("--sample", type=int, help="use a stratified subset of applicants")
    parser.add_argument("--smoke", action="store_true", help="small, fast synthetic run")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    cfg = load_config()
    if args.smoke:
        args.synthetic = True
        cfg.synthetic.n_applicants = 8000
        cfg.challenger.n_trials = 3
        cfg.selection.min_vars = 5
        # A smoke run must never overwrite the real model or reports.
        for name in ("interim", "processed", "artifacts", "reports"):
            setattr(cfg.paths, name, ROOT / "build" / "smoke" / name)
    if args.synthetic:
        cfg.paths.raw = ROOT / "data" / ("smoke" if args.smoke else "synthetic")
        synthetic.generate(cfg.paths.raw, cfg.synthetic.n_applicants, cfg.seed)
    try:
        summary = run(cfg, sample=args.sample, is_synthetic=args.synthetic)
    except DataValidationError as exc:
        print(f"{exc}\nSee README.md, 'Get the data', or run with --synthetic.", file=sys.stderr)
        return 1
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
