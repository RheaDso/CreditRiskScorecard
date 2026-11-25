"""Binning, selection, scaling and metric formulas checked against hand calculations."""

import math

import numpy as np
import pandas as pd
import pytest
from optbinning import OptimalBinning
from sklearn.linear_model import LogisticRegression

from scorecard import binning, scaling, selection, stability, strategy
from scorecard.config import load_config
from scorecard.model import fit_logistic

CFG = load_config()


def toy(n=4000, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.uniform(0, 1, n)
    y = (rng.random(n) < 0.3 * (1 - x) + 0.02).astype(int)
    x[:200] = np.nan
    return pd.Series(x, name="X"), pd.Series(y)


def test_fr_5_3_woe_formula():
    # bin 1: 8 good, 2 bad. bin 2: 2 good, 8 bad. Each side has 10 in total.
    woe, iv = binning.woe_iv(np.array([8, 2]), np.array([2, 8]))
    assert woe == pytest.approx([math.log(4), -math.log(4)])
    assert iv.sum() == pytest.approx(2 * 0.6 * math.log(4))


def test_fr_5_3_zero_count_adjustment():
    woe, _ = binning.woe_iv(np.array([10, 5]), np.array([0, 5]))
    assert np.isfinite(woe).all()


def test_fr_5_4_parity_with_optbinning():
    x, y = toy()
    ours = binning.fit_variable(x, y, CFG)
    theirs = OptimalBinning(name="X", user_splits=np.array(ours.splits),
                            user_splits_fixed=[True] * len(ours.splits),
                            monotonic_trend=None)
    theirs.fit(x.to_numpy(), y.to_numpy())
    table = theirs.binning_table.build()
    expected = table.loc[table["Count"] > 0, "WoE"].iloc[: len(ours.splits) + 1]
    assert ours.woe[: len(ours.splits) + 1] == pytest.approx(
        expected.astype(float).tolist(), abs=1e-6
    )


def test_fr_5_2_3_monotonic_min_size_missing_bin():
    x, y = toy()
    vb = binning.fit_variable(x, y, CFG)
    body = vb.table[vb.table["range"] != binning.MISSING]
    assert vb.labels[-1] == binning.MISSING
    assert (body["pct"] >= CFG.binning.min_bin_share - 1e-9).all()
    assert body["bad_rate"].is_monotonic_decreasing


def test_fr_5_5_override():
    x, y = toy()
    vb = binning.fit_variable(x, y, CFG, override=[0.25, 0.75])
    assert vb.splits == [0.25, 0.75]


def test_fr_5_7_transform_uses_train_bins_and_handles_unseen():
    x, y = toy()
    vb = binning.fit_variable(x, y, CFG)
    before = list(vb.woe)
    vb.transform(pd.Series([99.0, -99.0]))
    assert vb.woe == before
    category = pd.Series(["a", "b"] * 500, name="C")
    cb = binning.fit_variable(category, pd.Series([0, 0, 0, 1] * 250), CFG)
    assert cb.transform(pd.Series(["never seen"])).tolist() == [0.0]


def test_fr_6_filters():
    rng = np.random.default_rng(0)
    a = rng.normal(size=3000)
    frame = pd.DataFrame({"a": a, "copy": a + rng.normal(0, 0.1, 3000),
                          "b": rng.normal(size=3000)})
    assert selection.iv_filter({"a": 0.5, "z": 0.001}, 0.02) == ["a"]
    assert selection.correlation_filter(frame, ["a", "copy", "b"], 0.7) == ["a", "b"]
    kept = selection.vif_filter(frame, ["a", "copy", "b"], 5.0)
    assert selection.vif_table(frame, kept).max() < 5


def test_fr_6_5_wrong_sign_removed():
    rng = np.random.default_rng(0)
    frame = pd.DataFrame({"good": rng.normal(size=5000), "wrong": rng.normal(size=5000)})
    logit = -2 - frame["good"] + frame["wrong"]
    y = pd.Series((rng.random(5000) < 1 / (1 + np.exp(-logit))).astype(int))
    assert selection.sign_and_significance(frame, y, ["good", "wrong"], 0.05) == ["good"]


def test_fr_7_2_statsmodels_matches_sklearn():
    rng = np.random.default_rng(0)
    frame = pd.DataFrame(rng.normal(size=(3000, 2)), columns=["a", "b"])
    y = pd.Series((rng.random(3000) < 1 / (1 + np.exp(2 + frame["a"] + frame["b"]))).astype(int))
    ours = fit_logistic(frame, y, ["a", "b"])
    other = LogisticRegression(penalty=None, tol=1e-10, max_iter=1000).fit(frame, y)
    assert list(ours.coefficients.values()) == pytest.approx(other.coef_[0], abs=1e-4)
    assert ours.intercept == pytest.approx(other.intercept_[0], abs=1e-4)


def test_fr_8_1_2_base_score_and_pdo():
    assert scaling.score_from_odds(50, CFG.scaling) == pytest.approx(600)
    assert scaling.score_from_odds(100, CFG.scaling) == pytest.approx(620)


def test_fr_9_1_metrics_on_known_data():
    y = np.array([0, 0, 1, 1])
    p = np.array([0.1, 0.2, 0.8, 0.9])
    assert stability.gini(y, p) == pytest.approx(1.0)
    assert stability.ks_statistic(y, p) == pytest.approx(1.0)


def test_fr_9_4_psi():
    base = np.random.default_rng(0).normal(size=5000)
    assert stability.psi(base, base) == pytest.approx(0.0)
    assert stability.psi(base, base + 1.5) > 0.25


def test_fr_11_cutoff():
    y = np.array([1, 1, 0, 0])
    score = np.array([500, 520, 560, 600])
    table = strategy.cutoff_table(y, score, 5)
    assert table["approval_rate"].is_monotonic_decreasing
    assert strategy.recommend_cutoff(table, 0.05) == 525
    assert strategy.recommend_cutoff(table.head(1), 0.05) is None
