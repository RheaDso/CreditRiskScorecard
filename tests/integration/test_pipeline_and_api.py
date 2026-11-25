import json
import time

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from scorecard.api.main import create_app
from scorecard.scaling import score_from_odds


@pytest.fixture(scope="module")
def client(cfg, summary):
    return TestClient(create_app(cfg.paths.artifacts))


@pytest.fixture(scope="module")
def sample(cfg, card, summary):
    table = pd.read_parquet(cfg.paths.processed / "features.parquet").head(100)
    # The API takes raw inputs, so rebuild them from the validated application table.
    raw = pd.read_parquet(cfg.paths.interim / "application_train.parquet")
    merged = table.set_index("SK_ID_CURR")
    for column in card.required_inputs():
        if column not in merged:
            merged[column] = raw.set_index("SK_ID_CURR")[column]
    return merged


def payload(card, row):
    return {c: (None if pd.isna(row[c]) else row[c]) for c in card.required_inputs()}


def clean(record):
    return {k: (v.item() if isinstance(v, np.generic) else v) for k, v in record.items()}


def value_in_bin(vb, position):
    """A raw value that falls in the given bin."""
    if vb.labels[position] == "Missing":
        return np.nan
    if vb.categorical:
        return next(name for name, index in vb.groups.items() if index == position)
    return vb.splits[0] - 1 if position == 0 else vb.splits[position - 1]


def test_outputs_exist(cfg, summary):
    reports = ["data_quality.csv", "feature_dictionary.csv", "eda_summary.md",
               "binning_table.csv", "selection_funnel.csv", "model_summary.txt",
               "scorecard.csv", "scorecard.xlsx", "validation_metrics.csv", "decile_table.csv",
               "psi_table.csv", "challenger_comparison.csv", "cutoff_table.csv",
               "risk_bands.csv", "model_report.md", "model_report.html", "RUN_SUMMARY.md",
               "figures/shap_importance.png", "figures/shap_beeswarm.png",
               "figures/calibration.png"]
    assert not [f for f in reports if not (cfg.paths.reports / f).exists()]
    for f in ["binning.pkl", "model.pkl", "challenger.pkl", "metadata.json"]:
        assert (cfg.paths.artifacts / f).exists()


def test_fr_6_5_all_coefficients_negative(card):
    assert all(beta < 0 for beta in card.model.coefficients.values())


def test_fr_8_3_unrounded_points_equal_model_score(card, sample):
    features = card.prepare([clean(r) for r in sample.reset_index().to_dict("records")])
    p = card.probability(features)
    from_probability = [score_from_odds((1 - v) / v, card.scaling) for v in p]
    rounded = card.points(features).sum(axis=1).to_numpy()
    assert np.abs(rounded - from_probability).max() <= len(card.model.variables) * 0.5


def test_fr_11_3_bands_ordered(cfg, summary):
    bands = pd.read_csv(cfg.paths.reports / "risk_bands.csv").sort_values("band")
    assert bands["bad_rate"].is_monotonic_increasing


def test_fr_12_1_health(client):
    body = client.get("/health").json()
    assert body["status"] == "ok" and body["training_date"]


def test_fr_8_6_12_3_api_score_equals_batch_score(client, card, sample):
    expected = card.points(sample).sum(axis=1).tolist()
    got = [client.post("/score", json=clean(payload(card, row))).json()["score"]
           for _, row in sample.iterrows()]
    assert got == expected


def test_fr_12_2_validation_errors(client, card, sample):
    good = clean(payload(card, sample.iloc[0]))
    name = next(iter(good))
    missing = {k: v for k, v in good.items() if k != name}
    assert client.post("/score", json=missing).status_code == 422
    wrong = client.post("/score", json={**good, name: "not a number"})
    assert wrong.status_code == 422 and name in json.dumps(wrong.json())


def test_fr_12_4_reason_code_for_worst_attribute(card):
    var = card.model.variables[0]
    best = {v: int(np.argmax(card._points_for(v)[:-1])) for v in card.model.variables}
    frame = {}
    for v in card.model.variables:
        position = int(np.argmin(card._points_for(v)[:-1])) if v == var else best[v]
        frame[v] = [value_in_bin(card.bins[v], position)]
    result = card.score_features(pd.DataFrame(frame))[0]
    assert [r["variable"] for r in result.reason_codes] == [var]


def test_fr_12_6_batch_of_1000(client, card, sample):
    record = clean(payload(card, sample.iloc[0]))
    start = time.perf_counter()
    response = client.post("/score/batch", json={"applicants": [record] * 1000})
    assert response.status_code == 200 and len(response.json()) == 1000
    assert time.perf_counter() - start < 5
    too_many = client.post("/score/batch", json={"applicants": [record] * 1001})
    assert too_many.status_code == 422
