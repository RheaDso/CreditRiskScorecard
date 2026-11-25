import numpy as np
import pandas as pd
import pytest

from scorecard import features, ingest, synthetic
from scorecard.config import load_config
from scorecard.exceptions import DataValidationError
from scorecard.split import make_split


@pytest.fixture(scope="module")
def raw(tmp_path_factory):
    folder = tmp_path_factory.mktemp("raw")
    synthetic.generate(folder, n=2000, seed=1)
    return folder


@pytest.fixture(scope="module")
def tables(raw):
    t = {name: ingest.read_table(raw, name) for name in ingest.REQUIRED}
    t["application_train"] = ingest.fix_application(t["application_train"])
    return t


def test_fr_1_2_duplicate_id_fails(raw):
    app = ingest.read_table(raw, "application_train")
    ingest.validate_application(app)
    with pytest.raises(DataValidationError):
        ingest.validate_application(pd.concat([app, app.head(1)]))


def test_fr_1_3_anomalies_fixed():
    df = pd.DataFrame({"DAYS_EMPLOYED": [365243, -100], "CODE_GENDER": ["XNA", "F"]})
    out = ingest.fix_application(df)
    assert out["FLAG_DAYS_EMPLOYED_ANOM"].tolist() == [1, 0]
    assert np.isnan(out["DAYS_EMPLOYED"][0]) and pd.isna(out["CODE_GENDER"][0])


def test_fr_1_5_missing_column_and_file(tmp_path):
    with pytest.raises(DataValidationError, match="Missing file"):
        ingest.read_table(tmp_path, "bureau")
    (tmp_path / "bureau.csv").write_text("SK_ID_CURR\n1\n")
    with pytest.raises(DataValidationError, match="missing columns"):
        ingest.read_table(tmp_path, "bureau")


def test_fr_2_split_sizes_stratification_determinism(tables):
    cfg = load_config()
    app = tables["application_train"]
    split = make_split(app, cfg)
    assert split["SK_ID_CURR"].is_unique and len(split) == len(app)
    share = split["split"].value_counts(normalize=True)
    assert share["train"] == pytest.approx(0.6, abs=0.005)
    rates = split.merge(app).groupby("split")["TARGET"].mean()
    assert (rates - app["TARGET"].mean()).abs().max() < 0.003
    assert split.equals(make_split(app, cfg))
    assert split.loc[split["is_pot"], "split"].eq("test").all()


def test_fr_3_1_safe_division():
    out = features.safe_divide(pd.Series([1.0, 1.0]), pd.Series([0.0, np.nan]))
    assert out.isna().all()


def test_fr_3_2_3_4_features(tables):
    app = tables["application_train"].copy()
    loner = app.iloc[[0]].assign(SK_ID_CURR=1)  # applicant with no history
    app = pd.concat([app, loner], ignore_index=True)
    out = features.build_features(app, tables)
    assert len(out) == len(app) and out["SK_ID_CURR"].is_unique
    row = out[out["SK_ID_CURR"] == 1].iloc[0]
    assert row["BUR_COUNT"] == 0 and row["PREV_COUNT"] == 0
    assert row["HAS_BUREAU"] == 0 and row["HAS_PREV"] == 0
    assert np.isnan(row["INST_LATE_PAYMENT_RATE"])
    assert not np.isinf(out.select_dtypes("number").to_numpy()).any()
