"""One small synthetic pipeline run shared by all tests."""

import pytest

from scorecard import pipeline, synthetic
from scorecard.config import load_config
from scorecard.scoring import Scorecard


@pytest.fixture(scope="session")
def cfg(tmp_path_factory):
    root = tmp_path_factory.mktemp("run")
    config = load_config(root=root)
    config.challenger.n_trials = 2
    config.challenger.shap_background = 200
    config.selection.min_vars = 5  # small samples support fewer variables
    synthetic.generate(config.paths.raw, n=20000, seed=config.seed)
    return config


@pytest.fixture(scope="session")
def summary(cfg):
    return pipeline.run(cfg, is_synthetic=True)


@pytest.fixture(scope="session")
def card(cfg, summary):
    return Scorecard.load(cfg.paths.artifacts / "model.pkl")
