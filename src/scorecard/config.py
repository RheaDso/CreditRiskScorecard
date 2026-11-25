"""Load and validate config/config.yaml. No thresholds are hard-coded elsewhere."""

from pathlib import Path

import yaml
from pydantic import BaseModel

ROOT = Path(__file__).resolve().parents[2]


class Paths(BaseModel):
    raw: Path
    interim: Path
    processed: Path
    artifacts: Path
    reports: Path


class Split(BaseModel):
    train: float
    validation: float
    test: float
    pot_share_of_test: float


class Features(BaseModel):
    drop_missing_above: float
    rare_category_share: float


class Binning(BaseModel):
    fine_bins: int
    max_n_bins: int
    min_bin_share: float
    monotonic_trend: str
    overrides_file: Path


class Selection(BaseModel):
    min_iv: float
    suspicious_iv: float
    max_corr: float
    max_vif: float
    stepwise_min_gini_gain: float
    min_vars: int
    max_vars: int
    max_pvalue: float


class Scaling(BaseModel):
    base_score: float
    base_odds: float
    pdo: float


class Validation(BaseModel):
    psi_bins: int
    psi_warn: float
    psi_alert: float
    max_gini_gap: float
    targets: dict[str, float]


class Challenger(BaseModel):
    n_trials: int
    shap_background: int
    champion_uplift_threshold: float


class Strategy(BaseModel):
    cutoff_step: int
    target_bad_rate: float
    refer_band: int
    n_risk_bands: int


class Api(BaseModel):
    max_batch: int
    model_version: str


class Synthetic(BaseModel):
    n_applicants: int


class Config(BaseModel):
    seed: int
    paths: Paths
    split: Split
    features: Features
    binning: Binning
    selection: Selection
    scaling: Scaling
    validation: Validation
    challenger: Challenger
    strategy: Strategy
    api: Api
    synthetic: Synthetic


def load_config(path: Path | None = None, root: Path | None = None) -> Config:
    """Read the YAML file and resolve relative paths against `root`."""
    path = path or ROOT / "config" / "config.yaml"
    cfg = Config(**yaml.safe_load(path.read_text(encoding="utf-8")))
    base = root or ROOT
    for name in Paths.model_fields:
        setattr(cfg.paths, name, base / getattr(cfg.paths, name))
    cfg.binning.overrides_file = ROOT / cfg.binning.overrides_file
    return cfg
