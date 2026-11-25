"""One scoring path shared by the batch pipeline and the API, so they cannot disagree."""

import pickle
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from scorecard.binning import VariableBins
from scorecard.config import Scaling
from scorecard.features import FEATURE_INPUTS, application_features
from scorecard.model import LogisticModel
from scorecard.scaling import attribute_points

BANDS = "ABCDE"


@dataclass
class ScoreResult:
    score: int
    pd: float
    risk_band: str
    decision: str
    reason_codes: list[dict]
    points_breakdown: dict[str, int]


@dataclass
class Scorecard:
    model: LogisticModel
    bins: dict[str, VariableBins]
    scaling: Scaling
    cutoff: float = 0.0
    refer_band: int = 10
    band_edges: list[float] = field(default_factory=list)  # ascending, 4 values
    reason_messages: dict[str, str] = field(default_factory=dict)

    def _points_for(self, var: str) -> np.ndarray:
        """Rounded points per bin, plus a final entry for values with no bin (WoE 0)."""
        woe = np.append(np.array(self.bins[var].woe), 0.0)
        raw = attribute_points(
            self.model.coefficients[var], woe, self.model.intercept,
            len(self.model.variables), self.scaling,
        )
        return np.round(raw).astype(int)

    def points(self, features: pd.DataFrame) -> pd.DataFrame:
        """Points earned per variable. The score is the sum across columns."""
        return pd.DataFrame(
            {
                v: self._points_for(v)[self.bins[v].positions(features[v])]
                for v in self.model.variables
            },
            index=features.index,
        )

    def probability(self, features: pd.DataFrame) -> np.ndarray:
        woe = pd.DataFrame(
            {v: self.bins[v].transform(features[v]) for v in self.model.variables},
            index=features.index,
        )
        return self.model.predict_pd(woe)

    def band(self, scores: np.ndarray) -> np.ndarray:
        """A = highest scores (lowest risk), E = lowest scores."""
        position = np.searchsorted(self.band_edges, scores, side="right")
        return np.array(list(BANDS[::-1]))[position]

    def decision(self, score: int) -> str:
        if score >= self.cutoff + self.refer_band:
            return "APPROVE"
        if score < self.cutoff - self.refer_band:
            return "DECLINE"
        return "REFER"

    def score_features(self, features: pd.DataFrame) -> list[ScoreResult]:
        """Vectorised scoring of an already engineered feature table."""
        points = self.points(features)
        scores = points.sum(axis=1).to_numpy()
        probability = self.probability(features)
        bands = self.band(scores)
        # Reason codes: how far each variable is from its best possible points.
        best = np.array([self._points_for(v)[:-1].max() for v in self.model.variables])
        gaps = best - points.to_numpy()
        results = []
        for row, (score, p, band) in enumerate(zip(scores, probability, bands, strict=False)):
            order = np.argsort(-gaps[row], kind="stable")[:3]
            reasons = []
            for i in order:
                if gaps[row, i] <= 0:
                    continue
                var = self.model.variables[i]
                reasons.append(
                    {"variable": var, "points_lost": int(gaps[row, i]),
                     "message": self.reason_messages.get(var, f"Unfavourable value of {var}")}
                )
            results.append(
                ScoreResult(
                    score=int(score), pd=round(float(p), 4), risk_band=str(band),
                    decision=self.decision(int(score)), reason_codes=reasons,
                    points_breakdown={v: int(points.iat[row, i])
                                      for i, v in enumerate(self.model.variables)},
                )
            )
        return results

    def required_inputs(self) -> dict[str, type]:
        """Raw request fields the final model needs, with their types."""
        inputs: dict[str, type] = {}
        for var in self.model.variables:
            kind = str if self.bins[var].categorical else float
            for column in FEATURE_INPUTS.get(var, [var]):
                inputs[column] = float if var in FEATURE_INPUTS else kind
        return inputs

    def prepare(self, records: list[dict]) -> pd.DataFrame:
        """Raw records -> engineered features. History aggregates arrive pre-computed."""
        df = pd.DataFrame(records)
        features = application_features(df)
        for var in self.model.variables:
            if var not in features:
                features[var] = np.nan
            if not self.bins[var].categorical:
                features[var] = pd.to_numeric(features[var], errors="coerce")
        return features

    def score_records(self, records: list[dict]) -> list[ScoreResult]:
        return self.score_features(self.prepare(records))

    def score_applicant(self, record: dict) -> ScoreResult:
        return self.score_records([record])[0]


    def save(self, path: Path) -> None:
        with path.open("wb") as handle:
            pickle.dump(self, handle)

    @staticmethod
    def load(path: Path) -> "Scorecard":
        """Only load artifacts you produced yourself: unpickling runs code."""
        with path.open("rb") as handle:
            return pickle.load(handle)
