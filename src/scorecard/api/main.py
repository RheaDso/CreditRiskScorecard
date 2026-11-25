"""M12: scoring API.

    uvicorn scorecard.api.main:create_app --factory --port 8000
"""

import json
import os
from dataclasses import asdict
from pathlib import Path

from fastapi import FastAPI
from pydantic import BaseModel, Field

from scorecard.api.schemas import Health, ScoreResponse, applicant_model
from scorecard.config import ROOT, load_config
from scorecard.scoring import Scorecard


def create_app(artifacts_dir: Path | None = None) -> FastAPI:
    cfg = load_config()
    artifacts = artifacts_dir or Path(os.getenv("ARTIFACTS_DIR", str(ROOT / "artifacts")))
    card = Scorecard.load(artifacts / "model.pkl")
    metadata = json.loads((artifacts / "metadata.json").read_text(encoding="utf-8"))
    version = metadata["model_version"]

    Applicant = applicant_model(card.required_inputs())

    class Batch(BaseModel):
        applicants: list[Applicant] = Field(  # type: ignore[valid-type]
            ..., min_length=1, max_length=cfg.api.max_batch
        )

    app = FastAPI(title="Credit Scorecard API", version=version)

    @app.get("/health", response_model=Health)
    def health() -> Health:
        return Health(status="ok", model_version=version,
                      training_date=metadata["training_date"])

    @app.post("/score", response_model=ScoreResponse)
    def score(applicant: Applicant) -> dict:  # type: ignore[valid-type]
        record = applicant.model_dump()  # type: ignore[attr-defined]
        return {**asdict(card.score_applicant(record)), "model_version": version}

    @app.post("/score/batch", response_model=list[ScoreResponse])
    def score_batch(batch: Batch) -> list[dict]:
        records = [a.model_dump() for a in batch.applicants]  # type: ignore[attr-defined]
        return [{**asdict(r), "model_version": version} for r in card.score_records(records)]

    return app
