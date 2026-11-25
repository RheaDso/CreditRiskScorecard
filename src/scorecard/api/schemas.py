"""Request and response models. The request fields depend on the fitted model."""

from typing import Any

from pydantic import BaseModel, ConfigDict, Field, create_model


class ReasonCode(BaseModel):
    variable: str
    points_lost: int
    message: str


class ScoreResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    score: int
    pd: float
    risk_band: str
    decision: str
    reason_codes: list[ReasonCode]
    points_breakdown: dict[str, int]
    model_version: str


class Health(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    status: str
    model_version: str
    training_date: str


def applicant_model(inputs: dict[str, type]) -> type[BaseModel]:
    """Every input must be present; send null when the value is unknown.

    Bureau and history aggregates (BUR_*, INST_*, PREV_*) are passed pre-computed:
    the API never reads the raw history tables.
    """
    fields: dict[str, Any] = {}
    for name, kind in inputs.items():
        constraint = Field(..., allow_inf_nan=False) if kind is float else Field(...)
        fields[name] = (kind | None, constraint)
    return create_model("Applicant", **fields)
