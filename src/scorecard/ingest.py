"""M1: load raw CSVs, validate them and apply known data fixes."""

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import pandera.pandas as pa

from scorecard.exceptions import DataValidationError

log = logging.getLogger(__name__)

# Columns each table must contain. None means "load every column".
REQUIRED: dict[str, list[str]] = {
    "application_train": [
        "SK_ID_CURR", "TARGET", "AMT_INCOME_TOTAL", "AMT_CREDIT", "AMT_ANNUITY",
        "AMT_GOODS_PRICE", "DAYS_BIRTH", "DAYS_EMPLOYED", "CODE_GENDER",
        "CNT_FAM_MEMBERS", "DAYS_ID_PUBLISH", "DAYS_REGISTRATION",
        "EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3",
    ],
    "bureau": [
        "SK_ID_CURR", "CREDIT_ACTIVE", "CREDIT_TYPE", "DAYS_CREDIT", "CREDIT_DAY_OVERDUE",
        "AMT_CREDIT_SUM", "AMT_CREDIT_SUM_DEBT", "AMT_CREDIT_SUM_OVERDUE",
    ],
    "previous_application": [
        "SK_ID_CURR", "NAME_CONTRACT_STATUS", "AMT_APPLICATION", "AMT_CREDIT",
    ],
    "installments_payments": [
        "SK_ID_CURR", "DAYS_INSTALMENT", "DAYS_ENTRY_PAYMENT", "AMT_INSTALMENT", "AMT_PAYMENT",
    ],
}

APPLICATION_SCHEMA = pa.DataFrameSchema(
    {
        "SK_ID_CURR": pa.Column(int, unique=True, nullable=False),
        "TARGET": pa.Column(int, pa.Check.isin([0, 1])),
        "AMT_INCOME_TOTAL": pa.Column(float, pa.Check.gt(0), coerce=True),
        "AMT_CREDIT": pa.Column(float, pa.Check.gt(0), coerce=True),
    }
)


def read_table(raw_dir: Path, name: str) -> pd.DataFrame:
    """Read one CSV. Child tables load only needed columns to save memory."""
    path = raw_dir / f"{name}.csv"
    if not path.exists():
        raise DataValidationError(f"Missing file: {path}")
    required = REQUIRED.get(name, [])
    header = pd.read_csv(path, nrows=0).columns
    missing = [c for c in required if c not in header]
    if missing:
        raise DataValidationError(f"{path.name} is missing columns: {missing}")
    usecols = None if name.startswith("application") else required
    return pd.read_csv(path, usecols=usecols)


def validate_application(df: pd.DataFrame) -> pd.DataFrame:
    try:
        return APPLICATION_SCHEMA.validate(df)
    except pa.errors.SchemaError as exc:
        raise DataValidationError(f"application_train failed validation: {exc}") from exc


def fix_application(df: pd.DataFrame) -> pd.DataFrame:
    """Known issues: employment placeholder and the 'XNA' gender code."""
    df = df.copy()
    anomaly = df["DAYS_EMPLOYED"] == 365243
    df["FLAG_DAYS_EMPLOYED_ANOM"] = anomaly.astype(int)
    df["DAYS_EMPLOYED"] = df["DAYS_EMPLOYED"].astype(float).mask(anomaly)
    log.info("DAYS_EMPLOYED placeholder replaced: %d rows", anomaly.sum())

    xna = df["CODE_GENDER"] == "XNA"
    df["CODE_GENDER"] = df["CODE_GENDER"].mask(xna, np.nan)
    log.info("CODE_GENDER 'XNA' mapped to missing: %d rows", xna.sum())
    return df


def data_quality(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for col in df.columns:
        s = df[col]
        numeric = pd.api.types.is_numeric_dtype(s)
        rows.append(
            {
                "column": col,
                "dtype": str(s.dtype),
                "missing_pct": round(100 * s.isna().mean(), 2),
                "n_unique": s.nunique(),
                "min": s.min() if numeric else None,
                "max": s.max() if numeric else None,
            }
        )
    return pd.DataFrame(rows)


def ingest(raw_dir: Path, interim_dir: Path, reports_dir: Path) -> dict[str, pd.DataFrame]:
    """Run M1 and write Parquet copies of every table."""
    interim_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)
    tables = {name: read_table(raw_dir, name) for name in REQUIRED}
    tables["application_train"] = fix_application(
        validate_application(tables["application_train"])
    )
    if (raw_dir / "application_test.csv").exists():
        tables["application_test"] = fix_application(
            pd.read_csv(raw_dir / "application_test.csv")
        )
    for name, df in tables.items():
        df.to_parquet(interim_dir / f"{name}.parquet", index=False)
    data_quality(tables["application_train"]).to_csv(
        reports_dir / "data_quality.csv", index=False
    )
    return tables
