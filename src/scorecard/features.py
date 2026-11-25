"""M3: one row per applicant. No feature uses TARGET or other applicants' data."""

import numpy as np
import pandas as pd

EXT = ["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"]
KEYS = ["SK_ID_CURR", "TARGET"]
# Raw day offsets are replaced by readable *_YEARS features, so they are not candidates.
SUPERSEDED = ["DAYS_BIRTH", "DAYS_EMPLOYED", "DAYS_ID_PUBLISH", "DAYS_REGISTRATION"]

# Raw inputs behind each engineered application feature (used to build the API schema).
FEATURE_INPUTS: dict[str, list[str]] = {
    "CREDIT_INCOME_RATIO": ["AMT_CREDIT", "AMT_INCOME_TOTAL"],
    "ANNUITY_INCOME_RATIO": ["AMT_ANNUITY", "AMT_INCOME_TOTAL"],
    "CREDIT_TERM": ["AMT_ANNUITY", "AMT_CREDIT"],
    "GOODS_CREDIT_RATIO": ["AMT_GOODS_PRICE", "AMT_CREDIT"],
    "AGE_YEARS": ["DAYS_BIRTH"],
    "EMPLOYED_YEARS": ["DAYS_EMPLOYED"],
    "EMPLOYED_TO_AGE": ["DAYS_EMPLOYED", "DAYS_BIRTH"],
    "INCOME_PER_PERSON": ["AMT_INCOME_TOTAL", "CNT_FAM_MEMBERS"],
    "EXT_SOURCE_MEAN": EXT,
    "EXT_SOURCE_MIN": EXT,
    "EXT_SOURCE_MAX": EXT,
    "EXT_SOURCE_STD": EXT,
    "ID_PUBLISH_YEARS": ["DAYS_ID_PUBLISH"],
    "REGISTRATION_YEARS": ["DAYS_REGISTRATION"],
    "FLAG_DAYS_EMPLOYED_ANOM": ["DAYS_EMPLOYED"],
}

COUNT_COLUMNS = [
    "BUR_COUNT", "BUR_ACTIVE_COUNT", "BUR_CLOSED_COUNT", "PREV_COUNT", "PREV_REFUSED_COUNT",
]


def safe_divide(num: pd.Series, den: pd.Series) -> pd.Series:
    """Division that returns NaN, never inf, for a zero or missing denominator."""
    return num / den.where(den != 0)


def application_features(app: pd.DataFrame) -> pd.DataFrame:
    """Add ratio, age and external score features. Safe to call on a single record."""
    df = app.copy()
    needed = {c for cols in FEATURE_INPUTS.values() for c in cols}
    for col in needed:
        if col not in df:
            df[col] = np.nan
        df[col] = pd.to_numeric(df[col], errors="coerce").astype(float)

    anomaly = df["DAYS_EMPLOYED"] == 365243
    if "FLAG_DAYS_EMPLOYED_ANOM" not in df:
        df["FLAG_DAYS_EMPLOYED_ANOM"] = anomaly.astype(int)
    df.loc[anomaly, "DAYS_EMPLOYED"] = np.nan

    df["CREDIT_INCOME_RATIO"] = safe_divide(df["AMT_CREDIT"], df["AMT_INCOME_TOTAL"])
    df["ANNUITY_INCOME_RATIO"] = safe_divide(df["AMT_ANNUITY"], df["AMT_INCOME_TOTAL"])
    df["CREDIT_TERM"] = safe_divide(df["AMT_ANNUITY"], df["AMT_CREDIT"])
    df["GOODS_CREDIT_RATIO"] = safe_divide(df["AMT_GOODS_PRICE"], df["AMT_CREDIT"])
    df["AGE_YEARS"] = -df["DAYS_BIRTH"] / 365.25
    df["EMPLOYED_YEARS"] = -df["DAYS_EMPLOYED"] / 365.25
    df["EMPLOYED_TO_AGE"] = safe_divide(df["EMPLOYED_YEARS"], df["AGE_YEARS"])
    df["INCOME_PER_PERSON"] = safe_divide(df["AMT_INCOME_TOTAL"], df["CNT_FAM_MEMBERS"])
    df["EXT_SOURCE_MEAN"] = df[EXT].mean(axis=1)
    df["EXT_SOURCE_MIN"] = df[EXT].min(axis=1)
    df["EXT_SOURCE_MAX"] = df[EXT].max(axis=1)
    df["EXT_SOURCE_STD"] = df[EXT].std(axis=1)
    df["ID_PUBLISH_YEARS"] = -df["DAYS_ID_PUBLISH"] / 365.25
    df["REGISTRATION_YEARS"] = -df["DAYS_REGISTRATION"] / 365.25
    docs = [c for c in df.columns if c.startswith("FLAG_DOCUMENT_")]
    if docs:
        df["DOC_COUNT"] = df[docs].sum(axis=1)
    return df


def bureau_features(bureau: pd.DataFrame) -> pd.DataFrame:
    b = bureau.assign(
        active=bureau["CREDIT_ACTIVE"] == "Active",
        closed=bureau["CREDIT_ACTIVE"] == "Closed",
        dpd=bureau["CREDIT_DAY_OVERDUE"] > 0,
        age=-bureau["DAYS_CREDIT"],
    )
    out = b.groupby("SK_ID_CURR").agg(
        BUR_COUNT=("active", "size"),
        BUR_ACTIVE_COUNT=("active", "sum"),
        BUR_CLOSED_COUNT=("closed", "sum"),
        BUR_OVERDUE_SUM=("AMT_CREDIT_SUM_OVERDUE", "sum"),
        BUR_MAX_DPD_FLAG=("dpd", "max"),
        BUR_DAYS_CREDIT_MIN=("age", "max"),  # age of the oldest credit
        BUR_DAYS_CREDIT_MEAN=("age", "mean"),
        BUR_CREDIT_TYPE_NUNIQUE=("CREDIT_TYPE", "nunique"),
    )
    active = b[b["active"]].groupby("SK_ID_CURR")[["AMT_CREDIT_SUM_DEBT", "AMT_CREDIT_SUM"]].sum()
    out["BUR_DEBT_CREDIT_RATIO"] = safe_divide(
        active["AMT_CREDIT_SUM_DEBT"], active["AMT_CREDIT_SUM"]
    )
    out["BUR_MAX_DPD_FLAG"] = out["BUR_MAX_DPD_FLAG"].astype(int)
    return out.reset_index()


def previous_features(prev: pd.DataFrame) -> pd.DataFrame:
    p = prev.assign(
        approved=prev["NAME_CONTRACT_STATUS"] == "Approved",
        refused=prev["NAME_CONTRACT_STATUS"] == "Refused",
        ratio=safe_divide(prev["AMT_APPLICATION"], prev["AMT_CREDIT"]),
    )
    return (
        p.groupby("SK_ID_CURR")
        .agg(
            PREV_COUNT=("approved", "size"),
            PREV_APPROVED_RATE=("approved", "mean"),
            PREV_REFUSED_COUNT=("refused", "sum"),
            PREV_APP_CREDIT_RATIO_MEAN=("ratio", "mean"),
        )
        .reset_index()
    )


def instalment_features(inst: pd.DataFrame) -> pd.DataFrame:
    paid_day = inst["DAYS_ENTRY_PAYMENT"]
    i = pd.DataFrame(
        {
            "SK_ID_CURR": inst["SK_ID_CURR"],
            "late": (paid_day > inst["DAYS_INSTALMENT"]).astype(float).where(paid_day.notna()),
            "dpd": (paid_day - inst["DAYS_INSTALMENT"]).clip(lower=0),
            "under": (inst["AMT_PAYMENT"] < inst["AMT_INSTALMENT"])
            .astype(float)
            .where(inst["AMT_PAYMENT"].notna()),
            "ratio": safe_divide(inst["AMT_PAYMENT"], inst["AMT_INSTALMENT"]),
        }
    )
    return (
        i.groupby("SK_ID_CURR")
        .agg(
            INST_LATE_PAYMENT_RATE=("late", "mean"),
            INST_DPD_MEAN=("dpd", "mean"),
            INST_DPD_MAX=("dpd", "max"),
            INST_UNDERPAY_RATE=("under", "mean"),
            INST_PAYMENT_RATIO_MEAN=("ratio", "mean"),
        )
        .reset_index()
    )


def build_features(app: pd.DataFrame, tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Join application features with bureau, previous and instalment aggregates."""
    df = application_features(app)
    bur = bureau_features(tables["bureau"])
    prev = previous_features(tables["previous_application"])
    inst = instalment_features(tables["installments_payments"])
    for part in (bur, prev, inst):
        df = df.merge(part, on="SK_ID_CURR", how="left", validate="one_to_one")
    df["HAS_BUREAU"] = df["SK_ID_CURR"].isin(bur["SK_ID_CURR"]).astype(int)
    df["HAS_PREV"] = df["SK_ID_CURR"].isin(prev["SK_ID_CURR"]).astype(int)
    df[COUNT_COLUMNS] = df[COUNT_COLUMNS].fillna(0)
    return df.replace([np.inf, -np.inf], np.nan)


def columns_to_drop(train: pd.DataFrame, max_missing: float) -> list[str]:
    """Mostly-empty or constant columns, judged on TRAIN only to avoid leakage."""
    return [
        c
        for c in train.columns
        if c not in KEYS and (train[c].isna().mean() > max_missing or train[c].nunique() <= 1)
    ]


def source_of(feature: str) -> str:
    if feature.startswith(("BUR_", "HAS_BUREAU")):
        return "bureau"
    if feature.startswith("INST_"):
        return "instalments"
    if feature.startswith(("PREV_", "HAS_PREV")):
        return "previous"
    return "application"


def feature_dictionary(columns: list[str]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "name": columns,
            "source_table": [source_of(c) for c in columns],
            "formula": [
                "from " + ", ".join(FEATURE_INPUTS[c]) if c in FEATURE_INPUTS
                else ("aggregate by SK_ID_CURR" if source_of(c) != "application" else "raw column")
                for c in columns
            ],
            "business_meaning": [c.replace("_", " ").lower() for c in columns],
        }
    )
