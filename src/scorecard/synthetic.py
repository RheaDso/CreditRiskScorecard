"""Generate fake Home Credit style CSVs with a known risk signal.

Used for tests and for running the project without the Kaggle files.
Results obtained on this data say nothing about real borrowers.
"""

from pathlib import Path

import numpy as np
import pandas as pd


def _children(ids: np.ndarray, counts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Repeat each applicant id `count` times; return ids and the parent row index."""
    index = np.repeat(np.arange(len(ids)), counts)
    return ids[index], index


def _applications(n: int, rng: np.random.Generator, first_id: int) -> tuple[pd.DataFrame, dict]:
    """Build the application table plus the hidden drivers of default."""
    ids = np.arange(first_id, first_id + n)
    ext = rng.beta(4, 3, size=(n, 3))
    age = rng.uniform(21, 68, n)
    employed = np.minimum(rng.exponential(6, n), age - 18)
    income = rng.lognormal(11.9, 0.5, n).round(-2)
    credit = (income * rng.lognormal(1.1, 0.5, n)).round(-2)
    annuity = (credit * rng.uniform(0.03, 0.09, n)).round(-1)
    goods = (credit * rng.uniform(0.7, 1.0, n)).round(-2)
    education = rng.choice(
        ["Secondary", "Higher education", "Incomplete higher", "Lower secondary"],
        n, p=[0.65, 0.23, 0.07, 0.05],
    )
    hidden = {
        "ext": ext,
        "age": age,
        "employed": employed,
        "leverage": credit / income,
        "goods_ratio": goods / credit,
        "education": education,
        "debt_ratio": rng.beta(2, 3, n),
        "late_rate": rng.beta(1, 9, n),
        "underpay_rate": rng.beta(1, 12, n),
        "refused_rate": rng.beta(1, 4, n),
    }

    anomaly = rng.random(n) < 0.15  # pensioners / unemployed placeholder
    app = pd.DataFrame(
        {
            "SK_ID_CURR": ids,
            "CODE_GENDER": rng.choice(["F", "M", "XNA"], n, p=[0.65, 0.3497, 0.0003]),
            "AMT_INCOME_TOTAL": income,
            "AMT_CREDIT": credit,
            "AMT_ANNUITY": annuity,
            "AMT_GOODS_PRICE": goods,
            "CNT_FAM_MEMBERS": rng.integers(1, 6, n),
            "NAME_EDUCATION_TYPE": education,
            "DAYS_BIRTH": (-age * 365.25).astype(int),
            "DAYS_EMPLOYED": np.where(anomaly, 365243, (-employed * 365.25).astype(int)),
            "DAYS_ID_PUBLISH": -rng.integers(1, 6000, n),
            "DAYS_REGISTRATION": -rng.integers(1, 12000, n),
            "FLAG_DOCUMENT_2": rng.integers(0, 2, n),
            "FLAG_DOCUMENT_3": rng.integers(0, 2, n),
        }
    )
    for i, missing in enumerate([0.45, 0.02, 0.20]):
        app[f"EXT_SOURCE_{i + 1}"] = np.where(rng.random(n) < missing, np.nan, ext[:, i])
    return app, hidden


def _target(hidden: dict, rng: np.random.Generator) -> np.ndarray:
    """Draw defaults from a logistic function of the hidden drivers (~8% bad)."""
    education_effect = pd.Series(hidden["education"]).map(
        {"Higher education": -0.4, "Incomplete higher": 0.0, "Secondary": 0.1,
         "Lower secondary": 0.5}
    ).to_numpy()
    ext_effect = ((hidden["ext"] - 4 / 7) * np.array([-2.6, -3.2, -3.4])).sum(axis=1)
    logit = (
        ext_effect
        - 0.025 * (hidden["age"] - 44)
        - 0.05 * (np.minimum(hidden["employed"], 20) - 5)
        + 0.12 * (np.minimum(hidden["leverage"], 8) - 3.4)
        - 2.5 * (hidden["goods_ratio"] - 0.85)
        + 1.8 * (hidden["debt_ratio"] - 0.4)
        + 5.0 * (hidden["late_rate"] - 0.1)
        + 4.0 * (hidden["underpay_rate"] - 0.08)
        + 1.6 * (hidden["refused_rate"] - 0.2)
        + education_effect
    )
    low, high = -10.0, 5.0
    for _ in range(40):  # find the intercept that gives an 8% bad rate
        mid = (low + high) / 2
        rate = (1 / (1 + np.exp(-(mid + logit)))).mean()
        low, high = (mid, high) if rate < 0.08 else (low, mid)
    return (rng.random(len(logit)) < 1 / (1 + np.exp(-(low + logit)))).astype(int)


def _bureau(ids: np.ndarray, hidden: dict, rng: np.random.Generator) -> pd.DataFrame:
    counts = rng.poisson(3, len(ids))
    child_ids, parent = _children(ids, counts)
    m = len(child_ids)
    total = rng.lognormal(11.5, 0.8, m).round(-2)
    share = np.clip(hidden["debt_ratio"][parent] + rng.normal(0, 0.08, m), 0, 1)
    overdue = rng.random(m) < 0.03
    return pd.DataFrame(
        {
            "SK_ID_CURR": child_ids,
            "SK_ID_BUREAU": np.arange(m) + 5_000_000,
            "CREDIT_ACTIVE": rng.choice(["Active", "Closed"], m, p=[0.4, 0.6]),
            "CREDIT_TYPE": rng.choice(["Consumer credit", "Credit card", "Car loan"], m),
            "DAYS_CREDIT": -rng.integers(30, 2900, m),
            "CREDIT_DAY_OVERDUE": np.where(overdue, rng.integers(1, 90, m), 0),
            "AMT_CREDIT_SUM": total,
            "AMT_CREDIT_SUM_DEBT": (total * share).round(-2),
            "AMT_CREDIT_SUM_OVERDUE": np.where(overdue, (total * 0.05).round(), 0),
        }
    )


def _previous(ids: np.ndarray, hidden: dict, rng: np.random.Generator) -> pd.DataFrame:
    counts = rng.poisson(3, len(ids))
    child_ids, parent = _children(ids, counts)
    m = len(child_ids)
    refused = rng.random(m) < hidden["refused_rate"][parent]
    asked = rng.lognormal(11.5, 0.7, m).round(-2)
    return pd.DataFrame(
        {
            "SK_ID_PREV": np.arange(m) + 1_000_000,
            "SK_ID_CURR": child_ids,
            "NAME_CONTRACT_STATUS": np.where(refused, "Refused", "Approved"),
            "AMT_APPLICATION": asked,
            "AMT_CREDIT": (asked * rng.uniform(0.8, 1.2, m)).round(-2),
        }
    )


def _instalments(ids: np.ndarray, hidden: dict, rng: np.random.Generator) -> pd.DataFrame:
    counts = rng.poisson(14, len(ids))
    child_ids, parent = _children(ids, counts)
    m = len(child_ids)
    due_day = -rng.integers(30, 2000, m)
    late = rng.random(m) < hidden["late_rate"][parent]
    under = rng.random(m) < hidden["underpay_rate"][parent]
    amount = rng.lognormal(9.5, 0.5, m).round(2)
    delay = np.where(late, rng.integers(1, 40, m), -rng.integers(0, 15, m))
    return pd.DataFrame(
        {
            "SK_ID_PREV": child_ids,
            "SK_ID_CURR": child_ids,
            "DAYS_INSTALMENT": due_day,
            "DAYS_ENTRY_PAYMENT": due_day + delay,
            "AMT_INSTALMENT": amount,
            "AMT_PAYMENT": np.where(under, amount * 0.6, amount).round(2),
        }
    )


def generate(out_dir: Path, n: int = 5000, seed: int = 42) -> None:
    """Write application_train/test, bureau, previous and instalment CSVs."""
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    app, hidden = _applications(n, rng, first_id=100_000)
    app.insert(1, "TARGET", _target(hidden, rng))
    app.to_csv(out_dir / "application_train.csv", index=False)

    ids = app["SK_ID_CURR"].to_numpy()
    test_app, test_hidden = _applications(max(n // 5, 200), rng, first_id=900_000)
    test_app.to_csv(out_dir / "application_test.csv", index=False)

    all_ids = np.concatenate([ids, test_app["SK_ID_CURR"].to_numpy()])
    both = {k: np.concatenate([hidden[k], test_hidden[k]]) for k in hidden}
    _bureau(all_ids, both, rng).to_csv(out_dir / "bureau.csv", index=False)
    _previous(all_ids, both, rng).to_csv(out_dir / "previous_application.csv", index=False)
    _instalments(all_ids, both, rng).to_csv(out_dir / "installments_payments.csv", index=False)
