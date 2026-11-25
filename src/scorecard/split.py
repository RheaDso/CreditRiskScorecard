"""M2: stratified train / validation / test split plus a pseudo out-of-time flag."""

import pandas as pd
from sklearn.model_selection import train_test_split

from scorecard.config import Config


def make_split(app: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """Return SK_ID_CURR, split, is_pot."""
    s = cfg.split
    train, rest = train_test_split(
        app, train_size=s.train, stratify=app["TARGET"], random_state=cfg.seed
    )
    validation, test = train_test_split(
        rest,
        train_size=s.validation / (s.validation + s.test),
        stratify=rest["TARGET"],
        random_state=cfg.seed,
    )
    out = pd.concat(
        [
            pd.DataFrame({"SK_ID_CURR": part["SK_ID_CURR"], "split": name})
            for name, part in [("train", train), ("validation", validation), ("test", test)]
        ]
    ).sort_values("SK_ID_CURR", ignore_index=True)

    # No dates exist, so the highest ids inside TEST stand in for "later" applications.
    test_ids = out.loc[out["split"] == "test", "SK_ID_CURR"]
    pot_ids = test_ids.nlargest(int(round(len(test_ids) * s.pot_share_of_test)))
    out["is_pot"] = out["SK_ID_CURR"].isin(pot_ids)
    return out
