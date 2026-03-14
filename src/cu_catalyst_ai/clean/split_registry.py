from __future__ import annotations

import pandas as pd
from sklearn.model_selection import train_test_split


def assign_splits(df: pd.DataFrame, seed: int = 42, test_size: float = 0.2) -> pd.DataFrame:
    train_ids, test_ids = train_test_split(
        df["catalyst_id"], test_size=test_size, random_state=seed
    )
    out = df.copy()
    out["split"] = "train"
    out.loc[out["catalyst_id"].isin(test_ids), "split"] = "test"
    return out
