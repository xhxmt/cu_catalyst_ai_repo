from __future__ import annotations

import pandas as pd


def build_feature_table(
    df: pd.DataFrame, use_columns: list[str], categorical_columns: list[str]
) -> pd.DataFrame:
    out = df[["catalyst_id", "adsorption_energy", "split", *use_columns]].copy()
    out = pd.get_dummies(out, columns=categorical_columns, drop_first=False)
    return out
