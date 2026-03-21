from __future__ import annotations

import pandas as pd

NON_FEATURE_COLUMNS = {"catalyst_id", "adsorption_energy", "split", "element"}


def get_feature_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c not in NON_FEATURE_COLUMNS]
