from __future__ import annotations

import pandas as pd

KEY_COLUMNS = ["catalyst_id", "adsorbate", "facet"]


def drop_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    return df.drop_duplicates(subset=KEY_COLUMNS).reset_index(drop=True)
