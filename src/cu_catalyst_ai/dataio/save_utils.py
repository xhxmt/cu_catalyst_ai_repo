from __future__ import annotations

import pandas as pd

from cu_catalyst_ai.utils.io import write_table


def save_dataframe(df: pd.DataFrame, path: str) -> str:
    return str(write_table(df, path))
