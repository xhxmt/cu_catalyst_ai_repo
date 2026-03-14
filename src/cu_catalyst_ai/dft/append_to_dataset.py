from __future__ import annotations

import pandas as pd


def append_verified_dft_rows(df: pd.DataFrame, dft_rows: pd.DataFrame) -> pd.DataFrame:
    return pd.concat([df, dft_rows], ignore_index=True)
