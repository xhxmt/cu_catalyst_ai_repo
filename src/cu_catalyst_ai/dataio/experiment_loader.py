from __future__ import annotations

import pandas as pd

from cu_catalyst_ai.utils.io import read_table

REQUIRED_COLUMNS = {"catalyst_id", "measured_metric", "metric_name", "unit"}


def load_experiment_feedback(path: str) -> pd.DataFrame:
    df = read_table(path)
    missing = REQUIRED_COLUMNS.difference(df.columns)
    if missing:
        raise ValueError(f"Missing experiment columns: {sorted(missing)}")
    return df
