from __future__ import annotations

import pandas as pd


def add_structural_ratios(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["coordination_to_distance"] = out["coordination_number"] / out["avg_neighbor_distance"]
    return out
