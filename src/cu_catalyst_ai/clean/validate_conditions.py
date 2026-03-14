from __future__ import annotations

import pandas as pd

REQUIRED_COLUMNS = {
    "catalyst_id",
    "facet",
    "coordination_number",
    "avg_neighbor_distance",
    "electronegativity",
    "d_band_center",
    "surface_energy",
    "adsorption_energy",
}


def validate_required_columns(df: pd.DataFrame) -> pd.DataFrame:
    missing = REQUIRED_COLUMNS.difference(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    return df
