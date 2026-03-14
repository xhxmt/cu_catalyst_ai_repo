from __future__ import annotations

import pandas as pd


def normalize_units(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "unit_adsorption_energy" in out.columns:
        invalid = out["unit_adsorption_energy"].ne("eV")
        if invalid.any():
            raise ValueError("Only eV is supported in the current baseline pipeline.")
    return out
