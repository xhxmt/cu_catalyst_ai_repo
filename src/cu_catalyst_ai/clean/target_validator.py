"""Target definition and adsorbate governance checks.

Isolates rows whose ``target_definition`` column does not match the registered
target name, or whose ``adsorbate`` value does not match the required adsorbate
for that target.
"""

from __future__ import annotations

import pandas as pd

from cu_catalyst_ai.clean.governance import flag_rows


def validate_target_definition(
    df: pd.DataFrame,
    target_def_name: str,
    required_adsorbate: str = "CO",
) -> pd.DataFrame:
    """Flag rows whose target definition or adsorbate is inconsistent.

    Args:
        df: DataFrame with at least a ``target_definition`` column.
        target_def_name: The expected value of ``target_definition`` for every
            row (e.g. ``"co_adsorption_energy_ev_v1"``).
        required_adsorbate: Expected adsorbate value (default ``"CO"``).

    Returns:
        DataFrame with ``review_reason`` / ``review_stage`` columns added for
        any flagged rows.
    """
    out = df.copy()

    # --- target_definition mismatch ----------------------------------------
    if "target_definition" in out.columns:
        bad_def = out["target_definition"].ne(target_def_name) | out["target_definition"].isna()
        out = flag_rows(
            out,
            bad_def,
            reason=f"target_definition mismatch: expected '{target_def_name}'",
            stage="target_definition",
        )
    else:
        # Column is absent entirely — flag all rows
        out = flag_rows(
            out,
            pd.Series(True, index=out.index),
            reason="target_definition column missing",
            stage="target_definition",
        )

    # --- adsorbate mismatch ------------------------------------------------
    if "adsorbate" in out.columns:
        bad_ads = out["adsorbate"].ne(required_adsorbate) | out["adsorbate"].isna()
        out = flag_rows(
            out,
            bad_ads,
            reason=f"adsorbate mismatch: expected '{required_adsorbate}', got non-matching value",
            stage="target_definition",
        )

    return out
