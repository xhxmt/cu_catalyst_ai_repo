"""Provenance governance checks.

Isolates rows whose ``provenance`` column is absent, null, an empty string,
or whitespace-only.  These rows cannot be traced back to a source, which
makes them unsuitable for trusted training data.
"""

from __future__ import annotations

import pandas as pd

from cu_catalyst_ai.clean.governance import flag_rows


def validate_provenance(df: pd.DataFrame) -> pd.DataFrame:
    """Flag rows with missing or empty provenance.

    Args:
        df: DataFrame that should contain a ``provenance`` column.

    Returns:
        DataFrame with ``review_reason`` / ``review_stage`` columns added for
        rows without traceable provenance.
    """
    out = df.copy()

    if "provenance" not in out.columns:
        # Entire column absent — flag every row
        return flag_rows(
            out,
            pd.Series(True, index=out.index),
            reason="provenance column missing",
            stage="provenance",
        )

    # Null, empty-string, or whitespace-only
    bad = out["provenance"].isna() | out["provenance"].astype(str).str.strip().eq("")
    return flag_rows(out, bad, reason="provenance is missing or empty", stage="provenance")
