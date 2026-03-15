"""Unit normalisation for adsorption energy.

Converts ``adsorption_energy`` values to eV using the ``unit_adsorption_energy``
column as the declared unit.  Supported conversions:

==========  ============
Unit        Factor to eV
==========  ============
eV          1.0
meV         0.001
kJ/mol      0.010364
==========  ============

Rows with any other declared unit are **flagged for review** rather than
rejected globally.  The ``unit_adsorption_energy`` column of successfully
converted rows is set to ``"eV"``.
"""

from __future__ import annotations

import pandas as pd

from cu_catalyst_ai.clean.governance import flag_rows

# Map of supported unit strings → multiplication factor to reach eV
_UNIT_TO_EV: dict[str, float] = {
    "eV": 1.0,
    "meV": 0.001,
    "kJ/mol": 0.010364,
}


def normalize_units(
    df: pd.DataFrame,
    unit_conversions: dict[str, float] | None = None,
) -> pd.DataFrame:
    """Normalise adsorption energy to eV; flag rows with unknown units.

    Args:
        df: DataFrame that contains ``adsorption_energy`` and, optionally,
            ``unit_adsorption_energy`` columns.
        unit_conversions: Mapping of unit string → multiplication factor to eV.
            When provided (e.g. from ``cfg.target.supported_unit_conversions``),
            this overrides the built-in ``_UNIT_TO_EV`` table so that the target
            config becomes the single source of truth for unit handling.

    Returns:
        DataFrame with ``adsorption_energy`` converted to eV for rows with
        known units.  Rows with unknown units have ``review_reason`` /
        ``review_stage`` set via :func:`~cu_catalyst_ai.clean.governance.flag_rows`.
        The ``unit_adsorption_energy`` column for successfully converted rows is
        updated to ``"eV"``.
    """
    out = df.copy()

    if "unit_adsorption_energy" not in out.columns:
        # No unit column — assume everything is already eV (backward-compatible
        # with old datasets that omit this field).
        return out

    conversion_table = unit_conversions if unit_conversions is not None else _UNIT_TO_EV
    unit_col = out["unit_adsorption_energy"].fillna("").astype(str)

    # Identify unknown units
    known_mask = unit_col.isin(conversion_table)
    unknown_mask = ~known_mask

    if unknown_mask.any():
        # Build a human-readable summary of the unknown unit values
        unknown_units = unit_col[unknown_mask].unique().tolist()
        reason = f"unknown unit_adsorption_energy: {unknown_units}"
        out = flag_rows(out, unknown_mask, reason=reason, stage="unit_normalisation")

    # Convert known-unit rows (vectorised group-by approach)
    for unit, factor in conversion_table.items():
        row_mask = unit_col.eq(unit)
        if row_mask.any():
            # Only convert if adsorption_energy is numeric
            ae_numeric = pd.to_numeric(out.loc[row_mask, "adsorption_energy"], errors="coerce")
            convertible = row_mask & ae_numeric.notna()
            out.loc[convertible, "adsorption_energy"] = ae_numeric[convertible] * factor
            out.loc[convertible, "unit_adsorption_energy"] = "eV"

    return out
