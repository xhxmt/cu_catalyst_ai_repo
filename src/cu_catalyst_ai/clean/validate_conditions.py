"""Column-level and row-level validation for catalyst records.

Two distinct entry points:

* :func:`validate_required_columns` – structural check; **raises** if mandatory
  columns are absent.  Preserves backward-compatible behaviour for the demo
  source.

* :func:`validate_rows` – row-level governance; uses vectorised checks to flag
  hard-invalid and soft-review rows *without* raising globally.  Flagged rows
  are annotated with ``review_reason`` / ``review_stage`` and should later be
  separated by :func:`cu_catalyst_ai.clean.governance.split_good_review`.
"""

from __future__ import annotations

import pandas as pd

from cu_catalyst_ai.clean.governance import flag_rows

# Columns that must exist after the column-mapping step.
# Absence of any of these is a structural failure that prevents cleaning.
REQUIRED_COLUMNS: set[str] = {
    "catalyst_id",
    "facet",
    "adsorbate",
    "coordination_number",
    "avg_neighbor_distance",
    "electronegativity",
    "d_band_center",
    "surface_energy",
    "adsorption_energy",
    "provenance",
    "unit_adsorption_energy",
}


def validate_required_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Raise if any required structural column is missing.

    This check is intentionally strict: missing structural columns mean the
    data cannot be cleaned at all, so we fail fast rather than producing a
    silently corrupt output.

    Args:
        df: Raw (or column-mapped) DataFrame.

    Returns:
        The original *df* unchanged if all required columns are present.

    Raises:
        ValueError: If one or more required columns are absent.
    """
    missing = REQUIRED_COLUMNS.difference(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    return df


def validate_rows(
    df: pd.DataFrame,
    *,
    adsorption_energy_abs_max: float = 10.0,
    surface_energy_min: float = 0.0,
    electronegativity_min: float = 0.0,
    electronegativity_max: float = 4.0,
) -> pd.DataFrame:
    """Flag rows with hard-invalid or out-of-bounds values.

    Hard-invalid checks (structural impossibilities):
    - ``avg_neighbor_distance`` ≤ 0
    - ``coordination_number`` < 0
    - ``adsorption_energy`` cannot be cast to float

    Soft review-bound checks (physically suspect, requires human review):
    - ``|adsorption_energy|`` > *adsorption_energy_abs_max* (default 10 eV)
    - ``surface_energy`` ≤ *surface_energy_min* (default 0)
    - ``electronegativity`` ≤ *electronegativity_min* or > *electronegativity_max*

    Flagged rows are annotated via :func:`~cu_catalyst_ai.clean.governance.flag_rows`
    and should later be separated by
    :func:`~cu_catalyst_ai.clean.governance.split_good_review`.

    Args:
        df: DataFrame that has already passed :func:`validate_required_columns`
            and :func:`~cu_catalyst_ai.clean.normalize_units.normalize_units`.
        adsorption_energy_abs_max: Upper bound on absolute adsorption energy.
        surface_energy_min: Minimum allowed surface energy (exclusive).
        electronegativity_min: Minimum allowed electronegativity (exclusive).
        electronegativity_max: Maximum allowed electronegativity (inclusive).

    Returns:
        DataFrame with ``review_reason`` and ``review_stage`` columns added
        for any flagged rows.
    """
    out = df.copy()

    # ------------------------------------------------------------------
    # Hard-invalid: avg_neighbor_distance <= 0
    # ------------------------------------------------------------------
    if "avg_neighbor_distance" in out.columns:
        numeric_and = pd.to_numeric(out["avg_neighbor_distance"], errors="coerce")
        bad_and = numeric_and.le(0) | numeric_and.isna()
        out = flag_rows(out, bad_and, reason="avg_neighbor_distance <= 0", stage="hard_invalid")

    # ------------------------------------------------------------------
    # Hard-invalid: coordination_number < 0
    # ------------------------------------------------------------------
    if "coordination_number" in out.columns:
        numeric_cn = pd.to_numeric(out["coordination_number"], errors="coerce")
        bad_cn = numeric_cn.lt(0) | numeric_cn.isna()
        out = flag_rows(out, bad_cn, reason="coordination_number < 0", stage="hard_invalid")

    # ------------------------------------------------------------------
    # Hard-invalid: adsorption_energy non-numeric
    # ------------------------------------------------------------------
    if "adsorption_energy" in out.columns:
        numeric_ae = pd.to_numeric(out["adsorption_energy"], errors="coerce")
        bad_ae_numeric = numeric_ae.isna()
        out = flag_rows(
            out, bad_ae_numeric, reason="adsorption_energy is non-numeric", stage="hard_invalid"
        )
    else:
        # adsorption_energy is in REQUIRED_COLUMNS — if we reach here after
        # validate_required_columns then the column is always present.
        pass

    # ------------------------------------------------------------------
    # Soft bounds: adsorption energy magnitude
    # ------------------------------------------------------------------
    if "adsorption_energy" in out.columns:
        numeric_ae = pd.to_numeric(out["adsorption_energy"], errors="coerce")
        soft_ae = numeric_ae.abs().gt(adsorption_energy_abs_max) & numeric_ae.notna()
        out = flag_rows(
            out,
            soft_ae,
            reason=f"|adsorption_energy| > {adsorption_energy_abs_max} eV",
            stage="review_bounds",
        )

    # ------------------------------------------------------------------
    # Soft bounds: surface energy
    # ------------------------------------------------------------------
    if "surface_energy" in out.columns:
        numeric_se = pd.to_numeric(out["surface_energy"], errors="coerce")
        soft_se = numeric_se.le(surface_energy_min) & numeric_se.notna()
        out = flag_rows(
            out,
            soft_se,
            reason=f"surface_energy <= {surface_energy_min}",
            stage="review_bounds",
        )

    # ------------------------------------------------------------------
    # Soft bounds: electronegativity
    # ------------------------------------------------------------------
    if "electronegativity" in out.columns:
        numeric_en = pd.to_numeric(out["electronegativity"], errors="coerce")
        soft_en_lo = numeric_en.le(electronegativity_min) & numeric_en.notna()
        soft_en_hi = numeric_en.gt(electronegativity_max) & numeric_en.notna()
        out = flag_rows(
            out,
            soft_en_lo | soft_en_hi,
            reason=(
                f"electronegativity out of range ({electronegativity_min}, {electronegativity_max}]"
            ),
            stage="review_bounds",
        )

    return out
