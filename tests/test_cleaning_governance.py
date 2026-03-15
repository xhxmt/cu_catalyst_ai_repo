"""Tests for the cleaning governance layer.

Covers: unit normalisation, provenance validation, target definition
validation, hard-invalid and soft-bound row checks, and the
split_good_review split.
"""

from __future__ import annotations

import pandas as pd
import pytest

from cu_catalyst_ai.clean.governance import flag_rows, split_good_review
from cu_catalyst_ai.clean.normalize_units import normalize_units
from cu_catalyst_ai.clean.provenance_validator import validate_provenance
from cu_catalyst_ai.clean.target_validator import validate_target_definition
from cu_catalyst_ai.clean.validate_conditions import validate_required_columns, validate_rows

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

_VALID_ROW: dict = {
    "catalyst_id": "cu_0001",
    "element": "Cu",
    "facet": "111",
    "adsorbate": "CO",
    "coordination_number": 8.0,
    "avg_neighbor_distance": 2.55,
    "electronegativity": 1.9,
    "d_band_center": -1.6,
    "surface_energy": 1.55,
    "adsorption_energy": -0.6,
    "provenance": "test_db_v1",
    "unit_adsorption_energy": "eV",
    "target_definition": "co_adsorption_energy_ev_v1",
}


def _df(*overrides: dict) -> pd.DataFrame:
    """Build a one-row DataFrame from _VALID_ROW, applying any overrides."""
    row = {**_VALID_ROW}
    for o in overrides:
        row.update(o)
    return pd.DataFrame([row])


# ===========================================================================
# validate_required_columns
# ===========================================================================


def test_validate_required_columns_passes_valid_df() -> None:
    df = _df()
    out = validate_required_columns(df)
    assert len(out) == 1


def test_validate_required_columns_raises_on_missing() -> None:
    df = _df()
    df = df.drop(columns=["provenance"])
    with pytest.raises(ValueError, match="Missing required columns"):
        validate_required_columns(df)


# ===========================================================================
# normalize_units
# ===========================================================================


def test_normalize_units_ev_unchanged() -> None:
    df = _df()  # already eV
    out = normalize_units(df)
    assert abs(out.loc[0, "adsorption_energy"] - _VALID_ROW["adsorption_energy"]) < 1e-9
    assert out.loc[0, "unit_adsorption_energy"] == "eV"
    _, review = split_good_review(out)
    assert len(review) == 0


def test_normalize_units_mev_converted() -> None:
    """meV value should be multiplied by 0.001."""
    df = _df({"adsorption_energy": -600.0, "unit_adsorption_energy": "meV"})
    out = normalize_units(df)
    _, review = split_good_review(out)
    assert len(review) == 0
    assert abs(out.loc[0, "adsorption_energy"] - (-0.6)) < 1e-6
    assert out.loc[0, "unit_adsorption_energy"] == "eV"


def test_normalize_units_kjmol_converted() -> None:
    """kJ/mol value should be multiplied by 0.010364."""
    value_kjmol = -57.91  # ≈ -0.6 eV
    df = _df({"adsorption_energy": value_kjmol, "unit_adsorption_energy": "kJ/mol"})
    out = normalize_units(df)
    _, review = split_good_review(out)
    assert len(review) == 0
    expected = value_kjmol * 0.010364
    assert abs(out.loc[0, "adsorption_energy"] - expected) < 1e-6


def test_normalize_units_unknown_unit_flagged() -> None:
    """Rows with unrecognised units must be sent to review."""
    df = _df({"unit_adsorption_energy": "kcal/mol"})
    out = normalize_units(df)
    _, review = split_good_review(out)
    assert len(review) == 1
    assert "kcal/mol" in review.loc[0, "review_reason"]


# ===========================================================================
# validate_provenance
# ===========================================================================


def test_validate_provenance_accepts_valid() -> None:
    df = _df()
    out = validate_provenance(df)
    _, review = split_good_review(out)
    assert len(review) == 0


def test_validate_provenance_flags_missing() -> None:
    df = _df({"provenance": None})
    out = validate_provenance(df)
    _, review = split_good_review(out)
    assert len(review) == 1


def test_validate_provenance_flags_empty_string() -> None:
    df = _df({"provenance": ""})
    out = validate_provenance(df)
    _, review = split_good_review(out)
    assert len(review) == 1


def test_validate_provenance_flags_whitespace_only() -> None:
    df = _df({"provenance": "   "})
    out = validate_provenance(df)
    _, review = split_good_review(out)
    assert len(review) == 1


# ===========================================================================
# validate_target_definition
# ===========================================================================


def test_validate_target_definition_accepts_matching() -> None:
    df = _df()
    out = validate_target_definition(df, "co_adsorption_energy_ev_v1")
    _, review = split_good_review(out)
    assert len(review) == 0


def test_validate_target_definition_flags_mismatch() -> None:
    df = _df({"target_definition": "some_other_target"})
    out = validate_target_definition(df, "co_adsorption_energy_ev_v1")
    _, review = split_good_review(out)
    assert len(review) == 1


def test_validate_target_definition_flags_missing_column() -> None:
    df = _df()
    df = df.drop(columns=["target_definition"])
    out = validate_target_definition(df, "co_adsorption_energy_ev_v1")
    _, review = split_good_review(out)
    assert len(review) == 1


def test_validate_target_definition_flags_wrong_adsorbate() -> None:
    df = _df({"adsorbate": "H"})
    out = validate_target_definition(df, "co_adsorption_energy_ev_v1", required_adsorbate="CO")
    _, review = split_good_review(out)
    assert len(review) == 1


# ===========================================================================
# validate_rows (hard-invalid and soft bounds)
# ===========================================================================


def test_validate_rows_accepts_valid() -> None:
    df = _df()
    out = validate_rows(df)
    _, review = split_good_review(out)
    assert len(review) == 0


def test_validate_rows_flags_non_positive_neighbor_distance() -> None:
    df = _df({"avg_neighbor_distance": 0.0})
    out = validate_rows(df)
    _, review = split_good_review(out)
    assert len(review) == 1
    assert "avg_neighbor_distance" in review.loc[0, "review_reason"]


def test_validate_rows_flags_negative_coordination_number() -> None:
    df = _df({"coordination_number": -1.0})
    out = validate_rows(df)
    _, review = split_good_review(out)
    assert len(review) == 1


def test_validate_rows_flags_non_numeric_adsorption_energy() -> None:
    df = _df({"adsorption_energy": "N/A"})
    out = validate_rows(df)
    _, review = split_good_review(out)
    assert len(review) == 1
    assert "non-numeric" in review.loc[0, "review_reason"]


def test_validate_rows_flags_large_adsorption_energy() -> None:
    df = _df({"adsorption_energy": -15.0})
    out = validate_rows(df)
    _, review = split_good_review(out)
    assert len(review) == 1


def test_validate_rows_flags_zero_surface_energy() -> None:
    df = _df({"surface_energy": 0.0})
    out = validate_rows(df)
    _, review = split_good_review(out)
    assert len(review) == 1


def test_validate_rows_flags_out_of_range_electronegativity() -> None:
    df = _df({"electronegativity": 5.0})
    out = validate_rows(df)
    _, review = split_good_review(out)
    assert len(review) == 1


# ===========================================================================
# split_good_review (governance helpers)
# ===========================================================================


def test_split_good_review_no_flags() -> None:
    df = _df()
    clean, review = split_good_review(df)
    assert len(clean) == 1
    assert len(review) == 0


def test_split_good_review_all_flagged() -> None:
    df = _df()
    df = flag_rows(df, pd.Series([True]), reason="test", stage="test")
    clean, review = split_good_review(df)
    assert len(clean) == 0
    assert len(review) == 1


def test_split_good_review_mixed() -> None:
    df = pd.concat([_df(), _df({"avg_neighbor_distance": -1.0})], ignore_index=True)
    df = validate_rows(df)
    clean, review = split_good_review(df)
    assert len(clean) == 1
    assert len(review) == 1


# ===========================================================================
# Governance first-check-wins semantics
# ===========================================================================


def test_flag_rows_first_check_wins() -> None:
    """A row flagged by the first check retains the first reason."""
    df = _df()
    df = flag_rows(df, pd.Series([True]), reason="first reason", stage="stage1")
    df = flag_rows(df, pd.Series([True]), reason="second reason", stage="stage2")
    assert df.loc[0, "review_reason"] == "first reason"
    assert df.loc[0, "review_stage"] == "stage1"


# ===========================================================================
# adsorbate missing → goes to review (structural required column)
# ===========================================================================


def test_validate_required_columns_raises_on_missing_adsorbate() -> None:
    """adsorbate is now a structural required column; its absence must raise."""
    df = _df()
    df = df.drop(columns=["adsorbate"])
    with pytest.raises(ValueError, match="Missing required columns"):
        validate_required_columns(df)


# ===========================================================================
# Schema validation → review (validate_schema_rows)
# ===========================================================================


def test_validate_schema_rows_accepts_valid() -> None:
    """A fully valid row must pass schema validation without being flagged."""
    from cu_catalyst_ai.schemas.catalyst import validate_schema_rows  # noqa: PLC0415

    df = _df()
    out = validate_schema_rows(df)
    _, review = split_good_review(out)
    assert len(review) == 0


def test_validate_schema_rows_flags_wrong_element_type() -> None:
    """A row with element='Ag' (not 'Cu') must be flagged by schema validation."""
    from cu_catalyst_ai.schemas.catalyst import validate_schema_rows  # noqa: PLC0415

    df = _df({"element": "Ag"})
    out = validate_schema_rows(df)
    _, review = split_good_review(out)
    assert len(review) == 1
    assert review.loc[0, "review_stage"] == "schema_validation"


def test_validate_schema_rows_flags_non_numeric_coordination() -> None:
    """A row where coordination_number cannot be coerced to float must be flagged."""
    from cu_catalyst_ai.schemas.catalyst import validate_schema_rows  # noqa: PLC0415

    df = _df({"coordination_number": "not_a_float"})
    out = validate_schema_rows(df)
    _, review = split_good_review(out)
    assert len(review) == 1
    assert "schema_validation" in review.loc[0, "review_stage"]


def test_validate_schema_rows_skips_already_flagged() -> None:
    """Rows already flagged by an earlier stage must not be overwritten."""
    from cu_catalyst_ai.schemas.catalyst import validate_schema_rows  # noqa: PLC0415

    df = _df({"element": "Ag"})
    df = flag_rows(df, pd.Series([True]), reason="earlier stage", stage="provenance")
    out = validate_schema_rows(df)
    assert out.loc[0, "review_reason"] == "earlier stage"
    assert out.loc[0, "review_stage"] == "provenance"


# ===========================================================================
# normalize_units driven by target config (unit_conversions override)
# ===========================================================================


def test_normalize_units_uses_custom_conversions() -> None:
    """When unit_conversions is supplied, it overrides the built-in table."""
    from cu_catalyst_ai.clean.normalize_units import normalize_units  # noqa: PLC0415

    # Provide only 'eV' as supported; 'meV' is not in this custom table.
    custom_conversions = {"eV": 1.0}
    df = _df({"adsorption_energy": -600.0, "unit_adsorption_energy": "meV"})
    out = normalize_units(df, unit_conversions=custom_conversions)
    _, review = split_good_review(out)
    # meV is not in custom_conversions → must be flagged for review
    assert len(review) == 1
    assert "meV" in review.loc[0, "review_reason"]


def test_normalize_units_target_config_conversion_factor() -> None:
    """Unit conversion driven by target config dict must apply the correct factor."""
    from cu_catalyst_ai.clean.normalize_units import normalize_units  # noqa: PLC0415

    target_conversions = {"eV": 1.0, "meV": 0.001, "kJ/mol": 0.010364}
    df = _df({"adsorption_energy": -600.0, "unit_adsorption_energy": "meV"})
    out = normalize_units(df, unit_conversions=target_conversions)
    _, review = split_good_review(out)
    assert len(review) == 0
    assert abs(out.loc[0, "adsorption_energy"] - (-0.6)) < 1e-6


# ===========================================================================
# validate_rows driven by target config bounds
# ===========================================================================


def test_validate_rows_respects_custom_abs_max() -> None:
    """A tighter adsorption_energy_abs_max from target config must trigger review."""
    df = _df({"adsorption_energy": -3.0})
    # Default abs_max=10 → clean; custom abs_max=2 → review
    out_default = validate_rows(df, adsorption_energy_abs_max=10.0)
    _, review_default = split_good_review(out_default)
    assert len(review_default) == 0

    out_tight = validate_rows(df, adsorption_energy_abs_max=2.0)
    _, review_tight = split_good_review(out_tight)
    assert len(review_tight) == 1


# ===========================================================================
# clean all-empty → raises RuntimeError (via _run_clean logic)
# ===========================================================================


def test_split_good_review_all_flagged_produces_empty_clean() -> None:
    """When all rows are flagged, clean_df must be empty (prerequisite for RuntimeError)."""
    df = pd.concat([_df({"avg_neighbor_distance": -1.0})] * 3, ignore_index=True)
    df = validate_rows(df)
    clean_df, review_df = split_good_review(df)
    assert len(clean_df) == 0
    assert len(review_df) == 3
