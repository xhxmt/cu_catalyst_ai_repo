"""Unit tests for the element feature lookup table.

Covers:
- Known element returns correct values
- Unknown element returns all-None dict
- enrich_with_element_features injects columns correctly
- Missing element column raises KeyError
- Unknown elements in DataFrame produce NaN rows
"""

from __future__ import annotations

import math

import pandas as pd
import pytest

from cu_catalyst_ai.features.element_features import (
    _FEATURE_KEYS,
    enrich_with_element_features,
    get_element_features,
)

# ---------------------------------------------------------------------------
# get_element_features
# ---------------------------------------------------------------------------


def test_get_element_features_cu_d_band() -> None:
    """Cu d-band center matches Ruban 1997 Table 1 value (-2.67 eV)."""
    feat = get_element_features("Cu")
    assert feat["d_band_center"] is not None
    assert abs(feat["d_band_center"] - (-2.67)) < 0.01


def test_get_element_features_pt() -> None:
    """Pt should have all five features populated."""
    feat = get_element_features("Pt")
    for key in _FEATURE_KEYS:
        assert feat[key] is not None, f"Pt feature '{key}' should not be None"


def test_get_element_features_returns_all_keys() -> None:
    """Return dict must contain exactly the five expected keys."""
    feat = get_element_features("Ni")
    assert set(feat.keys()) == set(_FEATURE_KEYS)


def test_get_element_features_unknown_returns_none_dict() -> None:
    """Unknown element symbol must return all-None values (not raise)."""
    feat = get_element_features("Xx")
    for key in _FEATURE_KEYS:
        assert feat[key] is None, f"Unknown element key '{key}' should be None"


def test_get_element_features_unknown_does_not_raise() -> None:
    """get_element_features must never raise for any string input."""
    feat = get_element_features("")
    assert isinstance(feat, dict)


# ---------------------------------------------------------------------------
# enrich_with_element_features
# ---------------------------------------------------------------------------

_SAMPLE_DF = pd.DataFrame(
    {
        "catalyst_id": ["cu_001", "pt_001", "xx_001"],
        "element": ["Cu", "Pt", "Xx"],
        "adsorption_energy": [-0.6, -1.2, 0.0],
    }
)


def test_enrich_adds_all_feature_columns() -> None:
    """All five element feature columns must be present after enrichment."""
    out = enrich_with_element_features(_SAMPLE_DF.copy())
    for key in _FEATURE_KEYS:
        assert key in out.columns, f"Column '{key}' missing after enrichment"


def test_enrich_cu_d_band_correct() -> None:
    """Cu d_band_center must match expected value after enrichment."""
    out = enrich_with_element_features(_SAMPLE_DF.copy())
    cu_row = out[out["element"] == "Cu"].iloc[0]
    assert abs(cu_row["d_band_center"] - (-2.67)) < 0.01


def test_enrich_unknown_element_is_nan() -> None:
    """Unknown element 'Xx' must produce NaN for all feature columns."""
    out = enrich_with_element_features(_SAMPLE_DF.copy())
    xx_row = out[out["element"] == "Xx"].iloc[0]
    for key in _FEATURE_KEYS:
        assert math.isnan(xx_row[key]), f"Unknown element column '{key}' should be NaN"


def test_enrich_preserves_row_count() -> None:
    """Row count must not change after enrichment."""
    out = enrich_with_element_features(_SAMPLE_DF.copy())
    assert len(out) == len(_SAMPLE_DF)


def test_enrich_missing_element_column_raises() -> None:
    """DataFrame without 'element' column must raise KeyError."""
    df = pd.DataFrame({"catalyst_id": ["x"], "adsorption_energy": [0.0]})
    with pytest.raises(KeyError):
        enrich_with_element_features(df)


def test_enrich_does_not_mutate_input() -> None:
    """enrich_with_element_features must return a copy, not mutate the input."""
    original = _SAMPLE_DF.copy()
    enrich_with_element_features(original)
    assert "d_band_center" not in original.columns
