import math

import pandas as pd

from cu_catalyst_ai.clean.split_registry import assign_splits
from cu_catalyst_ai.dataio.mp_fetch import generate_demo_dataset
from cu_catalyst_ai.features.basic_features import (
    add_gcn,
    add_proxy_cn,
    add_surface_dband,
    build_feature_table,
)
from cu_catalyst_ai.features.structural_features import add_structural_ratios


def test_feature_table_contains_expected_columns() -> None:
    df = generate_demo_dataset(n_samples=20, seed=42)
    df = assign_splits(df, seed=42)
    df = add_structural_ratios(df)
    features = build_feature_table(
        df,
        use_columns=[
            "coordination_number",
            "avg_neighbor_distance",
            "electronegativity",
            "d_band_center",
            "surface_energy",
            "coordination_to_distance",
            "facet",
        ],
        categorical_columns=["facet"],
    )
    assert "facet_111" in features.columns
    assert "coordination_to_distance" in features.columns


def test_add_proxy_cn_maps_known_facets() -> None:
    """GCN values follow Calle-Vallejo 2015 for known facets; NaN and unknown use default."""
    df = pd.DataFrame({"facet": ["111", "310", None, "unknown"]})
    out = add_proxy_cn(df)
    assert "proxy_cn" in out.columns
    assert list(out["proxy_cn"]) == [7.5, 4.4, 6.0, 6.0], out["proxy_cn"].tolist()


def test_add_proxy_cn_noop_without_facet() -> None:
    """add_proxy_cn is a noop when the 'facet' column is absent."""
    df = pd.DataFrame({"electronegativity": [1.9, 2.0]})
    out = add_proxy_cn(df)
    assert "proxy_cn" not in out.columns
    # Original data unchanged
    assert list(out.columns) == ["electronegativity"]


def test_add_gcn_maps_known_facets() -> None:
    """add_gcn() returns Calle-Vallejo 2015 float values for known facets."""
    df = pd.DataFrame({"facet": ["111", "100", "211", "110", "310", None, "unknown"]})
    out = add_gcn(df)
    assert "gcn" in out.columns, "gcn column must be present"
    expected = [7.5, 6.7, 5.3, 6.0, 4.4, 6.0, 6.0]
    assert list(out["gcn"]) == expected, f"Got {out['gcn'].tolist()}"


def test_add_gcn_independent_of_proxy_cn() -> None:
    """add_gcn() writes to 'gcn' without touching 'proxy_cn'."""
    df = pd.DataFrame({"facet": ["111", "211"]})
    df = add_proxy_cn(df)
    df = add_gcn(df)
    assert "proxy_cn" in df.columns
    assert "gcn" in df.columns
    assert list(df["proxy_cn"]) == list(df["gcn"])


def test_add_gcn_noop_without_facet() -> None:
    """add_gcn() is a noop when 'facet' column is absent."""
    df = pd.DataFrame({"electronegativity": [1.8, 2.2]})
    out = add_gcn(df)
    assert "gcn" not in out.columns
    assert list(out.columns) == ["electronegativity"]


# ---------------------------------------------------------------------------
# Tests for add_surface_dband  (H-group feature)
# ---------------------------------------------------------------------------


def test_surface_dband_cu111_exact_value() -> None:
    """Cu(111) must return the literature value -2.67 eV (Ruban 1997)."""
    df = pd.DataFrame({"element": ["Cu"], "facet": ["111"], "d_band_center": [-2.67]})
    out = add_surface_dband(df)
    assert "surface_d_band_center" in out.columns
    assert abs(out["surface_d_band_center"].iloc[0] - (-2.67)) < 1e-9


def test_surface_dband_cu_all_facets() -> None:
    """All five Cu facets should return their defined values without fallback."""
    expected = {
        "111": -2.67,
        "100": -2.64,
        "211": -2.60,
        "310": -2.58,
        "511": -2.56,
    }
    df = pd.DataFrame(
        {
            "element": ["Cu"] * 5,
            "facet": list(expected.keys()),
            "d_band_center": [-2.67] * 5,
        }
    )
    out = add_surface_dband(df)
    for facet, val in expected.items():
        row = out[out["facet"] == facet].iloc[0]
        got = row["surface_d_band_center"]
        assert abs(got - val) < 1e-9, f"Cu({facet}): expected {val}, got {got}"


def test_surface_dband_supercell_aliases_map_to_111() -> None:
    """111-(NxN) supercell labels must resolve to the same value as 111."""
    df = pd.DataFrame(
        {
            "element": ["Cu", "Cu", "Cu", "Cu"],
            "facet": ["111", "111-(1x1)", "111-(2x2)", "111-(4x4)"],
            "d_band_center": [-2.67] * 4,
        }
    )
    out = add_surface_dband(df)
    vals = out["surface_d_band_center"].tolist()
    assert all(abs(v - (-2.67)) < 1e-9 for v in vals), f"supercell alias mismatch: {vals}"


def test_surface_dband_unknown_facet_falls_back_to_111() -> None:
    """Unknown facet for a metal in the table should fallback to metal's 111 value."""
    df = pd.DataFrame(
        {
            "element": ["Pd"],
            "facet": ["999"],  # not in table
            "d_band_center": [-1.83],
        }
    )
    out = add_surface_dband(df)
    # Pd(111) = -1.83
    assert abs(out["surface_d_band_center"].iloc[0] - (-1.83)) < 1e-9


def test_surface_dband_unknown_metal_returns_nan() -> None:
    """Unknown metal with no d_band_center should return NaN."""
    df = pd.DataFrame(
        {
            "element": ["XX"],
            "facet": ["111"],
            "d_band_center": [float("nan")],
        }
    )
    out = add_surface_dband(df)
    assert math.isnan(out["surface_d_band_center"].iloc[0])


def test_surface_dband_noop_without_required_columns() -> None:
    """add_surface_dband is a noop when 'element' or 'facet' columns are absent."""
    df_no_facet = pd.DataFrame({"element": ["Cu"], "d_band_center": [-2.67]})
    out = add_surface_dband(df_no_facet)
    assert "surface_d_band_center" not in out.columns

    df_no_element = pd.DataFrame({"facet": ["111"], "d_band_center": [-2.67]})
    out2 = add_surface_dband(df_no_element)
    assert "surface_d_band_center" not in out2.columns


def test_surface_dband_does_not_modify_d_band_center() -> None:
    """add_surface_dband must never overwrite the existing d_band_center column."""
    original_val = -2.67
    df = pd.DataFrame(
        {
            "element": ["Cu"],
            "facet": ["211"],
            "d_band_center": [original_val],
        }
    )
    out = add_surface_dband(df)
    assert out["d_band_center"].iloc[0] == original_val
    # Cu(211) surface value differs from bulk (-2.67 vs -2.60)
    assert abs(out["surface_d_band_center"].iloc[0] - (-2.60)) < 1e-9
